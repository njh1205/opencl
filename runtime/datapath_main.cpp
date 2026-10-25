// datapath_bench: per-symbol PL datapath -- the fp32 OpenCL kernels against the
// fp64 numpy oracle of tools/datapath_ref.py (which cross-checks its own
// conventions against the production modem in mix_precode/classical).
//
//   bits --qam_map--> s --tx_combine--> x --channel_apply--> y
//        --rx_combine(/diag_b)--> z --+--softdemod_llr--> llr --gmi--> GMI
//                                     `--modulo_fold--> zf
//                                          `--softdemod_folded--> llr (THP)
//
// The chain is driven through hpocl::Runtime, i.e. through exactly the wrappers
// a PS/PL deployment calls, so this gate also proves the datapath kernels are
// reachable from the shared runtime (they used to be reachable only from a
// private mini-runtime in this file, hence absent from the .xclbin path).
// Field order mirrors tools/datapath_ref.py::field_list.
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>
#include "hpocl_runtime.h"
#include "mcs_table.clh"   // host side of cqi_select: the integration check below

using cd = std::complex<double>;        // oracle precision
using cx = hpocl::cplx;                 // device precision (== cl float2)

namespace {
struct Reader {                               // little-endian, bounds-checked
  std::vector<uint8_t> buf; size_t o = 0; bool bad = false;
  explicit Reader(const char* path) {
    std::ifstream f(path, std::ios::binary);
    buf.assign((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
  }
  const uint8_t* take(size_t n) {
    if (o + n > buf.size()) { bad = true; return buf.data(); }
    const uint8_t* p = buf.data() + o; o += n; return p;
  }
  int i() { int v = 0; memcpy(&v, take(4), 4); return v; }
  double d() { double v = 0; memcpy(&v, take(8), 8); return v; }
  void u8(std::vector<uint8_t>& v) { memcpy(v.data(), take(v.size()), v.size()); }
  void cx_(std::vector<cd>& v) {
    const uint8_t* p = take(16 * v.size());
    for (cd& e : v) { double re, im; memcpy(&re, p, 8); memcpy(&im, p + 8, 8); p += 16; e = cd(re, im); }
  }
  void re(std::vector<double>& v) { memcpy(v.data(), take(8 * v.size()), 8 * v.size()); }
};

std::vector<cx> pack(const std::vector<cd>& v) {   // fp64 oracle -> fp32 device
  std::vector<cx> o(v.size());
  for (size_t i = 0; i < v.size(); ++i) o[i] = cx((float)v[i].real(), (float)v[i].imag());
  return o;
}
double err_c(const std::vector<cx>& got, const std::vector<cd>& exp) {
  double e = 0;
  for (size_t i = 0; i < exp.size(); ++i) e = fmax(e, std::abs(cd(got[i]) - exp[i]));
  return e;
}
template <class E>                       // E = double (oracle) or float (cl)
double err_f(const float* got, const E* exp, size_t n) {
  double e = 0;
  for (size_t i = 0; i < n; ++i) e = fmax(e, fabs((double)got[i] - (double)exp[i]));
  return e;
}
}  // namespace

int main(int argc, char** argv) {
  Reader rd(argc > 1 ? argv[1] : "build/datapath_vectors.bin");
  hpocl::Runtime rt;
  const char* dev = argc > 2 ? argv[2] : "cpu";
  // HPOCL_CLOPTS forwards extra -D macros to the device compiler, so a PL-style
  // build can be gated without editing this file: HPOCL_CLOPTS=-DLLR_CACHE_MAX=4
  // makes every m >= 6 case take softdemod_llr_k's recompute fallback.
  const char* copts = getenv("HPOCL_CLOPTS");
  // rate.cl rides along so the chain can end where a real link ends: per-stream
  // GMI -> the highest code rate that GMI supports.
  if (!rt.init(dev) ||
      !rt.load_program({"kernels/datapath.cl", "kernels/rate.cl"}, copts)) {
    fprintf(stderr, "[datapath] OpenCL init/build failed\n");
    return 2;
  }
  printf("[datapath_bench] OpenCL device: %s (fp32%s%s)\n", rt.device_name(),
         copts ? ", opts " : "", copts ? copts : "");

  const int grp = 2;                          // bits per demod work-item
  int ncase = rd.i(), fails = 0;
  for (int ci = 0; ci < ncase; ++ci) {
    int Nt = rd.i(), Nr = rd.i(), NRF = rd.i(), U = rd.i(), d = rd.i(),
        nsym = rd.i(), m = rd.i(), radius = rd.i(), has_diag = rd.i(),
        nedge = rd.i();
    double n0 = rd.d(), period = rd.d();
    const int T = U * d, ng = (m + grp - 1) / grp;
    const size_t nsymtot = (size_t)T * nsym, nllr = nsymtot * m,
                 nf = nsymtot + nedge, nllf = nf * m;

    std::vector<uint8_t> bits(nllr);
    std::vector<cd> Frf((size_t)Nt * NRF), Fbb((size_t)NRF * T), H((size_t)Nr * Nt),
        n((size_t)Nr * nsym), W((size_t)U * Nr * d), diag(T), es(nsymtot),
        ex((size_t)Nt * nsym), ey((size_t)Nr * nsym), ez(nsymtot), ezfin(nf), ezf(nf);
    std::vector<double> ellr(nllr), ellf(nllf), egmi(T), egmf(T);
    rd.u8(bits); rd.cx_(Frf); rd.cx_(Fbb); rd.cx_(H); rd.cx_(n); rd.cx_(W); rd.cx_(diag);
    rd.cx_(es); rd.cx_(ex); rd.cx_(ey); rd.cx_(ez); rd.cx_(ezfin); rd.cx_(ezf);
    rd.re(ellr); rd.re(ellf); rd.re(egmi); rd.re(egmf);
    if (rd.bad) { fprintf(stderr, "[datapath] truncated vectors file\n"); return 2; }

    std::vector<cx> cFrf = pack(Frf), cFbb = pack(Fbb), cH = pack(H), cn = pack(n),
                    cW = pack(W), cdiag = pack(diag), czfin = pack(ezfin);
    std::vector<cx> gs(nsymtot), gx((size_t)Nt * nsym), gy((size_t)Nr * nsym),
                    gz(nsymtot), gzf(nf);
    std::vector<float> gllr(nllr), gllf(nllf), gll0(nllr), ggmi(T), ggmf(T);

    // the whole chain, in order, through the deployed Runtime wrappers
    rt.qam_map(bits.data(), m, (int)nsymtot, gs.data());
    rt.tx_combine(cFrf.data(), Nt, NRF, cFbb.data(), T, gs.data(), nsym, gx.data());
    rt.channel_apply(cH.data(), Nr, Nt, gx.data(), nsym, cn.data(), gy.data());
    rt.rx_combine(cW.data(), U, Nr, d, gy.data(), nsym,
                  has_diag ? cdiag.data() : nullptr, gz.data());
    rt.softdemod_llr(gz.data(), (int)nsymtot, m, grp, (float)n0, gllr.data());
    rt.modulo_fold(czfin.data(), (int)nf, (float)period, gzf.data());
    rt.softdemod_folded(gzf.data(), (int)nf, m, grp, radius, (float)period,
                        (float)n0, gllf.data());
    rt.softdemod_folded(gz.data(), (int)nsymtot, m, grp, 0, (float)period,
                        (float)n0, gll0.data());   // radius 0 == parallel branch
    rt.gmi(gllr.data(), bits.data(), m, nsym, T, ggmi.data());
    rt.gmi(gllf.data(), bits.data(), m, nsym, T, ggmf.data());
    (void)ng;

    // Last hop of the chain: per-stream GMI -> CQI index / code rate. rate_bench
    // already gates mcs_select exhaustively against an independent numpy oracle
    // (10 cases, every table boundary, exact agreement), so what is asserted HERE
    // is the integration: the device selection must agree exactly with the host
    // side of the very same mcs_table.clh when both are fed the GMI the kernel
    // just produced. Sum SE is the bits/s/Hz a scheduler would report.
    const float kEps = 1e-5f;
    std::vector<hpocl::McsSel> sel_par(T), sel_fold(T);
    rt.mcs_select(ggmi.data(), T, kEps, sel_par.data());
    rt.mcs_select(ggmf.data(), T, kEps, sel_fold.data());
    int mcs_mism = 0, out_par = 0; double se_par = 0, se_fold = 0;
    for (int k = 0; k < T; ++k) {
      mcs_mism += (sel_par[k].index != cqi_select(ggmi[k], kEps)) +
                  (sel_fold[k].index != cqi_select(ggmf[k], kEps));
      se_par += sel_par[k].se; se_fold += sel_fold[k].se;
      out_par += sel_par[k].outage ? 1 : 0;
    }

    double eq = err_c(gs, es), etx = err_c(gx, ex), ech = err_c(gy, ey),
           erx = err_c(gz, ez), efo = err_c(gzf, ezf),
           el = err_f(gllr.data(), ellr.data(), nllr),
           elf = err_f(gllf.data(), ellf.data(), nllf),
           e0 = err_f(gll0.data(), gllr.data(), nllr),   // fp32 vs fp32
           eg = err_f(ggmi.data(), egmi.data(), T),
           egf = err_f(ggmf.data(), egmf.data(), T);
    double gm = 0, gmf = 0;
    for (int k = 0; k < T; ++k) { gm += egmi[k]; gmf += egmf[k]; }
    // Bounds, fp32 vs the fp64 oracle: 1e-4 for the linear stages (qam, tx,
    // channel, rx incl. the diag_b divide, modulo fold) and 1e-3 for the
    // parallel LLR / GMI as before. The folded branch gets 3e-3: its LLR sums
    // (2r+1)^2*M candidates, so both the subset-minimum search and the exp()
    // accumulation run ~25x longer in fp32 (measured ~2x the parallel error),
    // and gmi_fold inherits it through m bits. e0 is fp32 vs fp32 -- radius 0
    // must reproduce softdemod_llr_k -- hence the ULP-scale 1e-5.
    bool ok = eq < 1e-4 && etx < 1e-4 && ech < 1e-4 && erx < 1e-4 &&
              efo < 1e-4 && el < 1e-3 && elf < 3e-3 && e0 < 1e-5 && eg < 1e-3 &&
              egf < 3e-3 && mcs_mism == 0;
    fails += !ok;
    printf("case%d Nt=%d Nr=%d NRF=%d U=%d d=%d m=%d r=%d diag=%c snr=%4.1fdB: "
           "qam=%.1e tx=%.1e ch=%.1e rx=%.1e fold=%.1e llr=%.1e llrF=%.1e "
           "r0=%.1e gmi=%.1e gmiF=%.1e mcs=%d (GMI %.2f / folded %.2f of %d, "
           "SE %.2f/%.2f bpcu, outage %d/%d) %s\n",
           ci, Nt, Nr, NRF, U, d, m, radius, has_diag ? 'Y' : 'N', -10 * log10(n0),
           eq, etx, ech, erx, efo, el, elf, e0, eg, egf, mcs_mism, gm / T,
           gmf / T, m, se_par, se_fold, out_par, T, ok ? "PASS" : "FAIL");
  }
  if (rd.o != rd.buf.size()) { fprintf(stderr, "[datapath] vector file size mismatch\n"); ++fails; }
  printf(fails ? "[datapath_bench] FAIL\n" : "[datapath_bench] ALL PASS\n");
  return fails ? 1 : 0;
}
