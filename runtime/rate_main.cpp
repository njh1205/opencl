// rate_bench: the code-rate / MCS layer, three ways at once.
//
//   numpy oracle (tools/rate_ref.py, table retyped from 3GPP TS 38.214)
//     vs  host C++ (kernels/mcs_table.clh, compiled as C++ through -Ikernels)
//     vs  fp32 OpenCL kernel (kernels/rate.cl, the same .clh through the
//                              Runtime's include inliner)
//
// The host and the device compile the SAME header text, so the host column is
// not an independent implementation -- it is the proof that the two language
// front ends agree on it (the point of the .clh, exactly as for switch_law.clh).
// The numpy column is the independent one, and it is the one that catches a
// mis-typed table entry.
//
// What this layer is for: yang_unified_transceiver (the MART paper) is a
// MIMO-BICM model whose achievable-rate bound is evaluated for a given
// modulation and coding scheme, so the code rate is an outer parameter rather
// than something the precoder optimises. kernels/datapath.cl::gmi_k produces
// the per-stream GMI; this operator turns it into the highest (m, R) the link
// can actually carry, plus the margin an outer scheduler needs.
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>
#include "hpocl_runtime.h"
#include "mcs_table.clh"   // host side of the device table: cqi_select/cqi_se

using hpocl::McsSel;
using hpocl::Runtime;

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
  void i32(std::vector<int>& v) { memcpy(v.data(), take(4 * v.size()), 4 * v.size()); }
  void f32(std::vector<float>& v) { memcpy(v.data(), take(4 * v.size()), 4 * v.size()); }
  void f64(std::vector<double>& v) { memcpy(v.data(), take(8 * v.size()), 8 * v.size()); }
  void u8(std::vector<uint8_t>& v) { memcpy(v.data(), take(v.size()), v.size()); }
};

int maxdiff_i(const std::vector<int>& a, const std::vector<int>& b) {
  int m = 0;
  for (size_t i = 0; i < a.size(); ++i) m = std::max(m, std::abs(a[i] - b[i]));
  return m;
}
// A non-finite difference returns HUGE_VAL rather than being swallowed: fmax
// ignores a NaN operand, so `fmax(m, NaN)` would silently report the previous
// (finite) worst case and pass a case that produced garbage.
double maxdiff_d(const std::vector<float>& got, const std::vector<double>& exp) {
  double m = 0;
  for (size_t i = 0; i < exp.size(); ++i) {
    const double e = fabs((double)got[i] - exp[i]);
    if (!std::isfinite(e)) return HUGE_VAL;
    m = fmax(m, e);
  }
  return m;
}
}  // namespace

int main(int argc, char** argv) {
  Reader rd(argc > 1 ? argv[1] : "build/rate_vectors.bin");
  Runtime rt;
  const char* dev = argc > 2 ? argv[2] : "cpu";
  if (!rt.init(dev) || !rt.load_program({"kernels/rate.cl"})) {
    fprintf(stderr, "[rate] OpenCL init/build failed\n");
    return 2;
  }
  printf("[rate_bench] OpenCL device: %s (fp32)\n", rt.device_name());
  printf("[rate_bench] CQI table: %d entries, SE %.4f..%.4f bits/channel use\n",
         HPOCL_CQI_N - 1, cqi_se(1), cqi_se(HPOCL_CQI_N - 1));

  int ncase = rd.i(), fails = 0;
  double worst_cl = 0, worst_host = 0;
  for (int ci = 0; ci < ncase; ++ci) {
    const int K = rd.i();
    const float eps = (float)rd.d();
    std::vector<float> gmi(K);
    std::vector<int> eidx(K), em(K), er(K);
    std::vector<double> ese(K), emarg(K), einfo(K);
    std::vector<uint8_t> eout(K);
    rd.f32(gmi); rd.i32(eidx); rd.i32(em); rd.i32(er);
    rd.f64(ese); rd.f64(emarg); rd.f64(einfo); rd.u8(eout);
    if (rd.bad) { fprintf(stderr, "[rate] truncated vectors file\n"); return 2; }

    // ---- host side: the very same mcs_table.clh text the device compiles ----
    std::vector<int> hidx(K), hm(K), hr(K);
    std::vector<float> hse(K), hmarg(K), hinfo(K);
    std::vector<uint8_t> hout(K);
    for (int k = 0; k < K; ++k) {
      const int j = cqi_select(gmi[k], eps);
      hidx[k] = j;
      hout[k] = j < 0 ? 1 : 0;
      hm[k] = j < 0 ? 0 : kCqiM[j];
      hr[k] = j < 0 ? 0 : kCqiR[j];
      hse[k] = j < 0 ? 0.0f : cqi_se(j);
      hmarg[k] = j < 0 ? 0.0f : gmi[k] - hse[k];
      hinfo[k] = cqi_info_rate(gmi[k], j);
    }

    // ---- device side, through the deployed Runtime wrapper -----------------
    std::vector<McsSel> sel(K);
    rt.mcs_select(gmi.data(), K, eps, sel.data());
    std::vector<int> cidx(K), cm(K), cr(K);
    std::vector<float> cse(K), cmarg(K), cinfo(K);
    std::vector<uint8_t> cout_(K);
    for (int k = 0; k < K; ++k) {
      cidx[k] = sel[k].index; cm[k] = sel[k].m; cr[k] = sel[k].rate;
      cse[k] = sel[k].se; cmarg[k] = sel[k].margin; cinfo[k] = sel[k].info;
      cout_[k] = sel[k].outage ? 1 : 0;
    }

    // The discrete columns must match EXACTLY: an off-by-one in the table or a
    // flipped comparison direction is a wrong MCS, not a rounding difference.
    int d_idx_h = maxdiff_i(hidx, eidx), d_m_h = maxdiff_i(hm, em),
        d_r_h = maxdiff_i(hr, er);
    int d_idx_c = maxdiff_i(cidx, eidx), d_m_c = maxdiff_i(cm, em),
        d_r_c = maxdiff_i(cr, er);
    int d_out = 0;
    for (int k = 0; k < K; ++k) d_out += (cout_[k] != eout[k]) + (hout[k] != eout[k]);
    // The continuous columns are fp32 vs fp64: SE is exact (integer / 1024),
    // margin and info carry one fp32 subtraction, so 1e-5 is generous.
    double e_se = maxdiff_d(cse, ese), e_mg = maxdiff_d(cmarg, emarg),
           e_in = maxdiff_d(cinfo, einfo);
    double h_se = maxdiff_d(hse, ese), h_mg = maxdiff_d(hmarg, emarg),
           h_in = maxdiff_d(hinfo, einfo);
    double e_cl = fmax(e_se, fmax(e_mg, e_in));
    double e_hs = fmax(h_se, fmax(h_mg, h_in));
    worst_cl = fmax(worst_cl, e_cl);
    worst_host = fmax(worst_host, e_hs);

    double sum_se = 0; int nout = 0;
    for (int k = 0; k < K; ++k) { sum_se += ese[k]; nout += eout[k]; }
    bool ok = d_idx_h == 0 && d_m_h == 0 && d_r_h == 0 &&
              d_idx_c == 0 && d_m_c == 0 && d_r_c == 0 && d_out == 0 &&
              e_cl < 1e-5 && e_hs < 1e-5;
    fails += !ok;
    printf("case%d K=%-3d eps=%-6.0e: idx(h/cl) %d/%d  m %d/%d  R %d/%d  "
           "out_mism %d | se=%.1e marg=%.1e info=%.1e | host %.1e | "
           "sum SE %.3f bpcu, outage %d/%d %s\n",
           ci, K, eps, d_idx_h, d_idx_c, d_m_h, d_m_c, d_r_h, d_r_c, d_out,
           e_se, e_mg, e_in, e_hs, sum_se, nout, K, ok ? "PASS" : "FAIL");
  }
  if (rd.o != rd.buf.size()) {
    fprintf(stderr, "[rate] %zu trailing bytes (contract drift?)\n",
            rd.buf.size() - rd.o);
    ++fails;
  }
  printf("[rate_bench] worst kernel-vs-oracle %.2e, host-vs-oracle %.2e\n",
         worst_cl, worst_host);
  printf(fails ? "[rate_bench] FAIL\n" : "[rate_bench] ALL PASS\n");
  return fails ? 1 : 0;
}
