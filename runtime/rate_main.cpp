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
#include <vector>
#include "hpocl_runtime.h"
#include "hpocl_vecio.h"
#include "mcs_table.clh"   // host side of the device table: cqi_select/cqi_se

using hpocl::McsSel;
using hpocl::Runtime;

namespace {
// hpocl::VecReader (include/hpocl_vecio.h) is the shared bounds-checked reader;
// it names the float64/complex128/uint8 shapes but not the two integer-width
// columns this vector file carries, so those two go through its raw() entry
// point. Same memcpy, same bounds check, no second reader implementation.
void rd_f32(hpocl::VecReader& r, std::vector<float>& v) {
  r.raw(v.data(), sizeof(float) * v.size(), "float32[]");
}
void rd_i32(hpocl::VecReader& r, std::vector<int>& v) {
  r.raw(v.data(), sizeof(int32_t) * v.size(), "int32[]");
}

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
  const char* vec = argc > 1 ? argv[1] : "build/rate_vectors.bin";
  hpocl::VecReader rd(vec);
  if (!rd.ok()) {
    fprintf(stderr, "[rate] %s (run: python3 tools/rate_ref.py)\n", rd.error().c_str());
    return 2;
  }
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
    const float eps = (float)rd.f();
    // Pre-flight: K sizes the eight vectors below and comes from the file, and
    // every vector file in this repo is argv[1]. A wrong file reads a garbage K
    // and asks for gigabytes before any bounds check can fire (VecReader checks
    // each read, which is too late); take() validates without consuming.
    if (K < 1 || K > 65536) {
      fprintf(stderr, "[rate] case%d: implausible K=%d -- wrong vector file? "
              "(run: python3 tools/rate_ref.py)\n", ci, K);
      return 2;
    }
    if (!rd.take(12 * (size_t)K + 24 * (size_t)K + (size_t)K, "rate case"))
      return 2;
    std::vector<float> gmi(K);
    std::vector<int> eidx(K), em(K), er(K);
    std::vector<double> ese(K), emarg(K), einfo(K);
    std::vector<uint8_t> eout(K);
    rd_f32(rd, gmi); rd_i32(rd, eidx); rd_i32(rd, em); rd_i32(rd, er);
    rd.re(ese); rd.re(emarg); rd.re(einfo); rd.by(eout);
    if (rd.bad()) return 2;              // VecReader already printed the offset

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
  if (!rd.expect_done("rate_bench")) ++fails;   // contract drift vs the oracle
  printf("[rate_bench] worst kernel-vs-oracle %.2e, host-vs-oracle %.2e\n",
         worst_cl, worst_host);
  printf(fails ? "[rate_bench] FAIL\n" : "[rate_bench] ALL PASS\n");
  return fails ? 1 : 0;
}
