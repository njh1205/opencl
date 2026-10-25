// op_bench: validates every data-path operator of the stack against the CPU
// golden reference (ref/hp_ref.cpp) and prints PL-style profiling numbers.
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <random>
#include <vector>
#include "hpocl_runtime.h"
#include "hp_ref.h"

using namespace hpocl;
static std::mt19937 rng(1234);
static float rnd() { return std::uniform_real_distribution<float>(-1.f, 1.f)(rng); }
static std::vector<cplx> rc(size_t n, float s = 1.f) {
  std::vector<cplx> v(n);
  for (auto& x : v) x = cplx(rnd() * s, rnd() * s);
  return v;
}
static double frob(const std::vector<cplx>& v) {   // Frobenius norm, for the
  double s = 0;                                    // relative gates below
  for (auto& x : v) s += std::norm(x);
  return std::sqrt(s);
}
template <typename T>
static double maxerr(const std::vector<T>& a, const std::vector<T>& b) {
  double m = 0;
  for (size_t i = 0; i < a.size(); ++i) m = fmax(m, std::abs(a[i] - b[i]));
  return m;
}

int main(int argc, char** argv) {
  const char* pref = argc > 1 ? argv[1] : "cpu";
  Runtime rt;
  if (!rt.init(pref)) { fprintf(stderr, "opencl init failed\n"); return 2; }
  // HPOCL_CLOPTS forwards extra -D macros to the device compiler so a PL-style
  // build can be gated without editing this file: `make op_pl_bench` uses it to
  // raise WBB_LOCAL_MAX and exercise wbb_apply's local-memory tile.
  const char* copts = getenv("HPOCL_CLOPTS");
  if (!rt.load_program({"kernels/channel_cov.cl", "kernels/power_iter.cl",
                 "kernels/switch_threshold.cl", "kernels/wbb_apply.cl",
                 "kernels/thp_precoder.cl"}, copts)) return 2;
  printf("[op_bench] device: %s%s%s\n", rt.device_name(),
         copts ? ", opts " : "", copts ? copts : "");
  int fails = 0;
  const int M = 16, N = 16, S = 32, SC = 64, K = 8;

  // A hybrid array is rectangular by construction (N_RF chains x N_t antennas,
  // N_r receive antennas x N_t transmit), so the two matrix-shaped operators are
  // swept over tall and wide shapes as well as square ones. A transposed index
  // or a hidden M == N assumption shows up here and nowhere else in the suite.
  struct Shape { int m, n; const char* tag; };
  static const Shape kShapes[] = {{16, 16, "square"}, {8, 16, "tall"},
                                  {16, 8, "wide"}, {4, 12, "tall"},
                                  {12, 4, "wide"}, {1, 8, "1 row"},
                                  {8, 1, "1 col"}};
  { // channel_cov : R = (1/S) sum_s H_s H_s^H,  H [S][m][n] -> R [m][m]
    // RELATIVE gate: ||R||_F grows like sqrt(S)*m*n, so an absolute bound
    // measures the size of the input rather than the quality of the fp32
    // accumulation (M=16,N=64,S=32 reaches 1.14e-4 absolute at 4.4e-7 relative).
    double worst = 0; const Shape* ws = kShapes;
    for (const Shape& sh : kShapes) {
      auto H = rc((size_t)S * sh.m * sh.n);
      std::vector<cplx> R1, R2((size_t)sh.m * sh.m);
      hpref::channel_cov(H, sh.m, sh.n, S, R1);
      rt.channel_cov(H.data(), sh.m, sh.n, S, R2.data());
      double e = maxerr(R1, R2) / fmax(1.0, frob(R1));
      if (e > worst) { worst = e; ws = &sh; }
    }
    bool ok = worst < 1e-6; fails += !ok;
    printf("channel_cov   rel_err=%.2e %s (%d shapes, worst %dx%d %s, last %.1fus)\n",
           worst, ok ? "PASS" : "FAIL", (int)(sizeof kShapes / sizeof *kShapes),
           ws->m, ws->n, ws->tag, rt.last_us());
  }
  { // power_iter + deflate
    auto H = rc((size_t)S * M * N);
    std::vector<cplx> R; hpref::channel_cov(H, M, N, S, R);
    std::vector<cplx> v1, v2(M); float l1 = 0, l2 = 0;
    hpref::power_iter(R, M, 24, v1, l1);
    rt.power_iter(R.data(), M, 24, 1, v2.data(), &l2);
    double e = fmax(fabs(l1 - l2), maxerr(v1, v2)); bool ok = e < 1e-3;
    fails += !ok;
    std::vector<cplx> R2 = R;
    hpref::deflate(R, M, v1, l1);
    rt.deflate(R2.data(), M, 1, v2.data(), &l2);
    double e2 = maxerr(R, R2); ok = ok && e2 < 1e-3; fails += !(e2 < 1e-3);
    printf("power_iter    err=%.2e deflate_err=%.2e %s (%.1fus)\n", e, e2,
           ok ? "PASS" : "FAIL", rt.last_us());
  }
  { // capacity guard: M > PMAX must refuse loudly and NaN the outputs instead
    // of overrunning the kernel's __local scratch (kernels/power_iter.cl sizes
    // lv/lw at PMAX and the launch uses local size == M). The [hpocl]
    // power_iter line this prints on stderr IS the expected behaviour.
    const int MB = 65;                                   // PMAX + 1
    std::vector<cplx> R((size_t)MB * MB, cplx(0, 0));
    std::vector<cplx> v((size_t)MB, cplx(7.f, 7.f));     // sentinel, not NaN
    float l = 7.f;
    rt.power_iter(R.data(), MB, 4, 1, v.data(), &l);
    bool nan_all = std::isnan(l) && std::isnan(v[0].real()) &&
                   std::isnan(v[MB - 1].imag());
    bool ok = nan_all; fails += !ok;
    printf("power_iter cap M=%d>PMAX: %s\n", MB,
           ok ? "PASS (refused, outputs NaN)" : "FAIL (stale outputs)");
  }
  { // switch_threshold policy op (surrogate law, kernels/switch_law.clh)
    // nt/nrf and eta used to be 16/16 and 1.0, i.e. both factors the law carries
    // were multiplied by exactly 1 and never exercised. 64 antennas over 8 RF
    // chains gives nt/nrf = 8, and eta = 1 + (d-1)/(d(M-1)) at d=4, M=64 is the
    // main.tex value for a real configuration.
    const float nt = 64.f, nrf = 8.f, eta = 1.0119f;
    int C = 256;
    std::vector<float> lm(C), gm(C), md(C), t1(C), t2(C);
    std::vector<uint8_t> m1(C), m2(C);
    for (int i = 0; i < C; ++i) {
      lm[i] = 0.5f + fabs(rnd()); gm[i] = 0.5f + fabs(rnd()); md[i] = 20.f * rnd();
    }
    // degenerate rows: a rank-deficient channel drives g_min to 0, which is the
    // HPOCL_GMIN_FLOOR path (tau must stay finite, not inf/nan), and lambda_min
    // to 0, which drives tau to the HPOCL_TAULIN_FLOOR (-200 dB) path.
    gm[0] = 0.0f; gm[1] = 1e-30f; lm[2] = 0.0f; lm[3] = 1e-30f;
    hpref::switch_threshold(lm, gm, md, nt, nrf, eta, t1, m1);
    // boundary rows, pinned to the tau the reference actually computed:
    // main.tex says "THP if s >= tau", so metric == tau must select THP and one
    // float below must not. With continuous random metrics the probability of
    // hitting the boundary is zero, so >= and > were indistinguishable here.
    const int kAt = 4, kBelow = 5;
    md[kAt] = t1[kAt];
    md[kBelow] = std::nextafter(t1[kBelow], -HUGE_VALF);
    hpref::switch_threshold(lm, gm, md, nt, nrf, eta, t1, m1);
    rt.switch_threshold(lm.data(), gm.data(), md.data(), C, nt, nrf, eta,
                        t2.data(), m2.data());
    double e = maxerr(t1, t2); int mm = 0;
    for (int i = 0; i < C; ++i) mm += m1[i] != m2[i];
    // the boundary direction is asserted absolutely, not just for agreement:
    // both sides could be wrong the same way and still give mm == 0
    const bool at_thp = m1[kAt] == 1 && m2[kAt] == 1;
    const bool below_svd = m1[kBelow] == 0 && m2[kBelow] == 0;
    const bool finite = std::isfinite(t1[0]) && std::isfinite(t2[0]) &&
                        std::isfinite(t1[1]) && std::isfinite(t2[1]);
    bool ok = e < 1e-5 && mm == 0 && at_thp && below_svd && finite;
    fails += !ok;
    printf("switch_thr    err=%.2e mode_mismatch=%d nt/nrf=%.0f eta=%.5f "
           "gmin_floor=%s taulin_floor=%s boundary(>=)=%s %s\n",
           e, mm, nt / nrf, eta, finite ? "ok" : "FAIL",
           (t1[2] < -199.f && t1[2] > -201.f) ? "ok" : "FAIL",
           (at_thp && below_svd) ? "ok" : "FAIL", ok ? "PASS" : "FAIL");
  }
  { // wbb_apply : Y = W X,  W [m][n], X [SC][n] -> Y [SC][m]  (+ profile)
    double worst = 0; const Shape* ws = kShapes; double us = 0, gops = 0;
    for (const Shape& sh : kShapes) {
      auto W = rc((size_t)sh.m * sh.n), X = rc((size_t)SC * sh.n);
      std::vector<cplx> Y1, Y2((size_t)SC * sh.m);
      hpref::wbb_apply(W, sh.m, sh.n, X, SC, Y1);
      rt.wbb_apply(W.data(), sh.m, sh.n, X.data(), SC, Y2.data());
      double e = maxerr(Y1, Y2) / fmax(1.0, frob(Y1));   // relative, as above
      if (e > worst) { worst = e; ws = &sh; }
      if (sh.m == M && sh.n == N) {              // profile the reference shape
        us = rt.last_us();
        gops = 6.0 * SC * M * N / (us * 1e3);
      }
    }
    bool ok = worst < 1e-6; fails += !ok;
    printf("wbb_apply     rel_err=%.2e %s (%d shapes, worst %dx%d %s, %.1fus, "
           "%.2f GFLOP/s complex-MAC at %dx%d)\n", worst, ok ? "PASS" : "FAIL",
           (int)(sizeof kShapes / sizeof *kShapes), ws->m, ws->n, ws->tag, us,
           gops, M, N);
  }
  { // thp_precoder
    auto B = rc((size_t)K * K, 0.3f), F = rc((size_t)M * K), X = rc((size_t)SC * K);
    std::vector<cplx> Y1, Y2((size_t)SC * M);
    // tau = 1 pushes |x| ~ 1 past tau/2 so the centered-modulo fold is hit on
    // both axes (a regression check: the fold used to be tau^2 on the kernel)
    hpref::thp_precoder(B, F, X, SC, M, K, 1.0f, Y1);
    rt.thp_precoder(B.data(), F.data(), X.data(), SC, M, K, 1.0f, Y2.data());
    double e = maxerr(Y1, Y2); bool ok = e < 1e-4; fails += !ok;
    printf("thp_precoder  err=%.2e %s (%.1fus)\n", e, ok ? "PASS" : "FAIL", rt.last_us());
  }
  printf(fails ? "[op_bench] FAIL\n" : "[op_bench] ALL PASS\n");
  return fails ? 1 : 0;
}
