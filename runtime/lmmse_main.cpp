// lmmse_bench: LMMSE-CSI deployment of the closed-form threshold.
//   numpy oracle (tools/lmmse_ref.py, eps_s cross-checked against the
//   production full-covariance LMMSE pilot estimator) vs the fp64 C++ path:
//   lmmse_eps_s, the Wiener spectrum shrink, and the 4-step fixed point of
//   threshold_deploy_lmmse. tau may saturate to +-inf (uniform ordering).
#include <cstdio>
#include <cmath>
#include <vector>
#include <fstream>
#include <cstring>
#include "threshold_cf.h"

using namespace hpocl;

namespace {
struct Reader {
  std::vector<uint8_t> buf; size_t o = 0;
  explicit Reader(const char* p) {
    std::ifstream f(p, std::ios::binary);
    buf.assign((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
  }
  int i() { int v; memcpy(&v, buf.data() + o, 4); o += 4; return v; }
  double f() { double v; memcpy(&v, buf.data() + o, 8); o += 8; return v; }
  void re(std::vector<double>& v) { memcpy(v.data(), buf.data() + o, 8 * v.size()); o += 8 * v.size(); }
  bool done() const { return o == buf.size(); }
};
double err_tau(double a, double b) {
  if (std::isinf(a) || std::isinf(b)) return a == b ? 0.0 : HUGE_VAL;
  return fabs(a - b);
}
// relative conditioning of the closed-form denominator (same rule as cf_bench):
// near-flat spectra make the crossing point unidentifiable, count not compare
double flat_ratio(const double* lam, int K, int d, int M) {
  double lmin = lam[0], lmax = lam[0];
  for (int i = 0; i < K * d; ++i) lmax = fmax(lmax, lam[i]);
  if (lmax <= 1e-15) return 0.0;     // spectrum collapsed to ~0: trivially flat
  double gmin = 1e300;
  for (int i = 0; i < K * d; ++i) lmin = fmin(lmin, lam[i]);
  for (int k = 0; k < K; ++k) {
    double lg = 0;
    for (int i = 0; i < d; ++i) lg += log(fmax(lam[k * d + i], 1e-12));
    gmin = fmin(gmin, exp(lg / d));
  }
  const double eta = 1.0 + (double)(d - 1) / ((double)d * (M - 1.0));
  const double g = gmin / eta;
  return g > 0.0 ? (g - lmin) / g : 0.0;
}
}  // namespace

int main(int argc, char** argv) {
  Reader b(argc > 1 ? argv[1] : "build/lmmse_vectors.bin");
  int ncase = b.i(), fails = 0, nflat = 0;
  for (int ci = 0; ci < ncase; ++ci) {
    int M = b.i(), d = b.i(), K = b.i(), variant = b.i(), neig = b.i();
    double nv = b.f(); b.f();
    const int T = K * d;
    std::vector<double> eig(neig), lam_hat(T), lam_final(T);
    std::vector<double> one(1), two(2);
    b.re(eig); b.re(lam_hat); b.re(one);
    double eps_ref = one[0];
    b.re(lam_final); b.re(two);
    double tau_ref = two[0], flag_ref = two[1];

    CfConfig cfg; cfg.M = M; cfg.d = d; cfg.K = K;
    cfg.variant = variant ? ThpVariant::kMMSE_UCD : ThpVariant::kGMD;
    cfg.shrink_iters = 4;

    double eps = lmmse_eps_s(eig.data(), neig, nv);
    // 4-step fixed point, literal mirror of threshold_deploy_lmmse
    std::vector<double> lam_t(T);
    CfResult r = threshold_closed_form(lam_hat.data(), cfg);
    double rho = std::pow(10.0, r.tau_dB / 10.0) / T;
    for (int it = 0; it < cfg.shrink_iters; ++it) {
      shrink_spectrum(lam_hat.data(), T, rho, eps, lam_t.data());
      r = threshold_closed_form(lam_t.data(), cfg);
      rho = std::pow(10.0, r.tau_dB / 10.0) / T;
    }
    double e_eps = fabs(eps - eps_ref);         // Wiener scalar, bits-exact
    double e_tau = err_tau(r.tau_dB, tau_ref);   // the meaningful output
    // Exact flag agreement (0 = closed form, 1 = scan fallback, 2 = saturated
    // +-inf, 4 = capacity), same encoding as cf_ref.py tau_policy(),
    // CfResult::flag and kernels/threshold_cf.cl flags[c]. The old
    // (fell_back != 0) == (flag_ref != 0) form could not tell 1 from 2, and
    // fell_back is now exactly (flag == 1) -- so a saturated channel would have
    // compared false against flag_ref == 2 and failed this bench.
    bool flag_ok = (r.flag == (int)flag_ref);
    if (r.flag == 4) {
      fprintf(stderr, "[lmmse_bench] case%d rejected with flag 4 (K=%d d=%d "
              "M=%d exceeds kCapMaxT/kCapMaxK/kCapMaxD, or M is not a perfect "
              "square)\n", ci, K, d, M);
    }
    // Near-flat final spectra make the crossing unidentifiable (Psi sits at the
    // quadrature/Newton noise floor): the deployed budget can land on a fake
    // finite root where the converged reference saturates to +-inf. Measured on
    // the cf_bench pool at the deployed 20 steps: 1.0e-1 dB at 8 steps on the
    // flat_ratio = 3.0e-4 channel, 4.3e-14 dB at 20. Same rule as cf_bench --
    // count, do not compare those cases.
    bool flat_final = flat_ratio(lam_final.data(), K, d, M) < 0.05;
    if (flat_final) ++nflat;
    // The 4-step transient is path-dependent when the fixed point collapses to
    // +inf (strong shrink: spectrum -> 0, always SVD), so the shrunk spectrum
    // is only sanity-checked while the final threshold is finite.
    double e_lam = 0.0, scale = 1e-30;
    for (int i = 0; i < T; ++i) {
      e_lam = fmax(e_lam, fabs(lam_t[i] - lam_final[i]));
      scale = fmax(scale, fabs(lam_final[i]));
    }
    bool ok = e_eps < 1e-12 && (flat_final || (e_tau < 1e-3 && flag_ok &&
                                               (std::isinf(r.tau_dB) || e_lam < 1e-4 * scale)));
    fails += !ok;
    // flag= is now CfResult::flag (0/1/2/4), not fell_back, so the two numbers
    // printed are directly comparable. tau=+-inf prints as inf now that
    // threshold_closed_form returns INFINITY rather than HUGE_VAL, which is what
    // makes the isinf() test above and err_tau() meaningful.
    printf("case%d M=%3d d=%d K=%d %s nv=%.2g: eps=%.2e tau_err=%.3e lam=%.2e "
           "tau=%.5f%s flag=%d/%d %s\n",
           ci, M, d, K, variant ? "UCD" : "GMD", nv, e_eps, e_tau, e_lam / scale,
           r.tau_dB, std::isinf(r.tau_dB) ? " (sat)" : "", r.flag, (int)flag_ref,
           ok ? "PASS" : (flat_final ? "COUNT" : "FAIL"));
  }
  if (!b.done()) { printf("[lmmse_bench] trailing bytes\n"); ++fails; }
  printf(fails ? "[lmmse_bench] FAIL\n" : "[lmmse_bench] ALL PASS (flat=%d)\n", nflat);
  return fails ? 1 : 0;
}
