#pragma once
//-----------------------------------------------------------------------------
// Closed-form SVD/THP switching threshold (main.tex eq. cf) + safeguarded
// Newton refinement on the exact crossing equation (eq. phi) + LMMSE-CSI
// Wiener-shrinkage front end (eq. eps / shrink).
//
// PS-role code: plain C++17, no dependencies, fixed iteration counts only
// (newton_iters Newton steps, four shrink fixed-point steps) so it is
// cycle-budgetable on the Cortex-A53 and bit-repeatable against the PL OpenCL
// kernel. main.tex quotes five steps for its CDL-A pool (a pool that is not in
// this repo); the budget measured on the cf_bench pool is documented at
// newton_iters below, and all three copies must stay in step: this default,
// NEWTON_ITERS in kernels/threshold_cf.cl, and -DNEWTON_ITERS in
// opencl/build/build_xclbin.sh. tools/check_capacity.py fails the build on
// drift, so do not restate the value in prose here.
//-----------------------------------------------------------------------------
#include <vector>
#include <cstddef>

namespace hpocl {

enum class ThpVariant { kGMD = 0, kMMSE_UCD = 1 };

// Host mirror of the kernel's compiled-in capacity limits (kernels/
// threshold_cf.cl: MAXT / MAXK / MAXD). threshold_closed_form() rejects an
// out-of-capacity configuration with CfResult::flag = 4 and NaN thresholds, so
// the PS path refuses exactly the shapes the PL kernel refuses instead of
// returning a number the fabric cannot produce. kCapMaxK is kCapMaxT by default
// (K <= K*d since d >= 1); a PL build that passes -DMAXK=n must lower it too.
constexpr int kCapMaxT = 64;    // max streams per channel, T = K*d
constexpr int kCapMaxK = 64;    // max users K
constexpr int kCapMaxD = 128;   // max distance candidates in one of d0[]/d1[]

struct CfConfig {
  int M = 64;                 // square QAM order M = 2^m
  int d = 4;                  // streams per user
  int K = 8;                  // users
  ThpVariant variant = ThpVariant::kGMD;
  // Deployed fixed Newton budget: 20.
  //
  // History: this was 8, then moved to 20 on the strength of a table that read
  // "max |tau(iters) - tau(numpy,30)| on the cf_bench M=64/d=4/K=8 UCD config:
  // 8 -> 8.75e-2, 10 -> 2.20e-2, 12 -> 5.52e-3, 16 -> 3.46e-4, 20 -> 2.78e-5,
  // halving every step, i.e. pure bisection". That table does not reproduce on
  // the config it names. Re-measured (fp64 replica of this file, error against a
  // Brent root of the SAME Psi, cf_bench pool = tools/cf_ref.py seed 7, 5 configs
  // x 8 channels):
  //
  //  (a) the 39 channels cf_bench actually compares (flat_ratio >= 0.05):
  //        iters    3        5        8        10       16       20
  //        err      1.91e-1  2.13e-4  4.25e-6  1.06e-6  1.66e-8  4.97e-14  dB
  //      Superlinear, not bisection: on the flattest UCD channel (flat_ratio
  //      0.0759) psi_du = 1.47e-2 is not small at all, 11 consecutive Newton
  //      steps are ACCEPTED, and the residual reaches 2.6e-13 at it = 7. The
  //      "BISECT" entries after it = 11 are round-off (psi ~ 1e-15, step
  //      ~ 1e-13) rejected by an already-1e-13-wide bracket, i.e. the safeguard
  //      working, not failing. psi_du itself checks out against an independent
  //      central difference of psi_u to 1.0e-9..3.9e-5 relative over 40 samples.
  //      On that config max |tau(iters) - tau(numpy,30)| is FLAT at 2.78e-5 dB
  //      for every iters from 8 to 24 -- the old table's tail was that
  //      systematic host-vs-oracle offset (since fixed: the oracle now uses the
  //      same log-domain ln_deficit), not a convergence residual.
  //
  //  (b) the deeply near-flat channel cf_bench SKIPS (M=64,d=2,K=4 ch6 of the
  //      same pool, flat_ratio = 3.0e-4, denom = g_min/eta - lam_min = 3.35e-4):
  //        iters    3        5        8        10       12       16       20
  //        err      2.4      7.5e-1   1.0e-1   2.5e-2   6.4e-3   2.7e-13  4.3e-14
  //      Only 2 of 20 Newton steps are accepted, the rest bisect, and the error
  //      halves per step. THIS is the sequence the old table described, and
  //      8 -> 1.0e-1 dB is its 8-step entry.
  //
  // So 20 is required by regime (b), not by (a); 8 steps cost 4.25e-6 dB on
  // everything cf_bench measures. Regime (b) is real (main.tex:136 reports 7 of
  // 100 channels falling back, all at d=2, and ref/pipeline.cpp does not filter
  // on flatness), so keep 20 -- but note that cf_bench's flat_ratio < 0.05 skip
  // means the gate never sees the regime that justifies the budget. Cost of the
  // extra 12 iterations at (M,d,K) = (64,4,8): ~10.6 Mflop per Newton iteration
  // (L*ma*GH_N*L*(2*nimg+1) inner work x (T+K) deficits x 4 for psi_du's three
  // fills), i.e. ~212 Mflop per channel, ~0.21 s on a 1 GFLOP/s Cortex-A53;
  // M = 256 is ~1132 Mflop. A secant/Illinois safeguard instead of plain
  // bisection would very likely bring regime (b) under 8; that is the right
  // follow-up if the A53 budget cannot absorb 2.5x.
  // (main.tex makes no "<0.05 dB" claim -- its accuracy figure is 1.5e-3 dB max
  // over 200 CDL-A channels with five iterations, main.tex:167. The "<0.05 dB"
  // bound an earlier version of this comment attributed to the paper is
  // cf_bench's own maxdep tolerance.)
  int newton_iters = 20;
  int shrink_iters = 4;       // LMMSE fixed-point steps
};

struct CfResult {
  double tau_cf_dB = 0.0;     // closed form only (eq. cf)
  double tau_dB = 0.0;        // after Newton refinement on exact Phi
  double u_star = 0.0;        // root of Phi (u = delta^2 rho / 4)
  // method / status, the SAME encoding as flags[c] in kernels/threshold_cf.cl
  // and as the flag returned by tools/cf_ref.py tau_policy():
  //   0 = closed form (eq. cf) + Newton
  //   1 = near-flat spectrum: geometric scan of Psi supplied the start point
  //   2 = saturated: Psi keeps one sign over the whole scan, so the branch
  //       ordering is uniform and tau is +-INFINITY (-inf = always THP,
  //       +inf = always SVD). INFINITY, not HUGE_VAL, so std::isinf() is
  //       meaningful in cf_main / lmmse_main / pipeline.
  //   4 = configuration outside the compiled capacity (kCapMaxT/K/D, or M not a
  //       perfect square): thresholds are NaN and must not be used.
  int    flag = 0;
  bool   fell_back = false;   // exactly (flag == 1); kept for existing callers
  int    newton_used = 0;
};

// Deployed deficit-evaluator switch. The rationale and the measured numbers are
// next to the define in threshold_cf.cpp; it is declared HERE so that every
// consumer (runtime/cf_main.cpp, runtime/lmmse_main.cpp, ref/pipeline.cpp) can
// tell which evaluator it was compiled with and refuse a numpy oracle generated
// under the other one (tools/cf_ref.py reads HPOCL_DEFICIT_TABLE from the
// environment and stamps the setting into build/cf_vectors.bin).
#ifndef HPOCL_DEFICIT_TABLE
#define HPOCL_DEFICIT_TABLE 1
#endif

// bit-metric deficits, per main.tex eq.(1). No fitted constants on either path:
// the deployed evaluator is the generated ln-D table of include/deficit_table.h
// (converged quadrature, tabulated offline) inside its grid, the same analytic
// Lemma-4 tail C*Q(t) above it, and the live GH-48 rule below it or for a
// constellation the table does not cover. HPOCL_DEFICIT_TABLE=0 restores the
// pre-table path exactly (GH-48 up to kTQ, tail above).
double deficit_fin(double gamma, int M);   // finite Gray constellation stream
double deficit_mod(double gamma, int M);   // modulo-folded (THP) stream
// The two deficits from the LIVE Gauss-Hermite-48 rule alone: no table, no
// analytic-tail branch. This is the reference the deployed evaluator is gated
// against (cf_bench's maxq column, tools/check_deficit_table.py) and the
// fallback for an untabulated M or a t below DFT_TLO, so it stays compiled in
// and reachable -- a table that replaced the quadrature with no way to check it
// would be worse than no table.
double deficit_fin_quad(double gamma, int M);
double deficit_mod_quad(double gamma, int M);
// include/deficit_table.h metadata, so that no second copy of a grid literal
// exists anywhere: deficit_table_index() returns the row group of M in kDFT_M
// (row = 2*index + (mod ? 1 : 0)) or -1 when M is not tabulated, and
// deficit_table_grid() reports the covered t range, the point count and the
// number of constellations. With HPOCL_DEFICIT_TABLE=0 the index is always -1.
int  deficit_table_index(int M);
void deficit_table_grid(double* t_lo, double* t_hi, int* n_points, int* n_M);

// exact crossing function Phi(u) and its derivative; Psi is the
// cancellation-free log-domain balance solved by Newton on PL
double phi_u(double u, const double* lam /*K*d*/, const CfConfig& c);
double phi_du(double u, const double* lam, const CfConfig& c);
double psi_u(double u, const double* lam, const CfConfig& c);
double ln_deficit_fin(double gamma, int M);
double ln_deficit_mod(double gamma, int M);
double psi_du(double u, const double* lam, const CfConfig& c);

// LMMSE deployable scalar eps_s = tr(Re)/tr(Rh) from prior spectrum + pilot nv
double lmmse_eps_s(const double* eig_prior, int n, double nv);
// Wiener-shrink spectrum: lam_t = lam_hat / (1 + rho*lam_hat*eps_s)
void shrink_spectrum(const double* lam_hat, int n, double rho, double eps_s,
                     double* lam_t);

// top-level: per-user power-whitened squared singular values lam[K][d]
CfResult threshold_closed_form(const double* lam, const CfConfig& c);
// LMMSE-aware deployment: fixed point on rho with Wiener shrinkage
CfResult threshold_deploy_lmmse(const double* lam_hat, double eps_s,
                                const CfConfig& c);

}  // namespace hpocl
