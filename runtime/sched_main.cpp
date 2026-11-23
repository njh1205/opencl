// sched_bench: the DEVICE role of one scheduling instant against the PS role.
//
// Both roles run the same front_end() and design_instant() bodies from
// ref/pipeline.cpp (see runtime/hpocl_schedule.h); the only difference is the
// SvdProvider, so what this bench measures is exactly the question the board
// asks: does moving the decompositions and the threshold solve to fp32 kernels
// change the answer, and by how much?
//
//   host    schedule()          fp64 svd_cplx / threshold_closed_form
//   dev     schedule_device()   batched jacobi_svd / threshold_cf kernel, with
//                               the cost rule deciding which batches to launch
//   dev!    schedule_device()   force=true: EVERY batch launched, including the
//                               two nul x nul = 4x4 ones the cost rule keeps on
//                               the PS. Without this run the device path would
//                               only ever be gated on the two shapes that pay
//                               for themselves.
//
// Gauge. Three outputs are basis-dependent and are NOT compared entrywise:
//   Frf  columns carry a free phase (constant-modulus projection of a phased
//        singular vector), so columns are phase-aligned first;
//   Nk   the BD null space is 4-dimensional, so two correct bases differ by an
//        arbitrary unitary W, not by a diagonal phase;
//   Vd   differs by the same W^H.
// The physical precoder P_k = F_RF N_k V_D,k is invariant to both (N W * W^H Vd)
// and IS compared, which is also what pipeline_bench compares against the
// production oracle -- so this bench and that one gate the same quantity, from
// opposite ends. Everything else (sigma, B, R, Q, Fbb, rho, diagb, pscale, tau,
// mode) is gauge-invariant: sigma because singular values are, and the rest
// because they are functions of sigma and of the left basis U, which the null
// gauge does not touch.
//
// No numpy oracle here by design. The host role is already three-way gated
// against the production python by pipeline_bench (tools/pipeline_ref.py), so a
// second oracle would be a third implementation of the same instant; what is
// missing and what this adds is the device role. The chain is
// production python -> pipeline_ref.py -> pipeline_bench -> ref/pipeline.cpp
// -> sched_bench -> the kernels, and each link is a gate.
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>
#include "hpocl_schedule.h"
#include "hpocl_vecio.h"
#include "threshold_cf.h"     // HPOCL_DEFICIT_TABLE: host and device must agree

using namespace hpocl;

namespace {

// The geometries this bench exercises. The first five arguments are the ones
// tools/check_capacity.py parses (K, Nt, Nr, NRF, d) and checks against the
// decomp ceilings -- that gate is what stops a kernel from silently refusing a
// shape the control plane actually uses, which is how jacobi_svd shipped at
// 16x8 while null_basis needed 28x32.
// SCHED_CASE(K, Nt, Nr, NRF, d, M, variant, gamma_tot_dB, user_spread_dB)
// The FIRST FIVE arguments are the geometry tools/check_capacity.py parses and
// checks against the decomp ceilings. The last is a per-user large-scale gain
// spread applied to H: 0 means statistically identical users.
//
// THESE ARE THE USER'S ACTUAL CONFIGURATIONS, taken from
// mix_precode/{classical,ai}/config_pools/*.json -- not invented ones. The 21
// distinct (K,Nt,Nr,NRF,d) tuples over every pool span K in {2,3,4,8},
// Nt in {16,18,20,24,32,64,128}, Nr in {2,3,4,6,8}, NRF in {8,9,10,12,16,18,32},
// d in {2,3,4,8}, and ALL 21 satisfy the RF-BD feasibility test
// (K-1)*Nr < min(Nt,NRF) with nul >= d. The cases below cover the envelope:
// the largest Nt (128, where the analog factorisation is 4x128 and dominates),
// the largest K with the largest null-basis stack (28x32), Nr=8/d=8 (the widest
// whitening and the biggest stream count), Nr=2 (the narrowest), and the
// smallest config overall. M is 64 except where a pool uses 16, and gamma is
// chosen to land on both sides of tau so both design branches run.
//
// The last case is NOT from a pool. It is the geometry the README's cost model
// used to call "deployed" (K=8, Nt=16, Nr=4, NRF=32, d=4), which no pool
// contains and which is RF-BD INFEASIBLE: (K-1)*Nr = 28 >= min(Nt,NRF) = 16, so
// rank(A) = 16, null(A) = null(F_RF) and F_RF N_k == 0 -- measured against
// production python, which degenerates identically (its own null space is 32x16,
// ||F_RF N||_F = 2.6e-15, sigma = 3.6e-9). It is kept on purpose as the case
// that proves front_end()'s fe_flag fires, and it is the reason every cost number
// quoted at that shape had to be re-derived.
#define SCHED_CASE(K_, Nt_, Nr_, NRF_, d_, M_, var_, g_, sp_) \
  { K_, Nt_, Nr_, NRF_, d_, M_, var_, g_, sp_ }
struct Case { int K, Nt, Nr, NRF, d, M, variant; double gamma, spread; };
const Case kCases[] = {
  SCHED_CASE(8, 128, 4, 32, 4, 64, 0, 30.0, 0.0),  // paper_sweep, largest: T=32
  SCHED_CASE(8, 128, 4, 32, 3, 64, 1, 30.0, 0.0),  // letter_k8_d3, MMSE-UCD branch
  SCHED_CASE(8,  64, 4, 32, 2, 64, 0, 30.0, 0.0),  // generalization_pool_2_4_8
  SCHED_CASE(4,  64, 4, 16, 4, 64, 0, 30.0, 0.0),  // paper_sweep K=4
  SCHED_CASE(2,  32, 8, 16, 8, 64, 0, 30.0, 0.0),  // pool_2_4_8_fair: Nr=8, d=8
  SCHED_CASE(8,  32, 2, 16, 2, 16, 0, 22.0, 0.0),  // pool_2_4_8_fair: Nr=2, M=16
  SCHED_CASE(4,  32, 3, 16, 2, 64, 0, 20.0, 0.0),  // pool_v1: Nr=3, nul=7
  SCHED_CASE(2,  20, 4, 10, 3, 64, 0, 20.0, 0.0),  // pool_v1: Nt=20, NRF=10
  SCHED_CASE(2,  16, 4,  8, 4, 64, 0, 18.0, 0.0),  // smallest; nul == d == 4
  SCHED_CASE(8,  32, 4, 32, 4, 64, 0, 30.0, 6.0),  // 6 dB user spread
  SCHED_CASE(8,  16, 4, 32, 4, 64, 0, 30.0, 0.0),  // INFEASIBLE: fe_flag must be 1
};
const int kNCase = (int)(sizeof kCases / sizeof kCases[0]);

double err_c(const dcplx* a, const dcplx* b, size_t n) {
  double m = 0; for (size_t i = 0; i < n; ++i) m = fmax(m, std::abs(a[i] - b[i])); return m;
}
double err_r(const double* a, const double* b, size_t n) {
  double m = 0; for (size_t i = 0; i < n; ++i) m = fmax(m, fabs(a[i] - b[i])); return m;
}
// relative to the larger of the two magnitudes, so a field whose scale is 1e-20
// does not print as a huge absolute error or a tiny relative one by accident
double rel_c(const dcplx* a, const dcplx* b, size_t n) {
  double s = 0; for (size_t i = 0; i < n; ++i) s = fmax(s, std::abs(b[i]));
  return s > 0 ? err_c(a, b, n) / s : err_c(a, b, n);
}
double rel_r(const double* a, const double* b, size_t n) {
  double s = 0; for (size_t i = 0; i < n; ++i) s = fmax(s, fabs(b[i]));
  return s > 0 ? err_r(a, b, n) / s : err_r(a, b, n);
}
double err_tau(double a, double b) {          // +-inf when the branch order is uniform
  if (std::isinf(a) || std::isinf(b)) return a == b ? 0.0 : 1e30;
  if (std::isnan(a) || std::isnan(b)) return a == b ? 0.0 : 1e30;   // NaN == NaN: both refused
  return fabs(a - b);
}
// per-column phase alignment: columns of a constant-modulus precoder, and of
// any matrix whose columns are singular vectors, are free up to a phase.
void align_phase(std::vector<dcplx>& mine, const std::vector<dcplx>& ref,
                 int rows, int cols) {
  for (int j = 0; j < cols; ++j) {
    dcplx dot(0, 0);
    for (int r = 0; r < rows; ++r)
      dot += std::conj(ref[(size_t)r * cols + j]) * mine[(size_t)r * cols + j];
    if (std::abs(dot) > 1e-12) {
      dcplx ph = std::conj(dot / std::abs(dot));
      for (int r = 0; r < rows; ++r) mine[(size_t)r * cols + j] *= ph;
    }
  }
}
// P_k = F_RF N_k V_D,k  [Nt x d]: the physical precoder, invariant to the BD
// null gauge and to the singular-vector phase.
std::vector<dcplx> physical(const PipeOut& o, int k, const PipeCfg& c) {
  std::vector<dcplx> P((size_t)c.Nt * c.d, dcplx(0, 0));
  for (int i = 0; i < c.Nt; ++i)
    for (int j = 0; j < c.d; ++j) {
      dcplx a(0, 0);
      for (int t = 0; t < c.NRF; ++t) {
        dcplx nv(0, 0);
        for (int q = 0; q < o.nul; ++q)
          nv += o.Nk[((size_t)k * c.NRF + t) * o.nul + q] *
                o.Vd[((size_t)k * o.nul + q) * c.d + j];
        a += o.Frf[(size_t)i * c.NRF + t] * nv;
      }
      P[(size_t)i * c.d + j] = a;
    }
  return P;
}
void gen_H(const Case& cs, int seed, std::vector<dcplx>& H) {
  H.assign((size_t)cs.K * cs.Nr * cs.Nt, dcplx(0, 0));
  std::mt19937 g(seed);
  std::normal_distribution<double> n(0, 1);
  for (auto& x : H) x = dcplx(n(g), n(g)) / std::sqrt(2.0);
  // Per-user large-scale gains spanning `spread` dB, centred on 0 dB. This is
  // the knob that exposes the fp32 front end's dynamic-range limit: the
  // interference stack's Gram spans the same spread, and an fp32 Jacobi has
  // ~7 digits, so a spread much larger than the BD nullity's conditioning makes
  // the trailing singular vectors -- the ones null_basis keeps -- meaningless.
  // A first version of this bench used a 120 dB spread (1.5 dB x (k - K/2) with
  // K=8) and every deployed-geometry case diverged by 1e2-1e8; that was the
  // bench being unphysical, not the kernel being wrong, and the diagnostic at
  // the bottom of this file measures where the real limit is.
  if (cs.spread > 0) {
    for (int k = 0; k < cs.K; ++k) {
      const double e = cs.spread * (2.0 * k / (cs.K - 1) - 1.0) / 20.0;
      const double gk = pow(10.0, e);
      for (int i = 0; i < cs.Nr * cs.Nt; ++i)
        H[(size_t)k * cs.Nr * cs.Nt + i] *= gk;
    }
  }
}

// The gates. Each is the MEASURED worst case over the 11 cases below on this
// device, times ~5-10x margin for a different device/compiler -- the same
// convention decomp_bench documents for its 3e-4. Measured worst (fp32 device
// role vs fp64 PS role, K=8 Nt=128 Nr=4 NRF=32 d=4 being the hardest):
//   tau 3.7e-05 dB   sigma 8.9e-07   Frf 0.0 (both roles run the Gram analog
//   step on the PS, so Frf is identical)   P=Frf*N*Vd 1.9e-04 absolute on
//   |P| ~ 0.18   Frf*Fbb 5.7e-05   B/R 1.2e-06   Q 2.2e-05   rho 1.8e-06
//   pscale 2.0e-07   diagb 5.0e-07   |Frf| modulus 2.8e-17
// P is gated LOOSER than the transmitted Frf*Fbb, and that is not sloppiness:
// P = F_RF N V_D omits the whitener C^{-1/2}, whose eigenvalue floor damps
// exactly the null-space directions where an fp32 and an fp64 BD basis differ
// most. Measured, P is 3-40x more sensitive than Frf*Fbb on every case.
struct Tol {
  double tau_dB   = 5e-2;   // the fp32 threshold kernel's own documented budget
  double sigma    = 1e-4;   // relative
  double frf      = 1e-6;   // absolute; 0.0 today, kept tight on purpose
  double phys     = 1e-3;   // absolute, P_k = Frf N Vd (|P| ~ 0.18)
  double fbb      = 1e-3;   // relative, the TRANSMITTED Frf*Fbb
  double mat      = 1e-3;   // relative: B, R, Q
  double rho      = 1e-3;   // relative: rho_real, rho_tgt
  double scale    = 1e-5;   // relative: pscale, diagb
} tol;

// The same bench with the three Nt-class glue GEMMs on the fabric (roles "glue"
// and "glue!"). Measured the same way tol's limits were -- worst case over the 11
// cases on this device, then ~5-10x margin for a different device/compiler -- and
// the worst case is IDENTICAL for every field except `scale`, which is why only
// that one differs:
//   tau 3.9e-05 dB (costed 1.1e-05)   sigma 1.3e-05 (unchanged)   Frf 0.0
//   P=Frf*N*Vd 1.9e-04 (unchanged)    Frf*Fbb 5.8e-05 (unchanged)
//   B/R 6.0e-05   Q 6.9e-05   rho 2.2e-05 (costed 8.6e-06)
//   pscale 5.4e-09   diagb 1.52e-05 (costed 5.5e-06)  <- the one that moved
// diagb = pscale * diag(R) and pscale itself is BETTER with the glue offloaded
// (5.4e-9 vs 1.6e-15 is a different case, but pscale never exceeds 5.7e-7), so
// the movement is diag(R): the equalizer factor is a function of the whitened
// sigma and of V_D, i.e. of the null basis, and kGemmEff/kGemmFn make the INPUT
// to that basis fp32 as well as the Jacobi that factorises it. 1.52e-05 relative
// on a d x d gain matrix is 0.0015% -- it is the expected fp32 sensitivity, not a
// defect -- but it is 1.5x tol.scale, so the glue roles get 1e-4 (6.6x margin,
// the same convention as tol's 20x on a 5.0e-07 measurement). Every other field
// keeps tol's limit UNCHANGED, so a glue-specific regression in sigma or Frf*Fbb
// still fails at the SVD-only budget instead of hiding behind a looser one.
Tol tol_glue = []{ Tol t; t.scale = 1e-4; return t; }();

// And the same bench with EVERY Jacobi batch launched (`force` / `glue!`), which
// is a coverage role and not a deployment: the cost rule keeps these batches on
// the PS at every shape where launching them is not worth it, and at those shapes
// the two roles are bit-identical (case7 prints 0.0e+00 in every column). One
// field needs a wider bound there, and the reason is a property of the PROBLEM,
// not of the code:
//   P = F_RF N V_D, at K=2 Nt=20 Nr=4 NRF=10 d=3 (generalization_pool_v1), where
//   the interference stack is 4x10 so its six trailing singular values are EXACTLY
//   zero and the BD null basis N is defined only up to a 6x6 unitary rotation. Two
//   correct fp32 implementations may therefore return different bases, and P is
//   the one compared quantity the whitener C^{-1/2} does not damp (see tol.phys).
//   Measured, same inputs, same fp32 algorithm, two devices:
//     Apple M4 GPU   P = 1.1e-06      NVIDIA L4   P = 1.39e-02   (|P| ~ 0.18)
//   while on BOTH devices sigma agrees to <= 9.2e-07, the TRANSMITTED F_RF*F_BB to
//   <= 2.6e-06 and tau to <= 2.7e-05 dB. So the schedule and the transmitted
//   precoder are device-independent here; the un-whitened intermediate is not.
//   2e-2 is 1.4x the NVIDIA measurement -- a bound that says "this column is a
//   basis-dependent diagnostic in the forced role", not a claim of accuracy. The
//   costed roles keep tol.phys = 1e-3 unchanged, and they are what ships.
Tol tol_force = []{ Tol t; t.phys = 2e-2; return t; }();

int run_case(const Case& cs, int ci, Runtime& rt, bool have_cl) {
  PipeCfg c;
  c.K = cs.K; c.Nt = cs.Nt; c.Nr = cs.Nr; c.NRF = cs.NRF; c.d = cs.d;
  c.M = cs.M; c.gamma_tot_dB = cs.gamma; c.variant = cs.variant;
  std::vector<dcplx> H;
  gen_H(cs, 1234 + ci, H);
  const int T = cs.K * cs.d;
  const bool feasible = ((cs.K - 1) * cs.Nr) < std::min(cs.Nt, cs.NRF);

  // ---- PS role, per stage. front_end / threshold_closed_form / design_instant
  // are the three public pieces schedule() is made of, so timing them here is a
  // measurement of schedule(), not a second implementation of it.
  PipeOut oh; FrontEnd feh;
  auto st = std::chrono::steady_clock::now();
  front_end(H.data(), c, host_svd(), host_gemm(), feh, oh);
  auto en = std::chrono::steady_clock::now();
  double fe_h = std::chrono::duration<double, std::micro>(en - st).count();
  std::vector<double> lam((size_t)cs.K * cs.d);
  for (int i = 0; i < cs.K * cs.d; ++i) lam[i] = oh.sigma[i] * oh.sigma[i];
  CfConfig cf; cf.M = cs.M; cf.d = cs.d; cf.K = cs.K;
  cf.variant = cs.variant ? ThpVariant::kMMSE_UCD : ThpVariant::kGMD;
  st = std::chrono::steady_clock::now();
  CfResult rh = threshold_closed_form(lam.data(), cf);
  en = std::chrono::steady_clock::now();
  double th_h = std::chrono::duration<double, std::micro>(en - st).count();
  oh.tau_dB = rh.tau_dB;
  oh.mode = decide_mode(c, rh.tau_dB, rh.flag, &oh.cf_flag);
  st = std::chrono::steady_clock::now();
  design_instant(c, oh.mode, host_gemm(), feh, oh);
  en = std::chrono::steady_clock::now();
  double de_h = std::chrono::duration<double, std::micro>(en - st).count();

  printf("case%d K=%d Nt=%d Nr=%d NRF=%d d=%d M=%3d var=%d s=%+5.1fdB spread=%.0fdB%s\n",
         ci, cs.K, cs.Nt, cs.Nr, cs.NRF, cs.d, cs.M, cs.variant, cs.gamma,
         cs.spread, feasible ? "" : "  [RF-BD INFEASIBLE]");
  printf("        host role: front_end %.0f us, threshold %.0f us (flag %d), "
         "design %.0f us, total %.0f us | fe_flag=%d mode=%d tau=%.4f\n",
         fe_h, th_h, rh.flag, de_h, fe_h + th_h + de_h, oh.fe_flag, oh.mode,
         oh.tau_dB);
  if (oh.fe_flag != (feasible ? 0 : 1)) {
    printf("        FAIL fe_flag=%d but the geometry %s feasible\n", oh.fe_flag,
           feasible ? "IS" : "is NOT");
    return 1;
  }
  if (!have_cl) {
    printf("        NO DEVICE -- the device role is NOT tested by this run; "
           "this line is not a pass\n");
    return 0;
  }
  if (!feasible) {
    // Degenerate RF-BD: F_RF N_k == 0, so sigma is the eigenvalue floor's noise
    // and tau is meaningless. Both roles must AGREE that it is degenerate; the
    // numbers are not comparable because there is nothing to compare.
    DeviceSvd ds(rt); ds.force = true;
    PipeOut od;
    if (!schedule_device(H.data(), c, od, ds)) { printf("        refused\n"); return 1; }
    bool ok = (od.fe_flag == 1) && (od.nul == oh.nul);
    printf("        device role: fe_flag=%d (host %d), nul=%d/%d, sigma[0]=%.3e vs "
           "%.3e -- degenerate on both sides, numerics NOT gated %s\n",
           od.fe_flag, oh.fe_flag, od.nul, oh.nul, od.sigma[0], oh.sigma[0],
           ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
  }

  int fails = 0;
  // FOUR roles, because "does the device role agree with the PS role" stopped
  // being one question when the glue got a provider: the decompositions and the
  // three Nt-class GEMMs are offloaded by different cost decisions and carry
  // different accuracy risk (kGemmEff feeds the null basis, kGemmFn feeds the
  // whitening Gram, kGemmPower feeds one scalar). "costed"/"force" are exactly
  // what this bench ran before the glue provider existed -- sites = 0 keeps
  // every GEMM on the PS -- so the two new roles are an addition, not a change.
  struct Role { const char* name; bool svd_force; int sites; bool gemm_force; };
  static const int kAllSites = (1 << kGemmSites) - 1;
  static const Role kRoles[] = {
    {"costed", false, 0,        false},   // SVDs by cost rule, glue on the PS
    {"force",  true,  0,        false},   // every SVD batch launched
    {"glue",   false, kAllSites, false},  // + the three GEMMs, by cost rule
    {"glue!",  true,  kAllSites, true},   // everything offloaded, so the sites
                                          // the cost rule keeps on the PS at
                                          // Nt <= 32 are gated too
  };
  for (const Role& rl : kRoles) {
    DeviceSvd ds(rt);
    ds.force = rl.svd_force;
    DeviceGemm dg(rt);
    dg.force = rl.gemm_force;
    dg.sites = rl.sites;
    DeviceGemm* dgp = rl.sites ? &dg : nullptr;
    PipeOut od;
    auto a = std::chrono::steady_clock::now();
    if (!schedule_device(H.data(), c, od, ds, dgp)) {
      printf("        schedule_device refused\n"); return 1;
    }
    auto b = std::chrono::steady_clock::now();
    double dev_us = std::chrono::duration<double, std::micro>(b - a).count();

    bool disc = (od.mode == oh.mode) && (od.cf_flag == oh.cf_flag) &&
                (od.nul == oh.nul) && (od.fe_flag == oh.fe_flag);
    double e_tau = err_tau(od.tau_dB, oh.tau_dB);
    double e_sig = rel_r(od.sigma.data(), oh.sigma.data(), (size_t)cs.K * cs.d);
    double e_rho = fmax(rel_r(od.rho_real.data(), oh.rho_real.data(), (size_t)cs.K * cs.d),
                        rel_r(od.rho_tgt.data(), oh.rho_tgt.data(), (size_t)cs.K * cs.d));
    double e_ps  = rel_r(&od.pscale, &oh.pscale, 1);
    double e_dg  = rel_c(od.diagb.data(), oh.diagb.data(), (size_t)cs.K * cs.d);
    std::vector<dcplx> frf_d = od.Frf;
    align_phase(frf_d, oh.Frf, cs.Nt, cs.NRF);
    double e_frf = err_c(frf_d.data(), oh.Frf.data(), (size_t)cs.Nt * cs.NRF);
    double e_P = 0, e_B = 0, e_R = 0, e_Q = 0;
    for (int k = 0; k < cs.K; ++k) {
      std::vector<dcplx> Pd = physical(od, k, c), Ph = physical(oh, k, c);
      align_phase(Pd, Ph, cs.Nt, cs.d);
      e_P = fmax(e_P, err_c(Pd.data(), Ph.data(), (size_t)cs.Nt * cs.d));
      e_B = fmax(e_B, rel_c(od.B.data() + (size_t)k * cs.d * cs.d,
                            oh.B.data() + (size_t)k * cs.d * cs.d, (size_t)cs.d * cs.d));
      e_R = fmax(e_R, rel_c(od.R.data() + (size_t)k * cs.d * cs.d,
                            oh.R.data() + (size_t)k * cs.d * cs.d, (size_t)cs.d * cs.d));
      e_Q = fmax(e_Q, rel_c(od.Q.data() + (size_t)k * cs.Nr * cs.d,
                            oh.Q.data() + (size_t)k * cs.Nr * cs.d, (size_t)cs.Nr * cs.d));
    }
    // F_BB itself is NOT gateable entrywise, and this bench used to try. Fbb =
    // N C^{-1/2} V_D, and C = (F_RF N)^H (F_RF N) is rank-deficient whenever
    // nul > Nt, so C^{-1/2} carries 1/sqrt(eigenvalue floor) ~ 1e5 in the
    // directions F_RF annihilates. Those directions are UNOBSERVABLE -- they
    // contribute nothing to F_RF F_BB, which is what is transmitted -- but they
    // are exactly where an fp32 and an fp64 null basis differ most, so raw Fbb
    // is a gauge-like quantity: measured at K=3/Nt=6/NRF=12, max|Fbb| was 0.78
    // (host) against 1661 (device), a 2.1e3 relative difference, while
    // ||F_RF F_BB||_F^2 was 6.000000 on BOTH sides to six digits. Gate the
    // physical precoder instead, and report raw Fbb ungated.
    std::vector<dcplx> Ph((size_t)cs.Nt * T, dcplx(0, 0)), Pd((size_t)cs.Nt * T, dcplx(0, 0));
    for (int i = 0; i < cs.Nt; ++i)
      for (int j = 0; j < T; ++j) {
        dcplx ah(0, 0), ad(0, 0);
        for (int t = 0; t < cs.NRF; ++t) {
          ah += oh.Frf[(size_t)i * cs.NRF + t] * oh.Fbb[(size_t)t * T + j];
          ad += od.Frf[(size_t)i * cs.NRF + t] * od.Fbb[(size_t)t * T + j];
        }
        Ph[(size_t)i * T + j] = ah; Pd[(size_t)i * T + j] = ad;
      }
    double e_fbb = rel_c(Pd.data(), Ph.data(), (size_t)cs.Nt * T);
    double e_fbb_raw = rel_c(od.Fbb.data(), oh.Fbb.data(), (size_t)cs.NRF * T);
    double cm = 0;                       // constant modulus, which alignment hides
    for (auto& x : od.Frf) cm = fmax(cm, fabs(std::abs(x) - 1.0 / sqrt((double)cs.Nt)));

    // Per-role budget. Each relaxation is one field, traceable to the measurement
    // in the definition above: the glue roles move diag(R) (tol_glue.scale), and
    // forcing every Jacobi batch makes P basis-dependent on shapes whose null
    // space is degenerate by construction (tol_force.phys). Nothing else moves, so
    // a regression in any other field still fails at the shipped budget.
    Tol tl = tol;
    if (rl.sites) tl.scale = tol_glue.scale;
    if (rl.svd_force) tl.phys = tol_force.phys;
    struct Chk { const char* what; double v, lim; };
    Chk chks[] = {
      {"tau_dB", e_tau, tl.tau_dB}, {"sigma", e_sig, tl.sigma},
      {"Frf", e_frf, tl.frf},       {"|Frf|-1/sqrt(Nt)", cm, 1e-6},
      {"P=Frf*N*Vd", e_P, tl.phys}, {"Frf*Fbb", e_fbb, tl.fbb},
      {"B", e_B, tl.mat},           {"R", e_R, tl.mat},
      {"Q", e_Q, tl.mat},           {"rho", e_rho, tl.rho},
      {"pscale", e_ps, tl.scale},   {"diagb", e_dg, tl.scale},
    };
    bool okc = disc;
    for (auto& k : chks)
      if (!(k.v <= k.lim)) { okc = false; printf("        over gate: %s = %.3e > %.3e\n", k.what, k.v, k.lim); }
    fails += !okc;
    printf("        %-6s tau=%.1e sig=%.1e Frf=%.1e cm=%.1e P=%.1e FrfFbb=%.1e "
           "B=%.1e R=%.1e Q=%.1e rho=%.1e ps=%.1e diagb=%.1e  mode=%d/%d %s\n",
           rl.name, e_tau, e_sig, e_frf, cm, e_P, e_fbb, e_B,
           e_R, e_Q, e_rho, e_ps, e_dg, oh.mode, od.mode, okc ? "PASS" : "FAIL");
    printf("        %-6s (raw Fbb, ungated: %.3e -- its null(F_RF) component is "
           "unobservable, see the comment)\n", rl.name,
           e_fbb_raw);
    printf("        %-6s split: %d launch(es) %.2f Mflop + %d host batch(es) "
           "%.2f Mflop | wall %.0f us, of which kernel %.0f us (svd %.0f + "
           "threshold %.0f) => %.0f us PS glue+launch\n",
           rl.name, ds.launches, ds.flops_off / 1e6,
           ds.host_batches, ds.flops_host / 1e6, dev_us,
           ds.us_svd + ds.us_threshold, ds.us_svd, ds.us_threshold,
           dev_us - ds.us_svd - ds.us_threshold);
    if (rl.sites)
      printf("        %-6s glue:  %d launch(es) %.2f Mflop + %d host call(s) "
             "%.2f Mflop | kernel %.0f us, staging %.1fk complex elems (fp64->"
             "fp32->fp64, PS work the offload ADDS)\n",
             rl.name, dg.launches, dg.flops_off / 1e6, dg.host_calls,
             dg.flops_host / 1e6, dg.us_gemm, dg.elems_converted / 1e3);
    // The line above needs its caveat printed with it, or it reads as "the
    // device role is 40x slower", which is a property of this dev host's
    // driver and not of the code. Measured by decomposing one launch into its
    // phases (clCreateBuffer / clSetKernelArg / clEnqueueNDRange+clFinish /
    // clEnqueueReadBuffer / clReleaseMemObject): buffer create+release is
    // 13-32 us, readback ~1 ms, and clEnqueueNDRangeKernel+clFinish is the
    // whole story -- 50.2 ms for a 28x32 batch whose kernel EVENT time is
    // 241 us (208x), and 28.9 ms for a 4x4 batch whose event time is 0.3 us.
    // It also scales with the kernel's static private footprint, not with the
    // data: the same 4x4 C=8 launch costs 2.07 ms with DECOMP_MAXM/MAXD=16x8
    // (1.5 KB private per work-item) and 23.3 ms with 32x40 (22.8 KB), 11x.
    // On XRT the fixed cost is 10-50 us, not 50 ms, which is what min_flops is
    // derived from; quote the kernel-event column here, never the wall column.
    printf("          ^ wall includes this host's driver launch latency; the "
           "device-side number is the kernel column\n");
    for (size_t i = 0; i < ds.where.size(); ++i)
      printf("          svd batch %zu (%s) -> %s\n", i, ds.what[i].c_str(),
             ds.where[i]);
    for (size_t i = 0; i < dg.where.size(); ++i)
      printf("          %s -> %s\n", dg.what[i].c_str(), dg.where[i]);
  }

  // Per-site sensitivity, one role each: which GEMM costs the accuracy, and how
  // much. Run with the SVD cost rule as shipped so the only variable is the
  // single site enabled. This is the measurement tol_glue's per-field limits are
  // set from, and it is also what tells the board which sites to keep offloaded
  // if the PL budget is short (kGemmPower feeds one scalar; kGemmEff feeds the
  // null basis).
  printf("        per-site (one GEMM offloaded at a time, SVDs by cost rule):\n");
  for (int site = 0; site < kGemmSites; ++site) {
    DeviceSvd ds(rt);
    DeviceGemm dg(rt);
    dg.sites = 1 << site;
    PipeOut od;
    if (!schedule_device(H.data(), c, od, ds, &dg)) { printf("          refused\n"); return 1; }
    double e_sig = rel_r(od.sigma.data(), oh.sigma.data(), (size_t)cs.K * cs.d);
    double e_ps  = rel_r(&od.pscale, &oh.pscale, 1);
    double e_P = 0, e_fbb = 0;
    std::vector<dcplx> Ph((size_t)cs.Nt * T, dcplx(0, 0)), Pd((size_t)cs.Nt * T, dcplx(0, 0));
    for (int i = 0; i < cs.Nt; ++i)
      for (int j = 0; j < T; ++j) {
        dcplx ah(0, 0), ad(0, 0);
        for (int t = 0; t < cs.NRF; ++t) {
          ah += oh.Frf[(size_t)i * cs.NRF + t] * oh.Fbb[(size_t)t * T + j];
          ad += od.Frf[(size_t)i * cs.NRF + t] * od.Fbb[(size_t)t * T + j];
        }
        Ph[(size_t)i * T + j] = ah; Pd[(size_t)i * T + j] = ad;
      }
    e_fbb = rel_c(Pd.data(), Ph.data(), (size_t)cs.Nt * T);
    for (int k = 0; k < cs.K; ++k) {
      std::vector<dcplx> Pd2 = physical(od, k, c), Ph2 = physical(oh, k, c);
      align_phase(Pd2, Ph2, cs.Nt, cs.d);
      e_P = fmax(e_P, err_c(Pd2.data(), Ph2.data(), (size_t)cs.Nt * cs.d));
    }
    printf("          site %d (%s): tau=%.1e dB sig=%.1e P=%.1e FrfFbb=%.1e "
           "ps=%.1e -> %s\n", site,
           site == kGemmEff ? "H*F_RF  " : site == kGemmFn ? "F_RF*N_k" : "F_RF*F_BB",
           err_tau(od.tau_dB, oh.tau_dB), e_sig, e_P, e_fbb, e_ps,
           // where[site], NOT where[0]: front_end calls site 0 first, so where[0]
           // is site 0's decision on every run and printing it here reported
           // "host" for the two sites that WERE offloaded.
           (int)dg.where.size() > site ? dg.where[site] : "?");
  }
  return fails;
}

// Where does the fp32 front end stop tracking the fp64 one? The interference
// stack's Gram spans the per-user gain spread, and an fp32 Jacobi has ~7 digits,
// so the trailing singular vectors -- exactly the ones null_basis keeps -- lose
// meaning as the spread grows. This is a MEASUREMENT, not a gate: it is printed
// so the limit is a number in the README instead of an assumption, and so the
// gate cases above can be chosen inside it.
void spread_sweep(Runtime& rt, bool have_cl) {
  if (!have_cl) return;
  // generalization_pool_2_4_8's K=8 config: a real shape, rows=28 < NRF=32.
  Case cs{8, 64, 4, 32, 2, 64, 0, 30.0, 0.0};
  PipeCfg c;
  c.K = cs.K; c.Nt = cs.Nt; c.Nr = cs.Nr; c.NRF = cs.NRF; c.d = cs.d;
  c.M = cs.M; c.gamma_tot_dB = cs.gamma; c.variant = cs.variant;
  printf("\n[spread] fp32 vs fp64 front end against the per-user gain spread "
         "(K=8 Nt=64 Nr=4 NRF=32 d=2, from generalization_pool_2_4_8, "
         "force=true; relative errors). A MEASUREMENT, not a gate: it says how "
         "much per-user dynamic range the fp32 BD front end tracks before the "
         "trailing singular vectors -- the ones null_basis keeps -- stop being "
         "meaningful.\n");
  printf("  spread_dB |     sigma      P=Frf*N*Vd        Fbb           B      "
         "| tau_host  tau_dev  mode\n");
  for (double sp : {0.0, 6.0, 12.0, 20.0, 30.0, 40.0, 60.0, 120.0}) {
    cs.spread = sp;
    std::vector<dcplx> H; gen_H(cs, 4242, H);
    PipeOut oh; schedule(H.data(), c, oh);
    DeviceSvd ds(rt); ds.force = true;
    PipeOut od;
    if (!schedule_device(H.data(), c, od, ds)) continue;
    double e_sig = rel_r(od.sigma.data(), oh.sigma.data(), (size_t)c.K * c.d);
    double e_P = 0, e_B = 0;
    for (int k = 0; k < c.K; ++k) {
      std::vector<dcplx> Pd = physical(od, k, c), Ph = physical(oh, k, c);
      align_phase(Pd, Ph, c.Nt, c.d);
      e_P = fmax(e_P, err_c(Pd.data(), Ph.data(), (size_t)c.Nt * c.d));
      e_B = fmax(e_B, rel_c(od.B.data() + (size_t)k * c.d * c.d,
                            oh.B.data() + (size_t)k * c.d * c.d, (size_t)c.d * c.d));
    }
    double e_fbb = rel_c(od.Fbb.data(), oh.Fbb.data(), (size_t)c.NRF * c.K * c.d);
    printf("    %5.0f   | %10.3e %13.3e %11.3e %11.3e | %8.3f %8.3f  %d/%d\n",
           sp, e_sig, e_P, e_fbb, e_B, oh.tau_dB, od.tau_dB, oh.mode, od.mode);
  }
}
}  // namespace

int main(int argc, char** argv) {
  const char* pref = argc > 1 ? argv[1] : "cpu";
  Runtime rt;
  bool have_cl = rt.init(pref);
  if (have_cl) {
    // Same two conventions cf_main.cpp uses, and for the same reason: the host
    // and the device must solve the SAME Psi. threshold_cf.cl is fp32 by default
    // (the PL cannot build fp64: the ZU49DR's DSP48E2 has no floating point), so
    // an fp64-capable device has to ask for it; and HPOCL_DEFICIT_TABLE has to
    // reach the device or the host would solve the table Psi while the kernel
    // solves the quadrature one (up to 0.096 dB of tau apart).
    // HPOCL_CLOPTS wins here too, for the reason cf_main.cpp spells out: an
    // fp64-capable device with the 64 KB spec-minimum constant limit (every
    // NVIDIA GPU measured) cannot build the fp64 deficit table at all, so the
    // PL configuration has to be forceable from the environment.
    const char* copts_env = getenv("HPOCL_CLOPTS");
    std::string kopts = copts_env ? copts_env
                                  : (rt.device_fp64() ? "-DHPOCL_FP64" : "");
#if !HPOCL_DEFICIT_TABLE
    kopts += " -DHPOCL_DEFICIT_TABLE=0";
#endif
    // wbb_apply.cl is loaded for the GLUE roles: the device GemmProvider's three
    // sites are wbb_apply launches, and Runtime::has_kernel() is what decides
    // whether a site offloads. Leaving the source out would not fail loudly -- it
    // would quietly run every GEMM on the PS and report a pass.
    have_cl = rt.load_program({"kernels/decomp.cl", "kernels/threshold_cf.cl",
                               "kernels/wbb_apply.cl"},
                              kopts.c_str());
    if (have_cl)
      printf("[sched_bench] device: %s (fp64=%d, kernel opts '%s')\n",
             rt.device_name(), (int)rt.device_fp64(), kopts.c_str());
  }
  if (!have_cl)
    printf("[sched_bench] no OpenCL device/program -- the DEVICE ROLE IS NOT "
           "TESTED by this run\n");

  int fails = 0;
  for (int i = 0; i < kNCase; ++i) fails += run_case(kCases[i], i, rt, have_cl);
  spread_sweep(rt, have_cl);
  printf(fails ? "[sched_bench] FAIL (%d)\n" : "[sched_bench] ALL PASS\n", fails);
  return fails ? 1 : 0;
}
