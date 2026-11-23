// threshold_cf.cl -- closed-form SVD/THP switching threshold (tau), one
// work-item per user (fp32 by default, HPOCL_FP32).
// in lam [C][K*d] (descending per user) -> out tau_cf_dB / tau_dB [C] and
// flags [C] (bit0 = scan fallback used, 2 = saturated). Newton on Psi(u);
// near-flat spectra fall back to the shared scan in psi_scan.clh.

#if defined(HPOCL_FP64) && defined(HPOCL_FP32)
#error "HPOCL_FP64 and HPOCL_FP32 are mutually exclusive; pass at most one"
#endif
#ifndef HPOCL_FP64
#ifndef HPOCL_FP32
#define HPOCL_FP32 1             // the default datapath, see the header above
#endif
#endif
#include "gh_table.clh"        // GH-48 nodes/weights: the fallback quadrature
#include "deficit_table.clh"   // generated ln D grid: the deployed evaluator
#include "psi_scan.clh"        // near-flat fallback: PSI_SCAN_LO/HI, PSI_SCAN_N
// psi_scan.clh (included above) supplies the near-flat fallback scan
// (PSI_SCAN_LO/HI/N). `real` and PSI_BAND below are the fp64/fp32 selects.


#ifdef HPOCL_FP64
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
typedef double real;
#define PSI_BAND (real)1e-6      // psi dead band: fp64 round-off floor
#else
#pragma OPENCL EXTENSION cl_khr_fp64 : disable
typedef float real;
#define PSI_BAND (real)1e-3      // psi dead band: fp32 round-off floor
#endif
#define R_LOG(x) log(x)          // precision-generic, same spelling on both paths

// HPOCL_DEFICIT_TABLE selects the deficit evaluator: 0 = GH-48 quadrature,
// 1 = generated ln-D table + asymptotic tail (default, tools/gen_gh_table.py).

#ifndef HPOCL_DEFICIT_TABLE
#define HPOCL_DEFICIT_TABLE 1
#endif

#define MAXT 64          // max K*d streams (T = K*d)
#define MAXD 128         // max distance candidates in one LLR (5L when folded)
#ifndef MAXK             // max users K; sizes lm[]/lmu[]/lmd[]
#define MAXK MAXT        // K <= K*d <= MAXT since d >= 1 (see header)
#endif
// compile-time sanity: MAXK <= MAXT (K <= K*d since d >= 1).

typedef char hpocl_maxk_le_maxt_check[(MAXK <= MAXT) ? 1 : -1];

#ifndef NEWTON_ITERS
#define NEWTON_ITERS 20  // deployed fixed budget. 8 steps are enough for every
#endif

real softplus_bits(real t) {
  real ln2 = (real)0.6931471805599453;
  if (t < (real)0) return -t / ln2 + log2((real)1 + exp(t));
  return log2((real)1 + exp(-t));
}

real logsumexp_neg_d2(__private const real* d2, int n) {
  real m = d2[0];
  for (int i = 1; i < n; ++i) m = fmin(m, d2[i]);
  real s = (real)0;
  for (int i = 0; i < n; ++i) s += exp(-(d2[i] - m));
  return -m + R_LOG(s);
}

// axis_deficit: per-axis bit-metric (GMI) deficit on one PAM axis by
// Gauss-Hermite quadrature (no fitted constants).

real axis_deficit(int folded, real gamma, int L, int ma, real delta) {
  // capacity guard: (L+1)/2 levels x (folded ? 5 : 1) candidates per bit class
  // must fit in MAXD.

  if (((L + 1) / 2) * (folded ? 5 : 1) > MAXD) return (real)NAN;
  real sigma2 = (real)1 / ((real)2 * gamma);
  real sigma  = sqrt(sigma2);
  real inv2s2 = (real)0.5 / sigma2;
  real range  = (real)L * delta;
  int  nimg   = folded ? 2 : 0;
  real acc = (real)0;
  __private real d0[MAXD], d1[MAXD];
  for (int j = 0; j < L; ++j) {
    real xj = ((real)2 * j - (L - 1)) * delta * (real)0.5;
    int  gj = j ^ (j >> 1);
    for (int i = 0; i < GH_N; ++i) {
      real y = xj + sqrt((real)2) * sigma * (real)kGH_Z[i];
      real loss = (real)0;
      for (int b = 0; b < ma; ++b) {
        int n0 = 0, n1 = 0;
        for (int q = 0; q < L; ++q) {
          real xq = ((real)2 * q - (L - 1)) * delta * (real)0.5;
          int  cb = (q ^ (q >> 1)) >> b & 1;
          for (int l = -nimg; l <= nimg; ++l) {
            real dy = y - xq - (real)l * range;
            real d2 = dy * dy * inv2s2;
            if (cb == 0) d0[n0++] = d2; else d1[n1++] = d2;
          }
        }
        real llr = logsumexp_neg_d2(d1, n1) - logsumexp_neg_d2(d0, n0);
        real cj  = (real)((gj >> b) & 1);
        loss += softplus_bits(((real)2 * cj - (real)1) * llr);
      }
      acc += (real)kGH_W[i] * loss;
    }
  }
  return acc / ((real)L * sqrt((real)3.141592653589793));
}

// TQ: switch from quadrature to the analytic Q-tail at t = TQ*sqrt(gamma/2).

#define TQ ((real)5)
real q_tail(real t) { return (real)0.5 * erfc(t / sqrt((real)2)); }
// t2_of: t^2 = gamma*3/(M-1), the grid variable; the ln-D table keys on t2.

real t2_of(real gamma, int M) { return gamma * ((real)3 / (M - 1)); }
real c_const(int mod, int M) {
  real L = sqrt((real)M);
  real cmod = (real)4 * (real)3.141592653589793 / R_LOG((real)2);
  return mod == 1 ? cmod : cmod * (L - (real)1) / L;
}
// Generated ln-D table (tools/gen_gh_table.py): kDFT_M row groups, uniform
// ln-t grid over [DFT_TLO, DFT_THI] with DFT_DLNT spacing.

#define DFT_TLO2     ((real)DFT_TLO * (real)DFT_TLO)
#define DFT_THI2     ((real)DFT_THI * (real)DFT_THI)
#define DFT_INV_TLO2 ((real)1 / DFT_TLO2)
#define DFT_INV_DLNT ((real)1 / (real)DFT_DLNT)

int dft_index(int M) {           // row group of M in kDFT_M, -1 if untabulated
#if HPOCL_DEFICIT_TABLE
  for (int i = 0; i < DFT_NM; ++i)
    if (kDFT_M[i] == M) return i;
#else
  (void)M;                       // no unused-parameter warning with the table off
#endif
  return -1;
}

#if HPOCL_DEFICIT_TABLE
// dft_lookup_d: 4-point Lagrange interpolation of ln D on the uniform ln-t
// grid -- arithmetic/order identical to ref/threshold_cf.cpp and tools/cf_ref.py.

void dft_lookup_d(int row, real t2, __private real* val, __private real* dl) {
  // association: ((0.5*log(t2*DFT_INV_TLO2)) * DFT_INV_DLNT), identical everywhere.

  const real xr = (real)0.5 * R_LOG(t2 * DFT_INV_TLO2) * DFT_INV_DLNT;
  const int clamped = !(xr > (real)0) || xr > (real)(DFT_N - 1);
  real x = xr < (real)0 ? (real)0 : (xr > (real)(DFT_N - 1) ? (real)(DFT_N - 1) : xr);
  int i0 = (int)floor(x) - 1;
  if (i0 < 0) i0 = 0;
  if (i0 > DFT_N - 4) i0 = DFT_N - 4;
  const real f = x - (real)i0;
  const real A = f - (real)1, B = f - (real)2, C = f - (real)3;
  const real y0 = kDFT_LND[row][i0],     y1 = kDFT_LND[row][i0 + 1];
  const real y2 = kDFT_LND[row][i0 + 2], y3 = kDFT_LND[row][i0 + 3];
  const real w0 = -(A * B * C) / (real)6;
  const real w1 = f * B * C / (real)2;
  const real w2 = -(f * A * C) / (real)2;
  const real w3 = f * A * B / (real)6;
  *val = w0 * y0 + w1 * y1 + w2 * y2 + w3 * y3;
  if (clamped != 0) { *dl = (real)0; return; }
  const real d0 = -(B * C + A * C + A * B) / (real)6;   // w0'
  const real d1 = (B * C + f * C + f * B) / (real)2;    // w1'
  const real d2 = -(A * C + f * C + f * A) / (real)2;   // w2'
  const real d3 = (A * B + f * B + f * A) / (real)6;    // w3'
  *dl = (d0 * y0 + d1 * y1 + d2 * y2 + d3 * y3) * ((real)0.5 * DFT_INV_DLNT);
}

real dft_lookup(int row, real t2) {
  __private real v, dl;
  dft_lookup_d(row, t2, &v, &dl);
  return v;
}

// dlnQ_asym_dln_t: d lnQ_asym / d ln t, analytic.

real dlnQ_asym_dln_t(real t) {
  const real t2 = t * t, t4 = t2 * t2;
  const real c = (real)1 - (real)1 / t2 + (real)3 / t4;
  if (!(c > (real)0)) return -t2 - (real)1;
  return -t2 - (real)1 + ((real)2 / t2 - (real)12 / t4) / c;
}
#endif

// lnQ_asym: the deployed 3-term asymptotic series for ln Q(t).

real lnQ_asym(real t) {
  real l = (real)-0.5 * t * t - (real)0.5 * R_LOG((real)2 * (real)3.141592653589793) - R_LOG(t);
  real t2 = t * t;
  real c = (real)1 - (real)1 / t2 + (real)3 / (t2 * t2);
  return l + R_LOG(c > (real)0 ? c : (real)1);
}

// deficit_quad: GH-48 quadrature deficit -- the live path when the table is
// compiled out, and the generator of the table entries themselves.

real deficit_quad(int mod, real gamma, int M) {
  if (!(gamma > (real)0)) return log2((real)M);
  int L = (int)rint(sqrt((real)M));
  int ma = (int)rint(log2((real)L));
  real delta = sqrt((real)6 / (M - 1));
  return (real)2 * axis_deficit(mod, gamma, L, ma, delta);
}

// DEFICIT_TABLE_BRANCH selects the evaluator (quadrature / table / tail) at
// compile time; deficit_branch resolves the table row group at runtime.

#define DEFICIT_TABLE_BRANCH 0
#define DEFICIT_TAIL_BRANCH  1
#define DEFICIT_QUAD_BRANCH  2
// deficit_branch: evaluator for (t2, mi), mi = dft_index(M), -1 untabulated;
// t2 below DFT_TLO2 uses the asymptotic tail.

int deficit_branch(real t2, int mi) {
#if HPOCL_DEFICIT_TABLE
  if (mi >= 0) {
    if (t2 > DFT_THI2) return DEFICIT_TAIL_BRANCH;
    if (t2 > (real)0) return DEFICIT_TABLE_BRANCH;
    return DEFICIT_QUAD_BRANCH;
  }
#else
  (void)mi;
#endif
  return t2 > TQ * TQ ? DEFICIT_TAIL_BRANCH : DEFICIT_QUAD_BRANCH;
}

real deficit_fin(real gamma, int M) {
  // gamma <= 0 (zero/NaN singular value): deficit pinned to log2(M).

  if (!(gamma > (real)0)) return log2((real)M);
  const real t2 = t2_of(gamma, M);
  const int mi = dft_index(M);
  const int b = deficit_branch(t2, mi);
#if HPOCL_DEFICIT_TABLE
  if (b == DEFICIT_TABLE_BRANCH) return exp(dft_lookup(2 * mi, t2));
#endif
  if (b == DEFICIT_TAIL_BRANCH) return c_const(0, M) * q_tail(sqrt(t2));
  return deficit_quad(0, gamma, M);
}
real deficit_mod(real gamma, int M) {
  // same guard for the UCD gamma^* (dead user under MMSE-UCD).

  if (!(gamma > (real)0)) return log2((real)M);
  const real t2 = t2_of(gamma, M);
  const int mi = dft_index(M);
  const int b = deficit_branch(t2, mi);
#if HPOCL_DEFICIT_TABLE
  if (b == DEFICIT_TABLE_BRANCH) return exp(dft_lookup(2 * mi + 1, t2));
#endif
  if (b == DEFICIT_TAIL_BRANCH) return c_const(1, M) * q_tail(sqrt(t2));
  return deficit_quad(1, gamma, M);
}

real eta_factor(int d, int M) {
  return (real)1 + (real)(d - 1) / ((real)d * (M - 1));
}

// gmd_lg: sum_i log(max(lam_i, 1e-12)) -- the rho-independent part of GMD's
// g_k, computed once per channel in PsiPre.

real gmd_lg(__private const real* lu, int d) {
  real lg = (real)0;
  for (int i = 0; i < d; ++i) lg += R_LOG(fmax(lu[i], (real)1e-12));
  return lg;
}

void thp_point(__private const real* lu, int d, real rho, int variant,
               real* gam, real* dgam_drho, real eta, real lg_gmd) {
  // thp_point: gamma^* and d gamma^*/d rho for the selected variant:

  if (variant == 0) {                       // GMD: gamma^* = rho g / eta
    real g = exp(lg_gmd / d);
    *gam = rho * g / eta;
    *dgam_drho = g / eta;
    return;
  }
  real p = (real)1, msum = (real)0;         // MMSE-UCD equal output
  for (int i = 0; i < d; ++i) {
    p *= (real)1 + rho * lu[i];
    msum += lu[i] / ((real)1 + rho * lu[i]);
  }
  real geq = pow(p, (real)1 / d) - (real)1;
  *gam = geq / eta;
  *dgam_drho = ((real)1 + geq) * (msum / d) / eta;
}

// phi_sums: the Phi'(u) terms of main.tex:157-165. Dead but documented --
// nothing calls them; the kernel solves Psi(u) directly (psi_u/psi_du below).

void phi_sums(real u, __private const real* lam, int K, int d, int M,
              int variant, real* Sf, real* Sm, real* dSf, real* dSm) {
  real d2 = (real)6 / (M - 1);
  real rho = (real)4 * u / d2;
  real eta = eta_factor(d, M);
  real h = (real)1e-4;
  real sf = (real)0, sm = (real)0, dsf = (real)0, dsm = (real)0;
  for (int k = 0; k < K; ++k) {
    __private const real* lu = lam + k * d;
    for (int i = 0; i < d; ++i) {
      real g = (real)4 * u * lu[i] / d2;
      real D = deficit_fin(g, M);
      sf += D;
      // 0*inf guard: g == 0 (dead stream) makes D/g = +inf, pin the term instead.

      if (g > (real)0) {
        real dlog = (R_LOG(deficit_fin(g * ((real)1 + h), M)) -
                     R_LOG(deficit_fin(g * ((real)1 - h), M))) / ((real)2 * h);
        dsf += lu[i] * (D / g) * dlog;
      }
    }
    real gm, dgdrho;
    thp_point(lu, d, rho, variant, &gm, &dgdrho, eta, gmd_lg(lu, d));
    real D = deficit_mod(gm, M);
    sm += (real)d * D;                           // R_THP = Tm - d * sum_k D_mod
    // same 0*inf guard on the derivative path (gm == 0, dead user).

    if (gm > (real)0) {
      real dlog = (R_LOG(deficit_mod(gm * ((real)1 + h), M)) -
                   R_LOG(deficit_mod(gm * ((real)1 - h), M))) / ((real)2 * h);
      dsm += (real)d * dgdrho * (D / gm) * dlog;
    }
  }
  *Sf = sf; *Sm = sm; *dSf = ((real)4 / d2) * dsf; *dSm = ((real)4 / d2) * dsm;
}

real phi_u(real u, __private const real* lam, int K, int d, int M, int variant) {
  real sf, sm, dsf, dsm;
  phi_sums(u, lam, K, d, M, variant, &sf, &sm, &dsf, &dsm);
  return sf - sm;
}

// ln_deficit_t2_d: log-domain deficit + log-derivative; cancellation- and
// underflow-free -- the form Newton's psi_du solves on the fp32 device.

real ln_deficit_t2_d(int mod, real t2, int M, int mi, __private real* dl) {
  const int b = deficit_branch(t2, mi);
#if HPOCL_DEFICIT_TABLE
  if (b == DEFICIT_TABLE_BRANCH) {
    __private real v;
    dft_lookup_d(2 * mi + mod, t2, &v, dl);
    return v;
  }
#endif
  if (b == DEFICIT_TAIL_BRANCH) {
    const real t = sqrt(t2);
    // d/d ln t2 of (ln C + lnQ_asym) = (t dlnQ/d ln t)/2.

    *dl = (real)0.5 * dlnQ_asym_dln_t(t);
    return R_LOG(c_const(mod, M)) + lnQ_asym(t);
  }
  {
    const real k3 = (real)3 / (M - 1);
    const real v = R_LOG(deficit_quad(mod, t2 / k3, M));
    if (!(t2 > (real)0)) *dl = (real)0;          // dead stream: D pinned to log2(M)
    else {
      const real h = (real)1e-3;
      const real vp = R_LOG(deficit_quad(mod, t2 * ((real)1 + h) / k3, M));
      const real vm = R_LOG(deficit_quad(mod, t2 * ((real)1 - h) / k3, M));
      *dl = (vp - vm) / ((real)2 * h);
    }
    return v;
  }
}
real ln_deficit_t2(int mod, real t2, int M, int mi) {
  __private real dl;
  return ln_deficit_t2_d(mod, t2, M, mi, &dl);
}
real ln_deficit_impl(int mod, real gamma, int M) {
  return ln_deficit_t2(mod, t2_of(gamma, M), M, dft_index(M));
}
real ln_deficit_fin(real gamma, int M) { return ln_deficit_impl(0, gamma, M); }
real ln_deficit_mod(real gamma, int M) { return ln_deficit_impl(1, gamma, M); }

// PsiPre: per-channel precompute for the Psi hot path -- every member depends
// on (lam, K, d, M, variant) only, none on u.

typedef struct {
  int  nf, K, d, M, mi;
  int  variant;
  real k3, d2, eta, log_d;
  real cr[MAXT];        // 2*lam_i, so t_i^2 = u*cr[i]: no sqrt and no division
  real lg_gmd[MAXK];    // sum_i log(max(lam_i,1e-12)) per user: GMD's g_k
} PsiPre;

// nf = -1 marks a configuration outside compiled capacity (entry guard rejects it).

void psi_precompute(__private const real* lam, int K, int d, int M, int variant,
                    __private PsiPre* w) {
  w->K = K; w->d = d; w->M = M; w->variant = variant;
  const int nf = K * d;
  if (nf < 1 || nf > MAXT || K > MAXK || d < 1 || M < 4) {
    w->nf = -1; w->mi = -1;
    w->k3 = w->d2 = w->eta = w->log_d = (real)NAN;
    return;
  }
  w->nf = nf;
  w->mi = dft_index(M);
  w->k3 = (real)3 / (M - 1);
  w->d2 = (real)6 / (M - 1);
  w->eta = eta_factor(d, M);
  w->log_d = R_LOG((real)d);
  for (int i = 0; i < nf; ++i) w->cr[i] = (real)2 * lam[i];
  for (int k = 0; k < K; ++k) w->lg_gmd[k] = gmd_lg(lam + k * d, d);
}

// psi_u: Psi(u) for one user. lam is passed separately from PsiPre (MMSE-UCD's
// gamma^* is a product over (1 + rho*lam_i) rebuilt in the Newton loop).

real psi_u(real u, __private const real* lam, __private const PsiPre* w) {
  const int nf = w->nf, K = w->K, d = w->d, M = w->M;
  if (nf < 0) return (real)NAN;
  const real rho = (real)4 * u / w->d2;
  __private real lf[MAXT];
  __private real lm[MAXK];   // one entry per user k = 0..K-1, K <= MAXK
  real mf = (real)-1e30, mm = (real)-1e30;
  for (int k = 0; k < K; ++k) {
    __private const real* lu = lam + k * d;
    for (int i = 0; i < d; ++i) {
      // t_i^2 = u*cr_i (2*lam_i folded into PsiPre); gamma_i rebuilt only for derivatives.

      lf[k * d + i] = ln_deficit_t2(0, u * w->cr[k * d + i], M, w->mi);
      mf = fmax(mf, lf[k * d + i]);
    }
    real gm2, dg;
    thp_point(lu, d, rho, w->variant, &gm2, &dg, w->eta, w->lg_gmd[k]);
    lm[k] = w->log_d + ln_deficit_t2(1, gm2 * w->k3, M, w->mi);
    mm = fmax(mm, lm[k]);
  }
  real sf = (real)0, sm = (real)0;
  for (int i = 0; i < nf; ++i) sf += exp(lf[i] - mf);
  for (int i = 0; i < K; ++i) sm += exp(lm[i] - mm);
  return (mf + R_LOG(sf)) - (mm + R_LOG(sm));
}

// psi_du_central: 3-point central-difference dPsi/du, kept only as the
// cross-check for the analytic psi_du (3 Psi fills per call vs 1).

real psi_du_central(real u, __private const real* lam, __private const PsiPre* w) {
  const int nf = w->nf, K = w->K, d = w->d, M = w->M;
  if (nf < 0) return (real)NAN;
  real d2 = w->d2;
  real eta = w->eta;
  real h = (real)1e-3;
  real up = u * ((real)1 + h), dn = u * ((real)1 - h);
  __private real lf[MAXT], lfu[MAXT], lfd[MAXT];
  __private real lm[MAXK], lmu[MAXK], lmd[MAXK];   // indexed by k < K
  for (int pass = 0; pass < 3; ++pass) {
    real uu = (pass == 0) ? u : (pass == 1 ? up : dn);
    real rhou = (real)4 * uu / d2;
    for (int k = 0; k < K; ++k) {
      __private const real* lu = lam + k * d;
      for (int i = 0; i < d; ++i) {
        real v = ln_deficit_t2(0, uu * w->cr[k * d + i], M, w->mi);
        if (pass == 0) lf[k * d + i] = v;
        else if (pass == 1) lfu[k * d + i] = v;
        else lfd[k * d + i] = v;
      }
      real gm, dg;
      thp_point(lu, d, rhou, w->variant, &gm, &dg, eta, w->lg_gmd[k]);
      real v = w->log_d + ln_deficit_t2(1, gm * w->k3, M, w->mi);
      if (pass == 0) lm[k] = v; else if (pass == 1) lmu[k] = v; else lmd[k] = v;
    }
  }
  real mf = lf[0], mm = lm[0];
  for (int i = 0; i < nf; ++i) mf = fmax(mf, lf[i]);
  for (int i = 0; i < K; ++i) mm = fmax(mm, lm[i]);
  real sf = (real)0, sm = (real)0;
  for (int i = 0; i < nf; ++i) sf += exp(lf[i] - mf);
  for (int i = 0; i < K; ++i) sm += exp(lm[i] - mm);
  real duu = up - dn;
  real acc = (real)0;
  for (int i = 0; i < nf; ++i)
    acc += (exp(lf[i] - mf) / sf) * ((lfu[i] - lfd[i]) / duu);
  for (int i = 0; i < K; ++i)
    acc -= (exp(lm[i] - mm) / sm) * ((lmu[i] - lmd[i]) / duu);
  return acc;
}

// psi_du: deployed dPsi/du with per-term log-derivatives taken ANALYTICALLY --
// one Psi fill per call; agrees with psi_du_central.

real psi_du(real u, __private const real* lam, __private const PsiPre* w) {
  const int nf = w->nf, K = w->K, d = w->d, M = w->M;
  if (nf < 0) return (real)NAN;
  if (!(u > (real)0)) return (real)NAN;   // the central form is NaN here too
  const real rho = (real)4 * u / w->d2;
  __private real lf[MAXT], dlf[MAXT];   // ln D_fin and d ln D_fin / d ln u
  __private real lm[MAXK], dlm[MAXK];   // ln D_mod  and d ln D_mod  / d ln u
  real mf = (real)-1e30, mm = (real)-1e30;
  for (int k = 0; k < K; ++k) {
    __private const real* lu = lam + k * d;
    for (int i = 0; i < d; ++i) {
      __private real dl;
      lf[k * d + i] = ln_deficit_t2_d(0, u * w->cr[k * d + i], M, w->mi, &dl);
      dlf[k * d + i] = dl;                    // chain factor is exactly 1
      mf = fmax(mf, lf[k * d + i]);
    }
    real gm, dg;
    thp_point(lu, d, rho, w->variant, &gm, &dg, w->eta, w->lg_gmd[k]);
    __private real dl;
    lm[k] = w->log_d + ln_deficit_t2_d(1, gm * w->k3, M, w->mi, &dl);
    // gamma^* <= 0: dead user, derivative exactly 0 (otherwise rho*dg/gm is 0/0).

    dlm[k] = (gm > (real)0) ? dl * (rho * dg / gm) : (real)0;
    mm = fmax(mm, lm[k]);
  }
  real sf = (real)0, sm = (real)0;
  for (int i = 0; i < nf; ++i) sf += exp(lf[i] - mf);
  for (int i = 0; i < K; ++i) sm += exp(lm[i] - mm);
  real acc = (real)0;
  for (int i = 0; i < nf; ++i) acc += (exp(lf[i] - mf) / sf) * dlf[i];
  for (int i = 0; i < K; ++i) acc -= (exp(lm[i] - mm) / sm) * dlm[i];
  return acc / u;
}

// threshold_cf: the SVD/THP switching-threshold kernel. Guards, precompute,
// closed-form u0 with the below-grid short circuit, then scan fallback +
// bracketed Newton; outputs tau_cf_dB (closed form) and tau_dB (solved),
// flags: bit0 = fallback used, 2 = saturated ordering.
__kernel void threshold_cf(__global const real* restrict lam,  // [C][K*d]
                           int C, int K, int d, int M, int variant,
                           __global real* restrict tau_cf_dB,
                           __global real* restrict tau_dB,
                           __global int*  restrict flags)      // bit0 fallback
{
  int c = get_global_id(0);
  if (c >= C) return;
  int T = K * d;
  // hard capacity guards: spec[] / lm[] / deficit arrays are sized MAXT/MAXK/MAXD.

  int Lchk = (int)rint(sqrt((real)M));
  // non-square M modelled as Lchk-level PAM (ma = log2(Lchk) bits), then rejected.

  if (T < 1 || T > MAXT || K > MAXK || d < 1 || M < 4 || Lchk * Lchk != M ||
      5 * ((Lchk + 1) / 2) > MAXD) {                 // folded MAXD capacity
    tau_cf_dB[c] = (real)NAN; tau_dB[c] = (real)NAN; flags[c] = 4; return;
  }
  __private real spec[MAXT];
  for (int i = 0; i < T; ++i) spec[i] = lam[(size_t)c * T + i];

  // one PsiPre for the whole solve: fallback scan, bracket extension and Newton
  // all reuse it.

  __private PsiPre w;
  psi_precompute(spec, K, d, M, variant, &w);

  real lam_min = spec[0], lam_max = spec[0];
  for (int i = 1; i < T; ++i) {
    lam_min = fmin(lam_min, spec[i]);
    lam_max = fmax(lam_max, spec[i]);
  }
  int mult = 0;
  for (int i = 0; i < T; ++i)
    if (fabs(spec[i] - lam_min) <= (real)1e-9 * fmax((real)1, lam_min)) ++mult;
  if (mult == 0) mult = 1;

  real g_min = (real)1e300;
  for (int k = 0; k < K; ++k)               // lg is PsiPre's, same expression
    g_min = fmin(g_min, exp(w.lg_gmd[k] / d));

  int L = (int)rint(sqrt((real)M));
  real cmod = (real)4 * (real)3.141592653589793 / R_LOG((real)2);
  real cfin = cmod * (L - 1) / L;
  real eta  = eta_factor(d, M);
  real denom = g_min / eta - lam_min;
  real u0 = (real)-1;
  int fallback = 0;
  if (denom > (real)1e-12) {
    real num = R_LOG((real)d * cmod / ((real)mult * cfin)) -
               (real)0.5 * R_LOG(g_min / (eta * lam_min));
    u0 = num / denom;
  }
  // whole-spectrum-below-the-grid short circuit: if every t_i^2 lies below the
  // table grid, tau_cf is the asymptotic-limit closed form -- no scan needed.

  const int below_grid = (w.mi >= 0) && (lam_max > (real)0) && isfinite(lam_max) &&
                         ((real)2 * (real)PSI_SCAN_HI * lam_max < DFT_TLO2);
  if (!(u0 > (real)0) && !below_grid) {   // near-flat: first-transition scan
    // fallback scan: HPOCL_PSI_SCAN macro (same text in ref/threshold_cf.cpp and
    // tools/cf_ref.py) brackets the Psi sign change; Newton then refines inside
    // the bracket with the analytic psi_du.

    int found = 0;
    HPOCL_PSI_SCAN(real, psi_u(psa_u, spec, &w), PSI_BAND, u0, found);
    fallback = found;
  } else if (!(u0 > (real)0)) {
    fallback = 0;                           // short-circuited: flags[c] = 2 below
  }
  if (!(u0 > (real)0)) {                    // uniform ordering: saturate
    real inf = psi_u((real)1, spec, &w) > (real)0 ? -INFINITY : INFINITY;
    tau_cf_dB[c] = inf; tau_dB[c] = inf; flags[c] = 2; return;
  }

  real taucf = (real)10 * log10((real)4 * T * (M - 1) * u0 / (real)6);
  real lo = u0 / (real)4, hi = u0 * (real)4, u = u0;
  for (int e = 0; e < 6 && psi_u(lo, spec, &w) > (real)0; ++e) lo /= (real)4;
  for (int e = 0; e < 6 && psi_u(hi, spec, &w) < (real)0; ++e) hi *= (real)4;
  for (int it = 0; it < NEWTON_ITERS; ++it) {
    real f = psi_u(u, spec, &w);
    if (f <= (real)0) lo = u; else hi = u;
    real fp = psi_du(u, spec, &w);
    real un = u - f / (fp != (real)0 ? fp : (real)1);
    if (!(un > lo && un < hi)) un = (real)0.5 * (lo + hi);
    u = un;
  }
  tau_cf_dB[c] = taucf;
  tau_dB[c] = (real)10 * log10((real)4 * T * (M - 1) * u / (real)6);
  flags[c] = fallback;
}
