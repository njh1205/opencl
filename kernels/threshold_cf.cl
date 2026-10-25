//-----------------------------------------------------------------------------
// operator: threshold_cf  -- PL-side deployment of the closed-form SVD/THP
// switching threshold (main.tex eq. cf) + safeguarded Newton on the exact
// crossing equation (eq. phi). One work-item per scheduling instant/channel:
// the per-channel work is a scalar root find, so channel-level parallelism is
// what maps to the fabric; the inner quadrature is a fixed unrolled loop.
//
// Build flags: -DHPOCL_FP32 switches the datapath to single precision for
// area-constrained PL builds; default is double (PS CPU / fp64-capable PL).
//
// Compile-time capacity. Every private array below is fixed-size so the PL
// build gets a constant register/BRAM footprint; a channel whose (K,d,M) does
// not fit is rejected by the entry guard of threshold_cf() with
// flags[c] = 4 and NaN thresholds instead of overrunning the arrays
// (flags: 0 = closed form, 1 = scan fallback, 2 = saturated +-inf,
//          4 = configuration outside the compiled capacity, or M not a perfect
//              square). ref/threshold_cf.cpp CfResult::flag carries the same
//              encoding on the PS side -- keep the two predicates identical.
//
//   MAXT = 64    max streams per channel, T = K*d. Sizes spec[]/lf[]/lfu[]/lfd[].
//   MAXK = MAXT  max users K, sizes lm[]/lmu[]/lmd[]. K <= MAXK is forced by
//                K <= K*d <= MAXT because every user carries d >= 1 stream, so
//                MAXK == MAXT is the tightest bound that holds without knowing
//                d at compile time. Area-constrained PL builds that can
//                guarantee K <= n at the host may pass -DMAXK=n; the entry
//                guard then rejects K > n (a MAXK > MAXT typo is a compile
//                error, see the static check next to the define).
//   MAXD = 128   max distance candidates held in ONE of d0[]/d1[] inside the
//                Gauss-Hermite quadrature of axis_deficit(). Per label bit the
//                loop walks all L levels x (2*nimg+1) modulo images, nimg = 2
//                when folded (deficit_mod) and 0 when unfolded (deficit_fin),
//                and sorts each candidate into d0[]/d1[] by its Gray bit. So
//                the naive worst case is L*(2*nimg+1) = 5L entries in ONE array
//                (5L <= MAXD => L <= 25 => M <= 625), but that case cannot
//                occur: the reflected Gray label j^(j>>1) puts exactly L/2 of
//                the L levels in each bit class, so each array receives
//                    ceil(L/2) * (2*nimg+1)  =  5*ceil(L/2) ~ 2.5L  (folded)
//                                             ceil(L/2)   ~ 0.5L  (unfolded)
//                The guards use that exact bound:
//                    5*ceil(L/2) <= MAXD  =>  L <= 50  =>  M = L^2 <= 2500.
//                M in {4,16,64,256,1024} (L = 2,4,8,16,32 -> 5,10,20,40,80 per
//                array) fit; M = 4096 (L = 64 -> 160 > 128) does NOT and is
//                rejected with flags = 4. The unfolded path is never binding
//                (it needs only ceil(L/2) <= MAXD).
//-----------------------------------------------------------------------------
#include "gh_table.clh"        // GH-48 nodes/weights: the fallback quadrature
#include "deficit_table.clh"   // generated ln D grid: the deployed evaluator

#ifdef HPOCL_FP32
#pragma OPENCL EXTENSION cl_khr_fp64 : disable
typedef float real;
#define R_LOG(x) log(x)
#define PSI_BAND (real)1e-3      // psi dead band: fp32 round-off floor
#else
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
typedef double real;
#define R_LOG(x) log(x)
#define PSI_BAND (real)1e-6
#endif

//-----------------------------------------------------------------------------
// HPOCL_DEFICIT_TABLE -- which deficit evaluator this kernel deploys. Same
// switch, same three-way split and same defaults as ref/threshold_cf.cpp and
// tools/cf_ref.py; flip all three together.
//   1 (default): ln D from kernels/deficit_table.clh for a tabulated M with
//                DFT_TLO <= t <= DFT_THI, the analytic tail ln C + lnQ_asym(t)
//                above DFT_THI, and the GH-48 quadrature below DFT_TLO or for an
//                M the table does not cover. main.tex:169-171's "log D on a
//                uniform log-gamma grid of 1600 points per modulation and
//                branch, with the analytic tail C*Q(t) beyond it".
//   0:           the pre-table path, GH-48 up to TQ and the tail above it.
// The quadrature stays compiled in and reachable under BOTH settings: it is the
// fallback for an untabulated M and for t < DFT_TLO, and it is the reference
// cf_bench's maxq column checks the table against. The PL pays 62.5 KB of fp32
// __constant for the shipped grid (15 BRAM36 or 2 URAM288) and gets one log +
// four table reads + 15 flops per deficit instead of L*ma*GH_N*L*(2*nimg+1)
// quadrature work.
#ifndef HPOCL_DEFICIT_TABLE
#define HPOCL_DEFICIT_TABLE 1
#endif

#define MAXT 64          // max K*d streams (T = K*d)
#define MAXD 128         // max distance candidates in one LLR (5L when folded)
#ifndef MAXK             // max users K; sizes lm[]/lmu[]/lmd[]
#define MAXK MAXT        // K <= K*d <= MAXT since d >= 1 (see header)
#endif
// compile-time sanity: a -DMAXK larger than MAXT would make the K bound looser
// than the stream bound it is derived from; a negative array size fails the build.
typedef char hpocl_maxk_le_maxt_check[(MAXK <= MAXT) ? 1 : -1];

#ifndef NEWTON_ITERS
#define NEWTON_ITERS 20  // deployed fixed budget. 8 steps are enough for every
                         // channel cf_bench compares (flat_ratio >= 0.05: max
                         // 4.25e-6 dB vs a Brent root); 20 is needed by the
                         // deeply near-flat regime the bench skips, where only
                         // 2 of 20 Newton steps are accepted and the rest
                         // bisect (8 steps cost 1.0e-1 dB there). Measured
                         // sweeps, per-iteration cost and the accepted-step
                         // trace: newton_iters in ref/threshold_cf.h.
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

// per-axis bit-metric deficit by Gauss-Hermite quadrature (no fitted consts)
real axis_deficit(int folded, real gamma, int L, int ma, real delta) {
  // d0[]/d1[] capacity: per label bit each array receives the candidates of
  // one Gray bit class, i.e. ceil(L/2) levels x (2*nimg+1) modulo images
  // (nimg = 2 folded, 0 unfolded). The reflected Gray code splits the L levels
  // exactly in half, so ceil(L/2) is a tight upper bound for any L. Refuse M
  // too large for MAXD instead of writing past the array (threshold_cf()
  // already rejects such a channel with flags = 4; this keeps the helper safe
  // if it is ever called directly).
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

// Quadrature -> analytic-tail switch: t = delta*sqrt(gamma/2) (t_of below).
// SCOPE, since the ln-D table landed: TQ now governs ONLY the fallback path (an
// M that kernels/deficit_table.clh does not cover, or HPOCL_DEFICIT_TABLE=0).
// The deployed path switches quadrature -> table at DFT_TLO (1e-4, where GH-48
// is worth 1e-9 dB, so that handover costs nothing) and table -> tail at DFT_THI
// (32, where the plain C*Q(t) tail is worth 0.0052 dB, against the 0.3046 dB
// measured below for this switch). tools/check_capacity.py still parses the
// three TQ spellings, and they still have to agree.
// Was 10. GH-48's outermost node is 8.975315 (a 12.69 sigma noise span) and the
// dominant error event sits at z = t/sqrt(2) = 6.72 for t = 9.5, i.e. INSIDE the
// span, so the rule is not blind by truncation -- it fails on RESOLUTION: the
// softplus hinge is 1/(2*sqrt(2)*t) = 0.037 wide in z there against a local node
// spacing of 0.43. Measured GH-48 deficit error vs a converged reference:
// -0.0318 dB at t = 4, -0.1205 at t = 5, -0.6685 at t = 7, +1.6161 at t = 9.5,
// +1.9202 at t = 10; the analytic tail C*Q(t) is +0.2684 / +0.1841 / +0.1006 /
// +0.0566 / +0.0513 dB at the same t. The branches cross at t ~ 5.3, so 5 caps
// the worst-case deficit error at 0.184 dB where 10 cost 1.9202 dB. Raising the
// order fixes it monotonically (at t = 9.5: GH-96 +0.2319, GH-192 +0.0337,
// GH-384 -0.0072 dB), which a truncation story cannot explain; an earlier
// version of this comment said "~8.0" and "z ~ t", both wrong.
// The switch is still a HARD step in D (+0.3046 dB here, -1.8689 dB at 10), so
// Psi becomes a staircase: measured max step 0.0709 (TQ 5) vs 0.2584 (TQ 10),
// and 11 Psi sign changes on one cf_bench channel at TQ 10 vs exactly 1 at TQ 5
// on all 39 measurable channels. The "0.55 dB tau error at denom = 1e-2" an
// earlier version quoted does not reproduce: 0.000 dB, identical at both values.
// Full table, accuracy-vs-converged-reference numbers and the two rejected
// alternatives: kTQ in ref/threshold_cf.cpp and the decision record in
// README.md. tools/cf_ref.py carries the same value and tools/check_capacity.py
// fails the build on drift, so change all three together.
// Single macro: used by the fp32 and fp64 datapaths and by the ln-domain pair.
#define TQ ((real)5)
real q_tail(real t) { return (real)0.5 * erfc(t / sqrt((real)2)); }
real t_of(real gamma, int M) {
  return sqrt((real)6 / (M - 1)) * sqrt(gamma * (real)0.5);
}
real c_const(int mod, int M) {
  real L = sqrt((real)M);
  real cmod = (real)4 * (real)3.141592653589793 / R_LOG((real)2);
  return mod == 1 ? cmod : cmod * (L - (real)1) / L;
}
//---------------------------------------------------------------------------
// the generated ln-D table: row selection and interpolation
//---------------------------------------------------------------------------
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
// 4-point Lagrange interpolation of ln D on the uniform ln-t grid. IDENTICAL
// arithmetic, in the same order, as ref/threshold_cf.cpp dft_lookup() and
// tools/cf_ref.py _dft_lookup(); tools/gen_deficit_table.py interp_ln_deficit()
// is the reference copy. cf_bench's maxtab and maxcl columns compare the three,
// so a change here is a change in four places.
//
// Indexed by log(t/DFT_TLO)/DFT_DLNT rather than (log t - DFT_LNTLO)/DFT_DLNT
// BECAUSE this datapath may be fp32: the second form carries log(t)'s ~1e-6
// absolute error into a subtraction whose result is only 12.66, i.e. 1.3e-4
// grid steps (2.3e-3 dB of ln D at t = 32), while the ratio form logs a
// well-conditioned number and lands at 7.6e-6 steps (1.3e-4 dB) -- still 350x
// inside this kernel's 5e-2 dB budget against the fp64 host.
//
// The weights are the Lagrange basis prod_{m != k} (f-m)/(k-m) for nodes at
// f = 0,1,2,3: each is 1 at its own node, 0 at the other three, and they sum to
// 1, so the two clamped end intervals interpolate and never extrapolate.
real dft_lookup(int row, real t) {
  real x = R_LOG(t / (real)DFT_TLO) / (real)DFT_DLNT;
  x = x < (real)0 ? (real)0 : (x > (real)(DFT_N - 1) ? (real)(DFT_N - 1) : x);
  int i0 = (int)floor(x) - 1;
  if (i0 < 0) i0 = 0;
  if (i0 > DFT_N - 4) i0 = DFT_N - 4;
  const real f = x - (real)i0;
  const real w0 = -(f - (real)1) * (f - (real)2) * (f - (real)3) / (real)6;
  const real w1 = f * (f - (real)2) * (f - (real)3) / (real)2;
  const real w2 = -f * (f - (real)1) * (f - (real)3) / (real)2;
  const real w3 = f * (f - (real)1) * (f - (real)2) / (real)6;
  return w0 * kDFT_LND[row][i0] + w1 * kDFT_LND[row][i0 + 1] +
         w2 * kDFT_LND[row][i0 + 2] + w3 * kDFT_LND[row][i0 + 3];
}
#endif

// ln Q(t), the deployed 3-term asymptotic series. Defined here (it used to sit
// next to psi_u) because both the table's tail handover and the fallback's TQ
// switch use it.
real lnQ_asym(real t) {
  real l = (real)-0.5 * t * t - (real)0.5 * R_LOG((real)2 * (real)3.141592653589793) - R_LOG(t);
  real t2 = t * t;
  real c = (real)1 - (real)1 / t2 + (real)3 / (t2 * t2);
  return l + R_LOG(c > (real)0 ? c : (real)1);
}

//---------------------------------------------------------------------------
// the live GH-48 quadrature on its own: no table, no tail branch. Fallback and
// accuracy reference -- keep it reachable (cf_bench maxq compares it against the
// oracle's own quadrature, so a table that broke the model would show up there).
//---------------------------------------------------------------------------
real deficit_quad(int mod, real gamma, int M) {
  if (!(gamma > (real)0)) return log2((real)M);
  int L = (int)rint(sqrt((real)M));
  int ma = (int)rint(log2((real)L));
  real delta = sqrt((real)6 / (M - 1));
  return (real)2 * axis_deficit(mod, gamma, L, ma, delta);
}

// Which of the three evaluators applies at (gamma, M), plus the table row group
// (*mi_out = -1 when untabulated) and t. ONE selector for the linear and the
// log-domain form, so the two cannot take different branches -- a divergence
// there makes Psi and the linear Phi describe different crossings, which is the
// class of bug that once cost 2.78e-5 dB of systematic maxtau.
// gamma <= 0 gives t = NaN: every comparison fails, so it lands on
// DEFICIT_QUAD, where deficit_quad's own guard saturates the deficit at
// log2(M) bits. That is exactly the pre-table behaviour, and it is why
// ln_deficit_* needs no gamma guard (mirrors ref/threshold_cf.cpp).
#define DEFICIT_TABLE_BRANCH 0
#define DEFICIT_TAIL_BRANCH  1
#define DEFICIT_QUAD_BRANCH  2
int deficit_branch(real gamma, int M, __private int* mi_out, __private real* t_out) {
  const real t = t_of(gamma, M);
  *t_out = t;
  *mi_out = -1;
#if HPOCL_DEFICIT_TABLE
  const int mi = dft_index(M);
  if (mi >= 0) {
    *mi_out = mi;
    if (t >= (real)DFT_TLO && t <= (real)DFT_THI) return DEFICIT_TABLE_BRANCH;
    if (t > (real)DFT_THI) return DEFICIT_TAIL_BRANCH;
    return DEFICIT_QUAD_BRANCH;   // below the grid: GH-48, worth 1.1e-4 dB at
  }                               // t = 2 and 1e-9 dB at the grid edge
#endif
  return t > TQ ? DEFICIT_TAIL_BRANCH : DEFICIT_QUAD_BRANCH;
}

real deficit_fin(real gamma, int M) {
  // gamma <= 0 (or NaN): the singular value is exactly zero, sigma^2 = 1/(2g)
  // would be inf and d2 = dy^2 * (0.5/sigma2) = 0 * inf = NaN in the
  // quadrature. Mirror ref/threshold_cf.cpp: the deficit saturates at the full
  // label entropy log2(M) bits. !(g > 0) also catches NaN, and holds for both
  // real = float (HPOCL_FP32) and real = double.
  if (!(gamma > (real)0)) return log2((real)M);
  real t; int mi;
  const int b = deficit_branch(gamma, M, &mi, &t);
#if HPOCL_DEFICIT_TABLE
  if (b == DEFICIT_TABLE_BRANCH) return exp(dft_lookup(2 * mi, t));
#endif
  if (b == DEFICIT_TAIL_BRANCH) return c_const(0, M) * q_tail(t);
  return deficit_quad(0, gamma, M);
}
real deficit_mod(real gamma, int M) {
  // same guard as deficit_fin: gamma^*_k == 0 happens for a dead user under
  // MMSE-UCD (prod(1+rho*lam) == 1 -> geq = 0) and for GMD when lam == 0.
  if (!(gamma > (real)0)) return log2((real)M);
  real t; int mi;
  const int b = deficit_branch(gamma, M, &mi, &t);
#if HPOCL_DEFICIT_TABLE
  if (b == DEFICIT_TABLE_BRANCH) return exp(dft_lookup(2 * mi + 1, t));
#endif
  if (b == DEFICIT_TAIL_BRANCH) return c_const(1, M) * q_tail(t);
  return deficit_quad(1, gamma, M);
}

real eta_factor(int d, int M) {
  return (real)1 + (real)(d - 1) / ((real)d * (M - 1));
}

void thp_point(__private const real* lu, int d, real rho, int variant,
               real* gam, real* dgam_drho, real eta) {
  // returns gamma^* and d gamma^*/d rho (see host ref for the two forms)
  if (variant == 0) {                       // GMD: gamma^* = rho g / eta
    real lg = (real)0;
    for (int i = 0; i < d; ++i) lg += R_LOG(fmax(lu[i], (real)1e-12));
    real g = exp(lg / d);
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

// NOTE (dead but documented): phi_sums/phi_u implement the Phi'(u) formula of
// main.tex:157-165. Nothing calls them -- threshold_cf() below solves the
// log-domain balance Psi = ln S_fin - ln S_mod with psi_u/psi_du, which has the
// same root set as Phi = S_fin - S_mod but neither cancels nor underflows in
// fp32. Kept because the paper documents this form.
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
      // g == 0 (dead stream, lu[i] == 0) makes D/g = log2(M)/0 = +inf while
      // lu[i] == 0, so the product is 0 * inf = NaN. Skipping is exact: the
      // deficit guard pins D to log2(M) for every u, so dD/du = 0 and the term
      // is 0 anyway. Mirrors ref/threshold_cf.cpp phi_sums.
      if (g > (real)0) {
        real dlog = (R_LOG(deficit_fin(g * ((real)1 + h), M)) -
                     R_LOG(deficit_fin(g * ((real)1 - h), M))) / ((real)2 * h);
        dsf += lu[i] * (D / g) * dlog;
      }
    }
    real gm, dgdrho;
    thp_point(lu, d, rho, variant, &gm, &dgdrho, eta);
    real D = deficit_mod(gm, M);
    sm += (real)d * D;                           // R_THP = Tm - d * sum_k D_mod
    // same 0 * inf guard: gm == 0 for a dead user and dgdrho is 0 there.
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

// log-domain deficits + log-sum-exp balance: cancellation-free AND
// underflow-free; this is the form Newton solves on the fp32 PL datapath.
// lnQ_asym moved up next to deficit_branch, which also needs it.
real ln_deficit_impl(int mod, real gamma, int M) {
  real t; int mi;
  const int b = deficit_branch(gamma, M, &mi, &t);
#if HPOCL_DEFICIT_TABLE
  if (b == DEFICIT_TABLE_BRANCH) return dft_lookup(2 * mi + mod, t);
#endif
  if (b == DEFICIT_TAIL_BRANCH) return R_LOG(c_const(mod, M)) + lnQ_asym(t);
  return R_LOG(deficit_quad(mod, gamma, M));
}
real ln_deficit_fin(real gamma, int M) { return ln_deficit_impl(0, gamma, M); }
real ln_deficit_mod(real gamma, int M) { return ln_deficit_impl(1, gamma, M); }

real psi_u(real u, __private const real* lam, int K, int d, int M, int variant) {
  real d2 = (real)6 / (M - 1);
  real rho = (real)4 * u / d2;
  real eta = eta_factor(d, M);
  __private real lf[MAXT];
  __private real lm[MAXK];   // one entry per user k = 0..K-1, K <= MAXK
  real mf = (real)-1e30, mm = (real)-1e30;
  for (int k = 0; k < K; ++k) {
    __private const real* lu = lam + k * d;
    for (int i = 0; i < d; ++i) {
      lf[k * d + i] = ln_deficit_fin((real)4 * u * lu[i] / d2, M);
      mf = fmax(mf, lf[k * d + i]);
    }
    real gm2, dg;
    thp_point(lu, d, rho, variant, &gm2, &dg, eta);
    lm[k] = R_LOG((real)d) + ln_deficit_mod(gm2, M);
    mm = fmax(mm, lm[k]);
  }
  real sf = (real)0, sm = (real)0;
  for (int i = 0; i < K * d; ++i) sf += exp(lf[i] - mf);
  for (int i = 0; i < K; ++i) sm += exp(lm[i] - mm);
  return (mf + R_LOG(sf)) - (mm + R_LOG(sm));
}

real psi_du(real u, __private const real* lam, int K, int d, int M, int variant) {
  real d2 = (real)6 / (M - 1);
  real eta = eta_factor(d, M);
  real h = (real)1e-3;
  real up = u * ((real)1 + h), dn = u * ((real)1 - h);
  int nf = K * d;
  __private real lf[MAXT], lfu[MAXT], lfd[MAXT];
  __private real lm[MAXK], lmu[MAXK], lmd[MAXK];   // indexed by k < K
  for (int pass = 0; pass < 3; ++pass) {
    real uu = (pass == 0) ? u : (pass == 1 ? up : dn);
    real rhou = (real)4 * uu / d2;
    for (int k = 0; k < K; ++k) {
      __private const real* lu = lam + k * d;
      for (int i = 0; i < d; ++i) {
        real v = ln_deficit_fin((real)4 * uu * lu[i] / d2, M);
        if (pass == 0) lf[k * d + i] = v;
        else if (pass == 1) lfu[k * d + i] = v;
        else lfd[k * d + i] = v;
      }
      real gm, dg;
      thp_point(lu, d, rhou, variant, &gm, &dg, eta);
      real v = R_LOG((real)d) + ln_deficit_mod(gm, M);
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

__kernel void threshold_cf(__global const real* restrict lam,  // [C][K*d]
                           int C, int K, int d, int M, int variant,
                           __global real* restrict tau_cf_dB,
                           __global real* restrict tau_dB,
                           __global int*  restrict flags)      // bit0 fallback
{
  int c = get_global_id(0);
  if (c >= C) return;
  int T = K * d;
  // hard capacity guards (see the header): spec[] below is MAXT, lm[] in
  // psi_u/psi_du is MAXK, and deficit_mod() needs 5*ceil(L/2) <= MAXD.
  // Reject the channel instead of overrunning private memory.
  int Lchk = (int)rint(sqrt((real)M));
  // Lchk*Lchk != M: a non-square M is silently modelled as an Lchk-level PAM
  // with ma = rint(log2(Lchk)) bits per axis against a log2(M) bit label
  // entropy. Host, device and oracle all agree on that, so it is not a
  // cross-implementation mismatch -- it is just wrong, and cheaper to refuse
  // than to support. ref/threshold_cf.cpp applies the identical predicate
  // (kCapMaxT / kCapMaxK / kCapMaxD), and runtime/hpocl_runtime.cpp checks it
  // before the launch.
  if (T < 1 || T > MAXT || K > MAXK || d < 1 || M < 4 || Lchk * Lchk != M ||
      5 * ((Lchk + 1) / 2) > MAXD) {                 // folded MAXD capacity
    tau_cf_dB[c] = (real)NAN; tau_dB[c] = (real)NAN; flags[c] = 4; return;
  }
  __private real spec[MAXT];
  for (int i = 0; i < T; ++i) spec[i] = lam[(size_t)c * T + i];

  real lam_min = spec[0];
  for (int i = 1; i < T; ++i) lam_min = fmin(lam_min, spec[i]);
  int mult = 0;
  for (int i = 0; i < T; ++i)
    if (fabs(spec[i] - lam_min) <= (real)1e-9 * fmax((real)1, lam_min)) ++mult;
  if (mult == 0) mult = 1;

  real g_min = (real)1e300;
  for (int k = 0; k < K; ++k) {
    real lg = (real)0;
    for (int i = 0; i < d; ++i) lg += R_LOG(fmax(spec[k * d + i], (real)1e-12));
    g_min = fmin(g_min, exp(lg / d));
  }

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
  if (!(u0 > (real)0)) {                    // near-flat: geometric scan of Psi
    fallback = 1;                           // (160 points: coarser than PS)
    real ratio = pow((real)1e12, (real)1 / (real)160);
    real u = (real)1e-6, u_neg = (real)-1;
    for (int i = 0; i <= 160; ++i) {
      real f = psi_u(u, spec, K, d, M, variant);
      if (f <= -PSI_BAND) u_neg = u;
      else if (f >= PSI_BAND && u_neg > (real)0) { u0 = (real)0.5 * (u_neg + u); break; }
      u *= ratio;
    }
  }
  if (!(u0 > (real)0)) {                    // uniform ordering: saturate
    real inf = psi_u((real)1, spec, K, d, M, variant) > (real)0 ? -INFINITY : INFINITY;
    tau_cf_dB[c] = inf; tau_dB[c] = inf; flags[c] = 2; return;
  }

  real taucf = (real)10 * log10((real)4 * T * (M - 1) * u0 / (real)6);
  real lo = u0 / (real)4, hi = u0 * (real)4, u = u0;
  for (int e = 0; e < 6 && psi_u(lo, spec, K, d, M, variant) > (real)0; ++e) lo /= (real)4;
  for (int e = 0; e < 6 && psi_u(hi, spec, K, d, M, variant) < (real)0; ++e) hi *= (real)4;
  for (int it = 0; it < NEWTON_ITERS; ++it) {
    real f = psi_u(u, spec, K, d, M, variant);
    if (f <= (real)0) lo = u; else hi = u;
    real fp = psi_du(u, spec, K, d, M, variant);
    real un = u - f / (fp != (real)0 ? fp : (real)1);
    if (!(un > lo && un < hi)) un = (real)0.5 * (lo + hi);
    u = un;
  }
  tau_cf_dB[c] = taucf;
  tau_dB[c] = (real)10 * log10((real)4 * T * (M - 1) * u / (real)6);
  flags[c] = fallback;
}
