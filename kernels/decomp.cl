// decomp.cl -- per-user decomposition operators for the hybrid-precoding control
// plane (fp32, one work-item per user matrix). Mirrors ref/decomp.cpp, which is
// validated against the production python oracle at fp32 tolerance.
//   jacobi_svd : one-sided Jacobi SVD, fixed sweep budget
//   gmd_k      : Hager/Jiang GMD      -> right unitary P, equal-diag upper R
//   ucd_k      : MMSE-augmented waterfilled GMD -> V_d, rho_eq, upper R, pl
//   cbd_k      : GP-CBD paired block decomposition -> Q, block-diag R, P, act
// Private-array sizes are bounded per-kernel (DECOMP_/GMD_/CBD_MAXM/_MAXD below),
// so the PL build gets a fixed BRAM/reg footprint per kernel.

#pragma OPENCL EXTENSION cl_khr_fp64 : disable
// cmul/cconj come from cplx.clh below -- not retyped here: decomp.cl once carried
// non-static file-scope copies, and since Runtime::build() concatenates every .cl
// into one OpenCL program, that was a hard redefinition error once cplx.clh was linked in too.

#include "cplx.clh"

// Capacity ceilings -- one pair per kernel, #ifndef-guarded so a board build can
// override. Namespaced on purpose: Runtime::build() concatenates all .cl into ONE
// program, so two sources defining the same MAXD would silently resize each other's
// private arrays (tools/check_registry.py fails the build on a duplicated #define).
// OpenCL C 1.2 has no VLA, so each __private array is sized at its ceiling for every
// work-item; per-kernel ceilings keep the small kernels from paying for jacobi_svd's
// 32x40 geometry (23.4 KB vs 1.8..3.3 KB at their own ceilings).

#ifndef DECOMP_MAXM
#define DECOMP_MAXM 32   // jacobi_svd rows: (K-1)*Nr = 28, rounded up to 32
#endif
#ifndef DECOMP_MAXD
#define DECOMP_MAXD 40   // jacobi_svd cols: NRF = 32, rounded up to 40
#endif
#ifndef GMD_MAXM
#define GMD_MAXM    16   // gmd_k/ucd_k rows: nul = 4 today
#endif
#ifndef GMD_MAXD
#define GMD_MAXD     8   // streams per user (production caps d at 8)
#endif
#ifndef CBD_MAXM
#define CBD_MAXM    16   // cbd_k urows and vrows
#endif
#ifndef CBD_MAXD
#define CBD_MAXD     8
#endif

__kernel void jacobi_svd(__global const float2* restrict A,   // [C][m x d]
                         int C, int m, int d, int sweeps,
                         __global float2* restrict U,         // [C][m x d]
                         __global float*  restrict S,         // [C][d]
                         __global float2* restrict V)         // [C][d x d]
{
  int c = get_global_id(0);
  if (c >= C) return;                     // beyond the batch: not ours to write
  // Out-of-contract dimensions are refused loudly (S fills with NaN) instead of
  // overflowing __private storage: a bare return would hand the host an all-zero
  // U/S/V -- indistinguishable from a plausible wrong answer.

  if (m < 1 || d < 1 || m > DECOMP_MAXM || d > DECOMP_MAXD) {
    if (d >= 1 && d <= DECOMP_MAXD)
      for (int j = 0; j < d; ++j) S[c * d + j] = NAN;
    return;
  }
  __private float2 W[DECOMP_MAXM * DECOMP_MAXD], Vv[DECOMP_MAXD * DECOMP_MAXD];
  for (int i = 0; i < m * d; ++i) W[i] = A[c * m * d + i];
  for (int i = 0; i < d * d; ++i) Vv[i] = (i / d == i % d) ? (float2)(1, 0) : (float2)(0, 0);
  // per-pair rotation skip: tol = 1e-10 is ~1e-3 of the fp32 Gram noise floor and
  // scale-invariant (relative to sqrt(alpha*beta)); measured inert at fp32.

  const float tol = 1e-10f;

  // tol_exit = sweep-stop test, deliberately NOT tol: tol asks whether one rotation
  // is worth doing, tol_exit whether the whole matrix stopped improving. 1e-7 sits
  // just above the measured fp32 floor (sqrt(off)/||A||_F^2 ~ 1.5e-8..3.4e-8), so it
  // fires on the first sweep that cannot improve the fp32 answer (~2.0x vs 16 sweeps).

  const float tol_exit = 1e-7f;

  // Norm-invariant early exit: off = sum of squared Gram off-diagonals scales as
  // ||A||_F^4; comparing sqrt(off) against tol_exit * ||A||_F^2 makes the test
  // dimensionless (the old absolute test stopped after one pass once ||A||_F < ~1.5e-5).

  float gfro2 = 0.0f;
  for (int j = 0; j < d; ++j)
    for (int r = 0; r < m; ++r) gfro2 += dot(W[r * d + j], W[r * d + j]);

  for (int sw = 0; sw < sweeps; ++sw) {
    float off = 0.0f;
    for (int p = 0; p + 1 < d; ++p)
      for (int q = p + 1; q < d; ++q) {
        float alpha = 0, beta = 0; float2 gamma = (float2)(0, 0);
        for (int r = 0; r < m; ++r) {
          alpha += dot(W[r * d + p], W[r * d + p]);
          beta  += dot(W[r * d + q], W[r * d + q]);
          gamma += cmul(cconj(W[r * d + p]), W[r * d + q]);
        }
        off += dot(gamma, gamma);
        float mu = sqrt(dot(gamma, gamma));
        if (mu <= tol * sqrt(alpha * beta + 1e-30f)) continue;
        float tau = (alpha - beta) * 0.5f;
        float t = fabs(tau) <= 1e-30f ? 1.0f
                  : -sign(tau) * mu / (fabs(tau) + sqrt(tau * tau + mu * mu));
        float cc = 1.0f / sqrt(1.0f + t * t);
        float2 eith = gamma / mu;
        float2 jm = (t * cc) * cconj(eith);
        float2 jp = (t * cc) * eith;
        for (int r = 0; r < m; ++r) {
          float2 ap = W[r * d + p], aq = W[r * d + q];
          W[r * d + p] = cc * ap - cmul(jm, aq);
          W[r * d + q] = cmul(jp, ap) + cc * aq;
        }
        for (int r = 0; r < d; ++r) {
          float2 vp = Vv[r * d + p], vq = Vv[r * d + q];
          Vv[r * d + p] = cc * vp - cmul(jm, vq);
          Vv[r * d + q] = cmul(jp, vp) + cc * vq;
        }
      }
    // exit test and sweep loop: one sweep annihilates every column pair (p,q); the
    // rotation is skipped when the Gram off-diagonal is below tol (relative).

    if (sqrt(off) <= tol_exit * gfro2) break;
  }

  __private float Sn[DECOMP_MAXD]; int ord[DECOMP_MAXD];
  for (int j = 0; j < d; ++j) {
    float nrm = 0;
    for (int r = 0; r < m; ++r) nrm += dot(W[r * d + j], W[r * d + j]);
    Sn[j] = sqrt(nrm); ord[j] = j;
  }
  // Sort singular values DESCENDING (the contract gmd_dev / ucd_k / cbd_k assume);
  // insertion sort on an index array (d <= 32, ~d^2/2 compares once per work-item).

  for (int i = 0; i < d; ++i)
    for (int j = i + 1; j < d; ++j)
      if (Sn[ord[j]] > Sn[ord[i]]) { int t = ord[i]; ord[i] = ord[j]; ord[j] = t; }
  for (int j = 0; j < d; ++j) {
    float sj = Sn[ord[j]];
    S[c * d + j] = sj;
    for (int r = 0; r < m; ++r)
      // Zero-column cut: sj > 1e-12f keeps U well-defined on rank-deficient columns
      // (fp32 sweeps leave them at ~1e-8..1e-7*sigma_max; normalising by the noise would
      // emit a unit-norm column of garbage). ref/decomp.cpp uses the same 1e-12.

      U[c * m * d + r * d + j] = sj > 1e-12f ? W[r * d + ord[j]] / sj : (float2)(0, 0);
    for (int r = 0; r < d; ++r) V[c * d * d + r * d + j] = Vv[r * d + ord[j]];
  }
}

// gmd_dev: Hager/Jiang GMD core (left basis = I), shared by gmd_k / ucd_k.
// P0/P are [rows x d]; sigma[] must be sorted DESCENDING (the two-pointer
// pivot/flag construction is only valid in that order). Internally 1-BASED
// [1..pc] for df/perm/invperm, 0-BASED for P/R matrix columns -- the perm swap
// keeps the R column and P column permutation in lockstep.

void gmd_dev(__private const float2* P0, int rows, __private const float* sigma,
             int d, __private float2* R, __private float2* P) {
  for (int i = 0; i < rows * d; ++i) P[i] = P0[i];
  for (int i = 0; i < d * d; ++i) R[i] = (float2)(0, 0);
  __private float df[GMD_MAXD + 1], z[GMD_MAXD];
  __private int perm[GMD_MAXD + 1], invperm[GMD_MAXD + 1];
  for (int i = 0; i < d; ++i) { df[i + 1] = sigma[i]; perm[i] = i; invperm[i] = i; }
  perm[d] = d; invperm[d] = d;
  // tol  = positivity floor selecting pc (production value 1e-12, matches
  //        _apply_original_gmd_factors / ref/decomp.cpp).
  // dtol = production's denominator guard |d1-d2| <= dtol -> copysign(dtol, .).

  const float tol = 1e-12f;    // positivity floor -> pc      (production value)
  const float dtol = 1e-12f;   // denominator clamp -> frac   (production value)
  int pc = 0;
  for (int i = d; i >= 1; --i) if (df[i] >= tol) { pc = i; break; }
  if (pc == 1) { R[0] = (float2)(df[1], 0); return; }
  if (pc < 2) return;

  int large = 2, small = pc;
  // sbar = geometric mean of df[1..pc], load-bearing: it picks pivots and sets R's
  // diagonal. powf(prod, 1/pc) is the most accurate fp32 form (log-exp up to 8.4x
  // worse on R) but over/underflows outside ~1e-38..1e38, so the scale-relative
  // log form below is the fallback for badly-scaled spectra.

  float prod = 1.0f;
  for (int i = 1; i <= pc; ++i) prod *= df[i];
  float sbar;
  if (isfinite(prod) && prod > 0.0f) {
    sbar = pow(prod, 1.0f / (float)pc);
  } else {                                   // prod overflowed / underflowed
    // overflow/underflow fallback: df[1] is the largest, ratios df[i]/ref0 <= 1, so
    // the log sum cannot overflow.

    float ref0 = fmax(df[1], 1e-12f);
    float lsum = 0.0f;
    for (int i = 1; i <= pc; ++i) lsum += log(fmax(df[i] / ref0, 1e-38f));
    sbar = ref0 * exp(lsum / pc);            // ratios are <= 1 -> cannot overflow
  }

  for (int k = 1; k < pc; ++k) {
    bool flag = false; int pivot;
    if (df[k] >= sbar) { pivot = perm[small--]; flag = df[pivot] >= sbar; }
    else               { pivot = perm[large++]; flag = df[pivot] <= sbar; }
    int nx = k + 1;
    if (pivot != nx) {
      float td = df[nx]; df[nx] = df[pivot]; df[pivot] = td;
      int slot = invperm[nx]; perm[slot] = pivot; invperm[pivot] = slot;
      for (int r = 0; r < rows; ++r) {
        float2 tp = P[r * d + nx - 1]; P[r * d + nx - 1] = P[r * d + pivot - 1];
        P[r * d + pivot - 1] = tp;
      }
    }
    float d1 = df[k], d2 = df[nx], total = d1 + d2, c, s;
    // Short-circuit at ~1 fp32 ulp of sbar: when d1 == sbar the exact rotation is
    // (c,s) = (1,0); frac = (d1-sbar)/(d1-d2) would cancel two nearly equal numbers
    // (fp64 frac 1.75e-16 vs fp32 1.63e-7 on the tie spectrum, giving s = 4.54e-4).

    if (flag || fabs(d1 - sbar) <= 1.2e-7f * sbar) { c = 1.0f; s = 0.0f; }
    else {
      float den = d1 - d2;
      if (fabs(den) <= dtol) den = copysign(dtol, den != 0.0f ? den : 1.0f);
      float frac = (d1 - sbar) / den;
      s = sqrt(fmax(frac * (d1 + sbar) / total, 0.0f));
      c = sqrt(fmax((1.0f - frac) * (d2 + sbar) / total, 0.0f));
    }
    df[nx] = d1 * d2 / sbar;
    z[k] = s * c * (d2 - d1) * total / sbar;
    R[(k - 1) * d + (k - 1)] = (float2)(sbar, 0);
    for (int i = 1; i < k; ++i) {
      R[(i - 1) * d + (k - 1)] = (float2)(z[i] * c, 0);
      z[i] *= -s;
    }
    for (int r = 0; r < rows; ++r) {
      float2 pa = P[r * d + k - 1], pb = P[r * d + nx - 1];
      P[r * d + k - 1] = c * pa + s * pb;
      P[r * d + nx - 1] = -s * pa + c * pb;
    }
  }
  R[(pc - 1) * d + (pc - 1)] = (float2)(sbar, 0);
  for (int i = 1; i < pc; ++i) R[(i - 1) * d + (pc - 1)] = (float2)(z[i], 0);
}

__kernel void gmd_k(__global const float2* restrict P0,    // [C][rows x d]
                    __global const float*  restrict sigma, // [C][d]
                    int C, int rows, int d,
                    __global float2* restrict R, __global float2* restrict P)
{
  int c = get_global_id(0);
  if (c >= C) return;

  if (rows < 1 || d < 1 || rows > GMD_MAXM || d > GMD_MAXD) {
    if (d >= 1 && d <= GMD_MAXD)
      for (int i = 0; i < d * d; ++i) R[c * d * d + i] = (float2)(NAN, NAN);
    return;
  }
  __private float2 P0p[GMD_MAXM * GMD_MAXD], Rr[GMD_MAXD * GMD_MAXD], Pp[GMD_MAXM * GMD_MAXD];
  __private float sg[GMD_MAXD];
  for (int i = 0; i < rows * d; ++i) P0p[i] = P0[c * rows * d + i];
  for (int i = 0; i < d; ++i) sg[i] = sigma[c * d + i];
  gmd_dev(P0p, rows, sg, d, Rr, Pp);
  for (int i = 0; i < d * d; ++i) R[c * d * d + i] = Rr[i];
  for (int i = 0; i < rows * d; ++i) P[c * rows * d + i] = Pp[i];
}

__kernel void ucd_k(__global const float2* restrict Veff,  // [C][rows x d]
                    __global const float*  restrict sigma, // [C][d]
                    int C, int rows, int d, float snr, int waterfill, float min_pl,
                    __global float2* restrict Vd,      // [C][rows x d]
                    __global float*  restrict rho_eq,  // [C]
                    __global float2* restrict upper,   // [C][d x d]
                    __global float*  restrict pl_out)  // [C][d]
{
  int c = get_global_id(0);
  if (c >= C) return;

  if (rows < 1 || d < 1 || rows > GMD_MAXM || d > GMD_MAXD) {
    rho_eq[c] = NAN;
    if (d >= 1 && d <= GMD_MAXD)
      for (int i = 0; i < d; ++i) pl_out[c * d + i] = NAN;
    return;
  }
  const float alpha = 1.0f / fmax(snr, 1e-12f);
  __private float pl[GMD_MAXD], sg[GMD_MAXD], saug[GMD_MAXD];
  for (int i = 0; i < d; ++i) pl[i] = fmax(1.0f, min_pl);
  if (waterfill != 0) {
    // Waterfill contract: sigma sorted DESCENDING (as jacobi_svd / numpy SVD emit)
    // so the active streams are the strongest; the defensive insertion sort keeps
    // the active-set test correct for unsorted input, but full UCD with unsorted
    // sigma is still out of contract (saug must stay descending for gmd_dev).

    __private float sp[GMD_MAXD];
    __private int idx[GMD_MAXD];
    for (int i = 0; i < d; ++i) {
      sp[i] = fmax(sigma[c * d + i] * sigma[c * d + i], 1e-12f);
      idx[i] = i;
    }
    for (int i = 0; i < d; ++i)                       // insertion sort desc, d<=8
      for (int j = i + 1; j < d; ++j)
        if (sp[idx[j]] > sp[idx[i]]) { int t = idx[i]; idx[i] = idx[j]; idx[j] = t; }
    // Closed-form active-set scan, verbatim production. The comparison is
    // re-associated to `d > cnt*thr - inv` because `level > thr` with level=(d+inv)/cnt
    // loses the +d once thr ~ alpha/sp[0] exceeds d/eps32 (~3.4e7 at d=4), silently
    // marking every stream active with zero loading (measured max|dpl| 4.0 -> 0.0).

    int active = d; float inv_act = 0.0f;
    for (int cnt = d; cnt >= 1; --cnt) {
      float inv = 0;
      for (int i = 0; i < cnt; ++i) inv += alpha / sp[idx[i]];
      float thr = alpha / sp[idx[cnt - 1]];
      if ((float)d > (float)cnt * thr - inv) { active = cnt; inv_act = inv; break; }
    }
    // Loading clamp order: >= 0 FIRST, then min_pl on top (min_pl first could push pl
    // negative, and pl feeds sqrt below). The loading is re-associated the same way,
    // taking the difference before +d so the weakest active stream does not collapse to 0.

    for (int i = 0; i < d; ++i) {
      if (i < active) {
        float w = ((inv_act - (float)active * (alpha / sp[idx[i]])) + (float)d)
                  / (float)active;
        pl[i] = fmax(fmax(w, 0.0f), min_pl);
      } else {
        pl[i] = fmax(0.0f, min_pl);
      }
    }
  }
  for (int i = 0; i < d; ++i) {
    // defensive: never sqrt a negative loading, whatever min_pl the PS passes down.

    pl[i] = fmax(pl[i], 0.0f);
    sg[i] = sigma[c * d + i];
    float loaded = pl[i] * sg[i] * sg[i];
    saug[i] = sqrt(fmax(loaded + alpha, 1e-12f));
  }
  float t = 0;
  for (int i = 0; i < d; ++i) t += log(fmax(saug[i], 1e-12f));
  t = exp(t / d);

  __private float2 I[GMD_MAXD * GMD_MAXD], omega[GMD_MAXD * GMD_MAXD], up[GMD_MAXD * GMD_MAXD];
  for (int i = 0; i < d * d; ++i) I[i] = (i / d == i % d) ? (float2)(1, 0) : (float2)(0, 0);
  gmd_dev(I, d, saug, d, up, omega);
  for (int r = 0; r < rows; ++r)
    for (int cc = 0; cc < d; ++cc) {
      float2 acc = (float2)(0, 0);
      for (int k = 0; k < d; ++k)
        acc += cmul(Veff[c * rows * d + r * d + k], sqrt(pl[k]) * omega[k * d + cc]);
      Vd[c * rows * d + r * d + cc] = acc;
    }
  rho_eq[c] = fmax(t * t / alpha - 1.0f, 0.0f);
  for (int i = 0; i < d * d; ++i) upper[c * d * d + i] = up[i];
  for (int i = 0; i < d; ++i) pl_out[c * d + i] = pl[i];
}

// cbd_k: GP-CBD paired block decomposition, per user (ref/decomp.cpp::cbd_factors).
// in U [C][urows x d], V [C][vrows x d] SVD bases, sigma [C][d] DESCENDING, snr, pair_thr
// out Q [C][urows x d] rx combiner, P [C][vrows x d] tx factor,
//     R [C][d x d] block-diagonal: 2x2 upper-triangular blocks at (2k,2k+1) with
//       geom-mean diagonal sqrt(d1*d2) and coupling R[2k,2k+1] = d2-d1; 1x1 for the
//       unpaired middle/inactive tail. act [C] = selected active stream count.
// Downstream must take the block structure from `act`, not re-detect it from fp32 R:
// unpaired positions are analytically zero but sit at fp32 noise ~1e-7*||R||_F, above
// the 1e-9 gate the python caller re-detects 2x2 blocks with.

__kernel void cbd_k(__global const float2* restrict U,     // [C][urows x d]
                    __global const float2* restrict V,     // [C][vrows x d]
                    __global const float*  restrict sigma, // [C][d]
                    int C, int urows, int vrows, int d,
                    float snr, float pair_thr,
                    __global float2* restrict Q,           // [C][urows x d]
                    __global float2* restrict R,           // [C][d x d]
                    __global float2* restrict P,           // [C][vrows x d]
                    __global int*    restrict act)         // [C]
{
  int c = get_global_id(0);
  if (c >= C) return;
  // Out-of-contract dims => act = -1 (outside production's [1, d] range) and R = NaN.

  if (urows < 1 || vrows < 1 || d < 1 ||
      urows > CBD_MAXM || vrows > CBD_MAXM || d > CBD_MAXD) {
    act[c] = -1;
    if (d >= 1 && d <= CBD_MAXD)
      for (int i = 0; i < d * d; ++i) R[c * d * d + i] = (float2)(NAN, NAN);
    return;
  }

  __private float sg[CBD_MAXD], ss[CBD_MAXD], dd[CBD_MAXD];
  __private int perm[CBD_MAXD], row1[CBD_MAXD], row2[CBD_MAXD];
  for (int i = 0; i < d; ++i) sg[i] = sigma[c * d + i];

  // Active-count scan: fmax(sigma^2, 1e-12) is floored BEFORE the snr scaling and
  // snr clamped at 0, so a dead stream contributes 1e-12*snr instead of 0 --
  // load-bearing at high snr (production behaviour reproduced verbatim).

  int count = d;
  if (d >= 2) {
    float snrp = fmax(snr, 0.0f);
    for (int i = 0; i < d; ++i) ss[i] = fmax(sg[i] * sg[i], 1e-12f) * snrp;
    count = 1;                                   // python: 1 if none clears
    for (int k = d; k >= 1; --k)
      if (ss[0] * ss[k - 1] >= pair_thr) { count = k; break; }
  }
  act[c] = count;
  count = max(min(count, d), 1);                 // np.clip(active_count, 1, d)

  // Strongest/weakest interleave: block k gets (sigma[k], sigma[count-1-k]), so every
  // pair straddles the geometric mean; an odd count leaves the middle stream unpaired
  // as a 1x1 block; streams >= count keep their order and are appended.

  int n1 = 0, n2 = 0, np = 0;
  if (count % 2 == 0) {
    for (int i = 0; i < count / 2; ++i) row1[n1++] = i;
    for (int i = count - 1; i >= count / 2; --i) row2[n2++] = i;
    for (int k = 0; k < n1 && k < n2; ++k) {
      perm[np++] = row1[k]; perm[np++] = row2[k];
    }
    for (int i = count; i < d; ++i) perm[np++] = i;
  } else {
    int hmid = (count + 1) / 2;
    for (int i = 0; i < hmid; ++i) row1[n1++] = i;
    for (int i = count - 1; i >= hmid; --i) row2[n2++] = i;
    if (count < d) {                             // python: row2.append(extra)
      bool used = false;
      for (int k = 0; k < n1; ++k) used = used || row1[k] == count;
      for (int k = 0; k < n2; ++k) used = used || row2[k] == count;
      if (!used) row2[n2++] = count;
    }
    for (int k = 0; k < n1 && k < n2; ++k) {
      perm[np++] = row1[k]; perm[np++] = row2[k];
    }
    for (int i = 0; i < d; ++i) {                // python: the "not in perm" tail
      bool in = false;
      for (int k = 0; k < np; ++k) in = in || perm[k] == i;
      if (!in) perm[np++] = i;
    }
  }

  __private float2 q[CBD_MAXM * CBD_MAXD], p[CBD_MAXM * CBD_MAXD];
  for (int r = 0; r < urows; ++r)
    for (int i = 0; i < d; ++i)
      q[r * d + i] = U[c * urows * d + r * d + perm[i]];
  for (int r = 0; r < vrows; ++r)
    for (int i = 0; i < d; ++i)
      p[r * d + i] = V[c * vrows * d + r * d + perm[i]];
  for (int i = 0; i < d; ++i) dd[i] = sg[perm[i]];

  // tol = 1e-12f, taken identically by ref and kernel (no tolerance divergence
  // band in cbd_k).

  const float tol = 1e-12f;
  for (int idx = 0; idx < max(count - 1, 0); idx += 2) {
    float d1 = dd[idx], d2 = dd[idx + 1];
    float prod = d1 * d2;                         // the SAME product fmax sees
    float sbar = sqrt(fmax(prod, tol));           // pair geometric mean
    float total = d1 + d2, cc, sn;
    if (fabs(d1 - d2) <= tol || total <= tol) { cc = 1.0f; sn = 0.0f; }
    else if (prod < tol) {
      // prod < tol: clamp binds, sbar = sqrt(tol) = 1e-6, NOT sqrt(d1*d2), so the frac
      // identity does not hold and production's frac form is the contract -- even where
      // production is itself ill-conditioned and emits c > 1 (deliberately non-unitary).

      float frac = (d1 - sbar) / (d1 - d2);
      sn = sqrt(fmax(frac * (d1 + sbar) / total, 0.0f));
      cc = sqrt(fmax((1.0f - frac) * (d2 + sbar) / total, 0.0f));
    }
    else {
      // prod >= tol: production's frac form collapses to s^2 = d1/total, c^2 = d2/total;
      // the rearrangement avoids the catastrophic cancellation of 1-frac ~ sqrt(d2/d1)
      // once d1/d2 > ~1e6.

      sn = sqrt(fmax(d1 / total, 0.0f));
      cc = sqrt(fmax(d2 / total, 0.0f));
    }
    dd[idx + 1] = prod / fmax(sbar, tol);         // production's running update;

    float sb = fmax(sbar, tol);

    for (int r = 0; r < vrows; ++r) {
      float2 pa = p[r * d + idx], pb = p[r * d + idx + 1];
      p[r * d + idx]     = cc * pa + sn * pb;
      p[r * d + idx + 1] = -sn * pa + cc * pb;
    }

    for (int r = 0; r < urows; ++r) {
      float2 qa = q[r * d + idx], qb = q[r * d + idx + 1];
      q[r * d + idx]     = (cc * d1 * qa + sn * d2 * qb) / sb;
      q[r * d + idx + 1] = (cc * d1 * qb - sn * d2 * qa) / sb;
    }
  }

  for (int r = 0; r < urows; ++r)
    for (int i = 0; i < d; ++i) Q[c * urows * d + r * d + i] = q[r * d + i];
  for (int r = 0; r < vrows; ++r)
    for (int i = 0; i < d; ++i) P[c * vrows * d + r * d + i] = p[r * d + i];

  // R is recomputed from the factors exactly like production: (Q^H U) diag(sigma)
  // (V^H P), re-associated so it needs only CBD_MAXD^2 scratch instead of a
  // urows x vrows private array.

  __private float2 A[CBD_MAXD * CBD_MAXD], B[CBD_MAXD * CBD_MAXD];
  for (int i = 0; i < d; ++i)
    for (int k = 0; k < d; ++k) {
      float2 acc = (float2)(0, 0);
      for (int r = 0; r < urows; ++r)
        acc += cmul(cconj(q[r * d + i]), U[c * urows * d + r * d + k]);
      A[i * d + k] = acc;
    }
  for (int k = 0; k < d; ++k)
    for (int j = 0; j < d; ++j) {
      float2 acc = (float2)(0, 0);
      for (int r = 0; r < vrows; ++r)
        acc += cmul(cconj(V[c * vrows * d + r * d + k]), p[r * d + j]);
      B[k * d + j] = acc;
    }
  for (int i = 0; i < d; ++i)
    for (int j = 0; j < d; ++j) {
      float2 acc = (float2)(0, 0);
      for (int k = 0; k < d; ++k)
        acc += cmul(A[i * d + k] * sg[k], B[k * d + j]);
      R[c * d * d + i * d + j] = acc;
    }
}
