//-----------------------------------------------------------------------------
// operator family: decomp -- per-user decomposition operators for the
// hybrid-precoding control plane (fp32, one work-item per user matrix).
// Mirrors ref/decomp.cpp exactly; that file is validated against the
// production python oracle, so these kernels inherit the same contract at
// fp32 tolerance.
//
//   jacobi_svd : one-sided Jacobi, fixed sweep budget
//   gmd_k      : Hager/Jiang gmd.m  -> right unitary P, equal-diag upper R
//   ucd_k      : MMSE-augmented waterfilled GMD -> V_D, rho_equal, upper
//   cbd_k      : GP-CBD paired block decomposition -> Q, block-diag R, P,
//                active stream count
//
// Private-array sizes are bounded by DECOMP_MAXM/DECOMP_MAXD so the PL build gets fixed
// BRAM/reg footprint; d <= 8 streams per user on the target configurations.
//-----------------------------------------------------------------------------
#pragma OPENCL EXTENSION cl_khr_fp64 : disable
// cmul / cconj come from the shared header, not from file-scope copies. This
// was the last kernel source that still retyped them: cplx.clh's own header
// comment lists decomp.cl among the files the helpers were "retyped in", but
// the unification never reached here, so decomp.cl carried non-static
// file-scope `cmul`/`cconj`. Runtime::build() concatenates every .cl into ONE
// clCreateProgramWithSource, so adding decomp.cl to any program that also pulls
// in cplx.clh (op_bench's list is the obvious candidate) was a hard redefinition
// error waiting to happen. The two bodies were byte-identical to the header's
// `static inline` versions, so this is semantics-preserving. read_cl_source()
// resolves the include relative to this file (kernels/cplx.clh) for the ICDs
// that ignore -I, and v++ resolves it the same way.
#include "cplx.clh"

// Namespaced on purpose: threshold_cf.cl also defines MAXD (128, its LLR
// distance-candidate budget) and MAXT/MAXK. Runtime::build() concatenates every
// .cl it is given into ONE program, so two sources defining MAXD with different
// values would silently resize each other's private arrays -- a 16x overrun in
// threshold_cf's d0[]/d1[] if decomp.cl won. tools/check_registry.py now fails
// the build on any #define name shared by two kernel sources.
#define DECOMP_MAXD 8
#define DECOMP_MAXM 16

//-----------------------------------------------------------------------------
__kernel void jacobi_svd(__global const float2* restrict A,   // [C][m x d]
                         int C, int m, int d, int sweeps,
                         __global float2* restrict U,         // [C][m x d]
                         __global float*  restrict S,         // [C][d]
                         __global float2* restrict V)         // [C][d x d]
{
  int c = get_global_id(0);
  if (c >= C) return;                     // beyond the batch: not ours to write
  // Private arrays are sized DECOMP_MAXM/DECOMP_MAXD, so out-of-contract dimensions are
  // rejected instead of silently overflowing __private storage. The refusal is
  // LOUD: Runtime::jacobi_svd creates its device buffers with
  // CL_MEM_COPY_HOST_PTR from value-initialised host vectors, so a bare
  // `return` used to hand the caller an all-zero U/S/V -- indistinguishable
  // from a plausible wrong answer, and exactly the failure mode power_iter.cl
  // already sentinels against. S[] has extent C*d, so it is safe to fill
  // whenever d itself is in contract; if d is the offending argument no output
  // index can be formed without risking an overrun and we can only return.
  if (m < 1 || d < 1 || m > DECOMP_MAXM || d > DECOMP_MAXD) {
    if (d >= 1 && d <= DECOMP_MAXD)
      for (int j = 0; j < d; ++j) S[c * d + j] = NAN;
    return;
  }
  __private float2 W[DECOMP_MAXM * DECOMP_MAXD], Vv[DECOMP_MAXD * DECOMP_MAXD];
  for (int i = 0; i < m * d; ++i) W[i] = A[c * m * d + i];
  for (int i = 0; i < d * d; ++i) Vv[i] = (i / d == i % d) ? (float2)(1, 0) : (float2)(0, 0);
  // fp32 tolerance policy. eps(fp32) = 1.2e-7, so the Gram off-diagonal |gamma|
  // of an orthogonalised pair bottoms out around 1e-7*alpha. tol = 1e-10f is
  // ~1e-3 of that noise floor and drives the per-pair SKIP test, which is
  // relative to sqrt(alpha*beta) and therefore already scale-invariant.
  // Measured, the exact value is inert at fp32: tol = 1e-10f and tol = 1e-13f
  // run the same 16 sweeps and produce BIT-IDENTICAL S/U/V over 6
  // configurations (m,d,cond) = (6,4,1e3) (8,8,1e6) (16,8,1e2) (4,4,1e12)
  // (8,4,1e1) (16,8,1e6) -- max|dS| = max|dU| = max|dV| = 0.000e+00 -- because
  // the fp32 Gram off-diagonal never gets within 1e3 of either threshold. It is
  // kept at 1e-10f (no behavioural change) and the comment that used to claim
  // it "stops the sweep from chasing rounding" is removed: it does not.
  const float tol = 1e-10f;

  // Norm-invariant early exit. `off` is a sum of SQUARED Gram off-diagonals, so
  // it scales as ||A||_F^4; the previous ABSOLUTE test `off <= tol*tol` (=1e-20)
  // was therefore a test on the NORM of A, not on convergence, and it stopped
  // the sweep after a single pass once ||A||_F^4 fell under 1e-20. Measured,
  // full-rank 6x4 / 16x8:
  //   ||A||_F = 1.28e-4 : 3 sweeps, ||U^H U - I|| = 5.7e-6   (borderline)
  //   ||A||_F = 3.27e-5 : 2 sweeps, defect 8.7e-3, |dU| vs ref 7.9e-3
  //   ||A||_F = 1.11e-5 : 1 sweep,  defect 4.8e-1, |dU| vs ref 5.0e-1
  //   ||A||_F = 1.39e-5 (16x8) : 1 sweep, defect 5.5e-1, |dU| 9.8e-1
  //   ||A||_F = 3.42e-6 : 1 sweep,  defect 6.8e-1, |dU| vs ref 1.1e+0
  // i.e. below ||A||_F ~ 1.5e-5 the kernel emitted a U that is not orthonormal
  // at all, which cbd_k then multiplies by 1/sbar. Comparing sqrt(off) against
  // tol*||A||_F^2 makes the test dimensionless (both sides scale as ||A||_F^2).
  // ||W||_F == ||A||_F is invariant under the column rotations, so gfro2 is
  // accumulated once here. After the fix every row above runs the full budget:
  // defect 4.8e-1 -> 1.2e-7, |dU| 5.0e-1 -> 1.3e-7, |dS| 5.8e-7 -> 6.6e-13.
  // At O(1) norms nothing changes: 24/24 configurations (8 shapes x cond
  // 1e1/1e3/1e6) are BIT-IDENTICAL before/after. At fp32 the relative
  // off-diagonal bottoms out ~3e-9 > tol, so the exit no longer fires at all
  // and `sweeps` is a plain fixed budget -- the same thing the absolute form
  // already did for every norm the bench exercises.
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
    if (sqrt(off) <= tol * gfro2) break;   // scale-relative; see the note above
  }

  __private float Sn[DECOMP_MAXD]; int ord[DECOMP_MAXD];
  for (int j = 0; j < d; ++j) {
    float nrm = 0;
    for (int r = 0; r < m; ++r) nrm += dot(W[r * d + j], W[r * d + j]);
    Sn[j] = sqrt(nrm); ord[j] = j;
  }
  for (int i = 0; i < d; ++i)                 // insertion sort, d <= 8
    for (int j = i + 1; j < d; ++j)
      if (Sn[ord[j]] > Sn[ord[i]]) { int t = ord[i]; ord[i] = ord[j]; ord[j] = t; }
  for (int j = 0; j < d; ++j) {
    float sj = Sn[ord[j]];
    S[c * d + j] = sj;
    for (int r = 0; r < m; ++r)
      // Zero-column test: 1e-12f, and ref/decomp.cpp::svd_cplx now uses the
      // SAME 1e-12 (it used to be 1e-14 there), so U is defined identically on
      // both sides and decomp_bench's e_cl measures precision rather than a
      // threshold difference. The rationale: a rank-deficient column comes out
      // of the fp32 sweeps with norm ~1e-8..1e-7*sigma_max, and dividing it by
      // its own noise norm emits a UNIT-NORM column of garbage into U, which
      // cbd_k then amplifies by 1/sbar. Cutting at 1e-12 still keeps every
      // stream above -240 dB. Measured band where the two cuts used to disagree
      // (kernel zero, ref unit-norm garbage): sigma_j in (1e-14, 1e-12], which
      // the fp32 tail reaches for sigma_max in [1e-7, 1e-5] -- at sigma_max=1e-4
      // the tail is 8.1e-13, kernel column norm 0.0 vs ref 1.0.
      U[c * m * d + r * d + j] = sj > 1e-12f ? W[r * d + ord[j]] / sj : (float2)(0, 0);
    for (int r = 0; r < d; ++r) V[c * d * d + r * d + j] = Vv[r * d + ord[j]];
  }
}

//-----------------------------------------------------------------------------
// Hager/Jiang gmd.m device function (left basis = I), shared by gmd_k/ucd_k
//-----------------------------------------------------------------------------
// P0/P are [rows x d]: rows may exceed d (tall BD basis V in the pipeline).
// Indexing contract -- keep the two domains strictly separated:
//   * df/perm/invperm/pc are the Hager/Jiang 1-BASED domain [1..pc]
//     (df[0] is unused; perm[0]/invperm[0] are dummies);
//   * P/R matrix columns are 0-BASED [0..rows-1]/[0..d-1].
// Input contract: sigma[] must be sorted DESCENDING -- the Hager/Jiang
// two-pointer pivot/flag construction is only valid in that order. All
// internal callers satisfy this (jacobi_svd output, scipy SVD, ucd_k saug).
// The domains meet only in the mapping loops below; the perm swap keeps the
// R column and the P column permutation in lockstep (R col k-1 <-> P col k-1).
void gmd_dev(__private const float2* P0, int rows, __private const float* sigma,
             int d, __private float2* R, __private float2* P) {
  for (int i = 0; i < rows * d; ++i) P[i] = P0[i];
  for (int i = 0; i < d * d; ++i) R[i] = (float2)(0, 0);
  __private float df[DECOMP_MAXD + 1], z[DECOMP_MAXD];
  __private int perm[DECOMP_MAXD + 1], invperm[DECOMP_MAXD + 1];
  for (int i = 0; i < d; ++i) { df[i + 1] = sigma[i]; perm[i] = i; invperm[i] = i; }
  perm[d] = d; invperm[d] = d;
  // TWO independent decisions used to share one constant; they are now named
  // separately because they are not the same question and not the same answer.
  //
  // tol  -- "is this stream alive": sets the positive count pc, and pc selects
  //         the whole Hager/Jiang pivot structure. Value 1e-12f = production's
  //         `_apply_original_gmd_factors(tol=1e-12)` and ref/decomp.cpp, so all
  //         three agree. It used to be 1e-9f here on the argument that the fp32
  //         Jacobi residual (~1e-7*sigma_max) would otherwise be promoted to a
  //         live stream. Measured, that argument does not hold: the residual is
  //         ~1e-8..1e-7*sigma_max, i.e. ABOVE 1e-9 for every sigma_max >= 0.1,
  //         so both floors already agreed and 1e-9f bought nothing; it only
  //         fired for sigma in [1e-12, 1e-9) ABSOLUTE -- which the fp32 residual
  //         reaches when sigma_max is in [1e-5, 1e-2] -- and there it changed pc
  //         away from production: sigma=[3,2,1,9.99e-10] gave pc 4 (oracle) vs
  //         3 (kernel), max|dR| = 1.81 and max|dP| = 0.885 on a unitary P.
  //         1e-12f removes that band entirely. Measured, oracle-vs-kernel:
  //           sigma=[3,2,1,9.99e-10]  max|dR| 1.808 -> 3.13e-6, max|dP| 1.282 -> 7.26e-8
  //           sigma=[3,2,1,1.001e-12] max|dR| 1.816 -> 3.83e-5, max|dP| 1.280 -> 6.78e-8
  //           sigma=[5,4,3,2]e-10     (all four in the band, pc 4 -> 0)
  //                                   max|dR| 3.31e-10 -> 6.00e-17, max|dP| 1.080 -> 2.07e-7
  //         and no regression on 120 healthy spectra (see the sbar note below).
  // dtol -- production's denominator guard `|d1-d2| <= tol -> copysign(tol,.)`.
  //         Sharing the 1e-9f floor with pc silently widened that guard 1000x:
  //         for |d1-d2| in (1e-12, 1e-9] the oracle divides by the true den
  //         (frac ~ 0.5, c=s=0.7071) while the kernel divided by 1e-9 (frac ~ 0,
  //         c=1, s=0). Reachable only where an fp32 gap that small exists at
  //         all, i.e. ulp(sbar) < 1e-9, sbar <~ 8.3e-3: instrumented over 4000
  //         weak-channel spectra (sigma ~ 1e-5..1e-1), 2 of 11920 non-flag pivot
  //         steps land in (1e-12, 1e-9]. Worst max|dP| over those 4000 spectra
  //         3.30e-1 -> 2.02e-2 with dtol at production's 1e-12; max|dR| stays
  //         <= 8.9e-6 either way, so this is invisible in R and large in the
  //         precoder -- which is why it needs its own constant.
  const float tol = 1e-12f;    // positivity floor -> pc      (production value)
  const float dtol = 1e-12f;   // denominator clamp -> frac   (production value)
  int pc = 0;
  for (int i = d; i >= 1; --i) if (df[i] >= tol) { pc = i; break; }
  if (pc == 1) { R[0] = (float2)(df[1], 0); return; }
  if (pc < 2) return;

  int large = 2, small = pc;
  // Geometric mean of df[1..pc]. This value is load-bearing far out of
  // proportion to its size: it is compared against df[k] to pick the pivot and
  // it appears in the numerator of frac = (d1 - sbar)/(d1 - d2), so ONE ulp of
  // sbar becomes ~1e-3 of R when a pivoted pair lands close to sbar.
  // Measured over 8000 random spectra (scale 1e-8..1e8, pc 2..8), fp32 relative
  // error of sbar and the resulting worst max|dR| vs the oracle on 120 healthy
  // spectra:
  //   exp(sum(log(df))/pc)          worst 3.03e-6, 80% of cases > 0.5 ulp -> max|dR| 2.75e-3
  //   df1*exp(sum(log(df/df1))/pc)  worst 1.49e-7, 15% > 0.5 ulp          -> max|dR| 2.75e-3
  //   powf(prod(df), 1/pc)          13% > 0.5 ulp (best of the fp32-representable
  //                                 forms)                          -> max|dR| 3.27e-4
  // so powf() is the accurate form and the log-exp form was 8.4x worse on R and
  // 11.9x worse on P (max|dP| 7.74e-4 -> 6.51e-5); on the single worst healthy
  // spectrum (sigma=[8.907,7.947,3.089,2.678,1.745,0.850], where step k=5 pairs
  // 3.088435 with 3.088951 around sbar=3.088693) it is 54x: max|dR| 2.751e-3 ->
  // 5.076e-5. Newton-refining the log-exp estimate was tried and does not help
  // (worst sbar rel 1.49e-7 -> 1.32e-7, max|dR| unchanged). powf() is used, with
  // scale-relative log-exp as the FALLBACK for the one thing pow() cannot do:
  // prod(df) overflows fp32 above ~1e38 and underflows below ~1e-38, measured
  // on 30/120 badly-scaled spectra (sigma0 up to 1e12, pc up to 8) where the
  // bare pow() form returns inf/0 and destroys R. Cost: one soft-pow per gmd
  // call, control plane only, not the per-symbol datapath.
  float prod = 1.0f;
  for (int i = 1; i <= pc; ++i) prod *= df[i];
  float sbar;
  if (isfinite(prod) && prod > 0.0f) {
    sbar = pow(prod, 1.0f / (float)pc);
  } else {                                   // prod overflowed / underflowed
    // df[1] is the largest under the descending contract, and pc >= 2 implies
    // df[1] >= df[2] >= tol > 0; fmax() is belt-and-braces for a caller that
    // violates the contract, where a zero ref0 would make the ratio inf and sbar
    // NaN instead of merely wrong.
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
    // Short-circuit at ~1 fp32 ulp of sbar. When d1 == sbar exactly, frac == 0
    // and the exact answer is (c,s) = (1,0); but frac = (d1 - sbar)/(d1 - d2) is
    // a cancellation of two nearly equal numbers and sqrt() then amplifies the
    // residue as sqrt(eps): measured on sigma=[3,3,1,1], step k=2 has
    // d1 = d1_prev*d2_prev/sbar = sbar by construction, frac = 1.75e-16 in fp64
    // but 1.63e-7 in fp32, giving s = 0.0 vs s = 4.54e-4.
    // The threshold is deliberately ONE ulp (1.2e-7): it is the rounding level
    // of the quantity being tested, and wider is monotonically harmful. Measured
    // worst max|dR| over the tie family [3(1+e),3(1-e),1,1], e in [0,1e-2],
    // with the final pow-based sbar:
    //   no short-circuit 7.87e-4   1.2e-7*sbar 5.20e-4 (best)  3e-7*sbar 9.49e-4
    //   5e-7*sbar 1.64e-3          1e-6*sbar 1.64e-3           3e-6*sbar 3.00e-3
    // i.e. anything wider starts overriding GENUINE gaps of 1e-7..1e-6 relative,
    // where production's frac is real and non-zero. Effect on the exact tie
    // sigma=[3,3,1,1] alone: max|dR| 7.870e-4 -> 2.384e-7 (3301x) and
    // max|dP| 3.418e-4 -> 5.766e-8 (5928x); over the 5 degenerate bench spectra
    // max|dP| 3.42e-4 -> 1.94e-6 (176x). 120 healthy and 120 weak-channel
    // spectra are bit-for-bit unchanged by the short-circuit.
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
  // out-of-contract dimensions: refuse, and leave a loud sentinel rather than
  // the host's zero-fill (see the note in jacobi_svd). R[] has extent C*d*d.
  if (rows < 1 || d < 1 || rows > DECOMP_MAXM || d > DECOMP_MAXD) {
    if (d >= 1 && d <= DECOMP_MAXD)
      for (int i = 0; i < d * d; ++i) R[c * d * d + i] = (float2)(NAN, NAN);
    return;
  }
  __private float2 P0p[DECOMP_MAXM * DECOMP_MAXD], Rr[DECOMP_MAXD * DECOMP_MAXD], Pp[DECOMP_MAXM * DECOMP_MAXD];
  __private float sg[DECOMP_MAXD];
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
  // out-of-contract dimensions: refuse with a loud sentinel. rho_eq[] has
  // extent C (always safe once c < C) and pl_out[] has extent C*d.
  if (rows < 1 || d < 1 || rows > DECOMP_MAXM || d > DECOMP_MAXD) {
    rho_eq[c] = NAN;
    if (d >= 1 && d <= DECOMP_MAXD)
      for (int i = 0; i < d; ++i) pl_out[c * d + i] = NAN;
    return;
  }
  const float alpha = 1.0f / fmax(snr, 1e-12f);
  __private float pl[DECOMP_MAXD], sg[DECOMP_MAXD], saug[DECOMP_MAXD];
  for (int i = 0; i < d; ++i) pl[i] = fmax(1.0f, min_pl);
  if (waterfill != 0) {
    // Waterfilling contract: sigma must be sorted DESCENDING (as jacobi_svd /
    // numpy SVD both emit), so the first `active` streams are the strongest
    // ones -- identical to production _build_ucd_power_loading. The sort below
    // is defensive: a no-op on the canonical descending input, and it makes
    // this active-set test correct in isolation for unsorted input. NOTE the
    // full UCD is still out of contract for unsorted sigma -- saug must stay
    // DESCENDING for the downstream gmd_dev (Hager/Jiang pivoting assumes
    // descending order), so unsorted sigma fed to ucd_k is not supported.
    // idx maps the sorted order back to the original stream.
    __private float sp[DECOMP_MAXD];
    __private int idx[DECOMP_MAXD];
    for (int i = 0; i < d; ++i) {
      sp[i] = fmax(sigma[c * d + i] * sigma[c * d + i], 1e-12f);
      idx[i] = i;
    }
    for (int i = 0; i < d; ++i)                       // insertion sort desc, d<=8
      for (int j = i + 1; j < d; ++j)
        if (sp[idx[j]] > sp[idx[i]]) { int t = idx[i]; idx[i] = idx[j]; idx[j] = t; }
    // Water-level solve: production's closed-form active-set scan verbatim
    // (no iteration, no approximation) -- but the comparison and the loading
    // are RE-ASSOCIATED, because `level > thr` with level = (d + inv)/cnt is
    // cancellation-prone in fp32 exactly where it matters. At cnt == 1 the test
    // is `d + thr > thr`, unconditionally true in exact arithmetic, so
    // production always terminates with a well-defined active set. In fp32 the
    // `+d` is absorbed once thr = alpha/sp[0] exceeds d/eps32 (~3.4e7 at d=4),
    // level == thr bit for bit, NO cnt breaks, and `active` silently keeps its
    // initialiser d: every stream is marked active with zero loading.
    // Measured (spectrum [3,2,1,0.5], waterfill on): at snr=1e-10 production
    // gives active=1, pl=[4,0,0,0]; the un-re-associated fp32 form gave
    // active=4, pl=[0,0,0,0] -- max|d pl| = 4.0, i.e. 100% of the loading, and
    // rho_eq 9.0e-10 vs 0.0. The trigger is snr*sigma0^2 <~ 1.2e-7, so a WEAK
    // channel reaches it at ordinary SNR: sigma0=1e-2 diverges below -29 dB,
    // sigma0=1e-3 below -9 dB.
    //   `d > cnt*thr - inv` is the same inequality with the small quantity on
    // the side fp32 can still represent; cnt*thr - inv is exactly 0 at cnt==1,
    // so the break there is unconditional as in production. Residual band:
    // production's own fp64 loses the +d above thr ~ 1.8e16 (snr*sigma0^2 <
    // 5.6e-17, -162 dB) and then also returns active = d; that is unreachable
    // at fp32 and is documented rather than chased.
    int active = d; float inv_act = 0.0f;
    for (int cnt = d; cnt >= 1; --cnt) {
      float inv = 0;
      for (int i = 0; i < cnt; ++i) inv += alpha / sp[idx[i]];
      float thr = alpha / sp[idx[cnt - 1]];
      if ((float)d > (float)cnt * thr - inv) { active = cnt; inv_act = inv; break; }
    }
    // production _build_ucd_power_loading clamp order: the >= 0 clamp comes
    // FIRST and min_pl is applied on top of it. min_pl first would let a
    // negative min_pl push pl below zero, and pl feeds a sqrt four lines down.
    // The loading is re-associated the same way as the test above:
    //   level - thr_i = (d + inv)/active - thr_i = ((inv - active*thr_i) + d)/active
    // takes the DIFFERENCE before d is added, so the weakest active stream's
    // loading (exactly d/active at cnt == 1) does not collapse to 0. Measured:
    // with both rewrites the 18/45 active-set mismatches and the worst
    // max|d pl| = 6.0 over 45 (spectrum x snr) points all go to 0.0, and the
    // healthy regime is unchanged (max|d pl| 5.1e-8 at min_pl=0, 2.7e-7 at
    // min_pl=0.3 / -0.5).
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
    // defensive: never sqrt a negative loading, whatever min_pl the PS passes
    // down (the branches above cannot produce one, but a NaN here is silent
    // and poisons every stream of the user).
    pl[i] = fmax(pl[i], 0.0f);
    sg[i] = sigma[c * d + i];
    float loaded = pl[i] * sg[i] * sg[i];
    saug[i] = sqrt(fmax(loaded + alpha, 1e-12f));
  }
  float t = 0;
  for (int i = 0; i < d; ++i) t += log(fmax(saug[i], 1e-12f));
  t = exp(t / d);

  __private float2 I[DECOMP_MAXD * DECOMP_MAXD], omega[DECOMP_MAXD * DECOMP_MAXD], up[DECOMP_MAXD * DECOMP_MAXD];
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

//-----------------------------------------------------------------------------
// GP-CBD paired block decomposition (fp32), device mirror of ref/decomp.cpp::
// cbd_factors. Production: mix_precode/full_digital_mu/digital_precoder.py
// (_select_cbd_active_count + _apply_gp_cbd_factors +
//  design_cbd_transceiver_from_svd).
//
// Contract: U [C][urows x d] / V [C][vrows x d] are the SVD bases of the
// per-user reduced channel and sigma [C][d] is DESCENDING (the pairing and the
// active-count scan both assume sigma[0] is the strongest stream). One
// work-item per user. Outputs:
//   Q [C][urows x d] rx combiner chain, P [C][vrows x d] tx factor,
//   R [C][d x d] block-diagonal: 2x2 upper-triangular blocks at output indices
//       (2k, 2k+1) with geometric-mean diagonal sqrt(delta1*delta2) and
//       coupling R[2k,2k+1] = delta2 - delta1, 1x1 for the unpaired middle
//       stream of an odd active count and for the inactive tail,
//   act [C] selected active stream count (pre-clip, as production reports it).
//
// Downstream note: multiuser_simulation_environment._contiguous_cbd_blocks
// re-detects the 2x2 blocks from |R[i,i+1]| > 1e-9 * fmax(||R||_F, 1). A paired
// stream clears that gate by 7.9-8.7 orders of magnitude (measured margin
// 7.7e7 .. 5.6e8), but the analytically-zero *unpaired* positions are only zero
// up to fp32 rounding (~1e-7 * ||R||_F) and sit ABOVE the gate -- so on the PL
// the block structure must be taken from `act`, not re-detected from an fp32 R.
// Measured spur/gate, two input families:
//   * the 11 CBD vectors decomp_bench runs that have a paired block at all
//     (bases from the fp32 jacobi_svd, so the off-block residue is full fp32
//     noise). ON THE DEVICE (Apple M4, `make decomp_bench`, spur32 column):
//     56x, 57x, 60x, 64x, 66x, 74x, 94x, 98x, 111x, 118x, 134x. A numpy fp32
//     transcription of the same kernel gives 47x..111x on the same vectors --
//     same range, different per-vector values, because a transcription does not
//     reproduce the device's fused multiply-add and rounding. Quote the device
//     numbers; the "~100x" an earlier revision of this comment claimed is right
//     for this family but was never measured here.
//   * 7 synthetic cases with exactly-unitary bases (healthy d=4, ill-conditioned
//     1e-8/1e-9 at snr 1e6, tied [3,3,1,1], rank-deficient [3,2,0,0], odd active
//     count 5, d=8, m=16/d=8): 12.2x .. 32.1x -- the floor of the range, since a
//     clean basis leaves less residue to leak off-block.
// Either way the factor is >> 1, so re-detection is defeated and the conclusion
// stands. The number is now the one decomp_bench prints in its spur32 column,
// computed from the KERNEL's own fp32 R rather than argued.
//
// R is recomputed from the factors, exactly like production (Q^H (U Sigma V^H)
// P -- the incremental dd[] update is dead state and is not trusted). It is
// re-associated here as (Q^H U) Sigma (V^H P): the same product, but it needs
// only DECOMP_MAXD*DECOMP_MAXD scratch instead of a urows x vrows private array, keeping the
// PL footprint at the level the other decomp kernels use.
//-----------------------------------------------------------------------------
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
  // Private arrays are sized DECOMP_MAXM/DECOMP_MAXD: reject out-of-contract dimensions
  // instead of silently overflowing __private storage (same guard style as
  // jacobi_svd / gmd_k / ucd_k), and leave a loud sentinel. act[] has extent C
  // and -1 is outside the [1, d] range production can return, so it is an
  // unambiguous "kernel refused" marker; R[] has extent C*d*d.
  if (urows < 1 || vrows < 1 || d < 1 ||
      urows > DECOMP_MAXM || vrows > DECOMP_MAXM || d > DECOMP_MAXD) {
    act[c] = -1;
    if (d >= 1 && d <= DECOMP_MAXD)
      for (int i = 0; i < d * d; ++i) R[c * d * d + i] = (float2)(NAN, NAN);
    return;
  }

  __private float sg[DECOMP_MAXD], ss[DECOMP_MAXD], dd[DECOMP_MAXD];
  __private int perm[DECOMP_MAXD], row1[DECOMP_MAXD], row2[DECOMP_MAXD];
  for (int i = 0; i < d; ++i) sg[i] = sigma[c * d + i];

  // ---- _select_cbd_active_count -------------------------------------------
  // fmax(sigma^2, 1e-12) is floored BEFORE the snr scaling and snr is clamped at
  // 0, so a dead stream contributes 1e-12*snr instead of 0 -- that floor is
  // load-bearing, at high snr it can pull a rank-deficient stream back into the
  // active set (production behaviour, reproduced verbatim).
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

  // ---- strongest/weakest interleave permutation ---------------------------
  // Block k receives (sigma[k], sigma[count-1-k]), so every pair straddles the
  // geometric mean; an odd count leaves the middle stream (count-1)/2 unpaired
  // as a 1x1 block. Streams >= count keep their order and are appended.
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

  __private float2 q[DECOMP_MAXM * DECOMP_MAXD], p[DECOMP_MAXM * DECOMP_MAXD];
  for (int r = 0; r < urows; ++r)
    for (int i = 0; i < d; ++i)
      q[r * d + i] = U[c * urows * d + r * d + perm[i]];
  for (int r = 0; r < vrows; ++r)
    for (int i = 0; i < d; ++i)
      p[r * d + i] = V[c * vrows * d + r * d + perm[i]];
  for (int i = 0; i < d; ++i) dd[i] = sg[perm[i]];

  // ---- per-pair c/s and the g1/g2 rotations --------------------------------
  // tol = 1e-12f, the production default, kept identical to the fp64 reference
  // and to production's `_apply_gp_cbd_factors(tol=1e-12)`. It drives THREE
  // decisions here and only three: the exact-degeneracy branch
  // (|d1-d2| <= tol || total <= tol -> identity rotation), the floor on the
  // pair product inside sqrt(), and the floor on sbar where it is used as a
  // divisor. All three are taken identically by ref and kernel, so there is no
  // tolerance divergence band in cbd_k -- but see the clamp note below, which
  // IS a real band and is a property of production, not of the fp32 port.
  const float tol = 1e-12f;
  for (int idx = 0; idx < max(count - 1, 0); idx += 2) {
    float d1 = dd[idx], d2 = dd[idx + 1];
    float prod = d1 * d2;                         // the SAME product fmax sees
    float sbar = sqrt(fmax(prod, tol));           // pair geometric mean
    float total = d1 + d2, cc, sn;
    if (fabs(d1 - d2) <= tol || total <= tol) { cc = 1.0f; sn = 0.0f; }
    else if (prod < tol) {
      // CLAMP BINDS: sbar = sqrt(tol) = 1e-6, NOT sqrt(d1*d2), so the identity
      // used in the else-branch does not hold and production's frac form is the
      // contract (the oracle is the contract, even where the oracle is itself
      // ill-conditioned: for the pair (5e-7, 1e-7) production emits c = 2.031
      // > 1, i.e. a non-unitary p_cbd, and c^2+s^2 = 1 is NOT preserved here on
      // purpose). Reachability: needs d1*d2 < 1e-12 with |d1-d2| > 1e-12 and an
      // active_count that selects the pair, i.e. snr >= sqrt(1.7/(d1^2*d2^2)) --
      // 4.4e5 (56 dB) for a (3, 1e-13) pair, which decomp_ref.py already uses
      // as an snr, and sigma_max <~ 3.2e-3 for an fp32-Jacobi-derived spectrum
      // (sigma_min ~ 1e-7*sigma_max).
      // Measured worst over 11 clamped pairs, kernel-vs-oracle (errQ, errR,
      // errP) -- BEFORE this branch: (6.84e-1, 2.48e+0, 1.35e+0); AFTER:
      // (3.86e-2, 1.63e-1, 1.02e-7). On the pairs where production's frac is
      // itself representable the agreement becomes exact: (2e-6,1e-8) errQ
      // 5.32e-1 -> 7.47e-8, (1e-6,1e-9) errP 1.30e+0 -> 2.80e-8, (1e-3,1e-10)
      // errQ 4.61e-1 -> 1.08e-5. The residual 1.6e-1 on (3,1e-14) is fp32
      // evaluating production's own cancellation (3-1e-6)/(3-1e-14), not a
      // semantic difference.
      float frac = (d1 - sbar) / (d1 - d2);
      sn = sqrt(fmax(frac * (d1 + sbar) / total, 0.0f));
      cc = sqrt(fmax((1.0f - frac) * (d2 + sbar) / total, 0.0f));
    }
    else {
      // prod >= tol, so sbar^2 == d1*d2 and production's
      //   frac = (d1-sbar)/(d1-d2)
      //   s^2  = frac*(d1+sbar)/total    = (d1^2 - sbar^2)/(d1^2 - d2^2)
      //   c^2  = (1-frac)*(d2+sbar)/total = (sbar^2 - d2^2)/(d1^2 - d2^2)
      // collapses analytically to s^2 = d1/total, c^2 = d2/total. Verified:
      // identical to the frac form to 1.5e-13 over 2e5 random fp64 pairs, and
      // bit-identical to the pre-existing kernel on every unclamped pair tested
      // (worst (errQ,errR,errP) over 40 random unclamped spectra unchanged at
      // 2.29e-6). The rearrangement is load-bearing at fp32 because
      // 1-frac ~ sqrt(d2/d1) loses every significant digit to cancellation once
      // d1/d2 > ~1e6: for the pair (3, 1e-9) production's form in fp32 gives
      // max|g2_prod - g2_exact| = 1.14e-3 (vs 2.8e-12 in fp64) and c is off by
      // 2.1e-8, while this form reproduces the exact c to 2e-8.
      sn = sqrt(fmax(d1 / total, 0.0f));
      cc = sqrt(fmax(d2 / total, 0.0f));
    }
    dd[idx + 1] = prod / fmax(sbar, tol);         // production's running update;
                                                 // dead state, R is recomputed
    float sb = fmax(sbar, tol);
    // right basis: p[:, pair] @ g1,  g1 = [[c, -s], [s, c]]
    for (int r = 0; r < vrows; ++r) {
      float2 pa = p[r * d + idx], pb = p[r * d + idx + 1];
      p[r * d + idx]     = cc * pa + sn * pb;
      p[r * d + idx + 1] = -sn * pa + cc * pb;
    }
    // left basis: q[:, pair] @ g2,  g2 = [[c d1, -s d2], [s d2, c d1]]/sbar
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

  // ---- R = (Q^H U) diag(sigma) (V^H P), original bases, original sigma ----
  __private float2 A[DECOMP_MAXD * DECOMP_MAXD], B[DECOMP_MAXD * DECOMP_MAXD];
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
