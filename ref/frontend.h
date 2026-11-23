#pragma once
//-----------------------------------------------------------------------------
// Multi-user RF-BD front end (PS-role C++), port of
// mix_precode/classical/multiuser_simulation_environment.py:
//   effective_channels  : H_k F_RF
//   null_basis          : scipy.null_space convention (trailing right
//                         singular vectors), tol = max(M,N) eps sigma_max
//   power_whitening     : C = (F_RF N)^H (F_RF N), floored eigendecomposition,
//                         returns C^{-1/2}
//   whiten_channel      : Gbar_k = H_k F_RF N_k C_k^{-1/2}
//-----------------------------------------------------------------------------
#include <complex>

namespace hpocl {
using dcplx = std::complex<double>;

// The ONE complex GEMM primitive of the control-plane glue, in exactly
// kernels/wbb_apply.cl's form:
//     Y[sc*M + m] = sum_{n<N} W[m*N + n] * X[sc*N + n]      (NO conjugation)
// i.e. Y = X W^T with W the TRANSPOSED operand and both operands row-major.
// It is spelled this way -- and not as the textbook C = A B -- because that is
// the contract the existing device kernel already implements, so the PL role of
// every glue GEMM is one launch of a kernel that is already built, registered,
// capacity-guarded and gated (op_bench / op_pl_bench), instead of a 30th kernel
// with its own ceilings and its own gate.
//
// Three call sites use it, and they are 93% of the glue at the real pool shapes
// (docs/cost_model_real_shapes.md, K=8 Nt=128 Nr=4 NRF=32):
//   H_k F_RF   : W = F_RF^T [NRF x Nt], X = H [K*Nr x Nt]      1.05 Mflop
//   F_RF N_k   : W = stack_k(N_k^T) [K*nul x NRF], X = F_RF    1.05 Mflop
//   F_RF F_BB  : W = F_BB^T [T x NRF], X = F_RF [Nt x NRF]     1.05 Mflop
// The batch dimension of the first two is folded into SC, so "all K users" is
// one call and not K calls -- which is what makes the device role worth its
// launch overhead.
//
// The summation order is n ascending, which is what every loop it replaces ran
// in, so routing the glue through here is BIT-IDENTICAL on the host (measured:
// effective_channels below, and sched_bench's host role is unchanged to 0.0).
// `inline` in the header, and that is deliberate: ref/rx.cpp needs it for
// power_norm_scale, and rx_bench's link line is ref/rx.cpp + ref/decomp.cpp --
// putting the body in frontend.cpp would mean editing the Makefile's SRC_rx to
// add a translation unit that contributes one 8-line loop. One definition either
// way; this one just costs no link edge.
inline void gemm_wx(const dcplx* W, int M, int N, const dcplx* X, int SC,
                    dcplx* Y) {
  for (int sc = 0; sc < SC; ++sc)
    for (int m = 0; m < M; ++m) {
      dcplx acc(0, 0);
      const dcplx* w = W + (size_t)m * N;
      const dcplx* x = X + (size_t)sc * N;
      for (int n = 0; n < N; ++n) acc += w[n] * x[n];
      Y[(size_t)sc * M + m] = acc;
    }
}

void effective_channels(const dcplx* H, int K, int Nr, int Nt,
                        const dcplx* Frf, int NRF, dcplx* eff);
// N [NRF x NRF] worst case; returns nullity (columns written)
int null_basis(const dcplx* A, int rows, int NRF, dcplx* N);
void power_whitening(const dcplx* FN, int NRF, int n, double rel_floor,
                     double abs_floor, dcplx* inv_sqrt, double* eig_out);
void whiten_channel(const dcplx* reduced, int Nr, int n,
                    const dcplx* inv_sqrt, dcplx* Gbar);
// Shared-RF constant-modulus analog precoder (port of AnalogPrecoder):
// per-user right singular columns, DFT filler if short, unit-modulus
// projection, 1/sqrt(Nt) scale. Column phases are a gauge freedom of the
// whole downstream pipeline (eff/BD/whitening/Gbar are invariant).
void analog_precoder(const dcplx* H, int K, int Nr, int Nt, int NRF,
                     dcplx* Frf);

//-----------------------------------------------------------------------------
// The same four functions, SPLIT at the decomposition boundary.
//
// Why: every one of them is "one SVD plus cheap glue", and the SVD is 95% of the
// front end's cost (measured: at K=8, Nt=16, Nr=4, NRF=32 the eight null_basis
// factorisations are 91% of the 2.05 ms front end and the eight analog_precoder
// factorisations another 4%; the GEMMs and the 4x4 whitening are the rest). So
// the device-side control plane (runtime/hpocl_schedule.cpp) needs to run the
// SVDs as batched jacobi_svd launches and then execute THE SAME glue -- and a
// glue body copied into the device path is a second implementation that drifts.
// Each *_from_* / *_trailing function below is therefore the only definition of
// its glue, called by both paths; the unsplit wrappers above are just
// "host SVD, then glue".
//
// All of them are pure host code with no OpenCL dependency, so ref/ stays
// device-free and the split costs nothing on the PS-only build.
//-----------------------------------------------------------------------------

// analog_precoder's glue: V is [K][Nt x take] -- the ECONOMY right basis, take =
// min(NRF, Nr, Nt) columns per user in DESCENDING singular order, which is what
// svd_wide()/jacobi_svd emit for H_k [Nr x Nt] and what production's
// `svd(H_u, full_matrices=False)` returns. DFT-fills if the users supplied too
// few columns, projects to constant modulus and scales by 1/sqrt(Nt).
void analog_precoder_from_V(const dcplx* V, int K, int Nr, int Nt, int NRF,
                            dcplx* Frf);
// How many users' bases analog_precoder actually consumes (see the .cpp): the
// caller only has to factorise that many, which is what keeps the device batch
// the same size as the host loop.
int analog_users(int K, int Nr, int Nt, int NRF);

// null_basis's column selection: N = the trailing `nul` columns of V
// ([NRF x NRF], DESCENDING singular values), so N spans the null space of the
// interference stack. `nul` comes from the GEOMETRY (NRF - rows), not from a
// rank test on S -- see the fp32 rcond caveat in ref/decomp.h and the tol_exit
// note in kernels/decomp.cl: at 6 sweeps the fp32 rank margin Smin/Smax is
// 5.2e-10, four orders ABOVE null_basis's fp64 rcond (7.1e-15), so a rank test
// on device-computed singular values silently returns nul = 0. The host path
// can afford the rcond test because its SVD is fp64; both paths select the same
// trailing columns, because with S descending the set {j : S[j] <= tol} IS a
// suffix.
void null_basis_trailing(const dcplx* V, int NRF, int nul, dcplx* N);

// power_whitening's two halves: C = FN^H FN hermitian-symmetrised, and
// C^{-1/2} = V diag(1/sqrt(max(S, floor))) V^H from C's eigendecomposition
// (C is PSD, so its SVD IS its eigendecomposition and V is [n x n]).
void gram_hermitian(const dcplx* FN, int rows, int n, dcplx* C);
// C^{-1/2} = V diag(1/sqrt(S_floored)) V^H.
//   rel_floor/abs_floor are PRODUCTION's eigenvalue floor (1e-10 / 1e-12 in
//     build_physical_power_whitening) and are applied verbatim;
//   noise_rel is the noise floor of the arithmetic that produced S, as a
//     multiple of S[0]. An eigenvalue at or below it is not a small eigenvalue,
//     it is a structurally ZERO one that the decomposition cannot represent, so
//     it is snapped to zero before production's floor is applied -- otherwise
//     1/sqrt(noise) is applied to an eigenvector that is arbitrary.
//     Pass 8*eps of the SVD's precision (see SvdProvider::eps).
// The two are separate parameters because they answer separate questions, and
// because conflating them is a measured defect: C = (F_RF N)^H (F_RF N) has
// rank <= min(Nt, nul), so nul - min(Nt, nul) eigenvalues are structurally zero
// (at K=2/Nt=4/NRF=8 production's own eigh returns [-2.7e-16, -4.7e-17,
// 3.8e-17, 9.3e-17, 1.08, 1.63]). fp64 puts those at 2.1e-17..9.3e-17 x S[0],
// BELOW the 1e-10 floor, so they are floored. fp32 puts the same eigenvalues at
// 2.5e-09..3.7e-08 x S[0] -- ABOVE the 1e-10 floor, so they are NOT floored and
// the whitener divides by their noise. Measured consequence for the digital
// precoder Fbb = N C^{-1/2} V_D: relative error 2.8e+01 (K=2/Nt=4/NRF=8) and
// 1.1e+04 (K=3/Nt=6/NRF=12) against the fp64 role. With noise_rel = 8*eps32 =
// 9.5e-7, i.e. 26-380x above the measured fp32 band, Fbb agrees to ... (see
// sched_bench). For fp64 the change is PROVABLY bit-identical: production's
// floor 1e-10 is 13 orders above 8*eps64 = 1.8e-15, so max(S, floor) already
// returns floor for every S the snap can touch.
void inv_sqrt_from_eig(const dcplx* V, const double* S, int n, double rel_floor,
                       double abs_floor, double noise_rel, dcplx* inv_sqrt,
                       double* eig_out);
}  // namespace hpocl
