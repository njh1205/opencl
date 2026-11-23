#pragma once
//-----------------------------------------------------------------------------
// SVD / GMD / UCD / CBD decomposition operators -- PS-role C++ (fp64, no deps).
// Ports of mix_precode/classical/digital_precoder.py and
// mix_precode/full_digital_mu/digital_precoder.py:
//   svd_cplx    : one-sided Jacobi, fixed sweep budget (PL-friendly shape)
//   gmd_factors : Hager/Jiang gmd.m  -> right unitary P, equal-diag upper R
//   ucd_design  : MMSE-augmented, waterfilled GMD -> V_D, rho_equal, upper
//   cbd_factors : GP-CBD paired block decomposition -> Q, block-diag R, P
//   stream_snr  : realised per-stream SNR |diag(R(QR(H V_D)))|^2 * snr
//-----------------------------------------------------------------------------
#include <complex>
#include <vector>

namespace hpocl {
using dcplx = std::complex<double>;

// A [m x n], row-major. U [m x n], S [n] descending, V [n x n].
//
// CONTRACT: m may be SMALLER than n. This header used to say "m >= n", which is
// wrong and dangerously so: ref/frontend.cpp::null_basis calls it as
// svd_cplx(A, rows=(K-1)*Nr, n=NRF) = (28, 32) and analog_precoder called it as
// (Nr, Nt) = (4, 128), both wide. One-sided Jacobi orthogonalises COLUMNS, so a
// wide A is the natural case: V comes out [n x n] with n - m trailing columns
// spanning null(A) at singular value ~0, and U's corresponding columns are
// zeroed by the 1e-12 cut. Nothing in the algorithm requires m >= n.
// The cost, though, is quadratic in n -- n(n-1)/2 rotations per sweep and a
// private V of n^2 -- so for m << n use svd_wide() below instead.
void svd_cplx(const dcplx* A, int m, int n, dcplx* U, double* S, dcplx* V,
              int sweeps = 16);

// Economy SVD of a WIDE matrix A [m x n] with m << n, via the small Gram matrix:
//   G = A A^H  (m x m, Hermitian PSD)  ->  eigendecomposition  ->  sigma = sqrt
//   V[:,j] = A^H u_j / sigma_j         (only the `take` leading columns)
// U [m x m] (may be null), S [n] DESCENDING with the trailing n-m entries
// EXACTLY zero (rank(A) <= min(m,n), so those are true zeros, not omissions --
// which is what lets one provider contract serve both this and svd_cplx), and
// V [n x take] row-major. take is clamped to m.
//
// WHY THIS EXISTS: it is the shape the deployed configurations actually have.
// Over every config in mix_precode/{classical,ai}/config_pools/*.json the analog
// precoder factorises H_u [Nr x Nt] with Nr <= 8 and Nt up to 128, and one-sided
// Jacobi on that costs n(n-1)/2 = 8128 rotations per sweep and an n x n = 128x128
// private V. Measured on this host at K=8, Nt=128, Nr=4, NRF=32, d=4
// (paper_sweep.json), for the eight per-user SVDs:
//   svd_cplx (full 128x128 V)  162.5 ms   -- 87% of the 184 ms front end
//   svd_wide (Gram, 4x4 eig)     0.031 ms -- 5256x faster
//   leading-subspace agreement   5.4e-15 (fp64 roundoff)
// and 28x-5691x across the eight real configs. Production's AnalogPrecoder does
// the same thing implicitly: `svd(H_u, full_matrices=False)` returns vh as
// [min(Nr,Nt) x Nt], i.e. an ECONOMY V, and only its first min(NRF,Nr,Nt)
// columns are used. ref/frontend.cpp computing the full [Nt x Nt] V was the
// port's own waste, not a modelling difference.
//
// ACCURACY DOMAIN, and the reason null_basis must NOT use this: forming G squares
// the condition number, so a singular value at sigma_min/sigma_max = r becomes an
// eigenvalue at r^2. The LEADING triplets are what this is for and they are
// exact to roundoff; the trailing ones lose half the digits. fp64 keeps
// r >= 1e-8 usable, fp32 only r >= 1e-3. null_basis needs the SMALLEST singular
// vectors (they are the BD basis), which is the worst case for a Gram method and
// the best case for one-sided Jacobi -- so null_basis stays on svd_cplx and eats
// the n^2 private array. This split is by accuracy requirement, not by shape.
void svd_wide(const dcplx* A, int m, int n, int take, dcplx* U, double* S,
              dcplx* V, int sweeps = 16);

// GMD of diag(sigma) (Hager/Jiang gmd.m): the right basis P0 [rows x d] is
// rotated into P and the equal-diagonal upper factor R satisfies
// Q^H U diag(sigma) P = R for the SVD channel U diag(sigma) V^H. rows may
// exceed d -- the pipeline feeds the tall BD basis V [nul x d].
// The left basis is optional (Q0 [qrows x d] -> Q = q_gmd); pass null for the
// transmit-only use, and see ref/rx.h for the receive combiner w_ucd/q_gmd.
void gmd_factors(const dcplx* P0, int rows, const double* sigma, int d,
                 dcplx* R /*[d x d] upper, equal diag*/, dcplx* P /*[rows x d]*/,
                 const dcplx* Q0 = nullptr, int qrows = 0, dcplx* Q = nullptr);

// Strict UCD (design_ucd_transceiver_from_svd): waterfill -> MMSE augmentation
// -> GMD equalisation of the augmented spectrum. Veff [rows x d] -> Vd = p_ucd,
// upper = r_aug_ucd, rho_equal = t^2/alpha - 1 with t the geometric mean of the
// augmented singular values and alpha = 1/snr. The optional left basis
// U [urows x d] additionally yields Q = q_ucd; the receive combiner is then
// w_ucd = Q diag(1/diag(upper)) (ref/rx.h).
void ucd_design(const dcplx* Veff, int rows, const double* sigma, int d,
                double snr, bool waterfill, double min_pl, dcplx* Vd,
                double* rho_equal, dcplx* upper, double* power_loading,
                const dcplx* U = nullptr, int urows = 0, dcplx* Q = nullptr);

//-----------------------------------------------------------------------------
// GP-CBD -- paired block decomposition (production: full_digital_mu/
// digital_precoder.py :: _select_cbd_active_count + _apply_gp_cbd_factors +
// design_cbd_transceiver_from_svd).
//
// How many streams take part in the pairing: the SNR-scaled spectrum
//   ss[i] = max(sigma[i]^2, 1e-12) * max(snr, 0)
// is scanned from the full width downwards and the first count whose
// strongest*weakest product ss[0]*ss[count-1] reaches pair_threshold
// (production cbd_pair_threshold = 1.7) is returned; 1 if none does, and d
// itself when d < 2. Note both the 1e-12 floor on sigma^2 (applied BEFORE the
// snr scaling) and the max(snr, 0) clamp are load-bearing for rank-deficient
// spectra -- a dead stream contributes 1e-12*snr, not 0.
// Input contract: sigma DESCENDING (ss[0] must be the strongest stream).
int cbd_active_count(const double* sigma, int d, double snr,
                     double pair_threshold = 1.7);

// U [urows x d] left basis (SVD U), V [vrows x d] right basis (SVD V),
// sigma [d] descending. Emits Q [urows x d] (rx combiner chain), R [d x d] and
// P [vrows x d] (tx factor), plus the selected active count.
//
// Structure of R (production recomputes it as Q^H (U diag(sigma) V^H) P rather
// than trusting the incremental diagonal update -- so do we):
//   * 2x2 blocks at output indices (2k, 2k+1) for the paired streams,
//     R[2k,2k] = R[2k+1,2k+1] = sqrt(delta1*delta2) (geometric mean),
//     R[2k,2k+1] = delta2 - delta1 (<= 0 for a descending spectrum) and
//     R[2k+1,2k] = 0 -> upper-triangular inside the block;
//   * 1x1 for the unpaired middle stream of an odd active count and for the
//     inactive tail, which stays diagonal (in the permuted order);
//   * everything outside those blocks is exactly zero in exact arithmetic.
// The strongest/weakest interleave permutation puts (sigma[i], sigma[count-1-i])
// into block i, so each pair straddles the geometric mean and both rotated
// streams end up at the same SNR.
void cbd_factors(const dcplx* U, int urows, const dcplx* V, int vrows,
                 const double* sigma, int d, double snr, dcplx* Q, dcplx* R,
                 dcplx* P, int* active_count, double pair_threshold = 1.7);

// realised per-stream SNR of precoder Vd on channel H [m x n].
void stream_snr(const dcplx* H, int m, int n, const dcplx* Vd, int d,
                double snr, double* rho);

//-----------------------------------------------------------------------------
// fp32 side of the contract
//   The fp32 implementations live ONLY in kernels/decomp.cl (jacobi_svd, gmd_k,
//   ucd_k, cbd_k). This header used to announce a device-free `hpocl::f32::`
//   mirror layer that decomp_bench would cross-check against the fp64 functions
//   above; no such namespace ever existed in-tree, and the announcement has
//   been deleted rather than left describing a gate no test runs.
//   Consequence, stated plainly: on a host where Runtime::init finds no OpenCL
//   device (Apple dropped the CPU device, so arm64 macOS has none), decomp_bench
//   skips the whole kernel path and its `(!have_cl || e_cl < 3e-4)` gate passes
//   vacuously. The fp32 contract is then NOT tested at all. It is exercised only
//   where a device exists. decomp_main.cpp prints an explicit warning in that
//   case so a vacuous pass cannot be mistaken for a real one.
//   Every threshold in kernels/decomp.cl that differs from its counterpart here
//   is documented at the constant itself with the measured divergence band.
//-----------------------------------------------------------------------------

}  // namespace hpocl
