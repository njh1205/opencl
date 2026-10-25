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

// A [m x n], m >= n, row-major. U [m x n], S [n] descending, V [n x n].
void svd_cplx(const dcplx* A, int m, int n, dcplx* U, double* S, dcplx* V,
              int sweeps = 16);

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
