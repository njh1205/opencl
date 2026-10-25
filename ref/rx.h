#pragma once
//-----------------------------------------------------------------------------
// Receive-side control plane (PS-role C++17, fp64, no dependencies): the
// combiner/r-chain assembled once per scheduling instant, the digital power
// normalization and the THP receive diagonal. Ports of
// mix_precode/classical:
//   rx_combiner      <- MultiUserSimulationEnvironment.build_structured_digital_
//                       chain (q_local / r_local per strategy), delegating the
//                       Hager/Jiang work to ref/decomp.h -- nothing is
//                       re-implemented here
//   power_norm_scale <- correction_scale of the same function, equivalently
//                       MultiUserSimulationEnvironment.normalize_digital_precoder
//   rx_diag          <- diag(B) of thp_precoding.compute_effective_channel, the
//                       diag_b that thp_receive_equalized divides by (the
//                       receive modulo itself lives in ref/modem.cpp)
//
// Conventions:
//   * U [Nr x d], sigma [d] descending and V [rows x d] are the reduced SVD of
//     the power-whitened BD channel Gbar = U diag(sigma) V^H; V is the tall BD
//     null-space basis, so rows may exceed d and Gbar may be wide (rows > Nr is
//     a real pipeline shape).
//   * Q [Nr x d] is the receive combiner of the chain, production's q_local:
//     u_eff (SVD), q_gmd (GMD), w_ucd = q_ucd diag(1/safe(diag(r_aug_ucd)))
//     (MMSE-UCD, hence not unitary). The equalized stream vector is
//     z = Q^H y / diag_b, then the receive modulo.
//   * R [d x d] is the upper-triangular r_chain: diag(sigma), r_gmd,
//     r_aug_ucd. GMD and UCD equalize |diag(R)|, SVD does not.
//   * MMSE-UCD uses the DigitalStructuredPrecoder.__init__ defaults that the
//     environment forwards unchanged: ucd_waterfill = false and
//     ucd_min_power_loading = 0.0, i.e. power_loading = ones(d), and the
//     environment's alpha_override = 1/max(snr, 1e-12), i.e. alpha = 1/snr.
//-----------------------------------------------------------------------------
#include <complex>

namespace hpocl {
using dcplx = std::complex<double>;

// strategy: 's' SVD, 'g' GMD, 'u' MMSE-UCD (anything else leaves Q/R untouched,
// production raises ValueError). Writes Q [Nr x d] and R [d x d].
void rx_combiner(const dcplx* U, int Nr, const dcplx* V, int rows,
                 const double* sigma, int d, double snr, char strategy,
                 dcplx* Q, dcplx* R);

// sqrt(p_constraint / ||Frf Fbb||_F^2) for Frf [Nt x NRF] and Fbb [NRF x T];
// 1.0 when the power collapses (production only normalizes above 1e-12).
double power_norm_scale(const dcplx* Frf, int Nt, int NRF, const dcplx* Fbb,
                        int T, double p_constraint);

// diag_b [d] of the upper-triangular r_chain R [d x d].
void rx_diag(const dcplx* R, int d, dcplx* diag_b);
}  // namespace hpocl
