#pragma once
//-----------------------------------------------------------------------------
// Receive-side modem reference (PS-role C++17, fp64, no dependencies).
// Port of the gray_standard square-QAM paths of
// mix_precode/classical/{bicm_metrics.py,thp_precoding.py}:
//   qam_axis_spacing    <- thp_precoding.qam_axis_order_and_spacing
//   modulo_fold         <- thp_precoding.centered_modulo_complex
//   qam_table           <- bicm_metrics.get_constellation (16/64/256-QAM:
//                          per-axis Gray PAM, unit average power)
//   qam_map             <- bicm_metrics.modulate_bits_with_labeling
//   softdemod_parallel  <- bit metric of estimate_bicm_gmi_parallel_from_received
//   softdemod_folded    <- bit metric of estimate_bicm_gmi_thp_from_received over
//                          enumerate_thp_symbol_replicas (radius 2, 25 shifts)
//   gmi_from_llr        <- ideal rate m minus the per-bit
//                          stable_binary_logloss_from_signed_llr deficit
//
// Conventions pinned by the production code (checked by build/modem_bench):
//   * m = log2(M) bits per symbol, I-axis Gray bits (MSB first) then Q-axis;
//     llr[] and bits[] are flat [n*m] in that same order.
//   * LLR_b = logsumexp_{candidates with label bit b = 1} (-d2/n0)
//           - logsumexp_{candidates with label bit b = 0} (-d2/n0),
//     d2 = |y - candidate|^2, n0 = E|noise|^2 per complex sample. Natural log
//     domain, so a positive LLR favours bit value 1 (bit_llr_from_distances
//     with max_log=False, i.e. logsumexp and not max-log).
//   * the folded branch consumes an already modulo-folded y and scores the
//     whole periodic replica set, because the THP receive modulo destroys the
//     one-to-one map between y and the transmitted constellation point.
//   * GMI is in bits/symbol; production clamps it at zero, this port returns
//     the raw m - deficit (the oracle keeps every vector in the positive range).
//-----------------------------------------------------------------------------
#include <complex>
#include <cstdint>

namespace hpocl {
using dcplx = std::complex<double>;

// Modulation order M in {16, 64, 256} -> m in {4, 6, 8}; -1 if unsupported.
int qam_bits_per_symbol(int M);
// PAM points per axis (L = sqrt(M)), normalized minimum spacing 2/norm with
// norm = sqrt(2(L^2-1)/3), and the THP modulo period L*spacing.
// Zero if M is unsupported.
int    qam_axis_order(int M);
double qam_axis_spacing(int M);
double qam_modulo_period(int M);

// Unit-power gray_standard constellation: sym[M], bits[M*m].
void qam_table(int M, dcplx* sym, uint8_t* bits);
// Flat bit stream bits[n*m] -> symbols sym[n] (inverse of qam_table).
void qam_map(const uint8_t* bits, int M, dcplx* sym, int n);

// Centered modulo into [-period/2, period/2) on both axes.
void modulo_fold(const dcplx* z, int n, double period, dcplx* out);

// Per-bit logsumexp LLRs llr[n*m] from combined received samples y[n]:
// parallel branch (M candidates) and THP folded branch (25*M replicas).
void softdemod_parallel(const dcplx* y, double n0, int M, double* llr, int n);
void softdemod_folded(const dcplx* y, double n0, int M, double period,
                      double* llr, int n);

// GMI = m - mean_i sum_b log2(1 + exp(-(2*bits_b - 1) * llr_b)), the sample
// average over the given symbol sequence (no Monte Carlo draw).
double gmi_from_llr(const double* llr, const uint8_t* bits, int m, int n);
}  // namespace hpocl
