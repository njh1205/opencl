//-----------------------------------------------------------------------------
// ref/rx.cpp -- receive-side control plane. rx.h maps each entry point to the
// production code it ports; the GMD/UCD algebra itself is ref/decomp.h's.
//-----------------------------------------------------------------------------
#include "rx.h"

#include <algorithm>
#include <cmath>
#include <vector>

#include "decomp.h"
#include "frontend.h"   // gemm_wx: the one GEMM primitive, see frontend.h

namespace hpocl {
namespace {
// DigitalStructuredPrecoder.__init__ defaults, forwarded unchanged by the
// environment constructor: no waterfilling, no power-loading floor.
constexpr bool kUcdWaterfill = false;
constexpr double kUcdMinPowerLoading = 0.0;
// design_ucd_transceiver_from_svd: safe_diagonal = where(|diag| > 1e-12, diag,
// 1e-12 + 0j), guarding the 1/diag of w_ucd = q_ucd diag(1/safe_diagonal).
constexpr double kDiagFloor = 1e-12;
// normalize_digital_precoder / correction_scale: no normalization below this.
constexpr double kPowerFloor = 1e-12;
}  // namespace

void rx_combiner(const dcplx* U, int Nr, const dcplx* V, int rows,
                 const double* sigma, int d, double snr, char strategy,
                 dcplx* Q, dcplx* R) {
  if (strategy == 's') {                       // q = u_eff, r = diag(sigma)
    std::copy_n(U, (size_t)Nr * d, Q);
    std::fill_n(R, (size_t)d * d, dcplx(0.0, 0.0));
    for (int i = 0; i < d; ++i) R[(size_t)i * d + i] = dcplx(sigma[i], 0.0);
  } else if (strategy == 'g') {                // q_gmd / r_gmd, p_gmd unused
    gmd_factors(V, rows, sigma, d, R, nullptr, U, Nr, Q);
  } else if (strategy == 'u') {                // q_ucd -> w_ucd, p_ucd unused
    double rho_equal = 0.0;                    // ucd_design writes it, unused here
    ucd_design(V, rows, sigma, d, snr, kUcdWaterfill, kUcdMinPowerLoading,
               nullptr, &rho_equal, R, nullptr, U, Nr, Q);
    for (int c = 0; c < d; ++c) {
      dcplx x = R[(size_t)c * d + c];
      if (std::abs(x) <= kDiagFloor) x = dcplx(kDiagFloor, 0.0);
      const dcplx inv = dcplx(1.0, 0.0) / x;
      for (int r = 0; r < Nr; ++r) Q[(size_t)r * d + c] *= inv;
    }
  }
  // any other strategy: Q/R left untouched (production raises ValueError)
}

double power_norm_from_product(const dcplx* P, int Nt, int T,
                               double p_constraint) {
  double power = 0.0;                          // ||Frf Fbb||_F^2
  for (int i = 0; i < Nt; ++i)
    for (int t = 0; t < T; ++t) power += std::norm(P[(size_t)i * T + t]);
  return power > kPowerFloor ? std::sqrt(p_constraint / power) : 1.0;
}

double power_norm_scale(const dcplx* Frf, int Nt, int NRF, const dcplx* Fbb,
                        int T, double p_constraint) {
  // P[i][t] = sum_r Frf[i][r] Fbb[r][t] is gemm_wx with W = Fbb^T, X = Frf, so
  // the same call can go to the fabric (1.05 Mflop at Nt=128, 25-31% of the
  // glue). r ascending either way: same bits.
  std::vector<dcplx> FbbT((size_t)T * NRF), P((size_t)Nt * T);
  for (int r = 0; r < NRF; ++r)
    for (int t = 0; t < T; ++t) FbbT[(size_t)t * NRF + r] = Fbb[(size_t)r * T + t];
  gemm_wx(FbbT.data(), T, NRF, Frf, Nt, P.data());
  return power_norm_from_product(P.data(), Nt, T, p_constraint);
}

void rx_diag(const dcplx* R, int d, dcplx* diag_b) {
  for (int i = 0; i < d; ++i) diag_b[i] = R[(size_t)i * d + i];
}
}  // namespace hpocl
