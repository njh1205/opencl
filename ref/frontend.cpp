#include "frontend.h"
#include "decomp.h"
#include <cmath>
#include <limits>
#include <vector>
#include <algorithm>

namespace hpocl {

void effective_channels(const dcplx* H, int K, int Nr, int Nt, const dcplx* Frf,
                        int NRF, dcplx* eff) {
  for (int u = 0; u < K; ++u)
    for (int i = 0; i < Nr; ++i)
      for (int j = 0; j < NRF; ++j) {
        dcplx acc(0, 0);
        for (int t = 0; t < Nt; ++t)
          acc += H[(size_t)u * Nr * Nt + i * Nt + t] * Frf[(size_t)t * NRF + j];
        eff[(size_t)u * Nr * NRF + i * NRF + j] = acc;
      }
}

int null_basis(const dcplx* A, int rows, int NRF, dcplx* N) {
  std::vector<double> S(NRF);
  std::vector<dcplx> V(NRF * NRF);
  svd_cplx(A, rows, NRF, nullptr, S.data(), V.data());   // sweeps: decomp.h default
  // scipy.linalg.null_space's default rcond is eps * max(M, N); matching it
  // exactly is what makes the BD basis agree with the production oracle.
  const double tol = std::max(rows, NRF) * std::numeric_limits<double>::epsilon()
                     * (S[0] > 0 ? S[0] : 1.0);
  int nul = 0;
  for (int j = 0; j < NRF; ++j) nul += (S[j] <= tol);
  int c = 0;
  for (int j = 0; j < NRF && c < nul; ++j)
    if (S[j] <= tol) {
      for (int r = 0; r < NRF; ++r) N[(size_t)r * nul + c] = V[(size_t)r * NRF + j];
      ++c;
    }
  return nul;
}

void power_whitening(const dcplx* FN, int rows, int n, double rel_floor,
                     double abs_floor, dcplx* inv_sqrt, double* eig_out) {
  // C = FN^H FN, hermitian-symmetrised
  std::vector<dcplx> C(n * n);
  for (int i = 0; i < n; ++i)
    for (int j = 0; j < n; ++j) {
      dcplx acc(0, 0);
      for (int k = 0; k < rows; ++k)
        acc += std::conj(FN[(size_t)k * n + i]) * FN[(size_t)k * n + j];
      C[(size_t)i * n + j] = acc;
    }
  for (int i = 0; i < n; ++i)
    for (int j = 0; j < i; ++j) {
      dcplx m = 0.5 * (C[(size_t)i * n + j] + std::conj(C[(size_t)j * n + i]));
      C[(size_t)i * n + j] = m;
      C[(size_t)j * n + i] = std::conj(m);
    }
  // C is PSD: its SVD is an eigendecomposition; floor protects C^{-1/2}
  std::vector<double> S(n);
  std::vector<dcplx> V(n * n);
  svd_cplx(C.data(), n, n, nullptr, S.data(), V.data());   // sweeps: decomp.h default
  const double floor = std::max(abs_floor, rel_floor * S[0]);
  for (int i = 0; i < n; ++i) {
    if (eig_out) eig_out[i] = S[i];
    S[i] = std::max(S[i], floor);
  }
  // inv_sqrt = V diag(1/sqrt S) V^H
  for (int i = 0; i < n; ++i)
    for (int j = 0; j < n; ++j) {
      dcplx acc(0, 0);
      for (int k = 0; k < n; ++k)
        acc += V[(size_t)i * n + k] * (1.0 / std::sqrt(S[k])) *
               std::conj(V[(size_t)j * n + k]);
      inv_sqrt[(size_t)i * n + j] = acc;
    }
}

void whiten_channel(const dcplx* reduced, int Nr, int n, const dcplx* inv_sqrt,
                    dcplx* Gbar) {
  for (int i = 0; i < Nr; ++i)
    for (int j = 0; j < n; ++j) {
      dcplx acc(0, 0);
      for (int k = 0; k < n; ++k)
        acc += reduced[(size_t)i * n + k] * inv_sqrt[(size_t)k * n + j];
      Gbar[(size_t)i * n + j] = acc;
    }
}


void analog_precoder(const dcplx* H, int K, int Nr, int Nt, int NRF,
                     dcplx* Frf) {
  // candidate columns: per-user right singular vectors, capped at NRF each
  std::vector<dcplx> src(Nt * NRF, dcplx(0, 0));
  int col = 0;
  std::vector<dcplx> V(Nt * Nt);
  std::vector<double> S(Nt);
  for (int u = 0; u < K && col < NRF; ++u) {
    svd_cplx(H + (size_t)u * Nr * Nt, Nr, Nt, nullptr, S.data(), V.data());   // sweeps: decomp.h default
    int take = std::min(NRF, std::min(Nr, Nt));
    for (int j = 0; j < take && col < NRF; ++j, ++col)
      for (int r = 0; r < Nt; ++r) src[(size_t)r * NRF + col] = V[(size_t)r * Nt + j];
  }
  // DFT filler columns if the users supplied too few
  for (int c = col; c < NRF; ++c)
    for (int r = 0; r < Nt; ++r)
      src[(size_t)r * NRF + c] =
          std::exp(dcplx(0, 2 * M_PI * r * (c - col) / Nt));
  // constant-modulus projection
  for (int i = 0; i < Nt * NRF; ++i)
    Frf[i] = std::exp(dcplx(0, std::arg(src[i]))) / std::sqrt((double)Nt);
}

}  // namespace hpocl
