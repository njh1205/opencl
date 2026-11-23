#include "frontend.h"
#include "decomp.h"
#include <cmath>
#include <limits>
#include <vector>
#include <algorithm>

namespace hpocl {

void effective_channels(const dcplx* H, int K, int Nr, int Nt, const dcplx* Frf,
                        int NRF, dcplx* eff) {
  // eff[k][i][j] = sum_t H[k][i][t] Frf[t][j] IS gemm_wx with W = Frf^T, X = H
  // read as [K*Nr x Nt] and Y = eff read as [K*Nr x NRF], and the inner sum
  // runs over t ascending either way -- so this is the same bits, one
  // definition, and the device role can hand the same call to wbb_apply.
  std::vector<dcplx> FrfT((size_t)NRF * Nt);
  for (int t = 0; t < Nt; ++t)
    for (int j = 0; j < NRF; ++j) FrfT[(size_t)j * Nt + t] = Frf[(size_t)t * NRF + j];
  gemm_wx(FrfT.data(), NRF, Nt, H, K * Nr, eff);
}

void null_basis_trailing(const dcplx* V, int NRF, int nul, dcplx* N) {
  for (int c = 0; c < nul; ++c) {
    const int j = NRF - nul + c;                 // the trailing columns
    for (int r = 0; r < NRF; ++r) N[(size_t)r * nul + c] = V[(size_t)r * NRF + j];
  }
}

int null_basis(const dcplx* A, int rows, int NRF, dcplx* N) {
  std::vector<double> S(NRF);
  std::vector<dcplx> V(NRF * NRF);
  svd_cplx(A, rows, NRF, nullptr, S.data(), V.data());   // sweeps: decomp.h default
  // scipy.linalg.null_space's default rcond is eps * max(M, N); matching it
  // exactly is what makes the BD basis agree with the production oracle. This
  // test is fp64-only by construction -- see null_basis_trailing's note for why
  // the device path takes `nul` from the geometry instead.
  const double tol = std::max(rows, NRF) * std::numeric_limits<double>::epsilon()
                     * (S[0] > 0 ? S[0] : 1.0);
  int nul = 0;
  for (int j = 0; j < NRF; ++j) nul += (S[j] <= tol);
  // S is DESCENDING, so {j : S[j] <= tol} is a suffix and the rcond-selected
  // columns are exactly the trailing ones: same selection, one definition.
  null_basis_trailing(V.data(), NRF, nul, N);
  return nul;
}

void gram_hermitian(const dcplx* FN, int rows, int n, dcplx* C) {
  for (int i = 0; i < n; ++i)
    for (int j = 0; j < n; ++j) {
      dcplx acc(0, 0);
      for (int k = 0; k < rows; ++k)
        acc += std::conj(FN[(size_t)k * n + i]) * FN[(size_t)k * n + j];
      C[(size_t)i * n + j] = acc;
    }
  // Symmetrise: the product is Hermitian in exact arithmetic but not in fp, and
  // the eigendecomposition below assumes it (a Jacobi sweep on a non-Hermitian
  // matrix converges to something else entirely).
  for (int i = 0; i < n; ++i)
    for (int j = 0; j < i; ++j) {
      dcplx m = 0.5 * (C[(size_t)i * n + j] + std::conj(C[(size_t)j * n + i]));
      C[(size_t)i * n + j] = m;
      C[(size_t)j * n + i] = std::conj(m);
    }
}

void inv_sqrt_from_eig(const dcplx* V, const double* S, int n, double rel_floor,
                       double abs_floor, double noise_rel, dcplx* inv_sqrt,
                       double* eig_out) {
  // C is PSD, so its SVD IS its eigendecomposition and V is the eigenvector
  // matrix; the floor protects C^{-1/2} from the near-null directions the BD
  // basis leaves behind (rel_floor scales with the largest eigenvalue, so the
  // result is invariant to the channel gain).
  std::vector<double> Sf(S, S + n);
  const double floor = std::max(abs_floor, rel_floor * Sf[0]);
  // Snap the eigenvalues the decomposition cannot resolve to zero BEFORE
  // production's floor, so an fp32 eigendecomposition floors the structurally
  // zero directions at production's value instead of dividing by its own noise.
  // See the header for the measurement and for why this is bit-identical in
  // fp64 (floor >> noise there, so max() already returned floor).
  const double noise = noise_rel * (Sf[0] > 0 ? Sf[0] : 0.0);
  for (int i = 0; i < n; ++i) {
    if (eig_out) eig_out[i] = Sf[i];
    if (Sf[i] <= noise) Sf[i] = 0.0;
    Sf[i] = std::max(Sf[i], floor);
  }
  // inv_sqrt = V diag(1/sqrt S) V^H
  for (int i = 0; i < n; ++i)
    for (int j = 0; j < n; ++j) {
      dcplx acc(0, 0);
      for (int k = 0; k < n; ++k)
        acc += V[(size_t)i * n + k] * (1.0 / std::sqrt(Sf[k])) *
               std::conj(V[(size_t)j * n + k]);
      inv_sqrt[(size_t)i * n + j] = acc;
    }
}

void power_whitening(const dcplx* FN, int rows, int n, double rel_floor,
                     double abs_floor, dcplx* inv_sqrt, double* eig_out) {
  std::vector<dcplx> C((size_t)n * n);
  gram_hermitian(FN, rows, n, C.data());
  std::vector<double> S(n);
  std::vector<dcplx> V((size_t)n * n);
  svd_cplx(C.data(), n, n, nullptr, S.data(), V.data());   // sweeps: decomp.h default
  // This wrapper always uses the fp64 host SVD, so its noise floor is 8*eps64;
  // production's 1e-10 relative floor is 13 orders above it and the snap cannot
  // change any result (see inv_sqrt_from_eig).
  inv_sqrt_from_eig(V.data(), S.data(), n, rel_floor, abs_floor,
                    8.0 * 2.220446049250313e-16, inv_sqrt, eig_out);
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


void analog_precoder_from_V(const dcplx* V, int K, int Nr, int Nt, int NRF,
                            dcplx* Frf) {
  // candidate columns: per-user right singular vectors, capped at NRF each.
  // V is the ECONOMY basis [K][Nt x take] -- take columns, not Nt -- which is
  // what production's `svd(H_u, full_matrices=False)` returns and what
  // svd_wide() produces. Reading a full [Nt x Nt] V here was the port's own
  // waste: at Nt=128 it is 32x the memory and, on the Jacobi path, 87% of the
  // front end's time (see svd_wide in ref/decomp.h).
  std::vector<dcplx> src((size_t)Nt * NRF, dcplx(0, 0));
  int col = 0;
  const int take = std::min(NRF, std::min(Nr, Nt));
  for (int u = 0; u < K && col < NRF; ++u)
    for (int j = 0; j < take && col < NRF; ++j, ++col)
      for (int r = 0; r < Nt; ++r)
        src[(size_t)r * NRF + col] = V[((size_t)u * Nt + r) * take + j];
  // DFT filler columns if the users supplied too few
  for (int c = col; c < NRF; ++c)
    for (int r = 0; r < Nt; ++r)
      src[(size_t)r * NRF + c] =
          std::exp(dcplx(0, 2 * M_PI * r * (c - col) / Nt));
  // constant-modulus projection: only the PHASE of each candidate survives, so
  // a column gauge V_j -> V_j e^{it} becomes Frf_j -> Frf_j e^{it}. That is the
  // gauge freedom the whole downstream pipeline is invariant to (eff/BD/
  // whitening/Gbar all absorb it), and the reason sched_bench compares Frf up
  // to a per-column phase rather than entrywise.
  //
  // Spelled x/|x| and NOT production's exp(i*angle(x)), although the two are
  // algebraically the same map, because this loop is Nt*NRF = 4096 TRANSCENDENTAL
  // calls at the largest pool shape (K=8 Nt=128 NRF=32) and transcendentals are
  // the one thing a Cortex-A53 has no hardware for: they are the other half of
  // the projected 4.6 ms "glue+proj" PS budget in
  // docs/cost_model_real_shapes.md, and the reason the PS role misses a 1 ms
  // instant even after the three GEMMs go to the fabric. Measured on this host
  // (M4, 4096 projections x 2000 reps, /tmp/mpx3/proj_probe.cpp):
  //   exp(i*arg(x))/sqrt(Nt)    51.75 us
  //   x * (1/sqrt(Nt)) / |x|     5.86 us   -> 8.83x cheaper
  //   max |difference| = 3.10e-17 on entries of modulus 1/sqrt(128) = 0.0884
  //   (2.2 ulp) and 1.39e-16 on entries of modulus 0.5 (1.25 ulp), i.e. a couple
  //   of ulps of the ENTRY, and ~5x smaller than the Frf-vs-production error
  //   frontend_bench already reports for this step (7.2e-16 / 1.2e-15 / 1.3e-14
  //   over its three cases, dominated by the SVD basis, and reading the SAME
  //   before and after this change). Identical to the ulp at |x| = 1e-300 .. 1e0
  //   (std::abs is hypot, so the modulus neither overflows nor underflows).
  // Hoisting 1/sqrt(Nt) out of the loop and turning the per-entry division into a
  // multiplication is part of the 8.83x, not a separate change.
  //
  // The x == 0 branch keeps production's convention and is NOT a no-op: numpy's
  // angle(0) is 0, so exp(i*angle(0))/sqrt(Nt) = 1/sqrt(Nt), a unit-modulus
  // entry with phase 0. Returning 0 there instead would make Frf lose its
  // constant modulus -- the one property the whole RF chain exists to enforce --
  // for a structured channel whose singular vector has an exactly-zero entry.
  const double rs = 1.0 / std::sqrt((double)Nt);
  for (int i = 0; i < Nt * NRF; ++i) {
    const double m = std::abs(src[i]);
    Frf[i] = m > 0.0 ? src[i] * (rs / m) : dcplx(rs, 0.0);
  }
}

int analog_users(int K, int Nr, int Nt, int NRF) {
  // How many users' right singular bases can possibly reach Frf: each supplies
  // min(NRF, Nr, Nt) columns and the loop stops as soon as NRF are filled. The
  // rest is DFT filler, so factorising those users would be pure waste -- at
  // K=8, NRF=8, Nr=4 that is 6 of 8 SVDs. Both analog_precoder (host) and
  // front_end (provider-batched) use this, so the batch is the same size on
  // either path.
  const int take = std::max(1, std::min(NRF, std::min(Nr, Nt)));
  return std::min(K, (NRF + take - 1) / take);
}

void analog_precoder(const dcplx* H, int K, int Nr, int Nt, int NRF,
                     dcplx* Frf) {
  const int nu = analog_users(K, Nr, Nt, NRF);
  const int take = std::min(NRF, std::min(Nr, Nt));
  std::vector<dcplx> V((size_t)nu * Nt * take);
  std::vector<double> S(Nt);              // svd_wide fills all n, trailing zeros
  for (int u = 0; u < nu; ++u)
    // svd_wide, not svd_cplx: H_u is [Nr x Nt] with Nr <= 8 and Nt up to 128 in
    // every deployed config, and only its `take` LEADING right singular vectors
    // are used. Measured 5256x faster at K=8/Nt=128 with 5.4e-15 agreement
    // (ref/decomp.h).
    svd_wide(H + (size_t)u * Nr * Nt, Nr, Nt, take, nullptr, S.data(),
             V.data() + (size_t)u * Nt * take);   // sweeps: decomp.h default
  analog_precoder_from_V(V.data(), nu, Nr, Nt, NRF, Frf);
}

}  // namespace hpocl
