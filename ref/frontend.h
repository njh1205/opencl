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
}  // namespace hpocl
