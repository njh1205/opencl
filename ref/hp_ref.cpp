#include "hp_ref.h"
#include <cmath>
#include "hpocl_modulo.h"   // cmod: one definition shared with ref/modem.cpp and the kernels
#include "switch_law.clh"   // switching law: same text the device kernel compiles

namespace hpref {
using namespace hpocl;

void channel_cov(const std::vector<cplx>& H, int M, int N, int S,
                 std::vector<cplx>& R) {
  R.assign((size_t)M * M, cplx(0, 0));
  for (int s = 0; s < S; ++s) {
    const cplx* Hs = H.data() + (size_t)s * M * N;
    for (int i = 0; i < M; ++i)
      for (int j = 0; j < M; ++j) {
        cplx acc(0, 0);
        for (int n = 0; n < N; ++n) acc += Hs[(size_t)i * N + n] * std::conj(Hs[(size_t)j * N + n]);
        R[(size_t)i * M + j] += acc;
      }
  }
  for (auto& r : R) r /= (float)S;
}

void power_iter(const std::vector<cplx>& R, int M, int iters,
                std::vector<cplx>& v, float& lambda) {
  v.assign(M, cplx(0, 0));
  float n2 = 0;
  for (int i = 0; i < M; ++i) {
    v[i] = cplx(std::cos(i * 0.7f + 0.3f), std::sin(i * 1.1f));
    n2 += std::norm(v[i]);
  }
  for (auto& x : v) x /= std::sqrt(n2);
  lambda = 0;
  for (int t = 0; t < iters; ++t) {
    std::vector<cplx> w(M, cplx(0, 0));
    for (int i = 0; i < M; ++i)
      for (int j = 0; j < M; ++j) w[i] += R[(size_t)i * M + j] * v[j];
    float nn = 0;
    for (auto& x : w) nn += std::norm(x);
    lambda = std::sqrt(nn);
    for (auto& x : w) x /= lambda > 1e-20f ? lambda : 1e-20f;
    v.swap(w);
  }
}

void deflate(std::vector<cplx>& R, int M, const std::vector<cplx>& v,
             float lambda) {
  for (int i = 0; i < M; ++i)
    for (int j = 0; j < M; ++j)
      R[(size_t)i * M + j] -= lambda * (v[i] * std::conj(v[j]));
}

void switch_threshold(const std::vector<float>& lambda_min,
                      const std::vector<float>& g_min,
                      const std::vector<float>& metric_dB,
                      float nt, float nrf, float eta,
                      std::vector<float>& tau_dB, std::vector<uint8_t>& mode) {
  size_t c = lambda_min.size();
  tau_dB.resize(c); mode.resize(c);
  for (size_t i = 0; i < c; ++i) {
    tau_dB[i] = switch_tau_db(switch_tau_lin(lambda_min[i], g_min[i], nt, nrf, eta));
    mode[i] = (uint8_t)switch_selects_thp(metric_dB[i], tau_dB[i]);
  }
}

void wbb_apply(const std::vector<cplx>& W, int M, int N,
               const std::vector<cplx>& X, int SC, std::vector<cplx>& Y) {
  Y.assign((size_t)SC * M, cplx(0, 0));
  for (int sc = 0; sc < SC; ++sc)
    for (int m = 0; m < M; ++m) {
      cplx acc(0, 0);
      for (int n = 0; n < N; ++n) acc += W[(size_t)m * N + n] * X[(size_t)sc * N + n];
      Y[(size_t)sc * M + m] = acc;
    }
}

void thp_precoder(const std::vector<cplx>& B, const std::vector<cplx>& F,
                  const std::vector<cplx>& X, int SC, int M, int K, float tau,
                  std::vector<cplx>& Y) {
  Y.assign((size_t)SC * M, cplx(0, 0));
  for (int sc = 0; sc < SC; ++sc) {
    std::vector<cplx> b(K);
    for (int k = 0; k < K; ++k) {
      cplx acc = X[(size_t)sc * K + k];
      for (int j = 0; j < k; ++j) acc -= B[(size_t)k * K + j] * b[j];
      // centered modulo onto [-tau/2, tau/2): production
      // thp_precoding.centered_modulo, see include/hpocl_modulo.h.
      // Fold in double and narrow afterwards, exactly as the local lambda did
      // before: this is the golden reference, so factoring the expression out
      // must not change its rounding.
      b[k] = cplx(hpocl::cmod(std::complex<double>(acc), (double)tau));
    }
    for (int m = 0; m < M; ++m) {
      cplx acc(0, 0);
      for (int k = 0; k < K; ++k) acc += F[(size_t)m * K + k] * b[k];
      Y[(size_t)sc * M + m] = acc;
    }
  }
}
}  // namespace hpref
