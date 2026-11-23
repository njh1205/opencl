// thp_precoder.cl: Tomlinson-Harashima precoding - b = mod_tau(x - sum_{j<k} B[k][j] b[j]), y = F b.
// One work-item per subcarrier walks the K streams in order (the feedback loop is serial).
#include "cplx.clh"        // cmul/csub/cmod/cmac

// KMAX bounds per-work-item b[]; host enforces K <= KMAX too (defence in depth), NaN-fills Y otherwise
#define KMAX 16

// thp_precoder: B[K][K], F[M][K], X[SC][K] -> Y[SC][M]; sequential THP feedback + feedforward, one subcarrier per work-item.
__kernel void thp_precoder(__global const float2* restrict B,   // [K][K] fb
                           __global const float2* restrict F,   // [M][K] ff
                           __global const float2* restrict X,   // [SC][K]
                           int SC, int M, int K, float tau,
                           __global float2* restrict Y)         // [SC][M]
{
  int sc = get_global_id(0);
  if (sc >= SC) return;
  if (K > KMAX || K < 1) {
    for (int m = 0; m < M; ++m) Y[(size_t)sc * M + m] = (float2)(NAN, NAN);
    return;
  }

  float2 b[KMAX];
  for (int k = 0; k < K; ++k) {
    float2 acc = X[(size_t)sc * K + k];
    for (int j = 0; j < k; ++j) acc = csub(acc, cmul(B[(size_t)k * K + j], b[j]));
    b[k] = cmod(acc, tau);
  }
  for (int m = 0; m < M; ++m) {
    float2 acc = (float2)(0.0f, 0.0f);
    for (int k = 0; k < K; ++k) acc = cmac(acc, F[(size_t)m * K + k], b[k]);
    Y[(size_t)sc * M + m] = acc;
  }
}
