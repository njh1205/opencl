//-----------------------------------------------------------------------------
// operator: thp_precoder  (Tomlinson-Harashima branch)
//   b[k] = mod_tau( x[k] - sum_{j<k} B[k][j] b[j] )     sequential feedback
//   y    = F b                                          feedforward
// The feedback loop is inherently serial, so one work-item owns one
// subcarrier and walks the K streams in order -- the same "recurrent kernel"
// pattern used for KV-cache update kernels on GPU inference runtimes.
//-----------------------------------------------------------------------------
#include "cplx.clh"        // cmul / csub / cmod: shared with the rest of the datapath

// KMAX bounds the per-work-item feedback history b[KMAX]; the host passes K
// straight through, so an oversized K is rejected here rather than overflowing
// private memory. Runtime::thp_precoder refuses K > KMAX host-side (and
// NaN-fills Y), so this branch is defence in depth for any other caller -- and
// it writes a sentinel for the same reason: an untouched Y is indistinguishable
// from a legitimate one. tools/check_capacity.py keeps the two KMAX values equal.
#define KMAX 16

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
