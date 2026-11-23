// channel_cov.cl: channel covariance over S snapshots (fp32; complex = float2(re,im)).
#pragma OPENCL EXTENSION cl_khr_fp64 : disable
#include "cplx.clh"

// channel_cov: H[M x N] per snapshot (S snapshots) -> Hermitian R[M x M] = (1/S) sum_s H_s H_s^H
__kernel void channel_cov(__global const float2* restrict H,
                          int M, int N, int S,
                          __global float2* restrict R)
{
  int i = get_global_id(0);
  int j = get_global_id(1);
  if (i >= M || j >= M || i > j) return;   // upper triangle owns both entries

  float2 acc = (float2)(0.0f, 0.0f);
  for (int s = 0; s < S; ++s) {
    __global const float2* Hs = H + (size_t)s * M * N;
    for (int n = 0; n < N; ++n)
      acc = cmacr(acc, Hs[(size_t)i * N + n], Hs[(size_t)j * N + n]);
  }
  float inv = 1.0f / (float)S;
  float2 r = (float2)(acc.x * inv, acc.y * inv);
  R[(size_t)i * M + j] = r;
  if (i != j) R[(size_t)j * M + i] = (float2)(r.x, -r.y);
}
