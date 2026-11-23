// wbb_apply.cl: Y[sc] = W_BB * X[sc] - complex GEMV per subcarrier; local-memory tiling with direct-path fallback.
#include "cplx.clh"

#ifndef WBB_LOCAL_MAX
#define WBB_LOCAL_MAX 128       // float2 slots = 1 KB per work-group
#endif

// wbb_apply: W[M][N] x X[SC][N] -> Y[SC][M]; tiles the group's W/X rows into local memory when it fits, else direct global path.
__kernel void wbb_apply(__global const float2* restrict W,   // [M][N]
                        int M, int N,
                        __global const float2* restrict X,   // [SC][N]
                        int SC,
                        __global float2* restrict Y)         // [SC][M]
{
  int m  = get_global_id(0);
  int sc = get_global_id(1);
  if (m >= M || sc >= SC) return;

  const int LM = (int)get_local_size(0);        // work-items along m
  const int LS = (int)get_local_size(1);        // work-items along sc
  const int tile = (LM + LS) * N;
  float2 acc = (float2)(0.0f, 0.0f);

  if (LM * LS < 2 || tile > WBB_LOCAL_MAX) {
    __global const float2* wrow = W + (size_t)m * N;
    __global const float2* xrow = X + (size_t)sc * N;
    for (int n = 0; n < N; ++n) acc = cmac(acc, wrow[n], xrow[n]);
    Y[(size_t)sc * M + m] = acc;
    return;
  }

  // group base indices; the tile covers exactly this group's rows
  const int m0  = (int)get_group_id(0) * LM;
  const int s0  = (int)get_group_id(1) * LS;
  const int lm  = (int)get_local_id(0);
  const int ls  = (int)get_local_id(1);
  __local float2 wtile[WBB_LOCAL_MAX];          // LM rows of W, then LS rows of X
  __local float2* xtile = wtile + LM * N;

  // cooperative strided load; out-of-range rows skipped (compute side guards its indices)
  for (int e = lm * LS + ls; e < LM * N; e += LM * LS) {
    int r = e / N, n = e - r * N;
    if (m0 + r < M) wtile[e] = W[(size_t)(m0 + r) * N + n];
  }
  for (int e = lm * LS + ls; e < LS * N; e += LM * LS) {
    int r = e / N, n = e - r * N;
    if (s0 + r < SC) xtile[e] = X[(size_t)(s0 + r) * N + n];
  }
  barrier(CLK_LOCAL_MEM_FENCE);

  __local const float2* wrow = wtile + (size_t)lm * N;   // this work-item's W row
  __local const float2* xrow = xtile + (size_t)ls * N;   // this work-item's X row
  for (int n = 0; n < N; ++n) acc = cmac(acc, wrow[n], xrow[n]);
  Y[(size_t)sc * M + m] = acc;
}
