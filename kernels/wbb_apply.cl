//-----------------------------------------------------------------------------
// operator: wbb_apply   Y[sc] = W_BB * X[sc]   (complex GEMV per subcarrier)
// work-item : (m, sc). This is the hot data-path operator once the mode policy
// selects the SVD branch.
//
// LOCAL-MEMORY TILING. One output element per work-item means the naive form
// re-reads W[m][:] and X[sc][:] from global memory for every (m, sc) pair:
// 2*M*SC*N float2 of traffic for (M + SC)*N of data. A work-group covers
// LM x LS of those pairs but only (LM + LS)*N distinct operands, so the group
// stages them into local memory once -- one barrier per work-item, not per
// element -- and cuts the global traffic by 2*LM*LS/(LM+LS) (16x at 16x16).
//
// The arithmetic is untouched: the inner loop still runs n = 0..N-1 in order and
// still uses cmac, only the operands come from local instead of global memory,
// so the result is bit-identical to the untiled kernel (op_bench's relative
// error is unchanged by this rewrite).
//
// Capacity, and why the default is small. OpenCL C 1.2 has no variable-length
// __local array, so the tile is declared at WBB_LOCAL_MAX slots and the runtime
// gets that much local memory per work-group WHETHER OR NOT the group uses it --
// which is an occupancy limit, not a hint. Measured on the Apple M4 GPU at the
// reference shape (M=N=16, SC=64), complex MAC/s:
//     WBB_LOCAL_MAX = 4096 (32 KB)   59 GFLOP/s   (1.7 us)
//     WBB_LOCAL_MAX =  512 ( 4 KB)  142 GFLOP/s   (0.7 us)
//     WBB_LOCAL_MAX =  128 ( 1 KB)  338 GFLOP/s   (0.3 us)
//     untiled (the old kernel)      364 GFLOP/s   (0.3 us)
// i.e. on a device with a cache hierarchy the re-reads this tile avoids were
// already cheap, and a large static local array costs far more than it saves.
// The PL is the opposite case (no cache: every re-read is a DDR4 round trip over
// the AXI interconnect), which is where the 2*LM*LS/(LM+LS) traffic reduction is
// supposed to pay. So the default keeps the dev/CPU/GPU build at full occupancy
// and takes the direct path for any group that would not fit in 1 KB, and a PL
// build raises it to its BRAM budget:
//     v++ ... -DWBB_LOCAL_MAX=4096
// `make op_pl_bench` runs exactly that configuration as a gate, so the tiled
// path stays verified (bit-identical rel_err) instead of being dead code that
// only the fabric ever compiles.
// The kernel falls back to the direct global path whenever the tile would exceed
// WBB_LOCAL_MAX float2 or the group is degenerate (LM*LS < 2, where staging
// cannot pay for itself). The fallback is the old kernel verbatim.
//-----------------------------------------------------------------------------
#include "cplx.clh"

#ifndef WBB_LOCAL_MAX
#define WBB_LOCAL_MAX 128       // float2 slots = 1 KB per work-group; see above
#endif

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
  const int tile = (LM + LS) * N;               // float2 slots needed
  float2 acc = (float2)(0.0f, 0.0f);

  if (LM * LS < 2 || tile > WBB_LOCAL_MAX) {    // no benefit / no room: direct
    __global const float2* wrow = W + (size_t)m * N;
    __global const float2* xrow = X + (size_t)sc * N;
    for (int n = 0; n < N; ++n) acc = cmac(acc, wrow[n], xrow[n]);
    Y[(size_t)sc * M + m] = acc;
    return;
  }

  // group base indices: the tile holds exactly the rows this group touches
  const int m0  = (int)get_group_id(0) * LM;
  const int s0  = (int)get_group_id(1) * LS;
  const int lm  = (int)get_local_id(0);
  const int ls  = (int)get_local_id(1);
  __local float2 wtile[WBB_LOCAL_MAX];          // LM rows of W, then LS rows of X
  __local float2* xtile = wtile + LM * N;

  // cooperative load, strided over the whole group; out-of-range rows are
  // skipped (never read) because the compute side guards its own indices
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
