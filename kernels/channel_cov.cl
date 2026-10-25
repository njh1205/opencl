//-----------------------------------------------------------------------------
// operator: channel_cov   R = (1/S) * sum_s H_s H_s^H
// work-item  : one element (i,j) of R, and -- because R is Hermitian -- the
//              work-item with i < j writes BOTH R[i][j] and R[j][i] while the
//              i > j work-items exit immediately.
// FPGA note  : pure outer-product reduction, no barriers and no local memory, so
//              the arithmetic should map to DSP48E2 slices on Zynq PL
//              (XCZU49DR; DSP48E2, not DSP58 -- that is Versal). Not yet
//              measured: there is no Vitis synthesis report in-tree (README,
//              Zynq checklist 5).
//              The binding constraint on the fabric is TRAFFIC, not DSPs: even
//              after the Hermitian split below this kernel reads S*N*M*(M+1)
//              float2 from global memory for S*M*N of data -- 1.11 MB read for
//              65.5 KB of payload at M=N=16, S=32, a 17x amplification. At
//              ~2.4 GB/s per AXI port that is ~0.5 ms per covariance update,
//              i.e. a scheduling-instant-scale cost for an operator usually
//              assumed to be free. A local-memory tile over H_s would cut it by
//              ~M, but that needs a work-group shape the host does not currently
//              choose (it launches g = {M, M} with a null local size), so it is
//              recorded here rather than done: see the wbb_apply occupancy table
//              in README for what a static __local array costs when the tile
//              does not pay for itself.
//
// Why the Hermitian split: the naive M^2 work-items each stream two rows of
// every snapshot, i.e. 2*S*N*M^2 float2 of global traffic for S*M*N of data.
// Halving the number of working work-items halves that traffic and the multiply
// count, and needs neither local memory nor a barrier. It is also bit-exact:
// for a = H[i,n], b = H[j,n], cmacr accumulates
//     re += a.x*b.x + a.y*b.y,   im += a.y*b.x - a.x*b.y
// and swapping (i,j) for (j,i) gives the same products in the same n and s
// order, with re unchanged and im exactly negated (IEEE-754 multiply and add
// are commutative, and the final 1/S scale is applied identically). So
// R[j][i] == conj(R[i][j]) to the last bit, and op_bench's relative error is
// unchanged by this rewrite.
//-----------------------------------------------------------------------------
#pragma OPENCL EXTENSION cl_khr_fp64 : disable
#include "cplx.clh"

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
    for (int n = 0; n < N; ++n)          // acc += H[i,n] * conj(H[j,n])
      acc = cmacr(acc, Hs[(size_t)i * N + n], Hs[(size_t)j * N + n]);
  }
  float inv = 1.0f / (float)S;
  float2 r = (float2)(acc.x * inv, acc.y * inv);
  R[(size_t)i * M + j] = r;
  if (i != j) R[(size_t)j * M + i] = (float2)(r.x, -r.y);
}
