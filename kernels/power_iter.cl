//-----------------------------------------------------------------------------
// operator: power_iter / deflate
//   PL-friendly substitute for a full SVD: dominant singular pair of the
//   Hermitian covariance R via power iteration. The runtime launches this
//   N_s times with `deflate` in between = "iterative eigensolver" pattern
//   (same shape as Lanczos/LOBPCG kernels in GPU AI frameworks).
// work-group : one matrix (local size == M, capped at PMAX)
//
// BARRIER BUDGET (measured by counting, read before "optimising" this kernel):
//   the loop below uses exactly two barrier(CLK_LOCAL_MEM_FENCE) per iteration,
//   which is the minimum for this decomposition: every work-item reads all of
//   lv[] to form its row dot (needs the previous normalisation visible), and the
//   normalisation needs all of lw[] (needs all row dots visible). Parallelising
//   the normalisation as well would add a third barrier per iteration to save
//   4M of the 5M flops work-item 0 does serially, against M^2 flops done in
//   parallel -- an Amdahl bound of ~1.4x at M=16 before counting the extra
//   synchronisation, so it is deliberately not done.
//   What IS parallel is the initial vector: cos/sin are the most expensive
//   serial work in the kernel (~15% of a 24-iteration launch at M=16), so each
//   work-item now evaluates its own entry and only the M-term norm sum stays in
//   work-item 0. Same per-element expressions, same summation order, therefore
//   bit-identical to the previous single-work-item initialisation.
//   On the PL a barrier is a pipeline flush, so if Vitis reports the sync as the
//   bottleneck the alternative is one work-item per matrix with lv/lw in private
//   memory (zero barriers, zero local memory, 1 KB of on-chip storage per
//   work-item at PMAX=64) -- that trades the M-way intra-matrix parallelism for
//   nmat-way inter-matrix parallelism, and is only a win when nmat >= M. It is
//   NOT implemented here because the in-tree caller launches nmat=1.
//
// PMAX / local-size coupling (read before changing either):
//   runtime/hpocl_runtime.cpp launches this kernel with
//       size_t g[1] = {nmat * M};  size_t l[1] = {M};
//   i.e. one work-item per row of R and local size == M exactly. The kernel
//   then relies on two invariants:
//     (1) M <= PMAX, because lv[]/lw[] are __local[PMAX] and work-item 0
//         initialises (line "if (li == 0)") and re-normalises entries
//         0..M-1 on its own, and every work-item with li < M writes lw[li];
//     (2) local size >= M, so that a work-item exists for every row
//         (rows li >= M are never computed, and the group reduction in
//         work-item 0 would then read uninitialised local slots).
//   Violating (1) overruns __local memory (the runtime allocates only
//   CL_KERNEL_LOCAL_MEM_SIZE bytes per group) and corrupts neighbouring
//   groups; the guard below turns it into an explicit NaN sentinel on
//   v_out/lambda_out -- loud, bounded, and no uninitialised output -- instead
//   of silent garbage. Violating (2) is the case that needs care: there is no
//   work-item for rows [lsz, M), so the sentinel is written by GLOBAL id (which
//   is in bounds for any local size) rather than by (group, local) id, and
//   lambda_out is left alone because its extent nmat cannot be derived here.
//   PMAX also fixes the PL local-memory footprint:
//   3 * PMAX * sizeof(float2) = 768 B per work-group at PMAX = 64 (lv, lw and
//   the lp partial-norm array), plus 8 B of scalars.
//   A larger M therefore needs BOTH a larger PMAX here AND a host that keeps
//   local size == M within the device's CL_DEVICE_MAX_WORK_GROUP_SIZE.
//-----------------------------------------------------------------------------
#define PMAX 64          // max matrix dimension M == max work-group size used

__kernel void power_iter(__global const float2* restrict R,
                         int M, int iters,
                         __global float2* restrict v_out,
                         __global float*  restrict lambda_out)
{
  __local float2 lv[PMAX];
  __local float2 lw[PMAX];
  __local float  lp[PMAX];     // per-entry |.|^2 partials: init and loop share it
  __local float  lnorm;
  __local float  linv;
  int li = get_local_id(0);
  int g  = get_group_id(0);

  // capacity guard: see the PMAX / local-size coupling note above. All
  // work-items of the group take the same branch (M and the local size are
  // uniform), so there is no divergent-barrier hazard. `Rg` is formed only
  // after the guard: with lsz < M the group count exceeds nmat, so g*M*M can
  // point past the end of R and merely FORMING that pointer is already out of
  // bounds, even though nothing would dereference it on this path.
  int lsz = (int)get_local_size(0);
  if (M <= 0 || M > PMAX || lsz < M) {
    if (M > 0) {
      // Sentinel by GLOBAL id, not by (g, li). The host launches exactly
      // nmat*M work-items and v_out has extent nmat*M, so v_out[global_id] is
      // in bounds by construction for ANY local size. The previous
      // v_out[g*M + li] was wrong twice over when lsz < M: rows [lsz, M) of
      // every matrix had no work-item at all and stayed uninitialised (bV is
      // CL_MEM_WRITE_ONLY, hpocl_runtime.cpp), while groups with g >= nmat
      // wrote past the end. lambda_out has extent nmat, which is not derivable
      // from the arguments, so it is written only when the group count does
      // equal nmat -- i.e. when the local size is legal and it is M itself that
      // is out of contract.
      v_out[get_global_id(0)] = (float2)(NAN, NAN);
      if (lsz >= M && li == 0) lambda_out[g] = NAN;
    }
    return;
  }
  __global const float2* Rg = R + (size_t)g * M * M;

  // deterministic start vector (must match ref/hp_ref.cpp::power_iter entry for
  // entry, or op_bench compares two different iterations). The transcendentals
  // are spread over the work-items; the M-term norm sum stays serial in
  // work-item 0 and in the same 0..M-1 order, so the result is bit-identical to
  // computing the whole vector in work-item 0.
  if (li < M) {
    float2 v = (float2)(cos((float)li * 0.7f + 0.3f), sin((float)li * 1.1f));
    lv[li] = v;
    lp[li] = v.x * v.x + v.y * v.y;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  if (li == 0) {
    float n2 = 0.0f;
    for (int i = 0; i < M; ++i) n2 += lp[i];
    lnorm = sqrt(n2);
    linv = 1.0f / (lnorm > 1e-20f ? lnorm : 1e-20f);
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  if (li < M) lv[li] = (float2)(lv[li].x * linv, lv[li].y * linv);
  barrier(CLK_LOCAL_MEM_FENCE);

  float lam = 0.0f;
  for (int t = 0; t < iters; ++t) {
    // w = R v   (one row dot per work-item)
    if (li < M) {
      float2 acc = (float2)(0.0f, 0.0f);
      for (int j = 0; j < M; ++j) {
        float2 r = Rg[(size_t)li * M + j];
        float2 v = lv[j];
        acc.x += r.x * v.x - r.y * v.y;
        acc.y += r.x * v.y + r.y * v.x;
      }
      lw[li] = acc;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    // serial tail: the M-term norm sum and the M-entry rescale, both in
    // work-item 0 (see the BARRIER BUDGET note for why this is not split).
    if (li == 0) {
      float n2 = 0.0f;
      for (int i = 0; i < M; ++i) n2 += lw[i].x * lw[i].x + lw[i].y * lw[i].y;
      lnorm = sqrt(n2);
      float inv = 1.0f / (lnorm > 1e-20f ? lnorm : 1e-20f);
      for (int i = 0; i < M; ++i) lv[i] = (float2)(lw[i].x * inv, lw[i].y * inv);
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    lam = lnorm;
  }

  if (li < M) v_out[(size_t)g * M + li] = lv[li];
  if (li == 0) lambda_out[g] = lam;
}

__kernel void deflate(__global float2* restrict R, int M,
                      __global const float2* restrict v, float lambda)
{
  int i = get_global_id(0);
  int j = get_global_id(1);
  int g = get_global_id(2);   // matrix index in the batch
  if (i >= M || j >= M) return;
  __global const float2* vg = v + (size_t)g * M;
  float2 a = vg[i], b = vg[j];
  // R -= lambda * v v^H
  float2 d = (float2)(lambda * (a.x * b.x + a.y * b.y),
                      lambda * (a.y * b.x - a.x * b.y));
  size_t idx = (size_t)g * M * M + (size_t)i * M + j;
  R[idx] = (float2)(R[idx].x - d.x, R[idx].y - d.y);
}
