// power_iter.cl: power iteration over Hermitian R - dominant eigenpair (v_out, lambda_out) per matrix, plus deflate to strip it.
#define PMAX 64          // max matrix dim M == max work-group size; M <= PMAX

// power_iter: iterated R*v/|R*v| on nmat Hermitian MxM matrices -> dominant eigenpair per matrix (v_out, lambda_out).
__kernel void power_iter(__global const float2* restrict R,
                         int M, int iters,
                         __global float2* restrict v_out,
                         __global float*  restrict lambda_out)
{
  __local float2 lv[PMAX];
  __local float2 lw[PMAX];
  __local float  lp[PMAX];
  __local float  lnorm;
  __local float  linv;
  int li = get_local_id(0);
  int g  = get_group_id(0);

  // capacity guard: form Rg only after this check (with lsz < M, g*M*M can point past R)
  int lsz = (int)get_local_size(0);
  if (M <= 0 || M > PMAX || lsz < M) {
    if (M > 0) {
      // sentinel by global id: rows [lsz, M) have no work-item when lsz < M
      v_out[get_global_id(0)] = (float2)(NAN, NAN);
      if (lsz >= M && li == 0) lambda_out[g] = NAN;
    }
    return;
  }
  __global const float2* Rg = R + (size_t)g * M * M;

  // deterministic start vector; must match ref/hp_ref.cpp::power_iter bit-for-bit
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

// deflate: R -= lambda * v * v^H (rank-1 update), one (i, j) element per work-item.
__kernel void deflate(__global float2* restrict R, int M,
                      __global const float2* restrict v, float lambda)
{
  int i = get_global_id(0);
  int j = get_global_id(1);
  int g = get_global_id(2);   // matrix index in the batch
  if (i >= M || j >= M) return;
  __global const float2* vg = v + (size_t)g * M;
  float2 a = vg[i], b = vg[j];
  float2 d = (float2)(lambda * (a.x * b.x + a.y * b.y),
                      lambda * (a.y * b.x - a.x * b.y));
  size_t idx = (size_t)g * M * M + (size_t)i * M + j;
  R[idx] = (float2)(R[idx].x - d.x, R[idx].y - d.y);
}
