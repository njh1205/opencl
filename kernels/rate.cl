// rate.cl: per-stream GMI -> MCS selection (CQI index, m, r, SE, margin, info; fp32).
#include "mcs_table.clh"

// mcs_select_k: per-stream GMI -> CQI index / m / r / SE / margin / info; outage -> idx = -1, m = r = 0, out = 1
//   work-item: one stream, gid(0) = k < K
__kernel void mcs_select_k(__global const float* restrict gmi, int K, float eps,
                           __global int* restrict idx, __global int* restrict m,
                           __global int* restrict r, __global float* restrict se,
                           __global float* restrict marg,
                           __global float* restrict info,
                           __global uchar* restrict out) {
  int k = get_global_id(0);
  if (k >= K) return;

  const float g = gmi[k];
  const int j = cqi_select(g, eps);
  idx[k] = j;
  if (j < 0) {                       // outage: defined sentinels, not garbage
    m[k] = 0; r[k] = 0;
    se[k] = 0.0f; marg[k] = 0.0f; info[k] = 0.0f; out[k] = 1;
    return;
  }
  const float s = cqi_se(j);
  m[k] = kCqiM[j];
  r[k] = kCqiR[j];
  se[k] = s;
  marg[k] = g - s;
  info[k] = cqi_info_rate(g, j);
  out[k] = 0;
}
