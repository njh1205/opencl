// switch_threshold.cl: O(1) surrogate for the SVD/THP switch decision (fp32).
#include "switch_law.clh"

// switch_threshold: (lambda_min, g_min, metric_dB) -> tau_dB_out + mode_out (1 = THP, 0 = SVD)
__kernel void switch_threshold(__global const float* restrict lambda_min,
                               __global const float* restrict g_min,
                               __global const float* restrict metric_dB,
                               int count, float nt, float nrf, float eta,
                               __global float*  restrict tau_dB_out,
                               __global uchar*  restrict mode_out)
{
  int i = get_global_id(0);
  if (i >= count) return;
  float tau_db = switch_tau_db(switch_tau_lin(lambda_min[i], g_min[i], nt, nrf, eta));
  tau_dB_out[i] = tau_db;
  mode_out[i]   = (uchar)switch_selects_thp(metric_dB[i], tau_db);
}
