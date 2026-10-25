//-----------------------------------------------------------------------------
// operator: switch_threshold -- O(1) surrogate for the SVD/THP decision.
//   tau_dB = 10*log10( (N_t/N_RF) * eta * lambda_min / g_min )
//   mode   = metric_dB >= tau_dB ? 1 (THP) : 0 (SVD)
//
// NOT the deployed law: the pipeline decides with the closed-form/Newton
// threshold of kernels/threshold_cf.cl (see ref/pipeline.cpp), which is what
// main.tex eq. (cf) derives from the deficit model. This kernel is the cheap
// slot that needs only lambda_min and g_min -- useful as a PL-side pre-filter
// or as a sanity bound -- and it is kept honest by sharing switch_law.clh with
// the fp64 host reference instead of carrying a second copy of the formula.
// It has no production oracle of its own: op_bench checks kernel == host
// reference, not kernel == production policy.
//-----------------------------------------------------------------------------
#include "switch_law.clh"

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
