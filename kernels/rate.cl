//-----------------------------------------------------------------------------
// operator group: code-rate / MCS layer (the BICM outer parameter)
//
//   gmi_k (kernels/datapath.cl) -- per-stream GMI, bits per coded symbol
//        `--> mcs_select_k -- per-stream CQI index, m, R, SE, margin, outage
//
// Position in the chain: this is what turns the modem's information-theoretic
// metric into a link decision. The MART paper's transceiver design maximises
// the achievable rate for a *given* modulation and coding scheme, so the scheme
// has to come from somewhere -- in a real system from CQI feedback, here from
// the measured GMI through the table in kernels/mcs_table.clh.
//
// fp32 only, like the rest of the datapath; the table itself is integer, so the
// only rounding in this kernel is the SE division and the GMI comparison.
//-----------------------------------------------------------------------------
#include "mcs_table.clh"

//-----------------------------------------------------------------------------
// mcs_select_k : per-stream GMI -> selected MCS and its margins
//   in  gmi  : float[K]  generalized mutual information per stream (bits per
//                        coded symbol), i.e. the output of gmi_k
//       K    : number of streams
//       eps  : selection slack in bits (see cqi_select)
//   out idx  : int[K]    CQI index, -1 on outage
//       m    : int[K]    modulation order of the selected entry (0 on outage)
//       r    : int[K]    code rate x1024 of the selected entry (0 on outage)
//       se   : float[K]  m*r/1024, bits per channel use
//       marg : float[K]  gmi - se, the rate margin left over; 0 on outage
//                        (a margin against "no MCS" is not a rate, and leaving
//                        gmi there would propagate an inf/NaN GMI downstream)
//       info : float[K]  deliverable information rate, bits per channel use
//       out  : uchar[K]  1 on outage, 0 otherwise
//   work-item : one stream (1-D), gid(0) = k < K
// Every output is written on every path, including the K-overrun-free outage
// branch, so a caller never reads a stale buffer.
//-----------------------------------------------------------------------------
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
