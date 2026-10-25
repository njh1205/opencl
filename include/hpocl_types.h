#pragma once
//-----------------------------------------------------------------------------
// HPOCL - Heterogeneous Hybrid-Precoding Operator Stack (AI-RAN flavor)
//
// One OpenCL operator library, three execution targets:
//   [dev] x86/ARM CPU  : CL_DEVICE_TYPE_CPU      (algorithm bring-up, today)
//   [ps ] Cortex-A53   : host runtime on Zynq PS  (control plane, SVD flow)
//   [pl ] FPGA fabric  : CL_DEVICE_TYPE_ACCELERATOR via Vitis (.xclbin)
//
// Data layout conventions (SoA-free, AoS float2 = complex):
//   H  : [S][M][N]  float2   channel matrices (S snapshots, M=N_RF rows...)
//   R  : [M][M]     float2   Hermitian covariance H H^H
//   W  : [M][N]     float2   baseband precoder W_BB
//   X  : [SC][N]    float2   data-stream vectors per subcarrier
//   Y  : [SC][M]    float2   precoded output
//-----------------------------------------------------------------------------
#include <complex>
#include <vector>
#include <cstdint>

namespace hpocl {
using cplx = std::complex<float>;

struct PrecoderCfg {
  int M = 16;          // RF chains / precoder rows  (N_RF)
  int N = 16;          // antennas / columns         (N_t  subset per chain grp)
  int S = 32;          // channel snapshots averaged into covariance
  int SC = 64;         // subcarriers
  int K = 8;           // THP streams (sequential dimension)
  int iters = 24;      // power-iteration count for dominant singular pair
  float eta = 1.0f;    // power-normalization factor from main.tex
  float tau = 4.0f;    // THP modulo constellation period
};

// Per-stream result of the code-rate / MCS layer (kernels/rate.cl). The table
// and the selection rule live in kernels/mcs_table.clh so that the host
// reference and the device kernel compile the same text.
struct McsSel {
  int   index  = -1;     // CQI index, -1 == outage (nothing in the table fits)
  int   m      = 0;      // modulation order of the selected entry
  int   rate   = 0;      // code rate x1024 of the selected entry
  float se     = 0.0f;   // m*rate/1024, bits per channel use
  float margin = 0.0f;   // gmi - se, the rate margin left over
  float info   = 0.0f;   // deliverable information rate, bits per channel use
  bool  outage = true;   // index < 0
};

// The linear-domain switching law (surrogate slot) lives in kernels/switch_law.clh
// so that ref/hp_ref.cpp and kernels/switch_threshold.cl compile the same text.
// The deployed decision is the closed form in ref/threshold_cf.cpp.
}  // namespace hpocl
