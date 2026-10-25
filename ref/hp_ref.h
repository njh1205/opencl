#pragma once
// CPU golden reference for every HPOCL operator (no OpenCL dependency).
// Used by the bench to cross-validate kernel bit-behaviour on any host,
// and later as the PS-side fallback path when no PL accelerator is present.
#include "hpocl_types.h"

namespace hpref {
using namespace hpocl;

void channel_cov(const std::vector<cplx>& H, int M, int N, int S,
                 std::vector<cplx>& R);
void power_iter(const std::vector<cplx>& R, int M, int iters,
                std::vector<cplx>& v, float& lambda);
void deflate(std::vector<cplx>& R, int M, const std::vector<cplx>& v,
             float lambda);
void switch_threshold(const std::vector<float>& lambda_min,
                      const std::vector<float>& g_min,
                      const std::vector<float>& metric_dB,
                      float nt, float nrf, float eta,
                      std::vector<float>& tau_dB, std::vector<uint8_t>& mode);
void wbb_apply(const std::vector<cplx>& W, int M, int N,
               const std::vector<cplx>& X, int SC, std::vector<cplx>& Y);
void thp_precoder(const std::vector<cplx>& B, const std::vector<cplx>& F,
                  const std::vector<cplx>& X, int SC, int M, int K, float tau,
                  std::vector<cplx>& Y);
}  // namespace hpref
