#pragma once
//-----------------------------------------------------------------------------
// One scheduling instant of the PS control plane (appendix_zynq role):
//   analog F_RF -> effective channels -> per-user BD null basis ->
//   power whitening -> whitened spectrum -> closed-form threshold ->
//   mode policy (SVD vs THP) -> per-user structured design (V_D, B, rho).
// Composes ref/frontend.cpp + ref/decomp.cpp + ref/threshold_cf.cpp; every
// ingredient is individually validated against the production python.
//-----------------------------------------------------------------------------
#include "frontend.h"
#include <vector>

namespace hpocl {

struct PipeCfg {
  int K, Nt, Nr, NRF, d, M;   // geometry and QAM order M = 2^m
  double gamma_tot_dB;        // online decision variable s
  int variant;                // 0 = GMD, 1 = MMSE-UCD (THP branch flavour)
};

struct PipeOut {
  double tau_dB = 0;
  int mode = 0;               // 0 = SVD branch, 1 = THP branch
  // status of the threshold solve, in the SAME 0/1/2/4 encoding as
  // CfResult::flag / flags[c] in kernels/threshold_cf.cl (see ref/threshold_cf.h).
  // A caller MUST look at this: on 4 the threshold is NaN because the shape is
  // outside the compiled capacity, and `mode` is then a forced conservative SVD
  // fallback rather than a decision. 2 is a legitimate decision (tau = +-inf,
  // i.e. the branch ordering is uniform over the whole scan range).
  int cf_flag = 0;
  int nul = 0;                // BD nullity (per-user digital dimension)
  std::vector<dcplx> Frf;     // [Nt x NRF]
  std::vector<dcplx> Nk;      // [K][NRF x nul] BD bases
  std::vector<dcplx> Vd;      // [K][nul x d] per-user digital factor
  std::vector<dcplx> B;       // [K][d x d] THP feedback upper (identity: SVD)
  std::vector<double> sigma;  // [K][d] whitened singular values
  std::vector<double> rho_real, rho_tgt;  // [K][d]
  std::vector<dcplx> Q;       // [K][Nr x d] receive combiners (u_eff/q_gmd/w_ucd)
  std::vector<dcplx> R;       // [K][d x d] equalized upper factor (diag(s)/r_gmd/r_aug)
  std::vector<dcplx> Fbb;     // [NRF x T] N_k C_k^{-1/2} V_D,k, per-user blocks
                              // side by side (i-th RF, stream k*d+j at i*T+k*d+j),
                              // scaled by pscale
  std::vector<dcplx> diagb;   // [K][d] pscale * diag(R): the gain the receiver
                              //        divides by (thp_receive_equalized)
  double pscale = 1.0;        // digital power normalization sqrt(P/||Frf Fbb||^2)
};

void schedule(const dcplx* H /*[K][Nr x Nt]*/, const PipeCfg& c, PipeOut& o);

}  // namespace hpocl
