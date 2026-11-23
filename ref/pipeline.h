#pragma once
//-----------------------------------------------------------------------------
// One scheduling instant of the control plane (appendix_zynq role):
//   analog F_RF -> effective channels -> per-user BD null basis ->
//   power whitening -> whitened spectrum -> closed-form threshold ->
//   mode policy (SVD vs THP) -> per-user structured design (V_D, B, rho).
// Composes ref/frontend.cpp + ref/decomp.cpp + ref/threshold_cf.cpp; every
// ingredient is individually validated against the production python.
//
// TWO EXECUTION ROLES, ONE BODY. front_end() and design_instant() are written
// once here and take an SvdProvider, so the PS role (fp64 svd_cplx in a loop,
// host_svd()) and the device role (one batched jacobi_svd launch per SVD family,
// runtime/hpocl_schedule.cpp) execute the SAME glue and differ only in where the
// decompositions ran. `make sched_bench` gates the two against each other field
// by field, which is the only way "the kernels are validated" becomes "the board
// runs the control plane". Keeping the glue here also keeps ref/ device-free --
// the provider is a function pointer, so this file has no OpenCL dependency and
// still compiles and runs on a host with no CL runtime at all.
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
  // status of the RF-BD front end: 0 = the geometry is feasible and nul is the
  // true nullity; 1 = DEGENERATE, (K-1)*Nr >= min(Nt, NRF), in which case the
  // interference stack's rows all live in F_RF's row space, its rank is
  // min(Nt, NRF) rather than (K-1)*Nr, null(A) == null(F_RF) exactly, and
  // F_RF N_k == 0 to machine precision. The instant is still computed (the
  // eigenvalue floor keeps every field finite) but it is not a schedule: sigma
  // is ~1e-9 of the channel scale and tau is meaningless. A caller MUST look at
  // this, for the same reason it must look at cf_flag. See front_end().
  int fe_flag = 0;
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

//-----------------------------------------------------------------------------
// A batched SVD, and the ONLY thing that differs between the PS and PL roles.
//   A [C][m x n] row-major contiguous, U [C][m x n] (may be null: the caller of
//   the analog step does not need the left basis), S [C][n] DESCENDING,
//   V [C][n x n] with columns = right singular vectors. n MAY EXCEED m -- that
//   is precisely the BD front end's shape, A [(K-1)*Nr x NRF] = 28 x 32, and it
//   is why ref/decomp.h's old "m >= n" contract comment was wrong.
// One call is one launch on the device and a loop on the host; batching is what
// makes the offload worth its driver overhead (C=K=8 matrices per launch
// instead of 8 launches).
struct SvdProvider {
  // A is [C][m x n] row-major contiguous. `take` is how many LEADING right
  // singular columns the caller needs: take == n asks for the full basis and V
  // is [C][n x n]; take < n asks for the economy basis and V is [C][n x take].
  // U is [C][m x min(m,n)] and may be null. S is ALWAYS [C][n], descending, so a
  // caller can still do a rank test whichever algorithm served the request.
  //
  // `take` is in the contract because it is what makes the provider shape-aware,
  // and the deployed shapes are extremely asymmetric: the analog step wants the
  // 4 leading columns of a 4x128 factorisation (Nr <= 8, Nt up to 128 over every
  // config in mix_precode/*/config_pools), while the BD step wants ALL 32
  // columns of a 28x32 one -- specifically the trailing ones, which are the null
  // space. Those two want different algorithms: a Gram/eig form is 5256x cheaper
  // for the first and destroys the second (it squares the condition number, and
  // the null space lives at the bottom of the spectrum). Asking for `take` is
  // how one body of glue gets the right one from each role.
  void (*run)(void* ctx, const dcplx* A, int C, int m, int n, int take,
              dcplx* U, double* S, dcplx* V);
  void* ctx = nullptr;
  // Unit roundoff of the arithmetic this provider returns, i.e. the FINEST
  // scale at which its output means anything. front_end() derives the
  // whitening eigenvalue noise floor from it (8*eps, see
  // inv_sqrt_from_eig), which is what lets one body of glue serve an fp64 PS
  // role and an fp32 PL role without the fp32 role dividing by its own noise.
  // It is the precision the provider is CAPABLE of returning, not the precision
  // of any one batch: a provider that routes some batches to the host still
  // declares its coarsest, because the caller cannot know which batch it will
  // route, and the conservative choice only affects eigenvalues that are zero
  // anyway.
  double eps = 2.220446049250313e-16;   // DBL_EPSILON
};
SvdProvider host_svd();       // fp64 svd_cplx, sweeps = ref/decomp.h default

//-----------------------------------------------------------------------------
// The GLUE, and the SECOND thing that differs between the PS and PL roles.
//
// Why this exists (and why the header used to say the opposite): the glue was
// priced at "~0.13 Mflop for the whole instant" against 21.9 Mflop of Jacobi, at
// Nt=16. At the shapes the pools actually contain (mix_precode/*/config_pools,
// 21 distinct configs, Nt up to 128) it is **3.08-3.35 Mflop plus 4096
// atan2+cos+sin**, of which three GEMMs are 93% -- measured breakdown in
// docs/cost_model_real_shapes.md section 3. Projected onto a Cortex-A53 that is
// ~4.6 ms, i.e. ~40% of the hybrid PS budget and the single reason "one
// scheduling instant in 1 ms on the PS" fails. So the glue is no longer too
// small to matter, and it gets the same treatment the decompositions already
// have: one body, a provider, and a cost rule deciding which side runs it.
//
// The contract is kernels/wbb_apply.cl's, verbatim (see ref/frontend.h::gemm_wx
// for the layout and why this form and not C = A B):
//     Y[sc*M + m] = sum_{n<N} W[m*N + n] * X[sc*N + n]      (no conjugation)
// which means the DEVICE role of every glue GEMM is one launch of a kernel that
// already exists, is registered, capacity-guarded and gated -- no new kernel, no
// new __constant bytes (the budget is at 98.4% of the 64 KB spec floor in fp32
// and already over it in fp64, so "add a kernel" is not a free move here).
//-----------------------------------------------------------------------------
// Which glue GEMM a call is, so the device role can be gated SITE BY SITE
// (sched_bench runs each one alone: the three sites feed differently
// rank-sensitive consumers, and "the glue is fp32 now" is not one decision).
enum GemmSite {
  kGemmEff   = 0,   // H_k F_RF    -> the BD interference stack A (rank-critical)
  kGemmFn    = 1,   // F_RF N_k    -> the whitening Gram C (floor-critical)
  kGemmPower = 2,   // F_RF F_BB   -> pscale only (a single scalar consumer)
  kGemmSites = 3
};

struct GemmProvider {
  void (*run)(void* ctx, int site, const dcplx* W, int M, int N,
              const dcplx* X, int SC, dcplx* Y);
  void* ctx = nullptr;
};
GemmProvider host_gemm();     // fp64 gemm_wx; the PS role, and the fallback the
                              // device role uses when a call is not worth a launch

// The intermediates front_end() produces that PipeOut does not expose. Frf, Nk,
// sigma and nul live in PipeOut because the benches and the paper report them.
struct FrontEnd {
  std::vector<dcplx> eff;     // [K][Nr x NRF]   H_k F_RF
  std::vector<dcplx> inv;     // [K][nul x nul]  C_k^{-1/2}, floored
  std::vector<dcplx> Gbar;    // [K][Nr x nul]   power-whitened BD channel
  std::vector<dcplx> Vk;      // [K][nul x d]    gauge-pinned right basis (= V_D
                              //                   of the SVD branch)
  std::vector<dcplx> Uk;      // [K][Nr x d]     gauge-pinned left basis
};

// Steps 1-2. Fills fe and the PipeOut fields it owns (Frf, Nk, sigma, nul).
// Two providers, because two families of work are worth a launch and they are
// not the same family: the decompositions (SvdProvider) and the Nt-class GEMMs
// (GemmProvider). Everything else -- the constant-modulus projection, the gauge
// pin, the nullity selection, the transposes that put an operand in wbb form --
// is branchy or O(Nt*NRF) scalar work that costs more in launch overhead than in
// flops and stays on the PS in BOTH roles.
void front_end(const dcplx* H, const PipeCfg& c, const SvdProvider& svd,
               const GemmProvider& gemm, FrontEnd& fe, PipeOut& o);

// Canonical SVD gauge. (U_j e^{it}, V_j e^{it}) is a free phase, so pin it by
// making the largest entry of Gbar V (= sigma_j U_j) real positive; U_j is
// rotated by the SAME phase to keep Gbar = U diag(sigma) V^H exact. Referencing
// the receive side keeps the rule invariant to the BD-basis gauge, so V_D and
// the physical precoder F_RF N_k V_D,k are deterministic across the fp64 PS path
// and the fp32 PL kernel (and match LAPACK-based references entry by entry
// instead of up to a hidden diagonal phase). Without it the two roles' Vk/Uk are
// incomparable even when both are correct, which is why it is a shared function
// and not something each path does its own way.
//   G [Nr x nul], V [nul x d] and U [Nr x d] are rotated IN PLACE.
void svd_gauge_pin(const dcplx* G, int Nr, int nul, dcplx* V, dcplx* U, int d);

// Step 4: per-user structured design under the selected mode, plus the receive
// chain and the digital power normalization. Fills o.Vd/B/rho_real/rho_tgt/Q/R/
// Fbb/diagb/pscale from fe and o.Frf/o.Nk/o.sigma. `mode` is PipeOut::mode.
// Takes the GEMM provider for the same reason front_end does: pscale needs
// ||F_RF F_BB||_F^2, which at Nt=128 is 1.05 Mflop -- as much as either front-end
// GEMM -- and it is the one glue GEMM in this function.
void design_instant(const PipeCfg& c, int mode, const GemmProvider& gemm,
                    const FrontEnd& fe, PipeOut& o);

// The mode policy, i.e. the ONE definition of what tau means: THP when
// gamma_tot_dB >= tau, SVD otherwise, with flag 4 (shape outside the compiled
// capacity, tau = NaN) forced to the conservative SVD branch and reported
// through *out_flag, and flag 2 (tau = +-inf) still a legitimate decision.
// Shared by both roles so the device path cannot drift into a different policy;
// it prints the refusal itself, because a NaN threshold that silently selects a
// branch is the failure mode this exists to prevent.
int decide_mode(const PipeCfg& c, double tau_dB, int flag, int* out_flag);

// The whole PS-role instant: front_end(host_svd(), host_gemm()) + closed-form
// threshold + design_instant. runtime/hpocl_schedule.cpp::schedule_device() is
// the same three calls with a device SvdProvider, a device GemmProvider and the
// device threshold kernel.
void schedule(const dcplx* H, const PipeCfg& c, PipeOut& o);

}  // namespace hpocl
