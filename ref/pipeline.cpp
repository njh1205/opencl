#include "pipeline.h"
#include <cstdio>
#include "decomp.h"
#include "rx.h"
#include "threshold_cf.h"
#include <cmath>
#include <cstring>

namespace hpocl {

namespace {
// The PS role: fp64 svd_cplx, one matrix at a time. `U == nullptr` is honoured
// (svd_cplx only guards the write, so S and V are bit-identical either way) --
// the analog and null-basis steps do not need a left basis and allocating one
// would be (K*Nr*Nt + K*rows*NRF) of dead writes per instant.
void host_svd_run(void* /*ctx*/, const dcplx* A, int C, int m, int n, int take,
                  dcplx* U, double* S, dcplx* V) {
  // take < n on a wide matrix is the analog-precoder shape, and it is served by
  // the Gram form: measured 5256x faster than the full one-sided Jacobi at
  // 4x128 with 5.4e-15 agreement on the leading subspace (ref/decomp.h). It is
  // NOT used when take == n, because there the caller wants the trailing columns
  // too and a Gram matrix squares the condition number exactly where the null
  // space lives.
  if (take < n && m < n) {
    for (int c = 0; c < C; ++c)
      svd_wide(A + (size_t)c * m * n, m, n, take,
               U ? U + (size_t)c * m * m : nullptr, S + (size_t)c * n,
               V + (size_t)c * n * take);   // sweeps: decomp.h default
    return;
  }
  for (int c = 0; c < C; ++c)
    svd_cplx(A + (size_t)c * m * n, m, n,
             U ? U + (size_t)c * m * n : nullptr,
             S + (size_t)c * n, V + (size_t)c * n * n);  // sweeps: decomp.h default
}
}  // namespace

SvdProvider host_svd() { return SvdProvider{host_svd_run, nullptr}; }

// The PS role of the glue: ref/frontend.h's gemm_wx, which is the same
// expression the three call sites used to spell out inline, with the same
// ascending inner order -- so routing them through the provider moves no bits on
// the host (sched_bench's host role is the reference for that claim). The
// provider's whole job is to let the device role substitute ONE wbb_apply launch
// per site without a second copy of the glue existing to drift.
namespace {
void host_gemm_run(void* /*ctx*/, int /*site*/, const dcplx* W, int M, int N,
                   const dcplx* X, int SC, dcplx* Y) {
  gemm_wx(W, M, N, X, SC, Y);
}
}  // namespace

GemmProvider host_gemm() { return GemmProvider{host_gemm_run, nullptr}; }

void svd_gauge_pin(const dcplx* G, int Nr, int nul, dcplx* V, dcplx* U, int d) {
  for (int j = 0; j < d; ++j) {
    dcplx best(0, 0);
    for (int i = 0; i < Nr; ++i) {
      dcplx a(0, 0);
      for (int t = 0; t < nul; ++t) a += G[(size_t)i * nul + t] * V[(size_t)t * d + j];
      if (std::abs(a) > std::abs(best)) best = a;
    }
    if (std::abs(best) > 0.0) {
      dcplx ph = std::conj(best / std::abs(best));
      for (int r = 0; r < nul; ++r) V[(size_t)r * d + j] *= ph;
      for (int r = 0; r < Nr; ++r) U[(size_t)r * d + j] *= ph;
    }
  }
}

void front_end(const dcplx* H, const PipeCfg& c, const SvdProvider& svd,
               const GemmProvider& gemm, FrontEnd& fe, PipeOut& o) {
  const int K = c.K, Nt = c.Nt, Nr = c.Nr, NRF = c.NRF, d = c.d;

  // 1) shared analog precoder: ONE batched SVD of the per-user channels, then
  // the (SVD-free) column selection / DFT filler / constant-modulus projection.
  // Only the users whose columns can reach Frf are factorised (analog_users);
  // at K=8, NRF=8, Nr=4 that is 2 of 8, and the rest is DFT filler.
  const int nu = analog_users(K, Nr, Nt, NRF);
  const int take = std::min(NRF, std::min(Nr, Nt));   // == analog_precoder_from_V's
  std::vector<dcplx> Va((size_t)nu * Nt * take);      // ECONOMY: Nt x take, not Nt x Nt
  std::vector<double> Sa((size_t)nu * Nt);
  svd.run(svd.ctx, H, nu, Nr, Nt, take, nullptr, Sa.data(), Va.data());
  o.Frf.assign((size_t)Nt * NRF, dcplx(0, 0));
  analog_precoder_from_V(Va.data(), nu, Nr, Nt, NRF, o.Frf.data());

  // Effective channels H_k F_RF: a GEMM, not a decomposition. All K users go in
  // ONE call (W = F_RF^T, X = H read as [K*Nr x Nt], Y = eff), which is exactly
  // effective_channels() -- that function is now this call with the host
  // provider, so there is one definition -- and at Nt=128 it is 1.05 Mflop, 34%
  // of the glue. Cost rule and per-site gating: hpocl_schedule.cpp.
  std::vector<dcplx> FrfT((size_t)NRF * Nt);
  for (int t = 0; t < Nt; ++t)
    for (int j = 0; j < NRF; ++j)
      FrfT[(size_t)j * Nt + t] = o.Frf[(size_t)t * NRF + j];
  fe.eff.assign((size_t)K * Nr * NRF, dcplx(0, 0));
  gemm.run(gemm.ctx, kGemmEff, FrfT.data(), NRF, Nt, H, K * Nr, fe.eff.data());

  // 2) per-user BD basis. The interference stack for user k is every OTHER
  // user's effective channel, so all K stacks have the same shape
  // [(K-1)*Nr x NRF] and are factorised in ONE batched call -- that is the
  // single most expensive thing in the control plane (91% of the front end at
  // K=8, Nt=16, Nr=4, NRF=32) and the reason this file takes a provider.
  const int rows = (K - 1) * Nr;
  o.nul = NRF - rows;                       // nullity of the interference stack
  // RF-BD feasibility, and it is a structural fact about the geometry rather
  // than a numerical test. The interference stack is A = blkdiag(H_u) * (1 x
  // F_RF), so EVERY row of A is a combination of F_RF's rows: rank(A) <=
  // rank(F_RF) <= min(Nt, NRF), and null(A) = null(F_RF) whenever the two ranks
  // are equal. If rows >= min(Nt, NRF) they are equal, so F_RF N_k == 0
  // identically and the whitened channel is zero -- BD has no direction that
  // both nulls the interference and survives the analog stage. This is the
  // textbook BD condition (Nt > (K-1)*Nr) reaching through the RF chain, and it
  // is not a property of this port: measured against production python at
  // K=8, Nt=16, Nr=4, NRF=32 (rows = 28 > 16),
  //   rank(F_RF) = 16, rank(A) = 16 (NOT 28), production's own null space is
  //   (32 x 16) while nul = NRF - rows = 4, ||F_RF N||_F = 2.6e-15 against
  //   ||F_RF||_F = 5.66 (rel 4.5e-16), eig(C) = 1.4e-33 .. 1.7e-30, and
  //   sigma = 3.6e-9 -- production degenerates the same way.
  // At Nt = 32 the same K/Nr/NRF is feasible and healthy: rank(A) = 28 = rows,
  // ||F_RF N||_F = 0.386, eig(C) = 8.9e-3 .. 8.1e-2, sigma = 2.36 .. 0.26.
  // NOTE the geometry the README's cost model calls "deployed" (K=8, Nt=16,
  // Nr=4, NRF=32) is on the wrong side of this test; its front-end flop counts
  // are still real (the SVDs are still performed) but its schedule is not.
  o.fe_flag = (rows >= std::min(Nt, NRF)) ? 1 : 0;
  if (o.fe_flag)
    fprintf(stderr, "[pipeline] RF-BD DEGENERATE at K=%d Nt=%d Nr=%d NRF=%d: "
            "(K-1)*Nr = %d >= min(Nt,NRF) = %d, so rank(A) = %d < rows and "
            "F_RF*N_k == 0 -- nul = %d is not the nullity and the whitened "
            "spectrum is numerical noise (fe_flag=1). Need Nt > (K-1)*Nr.\n",
            K, Nt, Nr, NRF, rows, std::min(Nt, NRF), std::min(Nt, NRF), o.nul);
  const int nul = o.nul;
  std::vector<dcplx> A((size_t)K * rows * NRF);
  for (int k = 0; k < K; ++k) {
    int r0 = 0;
    for (int u = 0; u < K; ++u)
      if (u != k) {
        memcpy(A.data() + ((size_t)k * rows + r0) * NRF,
               fe.eff.data() + (size_t)u * Nr * NRF, sizeof(dcplx) * Nr * NRF);
        r0 += Nr;                           // r0 counts rows within the stack
      }
  }
  std::vector<dcplx> Vn((size_t)K * NRF * NRF);
  std::vector<double> Sn((size_t)K * NRF);
  // take == NRF: the BD basis IS the trailing columns, so the full right basis
  // is required and the provider must use the accuracy-preserving Jacobi.
  svd.run(svd.ctx, A.data(), K, rows, NRF, NRF, nullptr, Sn.data(), Vn.data());
  o.Nk.assign((size_t)K * NRF * nul, dcplx(0, 0));
  for (int k = 0; k < K; ++k)
    // `nul` comes from the GEOMETRY, not from a rank test on Sn: on fp32
    // singular values the fp64 rcond max(rows,NRF)*eps64*S0 = 7.1e-15 is four
    // orders BELOW the fp32 rank margin (Smin/Smax = 5.2e-10 at 6 sweeps,
    // measured), so a device-side rank test would silently return nul = 0 and
    // the user would get no streams at all. See ref/frontend.h and the tol_exit
    // note in kernels/decomp.cl. The host path's own null_basis() keeps the
    // rcond test because its SVD is fp64; both select the same trailing columns.
    null_basis_trailing(Vn.data() + (size_t)k * NRF * NRF, NRF, nul,
                        o.Nk.data() + (size_t)k * NRF * nul);

  // 3) power whitening and the whitened per-user channel. Both SVDs here are
  // nul x nul = 4 x 4 at the deployed geometry, i.e. ~17 kflop for all K of
  // them: the provider decides whether that is worth a launch, and the costed
  // answer is no (hpocl_schedule.cpp). The glue is identical either way.
  fe.inv.assign((size_t)K * nul * nul, dcplx(0, 0));
  fe.Gbar.assign((size_t)K * Nr * nul, dcplx(0, 0));
  fe.Vk.assign((size_t)K * nul * d, dcplx(0, 0));
  fe.Uk.assign((size_t)K * Nr * d, dcplx(0, 0));
  o.sigma.assign((size_t)K * d, 0.0);

  // F_RF N_k for ALL K users as ONE call: W = stack_k(N_k^T) [K*nul x NRF] (a
  // K*nul*NRF copy, 1024 elements at the largest pool shape), X = F_RF, so
  // Y[i][k*nul + j] = (F_RF N_k)[i][j] lands in a [Nt x K*nul] buffer and the
  // per-user blocks are copied out contiguous for gram_hermitian (K*Nt*nul =
  // 4096 copies, against the 1.05 Mflop this replaces). t ascending either way,
  // and complex multiplication is commutative in fp, so the host role is
  // bit-identical to the loop this replaced.
  // NOT rearranged as N_k^H (F_RF^H F_RF) N_k, which would need only a
  // [NRF x NRF] product: that squares F_RF's condition number before the
  // whitening eigendecomposition reads it, and C's structurally-zero eigenvalues
  // already sit at 2.5e-9..3.7e-8 x lam_max (see inv_sqrt_from_eig).
  std::vector<dcplx> NT((size_t)K * nul * NRF);
  for (int k = 0; k < K; ++k)
    for (int t = 0; t < NRF; ++t)
      for (int j = 0; j < nul; ++j)
        NT[((size_t)k * nul + j) * NRF + t] = o.Nk[((size_t)k * NRF + t) * nul + j];
  std::vector<dcplx> FN((size_t)K * Nt * nul), Yfn((size_t)Nt * K * nul),
                     Cw((size_t)K * nul * nul);
  gemm.run(gemm.ctx, kGemmFn, NT.data(), K * nul, NRF, o.Frf.data(), Nt,
           Yfn.data());
  for (int k = 0; k < K; ++k) {
    dcplx* FNk = FN.data() + (size_t)k * Nt * nul;
    for (int i = 0; i < Nt; ++i)
      for (int j = 0; j < nul; ++j)
        FNk[(size_t)i * nul + j] = Yfn[(size_t)i * K * nul + k * nul + j];
    gram_hermitian(FNk, Nt, nul, Cw.data() + (size_t)k * nul * nul);
  }
  std::vector<dcplx> Vw((size_t)K * nul * nul);
  std::vector<double> Sw((size_t)K * nul);
  svd.run(svd.ctx, Cw.data(), K, nul, nul, nul, nullptr, Sw.data(), Vw.data());
  for (int k = 0; k < K; ++k)
    // 1e-10 / 1e-12 are production's floors (build_physical_power_whitening);
    // 8*eps is the provider's noise floor, which is what makes the fp32 role
    // floor its structurally-zero eigenvalues at production's value instead of
    // dividing by fp32 noise (measured: see inv_sqrt_from_eig).
    inv_sqrt_from_eig(Vw.data() + (size_t)k * nul * nul, Sw.data() + (size_t)k * nul,
                      nul, 1e-10, 1e-12, 8.0 * svd.eps,
                      fe.inv.data() + (size_t)k * nul * nul, nullptr);

  for (int k = 0; k < K; ++k) {
    const dcplx* N = o.Nk.data() + (size_t)k * NRF * nul;
    const dcplx* inv = fe.inv.data() + (size_t)k * nul * nul;
    dcplx* G = fe.Gbar.data() + (size_t)k * Nr * nul;
    for (int i = 0; i < Nr; ++i)
      for (int j = 0; j < nul; ++j) {
        dcplx acc(0, 0);
        for (int t = 0; t < NRF; ++t)
          acc += fe.eff[(size_t)k * Nr * NRF + i * NRF + t] * N[(size_t)t * nul + j];
        G[(size_t)i * nul + j] = acc;        // reduced = H_k,eff N_k
      }
    std::vector<dcplx> red(G, G + (size_t)Nr * nul);   // then Gbar = reduced C^{-1/2}
    for (int i = 0; i < Nr; ++i)
      for (int j = 0; j < nul; ++j) {
        dcplx acc(0, 0);
        for (int t = 0; t < nul; ++t)
          acc += red[(size_t)i * nul + t] * inv[(size_t)t * nul + j];
        G[(size_t)i * nul + j] = acc;
      }
  }
  // The whitened channel's reduced SVD: [Nr x nul] = 4 x 4 at the deployed
  // geometry, again batched, again the provider's call whether to launch.
  std::vector<dcplx> Ufull((size_t)K * Nr * nul), Vfull((size_t)K * nul * nul);
  std::vector<double> Sall((size_t)K * nul);
  svd.run(svd.ctx, fe.Gbar.data(), K, Nr, nul, nul, Ufull.data(), Sall.data(),
          Vfull.data());
  for (int k = 0; k < K; ++k) {
    dcplx* V = fe.Vk.data() + (size_t)k * nul * d;
    dcplx* Um = fe.Uk.data() + (size_t)k * Nr * d;
    double* sg = o.sigma.data() + (size_t)k * d;
    for (int i = 0; i < d; ++i) sg[i] = Sall[(size_t)k * nul + i];
    for (int r = 0; r < nul; ++r)
      for (int j = 0; j < d; ++j)
        V[(size_t)r * d + j] = Vfull[((size_t)k * nul + r) * nul + j];
    for (int r = 0; r < Nr; ++r)
      for (int j = 0; j < d; ++j)
        Um[(size_t)r * d + j] = Ufull[((size_t)k * Nr + r) * nul + j];
    svd_gauge_pin(fe.Gbar.data() + (size_t)k * Nr * nul, Nr, nul, V, Um, d);
  }
}

void design_instant(const PipeCfg& c, int mode, const GemmProvider& gemm,
                    const FrontEnd& fe, PipeOut& o) {
  const int K = c.K, NRF = c.NRF, d = c.d, nul = o.nul;
  const int T = K * d;
  const double snr = std::pow(10.0, c.gamma_tot_dB / 10.0) / T;

  o.Vd.assign((size_t)K * nul * d, dcplx(0, 0));
  o.B.assign((size_t)K * d * d, dcplx(0, 0));
  o.rho_real.assign((size_t)K * d, 0.0);
  o.rho_tgt.assign((size_t)K * d, 0.0);
  o.Q.assign((size_t)K * c.Nr * d, dcplx(0, 0));
  o.R.assign((size_t)K * d * d, dcplx(0, 0));
  o.Fbb.assign((size_t)K * NRF * d, dcplx(0, 0));
  o.diagb.assign((size_t)K * d, dcplx(0, 0));

  const char strat = mode ? (c.variant ? 'u' : 'g') : 's';
  for (int k = 0; k < K; ++k) {
    const dcplx* V = fe.Vk.data() + (size_t)k * nul * d;
    const double* sg = o.sigma.data() + (size_t)k * d;
    dcplx* Vd = o.Vd.data() + (size_t)k * nul * d;
    dcplx* Bk = o.B.data() + (size_t)k * d * d;
    double* rr = o.rho_real.data() + (size_t)k * d;
    double* rt = o.rho_tgt.data() + (size_t)k * d;
    if (strat == 's') {
      memcpy(Vd, V, sizeof(dcplx) * nul * d);
      for (int i = 0; i < d; ++i) { rt[i] = sg[i] * sg[i] * snr; }
      for (int i = 0; i < d; ++i) Bk[(size_t)i * d + i] = dcplx(1, 0);
    } else if (strat == 'g') {
      std::vector<dcplx> R((size_t)d * d), P((size_t)nul * d);
      gmd_factors(V, nul, sg, d, R.data(), P.data());
      memcpy(Vd, P.data(), sizeof(dcplx) * nul * d);
      memcpy(Bk, R.data(), sizeof(dcplx) * d * d);
      double g = 0;
      for (int i = 0; i < d; ++i) g += log(fmax(sg[i] * sg[i] * snr, 1e-12));
      g = exp(g / d);
      for (int i = 0; i < d; ++i) rt[i] = g;
    } else {
      std::vector<dcplx> R((size_t)d * d);
      double req = 0;
      ucd_design(V, nul, sg, d, snr, false, 0.0, Vd, &req, R.data(), nullptr);
      memcpy(Bk, R.data(), sizeof(dcplx) * d * d);
      for (int i = 0; i < d; ++i) rt[i] = req;
    }
    stream_snr(fe.Gbar.data() + (size_t)k * c.Nr * nul, c.Nr, nul, Vd, d, snr, rr);

    // receive side of the chain: combiner Q_k, equalized factor R_k and the
    // power-normalised digital block for the datapath (ref/rx.h)
    rx_combiner(fe.Uk.data() + (size_t)k * c.Nr * d, c.Nr, V, nul, sg, d, snr,
                strat, o.Q.data() + (size_t)k * c.Nr * d,
                o.R.data() + (size_t)k * d * d);
    const dcplx* inv = fe.inv.data() + (size_t)k * nul * nul;
    const dcplx* Nk0 = o.Nk.data() + (size_t)k * NRF * nul;
    for (int i = 0; i < NRF; ++i)                 // Fbb is [NRF x T] with the
      for (int j = 0; j < d; ++j) {               // per-user blocks side by side
        dcplx a(0, 0);
        for (int t = 0; t < nul; ++t) {
          dcplx b(0, 0);
          for (int q = 0; q < nul; ++q)
            b += inv[(size_t)t * nul + q] * Vd[(size_t)q * d + j];
          a += Nk0[(size_t)i * nul + t] * b;
        }
        o.Fbb[(size_t)i * T + (size_t)k * d + j] = a;
      }
  }
  // digital power normalization (production: p_constraint = total_streams).
  // The same scale multiplies the digital blocks and diag(R) -- the receiver
  // divides by pscale*diag(R) (thp_receive_equalized) -- while the THP
  // feedback b1 = R/diag(R) is scale-invariant.
  // pscale = sqrt(P / ||F_RF F_BB||_F^2), production's p_constraint = T. The
  // product is gemm_wx with W = F_BB^T [T x NRF] and X = F_RF, i.e. 1.05 Mflop at
  // Nt=128 and the only glue GEMM in this function; the norm, the 1e-12 floor and
  // the sqrt stay in ref/rx.cpp so the RULE has one definition
  // (power_norm_from_product, which power_norm_scale also calls).
  std::vector<dcplx> FbbT((size_t)T * NRF), P((size_t)c.Nt * T);
  for (int r = 0; r < NRF; ++r)
    for (int t = 0; t < T; ++t)
      FbbT[(size_t)t * NRF + r] = o.Fbb[(size_t)r * T + t];
  gemm.run(gemm.ctx, kGemmPower, FbbT.data(), T, NRF, o.Frf.data(), c.Nt,
           P.data());
  o.pscale = power_norm_from_product(P.data(), c.Nt, T, T);
  for (int k = 0; k < K; ++k)
    for (int j = 0; j < d; ++j) {
      o.diagb[(size_t)k * d + j] =
          o.pscale * o.R[(size_t)k * d * d + (size_t)j * d + j];
      for (int i = 0; i < NRF; ++i)
        o.Fbb[(size_t)i * T + (size_t)k * d + j] *= o.pscale;
    }
}

// The mode policy, shared by both roles: one definition of "what does tau mean".
int decide_mode(const PipeCfg& c, double tau_dB, int flag, int* out_flag) {
  *out_flag = flag;
  if (flag == 4) {
    // Outside the compiled capacity (T > MAXT, K > MAXK, 5*ceil(L/2) > MAXD, or
    // M not a perfect square): tau is NaN. `gamma >= NaN` is false, so the plain
    // comparison below would silently select the SVD branch and the caller would
    // see a plausible-looking schedule for a shape the fabric cannot produce.
    // Say so, keep the conservative branch, and record it in o.cf_flag.
    fprintf(stderr, "[pipeline] threshold_cf refused (M=%d K=%d d=%d): outside "
            "the compiled capacity -- forcing the SVD branch, cf_flag=4\n",
            c.M, c.K, c.d);
    return 0;
  }
  // flag 0/1 are finite thresholds; flag 2 is +-INFINITY, where the comparison
  // is still the right decision (-inf => always THP, +inf => always SVD).
  return (c.gamma_tot_dB >= tau_dB) ? 1 : 0;
}

void schedule(const dcplx* H, const PipeCfg& c, PipeOut& o) {
  FrontEnd fe;
  front_end(H, c, host_svd(), host_gemm(), fe, o);

  // closed-form switching threshold on the whitened spectrum
  std::vector<double> lam((size_t)c.K * c.d);
  for (size_t i = 0; i < lam.size(); ++i) lam[i] = o.sigma[i] * o.sigma[i];
  CfConfig cf;
  cf.M = c.M; cf.d = c.d; cf.K = c.K;
  cf.variant = c.variant ? ThpVariant::kMMSE_UCD : ThpVariant::kGMD;
  CfResult r = threshold_closed_form(lam.data(), cf);
  o.tau_dB = r.tau_dB;
  o.mode = decide_mode(c, r.tau_dB, r.flag, &o.cf_flag);

  design_instant(c, o.mode, host_gemm(), fe, o);
}

}  // namespace hpocl
