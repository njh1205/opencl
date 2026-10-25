#include "pipeline.h"
#include <cstdio>
#include "decomp.h"
#include "rx.h"
#include "threshold_cf.h"
#include <cmath>
#include <cstring>

namespace hpocl {

void schedule(const dcplx* H, const PipeCfg& c, PipeOut& o) {
  const int K = c.K, Nt = c.Nt, Nr = c.Nr, NRF = c.NRF, d = c.d;
  const int T = K * d;
  const double snr = std::pow(10.0, c.gamma_tot_dB / 10.0) / T;

  // 1) shared analog precoder and effective channels
  o.Frf.assign((size_t)Nt * NRF, dcplx(0, 0));
  analog_precoder(H, K, Nr, Nt, NRF, o.Frf.data());
  std::vector<dcplx> eff((size_t)K * Nr * NRF);
  effective_channels(H, K, Nr, Nt, o.Frf.data(), NRF, eff.data());

  // 2) per-user BD basis, whitening, whitened spectrum
  const int rows = (K - 1) * Nr;
  o.nul = NRF - rows;                       // nullity of the interference stack
  o.Vd.assign((size_t)K * o.nul * d, dcplx(0, 0));
  o.B.assign((size_t)K * d * d, dcplx(0, 0));
  o.sigma.assign((size_t)K * d, 0.0);
  o.rho_real.assign((size_t)K * d, 0.0);
  o.rho_tgt.assign((size_t)K * d, 0.0);
  std::vector<dcplx> Gbar((size_t)K * Nr * o.nul);
  std::vector<dcplx> Vk((size_t)K * o.nul * d);
  std::vector<std::vector<dcplx>> invs;           // per-user C_k^{-1/2}
  std::vector<dcplx> Uk((size_t)K * Nr * d);
  o.Nk.assign((size_t)K * NRF * o.nul, dcplx(0, 0));
  o.Q.assign((size_t)K * Nr * d, dcplx(0, 0));
  o.R.assign((size_t)K * d * d, dcplx(0, 0));
  o.Fbb.assign((size_t)K * NRF * d, dcplx(0, 0));
  o.diagb.assign((size_t)K * d, dcplx(0, 0));

  for (int k = 0; k < K; ++k) {
    std::vector<dcplx> A((size_t)rows * NRF);
    int r0 = 0;
    for (int u = 0; u < K; ++u)
      if (u != k) {
        memcpy(A.data() + (size_t)r0 * NRF,          // r0 counts rows
               eff.data() + (size_t)u * Nr * NRF, sizeof(dcplx) * Nr * NRF);
        r0 += Nr;
      }
    dcplx* N = o.Nk.data() + (size_t)k * NRF * o.nul;
    null_basis(A.data(), rows, NRF, N);

    std::vector<dcplx> FN((size_t)Nt * o.nul), inv((size_t)o.nul * o.nul);
    for (int i = 0; i < Nt; ++i)
      for (int j = 0; j < o.nul; ++j) {
        dcplx acc(0, 0);
        for (int t = 0; t < NRF; ++t)
          acc += o.Frf[(size_t)i * NRF + t] * N[(size_t)t * o.nul + j];
        FN[(size_t)i * o.nul + j] = acc;
      }
    power_whitening(FN.data(), Nt, o.nul, 1e-10, 1e-12, inv.data(), nullptr);
    invs.push_back(inv);
    dcplx* G = Gbar.data() + (size_t)k * Nr * o.nul;
    for (int i = 0; i < Nr; ++i)
      for (int j = 0; j < o.nul; ++j) {
        dcplx acc(0, 0);
        for (int t = 0; t < NRF; ++t)
          acc += eff[(size_t)k * Nr * NRF + i * NRF + t] * N[(size_t)t * o.nul + j];
        G[(size_t)i * o.nul + j] = acc;      // reduced = H_k,eff N_k
      }
    std::vector<dcplx> red(G, G + (size_t)Nr * o.nul);   // then Gbar = reduced C^{-1/2}
    for (int i = 0; i < Nr; ++i)
      for (int j = 0; j < o.nul; ++j) {
        dcplx acc(0, 0);
        for (int t = 0; t < o.nul; ++t)
          acc += red[(size_t)i * o.nul + t] * inv[(size_t)t * o.nul + j];
        G[(size_t)i * o.nul + j] = acc;
      }
    dcplx* V = Vk.data() + (size_t)k * o.nul * d;
    double* sg = o.sigma.data() + (size_t)k * d;
    std::vector<double> Sall(o.nul);
    std::vector<dcplx> Vfull((size_t)o.nul * o.nul), Ufull((size_t)Nr * o.nul);
    svd_cplx(G, Nr, o.nul, Ufull.data(), Sall.data(), Vfull.data());  // sweeps: decomp.h default
    for (int i = 0; i < d; ++i) sg[i] = Sall[i];
    for (int r = 0; r < o.nul; ++r)
      for (int j = 0; j < d; ++j) V[(size_t)r * d + j] = Vfull[(size_t)r * o.nul + j];
    dcplx* Um = Uk.data() + (size_t)k * Nr * d;
    for (int r = 0; r < Nr; ++r)
      for (int j = 0; j < d; ++j) Um[(size_t)r * d + j] = Ufull[(size_t)r * o.nul + j];

    // Canonical SVD gauge. (U_j e^{it}, V_j e^{it}) is a free phase, so pin it
    // by making the largest entry of Gbar V (= sigma_j U_j) real positive;
    // U_j is rotated by the same phase to keep Gbar = U diag(sigma) V^H exact.
    // Referencing the receive side keeps the rule invariant to the BD-basis
    // gauge, so V_D and the physical precoder F_RF N_k V_D,k are deterministic
    // across the fp64 PS path and the fp32 PL kernel (and match LAPACK-based
    // references entry by entry instead of up to a hidden diagonal phase).
    for (int j = 0; j < d; ++j) {
      dcplx best(0, 0);
      for (int i = 0; i < Nr; ++i) {
        dcplx a(0, 0);
        for (int t = 0; t < o.nul; ++t) a += G[(size_t)i * o.nul + t] * V[(size_t)t * d + j];
        if (std::abs(a) > std::abs(best)) best = a;
      }
      if (std::abs(best) > 0.0) {
        dcplx ph = std::conj(best / std::abs(best));
        for (int r = 0; r < o.nul; ++r) V[(size_t)r * d + j] *= ph;
        for (int r = 0; r < Nr; ++r) Um[(size_t)r * d + j] *= ph;
      }
    }
  }

  // 3) closed-form switching threshold on the whitened spectrum
  std::vector<double> lam((size_t)K * d);
  for (int i = 0; i < K * d; ++i) lam[i] = o.sigma[i] * o.sigma[i];
  CfConfig cf;
  cf.M = c.M; cf.d = d; cf.K = K;
  cf.variant = c.variant ? ThpVariant::kMMSE_UCD : ThpVariant::kGMD;
  CfResult r = threshold_closed_form(lam.data(), cf);
  o.tau_dB = r.tau_dB;
  o.cf_flag = r.flag;
  if (r.flag == 4) {
    // Outside the compiled capacity (T > MAXT, K > MAXK, 5*ceil(L/2) > MAXD, or M
    // not a perfect square): tau is NaN. `gamma >= NaN` is false, so the plain
    // comparison below would silently select the SVD branch and the caller would
    // see a plausible-looking schedule for a shape the fabric cannot produce.
    // Say so, keep the conservative branch, and record it in o.cf_flag.
    fprintf(stderr, "[pipeline] threshold_cf refused (M=%d K=%d d=%d): outside "
            "the compiled capacity -- forcing the SVD branch, cf_flag=4\n",
            c.M, K, d);
    o.mode = 0;
  } else {
    // flag 0/1 are finite thresholds; flag 2 is +-INFINITY, where the comparison
    // is still the right decision (-inf => always THP, +inf => always SVD).
    o.mode = (c.gamma_tot_dB >= r.tau_dB) ? 1 : 0;
  }

  // 4) per-user structured design under the selected mode
  const char strat = o.mode ? (c.variant ? 'u' : 'g') : 's';
  for (int k = 0; k < K; ++k) {
    dcplx* V = Vk.data() + (size_t)k * o.nul * d;
    double* sg = o.sigma.data() + (size_t)k * d;
    dcplx* Vd = o.Vd.data() + (size_t)k * o.nul * d;
    dcplx* Bk = o.B.data() + (size_t)k * d * d;
    double* rr = o.rho_real.data() + (size_t)k * d;
    double* rt = o.rho_tgt.data() + (size_t)k * d;
    if (strat == 's') {
      memcpy(Vd, V, sizeof(dcplx) * o.nul * d);
      for (int i = 0; i < d; ++i) { rt[i] = sg[i] * sg[i] * snr; }
      for (int i = 0; i < d; ++i) Bk[i * d + i] = dcplx(1, 0);
    } else if (strat == 'g') {
      std::vector<dcplx> R(d * d), P(o.nul * d);
      gmd_factors(V, o.nul, sg, d, R.data(), P.data());
      memcpy(Vd, P.data(), sizeof(dcplx) * o.nul * d);
      memcpy(Bk, R.data(), sizeof(dcplx) * d * d);
      double g = 0;
      for (int i = 0; i < d; ++i) g += log(fmax(sg[i] * sg[i] * snr, 1e-12));
      g = exp(g / d);
      for (int i = 0; i < d; ++i) rt[i] = g;
    } else {
      std::vector<dcplx> R(d * d);
      double req = 0;
      ucd_design(V, o.nul, sg, d, snr, false, 0.0, Vd, &req, R.data(), nullptr);
      memcpy(Bk, R.data(), sizeof(dcplx) * d * d);
      for (int i = 0; i < d; ++i) rt[i] = req;
    }
    stream_snr(Gbar.data() + (size_t)k * Nr * o.nul, Nr, o.nul, Vd, d, snr, rr);

    // receive side of the chain: combiner Q_k, equalized factor R_k and the
    // power-normalised digital block for the datapath (ref/rx.h)
    rx_combiner(Uk.data() + (size_t)k * Nr * d, Nr, V, o.nul, sg, d, snr, strat,
                o.Q.data() + (size_t)k * Nr * d, o.R.data() + (size_t)k * d * d);
    const std::vector<dcplx>& inv = invs[k];
    const dcplx* Nk0 = o.Nk.data() + (size_t)k * NRF * o.nul;
    for (int i = 0; i < NRF; ++i)                 // Fbb is [NRF x T] with the
      for (int j = 0; j < d; ++j) {               // per-user blocks side by side
        dcplx a(0, 0);
        for (int t = 0; t < o.nul; ++t) {
          dcplx b(0, 0);
          for (int q = 0; q < o.nul; ++q)
            b += inv[(size_t)t * o.nul + q] * Vd[(size_t)q * d + j];
          a += Nk0[(size_t)i * o.nul + t] * b;
        }
        o.Fbb[(size_t)i * T + (size_t)k * d + j] = a;
      }
  }
  // digital power normalization (production: p_constraint = total_streams).
  // The same scale multiplies the digital blocks and diag(R) -- the receiver
  // divides by pscale*diag(R) (thp_receive_equalized) -- while the THP
  // feedback b1 = R/diag(R) is scale-invariant.
  o.pscale = power_norm_scale(o.Frf.data(), Nt, NRF, o.Fbb.data(), T, T);
  for (int k = 0; k < K; ++k)
    for (int j = 0; j < d; ++j) {
      o.diagb[(size_t)k * d + j] = o.pscale * o.R[(size_t)k * d * d + (size_t)j * d + j];
      for (int i = 0; i < NRF; ++i)
        o.Fbb[(size_t)i * T + (size_t)k * d + j] *= o.pscale;
    }
}

}  // namespace hpocl
