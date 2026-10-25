// cf_bench: three-way validation of the closed-form switching threshold:
//   numpy reference (tools/cf_ref.py)  vs  C++ PS path  vs  OpenCL kernel.
//
// Columns, and what each one is evidence FOR:
//   quad  the LIVE GH-48 quadrature, C++ vs oracle, at three spot gammas. Kept
//         on deficit_*_quad (never the deployed evaluator) so the column still
//         tests axis_deficit after the ln-D table landed -- the quadrature is
//         the fallback for an untabulated M and for t < DFT_TLO, and this is the
//         gate that proves it still works.
//   tab   the DEPLOYED evaluator (table / tail / quadrature, whichever branch
//         applies), C++ vs oracle, at 27 probes straddling both grid edges.
//   l4    Lemma 4 straight off the C++ values: D_mod/D_fin must equal L/(L-1)
//         for every probe at t >= 5. Needs no oracle, so it catches a wrong,
//         transposed or stale table even if C++ and the oracle agreed on it.
//   hnd   the step the deployed function takes at its two handovers, measured
//         here rather than quoted from a comment: table -> analytic tail at
//         DFT_THI and quadrature -> table at DFT_TLO. This is the discontinuity
//         that used to be 0.3046 dB at TQ = 5 and made Psi a staircase.
//   cf    the closed-form seed u*_0 (no deficit is involved, so the table cannot
//         move it); tau the deployed 20-step Newton root vs the 30-step oracle;
//   cl    the kernel against the fp64 host; dep 20 Newton steps vs 8.
#include <cstdio>
#include <cstring>
#include <cmath>
#include <vector>
#include <string>
#include <fstream>
#include "hpocl_runtime.h"
#include "threshold_cf.h"

using namespace hpocl;

// relative conditioning of the closed-form denominator (g_min/eta - lam_min)
// over g_min/eta. Below ~0.05 the spectrum is near-flat: Psi then stays at
// round-off level over a wide u range, so the crossing point is not
// identifiable (both branches are within a fraction of a dB).
static double flat_ratio(const double* lam, int K, int d, int M) {
  double lmin = lam[0], gmin = 1e300;
  for (int i = 0; i < K * d; ++i) lmin = fmin(lmin, lam[i]);
  for (int k = 0; k < K; ++k) {
    double lg = 0;
    for (int i = 0; i < d; ++i) lg += log(fmax(lam[k * d + i], 1e-12));
    gmin = fmin(gmin, exp(lg / d));
  }
  const double eta = 1.0 + (double)(d - 1) / ((double)d * (M - 1.0));
  const double g = gmin / eta;
  return g > 0.0 ? (g - lmin) / g : 0.0;
}

// tau saturates to +-inf when Psi keeps one sign (uniform branch ordering)
static double err_tau(double a, double b) {
  if (std::isinf(a) || std::isinf(b)) return a == b ? 0.0 : HUGE_VAL;
  return fabs(a - b);
}

struct Bin { const uint8_t* p; size_t n; size_t o = 0;
  template <typename T> T rd() { T v; memcpy(&v, p + o, sizeof(T)); o += sizeof(T); return v; }
  const void* ptr(size_t bytes) { const void* q = p + o; o += bytes; return q; }
};

int main(int argc, char** argv) {
  const char* path = argc > 1 ? argv[1] : "build/cf_vectors.bin";
  const char* pref = argc > 2 ? argv[2] : "cpu";
  std::ifstream f(path, std::ios::binary);
  if (!f) { fprintf(stderr, "missing %s (run: python3 tools/cf_ref.py)\n", path); return 2; }
  std::vector<uint8_t> buf((std::istreambuf_iterator<char>(f)), {});
  Bin b{buf.data(), buf.size()};

  Runtime rt;
  if (!rt.init(pref)) { fprintf(stderr, "opencl init failed\n"); return 2; }
  // HPOCL_DEFICIT_TABLE has to reach the DEVICE too, or the "one macro restores
  // the quadrature" promise is false in the most confusing way possible: the
  // kernel defaults the switch to 1, so a host built with =0 would solve the
  // quadrature Psi while the device solves the table Psi. The two evaluators
  // differ by up to 0.096 dB of tau (README's decision record), which shows up
  // as maxcl = 8.8e-2 / 9.6e-2 dB -- measured, on this device, and large enough
  // to fail the 5e-2 bound while every host-side column looks perfect.
  std::string kopts = rt.device_fp64() ? "" : "-DHPOCL_FP32";
#if !HPOCL_DEFICIT_TABLE
  kopts += " -DHPOCL_DEFICIT_TABLE=0";
#endif
  if (!rt.load_program({"kernels/threshold_cf.cl"}, kopts.c_str())) return 2;
  double cl_tol = rt.device_fp64() ? 1e-6 : 5e-2;
  printf("[cf_bench] device: %s (fp64=%d, kernel opts '%s')\n", rt.device_name(),
         (int)rt.device_fp64(), kopts.c_str());

  // 'DFT1': tools/cf_ref.py stamps the format and the evaluator it was run with,
  // so a vectors file generated under the other HPOCL_DEFICIT_TABLE setting is
  // refused here instead of showing up as an unexplained accuracy failure.
  const int magic = b.rd<int>();
  if (magic != 0x32544644) {
    fprintf(stderr, "[cf_bench] %s: bad magic 0x%08x (expected 0x32544644 'DFT2'; "
            "0x31544644 'DFT1' is the older format whose ln-D probe block carried "
            "t instead of gamma) -- regenerate: python3 tools/cf_ref.py\n",
            path, magic);
    return 2;
  }
  int ncfg = b.rd<int>();
  const int oracle_table = b.rd<int>();
#if HPOCL_DEFICIT_TABLE
  const int host_table = 1;
#else
  const int host_table = 0;
#endif
  if (oracle_table != host_table) {
    fprintf(stderr, "[cf_bench] %s was generated with HPOCL_DEFICIT_TABLE=%d but "
            "this binary was compiled with %d -- the two would solve different "
            "deficits. Regenerate the vectors or rebuild with the same setting.\n",
            path, oracle_table, host_table);
    return 2;
  }
  double dft_tlo = 0.0, dft_thi = 0.0;
  int dft_n = 0, dft_nm = 0;
  deficit_table_grid(&dft_tlo, &dft_thi, &dft_n, &dft_nm);
  if (host_table)
    printf("[cf_bench] deficit evaluator: ln-D table -- %d constellations x 2 "
           "branches x %d points, t in [%g,%g], analytic tail above, GH-48 below\n",
           dft_nm, dft_n, dft_tlo, dft_thi);
  else
    printf("[cf_bench] deficit evaluator: GH-48 quadrature to TQ, analytic tail "
           "above (HPOCL_DEFICIT_TABLE=0)\n");
  int fails = 0;
  int only = argc > 3 ? atoi(argv[3]) : -1;
  bool verb = getenv("CF_VERBOSE") != nullptr;
  for (int ci = 0; ci < ncfg; ++ci) {
    if (only >= 0 && ci != only) {  // consume blobs without running
      int Mq = b.rd<int>(), dq = b.rd<int>(), Kq = b.rd<int>();
      b.rd<int>(); int Cq = b.rd<int>();
      b.ptr(sizeof(double) * Cq * Kq * dq + 2 * sizeof(double) * Cq +
            sizeof(int) * Cq + 3 * sizeof(double) + 2 * 3 * sizeof(double));
      const int ntq = b.rd<int>();                 // ln-D probe block: gamma,
      b.ptr(3 * sizeof(double) * ntq);             // ln D_fin, ln D_mod
      (void)Mq; continue;
    }
    int M = b.rd<int>(), d = b.rd<int>(), K = b.rd<int>(), var = b.rd<int>(), C = b.rd<int>();
    std::vector<double> lam((size_t)C * K * d);
    memcpy(lam.data(), b.ptr(sizeof(double) * lam.size()), sizeof(double) * lam.size());
    std::vector<double> tcf(C), tau_ref(C);
    memcpy(tcf.data(), b.ptr(sizeof(double) * C), sizeof(double) * C);
    std::vector<int> pyflag(C);      // 0 closed form, 1 scan, 2 saturated, 4 capacity
    memcpy(pyflag.data(), b.ptr(sizeof(int) * C), sizeof(int) * C);
    memcpy(tau_ref.data(), b.ptr(sizeof(double) * C), sizeof(double) * C);
    double gs[3], dfin[3], dmod[3];
    memcpy(gs, b.ptr(sizeof(gs)), sizeof(gs));
    memcpy(dfin, b.ptr(sizeof(dfin)), sizeof(dfin));
    memcpy(dmod, b.ptr(sizeof(dmod)), sizeof(dmod));
    const int nt = b.rd<int>();
    // gp[] is the GAMMA at which the oracle evaluated ln D, not the t it was
    // derived from, so the two sides feed ln_deficit_* the identical bit
    // pattern. Reconstructing g here as 2*t*t*(M-1)/6 instead looked harmless
    // and was not: tools/cf_ref.py built its g as 2*(t/delta)^2, which differs
    // by 1 ulp, and a TP probe sits exactly on DFT_THI -- at M = 256 / t = 32.0
    // the oracle recovered t = 32.0 (table branch) while this file recovered
    // 32.00000000000001 (tail branch), so maxtab measured the handover step,
    // 1.20e-3 nats, against a 1e-9 gate.
    std::vector<double> gp(nt), tplnfin(nt), tplnmod(nt);
    memcpy(gp.data(), b.ptr(sizeof(double) * nt), sizeof(double) * nt);
    memcpy(tplnfin.data(), b.ptr(sizeof(double) * nt), sizeof(double) * nt);
    memcpy(tplnmod.data(), b.ptr(sizeof(double) * nt), sizeof(double) * nt);

    // 1) quadrature cross-check C++ vs numpy, on deficit_*_quad: the live GH-48
    //    rule with no table and no tail branch, so this column keeps testing
    //    axis_deficit itself. The oracle emits the same two functions.
    double maxq = 0;
    for (int i = 0; i < 3; ++i) {
      maxq = fmax(maxq, fabs(deficit_fin_quad(gs[i], M) - dfin[i]) / fmax(1e-300, dfin[i]));
      maxq = fmax(maxq, fabs(deficit_mod_quad(gs[i], M) - dmod[i]) / fmax(1e-300, dmod[i]));
    }

    // 1b) the DEPLOYED evaluator, C++ vs oracle, at probes that straddle both
    //     grid edges so a branch-selection off-by-one cannot hide; and the
    //     Lemma-4 ratio computed from the C++ values alone (no oracle involved),
    //     which is what catches a wrong or transposed table. g inverts t_of:
    //     t = sqrt(6/(M-1))*sqrt(g/2)  =>  g = 2 t^2 (M-1)/6.
    double maxtab = 0.0, maxl4 = 0.0;
    const int Lp = (int)std::lround(std::sqrt((double)M));
    const double want_ratio = (double)Lp / (double)(Lp - 1);
    for (int i = 0; i < nt; ++i) {
      const double g = gp[i];
      const double lf = ln_deficit_fin(g, M), lm = ln_deficit_mod(g, M);
      maxtab = fmax(maxtab, fabs(lf - tplnfin[i]));
      maxtab = fmax(maxtab, fabs(lm - tplnmod[i]));
      // t only selects which probes the Lemma-4 ratio applies to, so a 1-ulp
      // difference here can at most include or exclude a probe at t = 5, where
      // the ratio already holds to ~1e-13. Same spelling as threshold_cf.cpp
      // t_of(): sqrt(6/(M-1)) * sqrt(gamma*0.5).
      const double t = std::sqrt(6.0 / (M - 1.0)) * std::sqrt(g * 0.5);
      if (t >= 5.0)   // the ratio is exact only above t ~ 2.3..3.7 (see
        maxl4 = fmax(maxl4, fabs(std::exp(lm - lf) - want_ratio) / want_ratio);
    }

    // 1c) the two handover steps of the deployed function, measured here rather
    //     than quoted from a comment. The step is the difference between the two
    //     BRANCHES at one and the same t, read off as
    //     |ln D(edge*(1-eps)) - ln D(edge*(1+eps))| with eps small enough that
    //     the smooth part of ln D across the interval (|d ln D/d ln t| * 2*eps
    //     = t^2 * 2e-9, i.e. 2.0e-6 nats at t = 32) stays 500x below the step
    //     being measured. At eps = 1e-6 drift and step would be the same size
    //     and the number would mean nothing.
    double hnd = 0.0;
    for (int e = 0; e < 2 && dft_thi > 0.0; ++e) {
      const double edge = e == 0 ? dft_tlo : dft_thi;
      const double eps = 1e-9, k = 2.0 * edge * edge * (M - 1) / 6.0;
      const double ga = k * (1.0 - eps) * (1.0 - eps);
      const double gb = k * (1.0 + eps) * (1.0 + eps);
      hnd = fmax(hnd, 4.342944819 * fabs(ln_deficit_fin(ga, M) - ln_deficit_fin(gb, M)));
      hnd = fmax(hnd, 4.342944819 * fabs(ln_deficit_mod(ga, M) - ln_deficit_mod(gb, M)));
    }
    // 2) C++ threshold vs numpy. One flag encoding across all four sites
    //    (cf_ref.py tau_policy, CfResult::flag, kernels/threshold_cf.cl
    //    flags[c], and this file): 0 = closed form, 1 = scan fallback,
    //    2 = saturated +-inf, 4 = outside the compiled capacity. Near-flat
    //    channels are counted, not compared.
    CfConfig cfg; cfg.M = M; cfg.d = d; cfg.K = K;
    cfg.variant = var ? ThpVariant::kMMSE_UCD : ThpVariant::kGMD;
    double maxcf = 0, maxtau = 0, maxdep = 0;
    int nroot = 0, ncomp = 0, nfb = 0, nflat = 0, nskip = 0, nfbm = 0, nflag = 0;
    for (int c = 0; c < C; ++c) {
      const double* lc = lam.data() + (size_t)c * K * d;
      const double fr = flat_ratio(lc, K, d, M);
      CfResult r = threshold_closed_form(lc, cfg);
      // Capacity rejection must fail loudly, and it must be tested BEFORE the
      // flat_ratio / pyflag skip below: otherwise a rejected channel that also
      // happens to be near-flat is skipped and the NaN threshold passes.
      if (r.flag == 4) {
        fprintf(stderr, "[cf_bench] HOST rejected case %d with flag 4 "
                "(K=%d d=%d M=%d exceeds kCapMaxT/kCapMaxK/kCapMaxD, or M is "
                "not a perfect square)\n", c, K, d, M);
        ++nfbm; ++fails; continue;
      }
      // exact flag agreement, not the old (pyflag != 0) != fell_back: that
      // loose form could not tell 1 (scan fallback) from 2 (saturated), and
      // CfResult::fell_back is now exactly (flag == 1). Measured 0/40 exact
      // mismatches on this pool.
      if (pyflag[c] != r.flag) ++nflag;               // method must agree
      if (pyflag[c]) ++nfb;
      if (fr < 0.05) ++nflat;
      if (verb) printf("   ch%d flag=%d flat=%.3f tau_cpp=%.9f tau_py=%.9f\n",
                       c, pyflag[c], fr, r.tau_dB, tau_ref[c]);
      if (fr < 0.05 || pyflag[c] == 2) continue;
      maxtau = fmax(maxtau, err_tau(r.tau_dB, tau_ref[c]));
      if (pyflag[c] == 0) maxcf = fmax(maxcf, fabs(r.tau_cf_dB - tcf[c]));
      ++ncomp;
      // Deployed Newton budget (cfg.newton_iters, 20) against an 8-step run of
      // the SAME code path. This used to build a second CfConfig with
      // newton_iters = 20 and compare it against the default, which had also
      // become 20 -- so maxdep was identically 0.000 and the `maxdep < bound`
      // gate below could not fail.
      //
      // maxdep is EVIDENCE for the budget, not a tolerance on the deployed
      // answer, so its bound only has to catch a diverging safeguard. Measured
      // on the device (Apple M4, `make cf_bench`, flat_ratio >= 0.05 population):
      //   M=64 d=4 K=8 UCD   8.7e-2 dB   <- one channel, see below
      //   the other four configs  0.000 dB
      // The UCD number is above the 0.05 dB accuracy target the README quotes,
      // which is precisely the argument for deploying 20 steps instead of 8.
      // Convergence ladder on that same config, each rung against the next:
      //   8 -> 20 steps : 8.7e-2 dB      (this column)
      //   20 -> 30 steps: 2.2e-5 dB      (the maxtau column; the oracle runs 30)
      // so each budget increase buys ~3-4 orders here. It is a flat-Psi channel
      // (M=64/d=4/K=8 UCD ch3, flat_ratio 0.162 -- NOT near-flat enough to be
      // skipped), not a formula divergence: the other 39 compared channels agree
      // with the oracle to 2.1e-14 dB, which they could not if the two sides
      // solved different functions.
      CfConfig cfg8 = cfg; cfg8.newton_iters = 8;
      CfResult r8 = threshold_closed_form(lc, cfg8);
      maxdep = fmax(maxdep, err_tau(r.tau_dB, r8.tau_dB));
      const double u = r.u_star;
      if ((psi_u(u * 0.99, lc, cfg) <= 0.0 && psi_u(u * 1.01, lc, cfg) >= 0.0) ||
          fabs(psi_u(u, lc, cfg)) < 1e-6) ++nroot;
    }
    // 3) OpenCL kernel vs C++ (same conditioning rule; the fp32 datapath may
    //    take the scan fallback on a different channel subset, which is method
    //    divergence rather than a precision failure -> counted as fbm)
    std::vector<double> cl_cf(C), cl_tau(C); std::vector<int> cl_fl(C);
    rt.threshold_cf(lam.data(), C, K, d, M, var, cl_cf.data(), cl_tau.data(), cl_fl.data());
    double maxcl = 0, maxcl_cf = 0;
    for (int c = 0; c < C; ++c) {
      const double* lc = lam.data() + (size_t)c * K * d;
      CfResult r = threshold_closed_form(lc, cfg);
      // flag 4 is the capacity rejection (T > MAXT, K > MAXK, a constellation
      // whose folded candidate count exceeds MAXD, or M not a perfect square);
      // its tau is NaN by design. That is a build-configuration error, not a
      // fallback, so it must fail loudly instead of being counted as a method
      // mismatch -- and it is tested BEFORE the skip, on both sides, so a
      // near-flat over-capacity channel cannot slip through.
      if (cl_fl[c] == 4 || r.flag == 4) {
        fprintf(stderr, "[cf_bench] %s rejected case %d with flag 4 "
                "(K=%d d=%d M=%d exceeds the compiled MAXT/MAXK/MAXD capacity "
                "or M is not a perfect square)\n",
                cl_fl[c] == 4 ? "kernel" : "host", c, K, d, M);
        ++nfbm; ++fails; continue;
      }
      if (pyflag[c] || flat_ratio(lc, K, d, M) < 0.05) { ++nskip; continue; }
      if (cl_fl[c] != r.flag) { ++nfbm; continue; }
      if (verb) printf("   ch%d cl tau_cf=%.6f cl_cf=%.6f tau=%.6f cl_tau=%.6f fb=%d\n",
                       c, r.tau_cf_dB, cl_cf[c], r.tau_dB, cl_tau[c], cl_fl[c]);
      maxcl = fmax(maxcl, err_tau(cl_tau[c], r.tau_dB));
      maxcl_cf = fmax(maxcl_cf, fabs(cl_cf[c] - r.tau_cf_dB));
    }
    double us = rt.last_us();
    // maxtab / maxl4 bounds. maxtab compares two independent implementations of
    // the SAME branch logic and the SAME 4-point Lagrange arithmetic on the SAME
    // 1600 fp64 values, so its floor is the log(t) round trip (the probe gamma is
    // built as 2t^2(M-1)/6 and read back through t_of): measured here as
    // <= 1e-13 nats, and 1e-9 leaves four orders of margin while still failing on
    // an off-by-one row or column (which costs >= 4 nats). maxl4 is a property of
    // the model, not of an implementation -- Lemma 4 says D_mod/D_fin = L/(L-1)
    // exactly, and the converged deficit obeys it to 5.5e-14 for every grid
    // t >= 5 (tools/check_deficit_table.py); 1e-9 is the tolerance cf_ref.py has
    // always used for its own Lemma-4 probe. hnd is the deployed function's
    // handover step: 0.0052 dB measured at DFT_THI = 32 and ~1e-15 dB at
    // DFT_TLO, against 0.3046 dB for the TQ = 5 quadrature->tail switch it
    // replaces. 6e-3 dB is that 0.0052 with 15% margin, so regenerating the table
    // with a lower DFT_THI fails here and forces the trade to be re-measured.
    //
    // maxtau was 1e-4 dB, sized for a 2.78e-5 dB systematic offset between host
    // and oracle: above TQ the host solves on log(C) + lnQ_asym(t) while the
    // oracle used log(C * 0.5 * erfc), i.e. the two were different functions.
    // cf_ref.py now ports ln_deficit_fin/ln_deficit_mod verbatim, so they are
    // the same function, and the offset is GONE where it lived: measured on the
    // device, 39 of the 40 compared channels now agree to 2.1e-14 .. 3.6e-14 dB
    // (they were 4.4e-5 / 5.1e-7 / 2.8e-5 / 2.2e-7 per config before).
    // The one exception is a convergence residual, not a formula difference:
    // M=64/d=4/K=8 UCD ch3 (flat_ratio 0.162) still differs by 2.16e-5 dB
    // because the deployed 20-step budget has not reached the oracle's 30-step
    // root on that flat Psi -- see the ladder in the cfg8 block above. 1e-4 dB
    // therefore stays the bound: 4.6x margin over the measured residual and
    // 500x inside the 0.05 dB accuracy target. A 1e-9 bound would be asserting
    // machine-precision convergence of a FIXED budget, which is not what is
    // deployed and not what this column should claim. The ln-D table does not
    // change what maxtau measures -- host and oracle still evaluate the same
    // function, now a table lookup instead of a quadrature -- but it does change
    // the numbers, because the deployed Psi is a different (smoother, more
    // accurate) function: the staircase step at the old TQ = 5 switch is gone.
    // Re-measure the column on the device after this change; the bound stays
    // 1e-4 dB, which the oracle-side 30-step root still justifies.
    bool ok = maxq < 1e-9 && maxcf < 1e-9 && maxtau < 1e-4 && nflag == 0 &&
              maxcl < cl_tol && maxdep < 1.0 &&
              maxtab < 1e-9 && maxl4 < 1e-9 && hnd < 6e-3;
    if (!ok) ++fails;
    printf("cfg(M=%3d d=%d K=%d %s): quad=%.2e tab=%.2e l4=%.1e hnd=%.2e "
           "cf=%.2e tau=%.2e cl=%.2e "
           "roots=%d/%d fb=%d flat=%d skip=%d fbm=%d flagmis=%d dep=%.3f cf_only=%.1e %.0fus %s\n",
           M, d, K, var ? "UCD" : "GMD", maxq, maxtab, maxl4, hnd,
           maxcf, maxtau, maxcl,
           nroot, ncomp, nfb, nflat, nskip, nfbm, nflag, maxdep, maxcl_cf, us,
           ok ? "PASS" : "FAIL");
  }
  printf(fails ? "[cf_bench] FAIL\n" : "[cf_bench] ALL PASS\n");
  return fails ? 1 : 0;
}
