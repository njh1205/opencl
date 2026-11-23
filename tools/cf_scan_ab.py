#!/usr/bin/env python3
"""cf_scan_ab -- gate the near-flat fallback on the spectra `cf_bench` skips.

WHY THIS EXISTS. cf_bench's pool is 38 flag-0 + 2 flag-2 + 0 flag-1 out of 40,
and its tau column SKIPS every channel with flat_ratio < 0.05 or pyflag == 2.
So the near-flat fallback -- the only code kernels/psi_scan.clh owns -- has no
tau comparison anywhere in `make check`. This script is that missing coverage: a
FIXED case list of degenerate / near-flat / below-grid spectra plus a
scale-swept random pool, solved under two fallbacks, with Psi fills counted by
instrumentation rather than estimated.

MODES
  (default)      A/B: the pre-change 401-point scan against the shipped
                 PSI_SCAN_N-point scan, in the oracle and (with --host) in the
                 fp64 C++ ref. Reports per-case fills, flags, tau, and the
                 aggregate max|dtau| / flag-mismatch count.
  --sweep N,...   the resolution table kernels/psi_scan.clh quotes to justify
                 PSI_SCAN_N: for each N, the whole tau_policy (scan + bracket
                 extension + Newton + saturation) against the N = 401 baseline,
                 with average fills. This is the reproduction path for that table
                 and the procedure the static_assert in ref/threshold_cf.cpp
                 points at when someone moves the resolution.
  --countexamples the three Psi profiles that killed the "fixed 40-step
                 bisection on ln u" of README's PL list item 7: the structural
                 zero of Psi at u -> 0 (plus the scale sweep that shows an
                 endpoint test is not scale-covariant), a non-monotone Psi whose
                 top endpoint is inside the dead band, and the 7-decade in-band
                 plateau where a bisection and a scan hand Newton different
                 basins.
  --host          also measure the fp64 C++ ref, compiled here into a throwaway
                 binary from --new-source (default the repo's
                 ref/threshold_cf.cpp) and, if given, --old-source. Fills come
                 from counters this script inserts into COPIES of psi_u/psi_du;
                 no source in the repo is touched. --old-source needs a pristine
                 pre-change threshold_cf.cpp, which the repo no longer contains
                 -- that is the one input this script cannot regenerate for
                 itself.

EXAMPLES
  python3 tools/cf_scan_ab.py
  python3 tools/cf_scan_ab.py --sweep 401,321,241,161,121,81,61,41,21,11
  python3 tools/cf_scan_ab.py --countexamples
  python3 tools/cf_scan_ab.py --host --old-source /tmp/old_hpocl_v1/ref/threshold_cf.cpp \\
      --rounds 5 --reps 51

Exit status is 1 if the two arms disagree on any flag (a decision change: finite
tau against +-inf), or on any tau by more than --tol dB (default 1e-3, i.e. 50x
inside cf_bench's 5e-2 gate and 2.6x inside the 0.0004 dB README's decision
record measured between a 161-point and a 401-point scan of the same range).
"""
import argparse
import math
import os
import statistics
import subprocess
import sys
import tempfile

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
import cf_ref  # noqa: E402  (after the sys.path insert)

# ---------------------------------------------------------------------------
# the fixed case list. Four classes, because the fallback has four behaviours to
# get right and cf_bench covers at most one of them:
#   flat    every stream equal -> denom = g_min/eta - lam_min < 0 exactly, so the
#           closed form is refused and Psi is a shallow plateau: the class
#           cf_bench's pool NEVER produces a root in (0 of 40 flag-1).
#   zero    one or two zero streams (a rank-deficient channel): lam_min = 0 makes
#           the closed form's log(g_min/(eta*lam_min)) infinite, so u0 = -inf and
#           the fallback runs. README's decision record measured exactly these
#           degenerate gamma = 0 shapes (1 and 2 zero streams, and 1 zero + flat
#           rest) at both TQ values and both dead bands; 8 of the 24 here are the
#           "1 zero + flat rest" variant.
#   below   the whole spectrum under the ln-D grid (lam ~ 1e-18): the
#           whole-spectrum-below-the-grid short circuit fires on the two compiled
#           sides and NOT in the oracle, so this class is the independent check of
#           that proof. flag 2, tau = +-inf.
#   ctrl    well conditioned: the fallback must not run at all, so the fill count
#           here is the control that proves a change is scoped to the fallback.
# K is 4 throughout (T = K*d in {8, 16}); cf_bench already covers K = 8 controls.
# ---------------------------------------------------------------------------
SEED = 20261026


def build_cases():
    rng = np.random.default_rng(SEED)
    cases = []

    def add(name, M, d, K, variant, lam):
        cases.append(dict(name=name, M=M, d=d, K=K, variant=variant,
                          lam=np.asarray(lam, dtype=np.float64)))

    def loguni(n, lo=-2.0, hi=2.0):
        return 10.0 ** rng.uniform(lo, hi, n)

    for M in (16, 64):
        for d in (2, 4):
            for v in (0, 1):
                for sc in (1.0, 1e-3):     # flat: the crossing u scales with 1/lam
                    add("flat_M%d_d%d_%s_s%g" % (M, d, "UCD" if v else "GMD", sc),
                        M, d, 4, v, np.full(4 * d, sc))
    for M in (16, 64):
        for d in (2, 4):
            for v in (0, 1):
                T = 4 * d
                lam = loguni(T)
                l1 = lam.copy(); l1[0] = 0.0
                l2 = lam.copy(); l2[0] = l2[1] = 0.0
                lf = np.full(T, 1.0); lf[0] = 0.0
                add("zero1_M%d_d%d_%s" % (M, d, "UCD" if v else "GMD"), M, d, 4, v, l1)
                add("zero2_M%d_d%d_%s" % (M, d, "UCD" if v else "GMD"), M, d, 4, v, l2)
                add("zeroflat_M%d_d%d_%s" % (M, d, "UCD" if v else "GMD"), M, d, 4, v, lf)
    # below the ln-D grid: the shape ref/threshold_cf.cpp's below_grid comment
    # quotes (lam spanning 30.22 dB around 2.25e-18), so 2*PSI_SCAN_HI*lam_max
    # = 1.5e-10 against kDftTlo2 = 1e-8.
    span = np.linspace(-15.11, 15.11, 16)          # 30.22 dB
    add("below_M64_d4_GMD", 64, 4, 4, 0, 2.25e-18 * 10.0 ** (span / 20.0))
    add("below_M64_d4_UCD", 64, 4, 4, 1, 2.25e-18 * 10.0 ** (span / 20.0))
    add("below_M16_d2_GMD", 16, 2, 4, 0, 1e-17 * 10.0 ** (span[:8] / 20.0))
    for M, d in ((16, 2), (64, 4), (256, 4)):
        add("ctrl_M%d_d%d" % (M, d), M, d, 4, 0, loguni(4 * d, -1.0, 1.0))
    return cases


def case_class(name):
    if name.startswith("zeroflat"):
        return "zero"
    for k in ("flat", "zero", "below", "ctrl"):
        if name.startswith(k):
            return k
    return "pool"


def random_pool(nper=12, seed=7):
    """A scale-swept pool drawn exactly like tools/cf_ref.py main() does (seed 7,
    lognormal 0.8 descending, plus the near-flat 1 + 0.02N config), times 10^k for
    k in -6..6 so it visits the below-grid and saturated regimes the shipped pool
    never reaches, plus harder near-flat / rank-deficient shapes."""
    rng = np.random.default_rng(seed)
    out = []
    cfgs = [(16, 2, 4, 0), (64, 4, 8, 0), (64, 4, 8, 1), (256, 4, 8, 0),
            (64, 2, 4, 0)]
    for (M, d, K, var) in cfgs:
        if (M, d, K) == (64, 2, 4):
            lam = np.abs(1.0 + 0.02 * rng.normal(0.0, 1.0, (nper, K * d)))
        else:
            lam = np.sort(np.exp(rng.normal(0.0, 0.8, (nper, K * d))), axis=1)[:, ::-1]
        for i in range(nper):
            for k in (-6, -3, 0, 3, 6):
                out.append(dict(name="pool_M%d_d%d_K%d_v%d_i%d_s%+d" % (M, d, K, var, i, k),
                                M=M, d=d, K=K, variant=var, lam=lam[i] * 10.0 ** k))
    for M, d, K in ((16, 2, 4), (64, 4, 4), (64, 2, 8), (256, 4, 4)):
        for v in (0, 1):
            for i in range(nper):
                T = K * d
                kind = i % 4
                if kind == 0:
                    base = np.abs(1.0 + 0.02 * rng.normal(0.0, 1.0, T))
                else:
                    base = np.exp(rng.uniform(-2, 2, T)) * 10.0 ** rng.integers(-4, 5)
                if kind == 2:
                    base[0] = 0.0
                if kind == 3:
                    base[0] = base[1] = 0.0
                out.append(dict(name="flat_M%d_d%d_K%d_v%d_i%d" % (M, d, K, v, i),
                                M=M, d=d, K=K, variant=v, lam=base))
    return out


# ---------------------------------------------------------------------------
# oracle side
# ---------------------------------------------------------------------------
def root_scan_n(lam, M, d, K, variant, n):
    """First-transition scan of Psi on an n-point geometric grid over
    [PSI_SCAN_LO, PSI_SCAN_HI]. n = 401 is the pre-change ref/threshold_cf.cpp
    scan_root verbatim (ratio 1e12**(1/400), arithmetic-midpoint u0); n =
    cf_ref.PSI_SCAN_N is what ships. Kept here, parameterised, so --sweep can
    re-measure the resolution table without touching the shipped code."""
    u0, _ = cf_ref.tau_closed_form(lam, M, d, K)
    if u0 is not None:
        return u0, 0
    band = cf_ref.PSI_BAND
    ratio = (cf_ref.PSI_SCAN_HI / cf_ref.PSI_SCAN_LO) ** (1.0 / (n - 1))
    u, u_neg = cf_ref.PSI_SCAN_LO, None
    with np.errstate(divide="ignore", invalid="ignore"):
        for _ in range(n):
            f = cf_ref.psi(lam, u, M, d, K, variant)
            if f <= -band:
                u_neg = u
            elif f >= band and u_neg is not None:
                return 0.5 * (u_neg + u), 1
            u *= ratio
    return None, 2


def root_bisect(lam, M, d, K, variant, n):
    """The REJECTED alternative, kept runnable: the same n-point grid to find the
    bracket, then PSI_BISECT_STYLE 40 ln-u bisection steps inside it, with the
    in-band collapse. --countexamples shows what it costs (up to 30.6 dB of tau
    on a 7-decade dead-band plateau, because the collapse rule puts u0 in the
    middle of the plateau and Newton lands in a different basin)."""
    u0, _ = cf_ref.tau_closed_form(lam, M, d, K)
    if u0 is not None:
        return u0, 0
    band = cf_ref.PSI_BAND
    vlo = math.log(cf_ref.PSI_SCAN_LO)
    dv = (math.log(cf_ref.PSI_SCAN_HI) - vlo) / (n - 1)
    vn = None
    found = None
    with np.errstate(divide="ignore", invalid="ignore"):
        for j in range(n):
            v = vlo + dv * j
            f = cf_ref.psi(lam, math.exp(v), M, d, K, variant)
            if f <= -band:
                vn = v
            elif f >= band and vn is not None and found is None:
                found = (vn, v)
    if found is None:
        return None, 2
    a, b = found
    for _ in range(40):
        m = 0.5 * (a + b)
        f = cf_ref.psi(lam, math.exp(m), M, d, K, variant)
        if f <= -band:
            a = m
        elif f >= band:
            b = m
        else:
            a = b = m
    return math.exp(0.5 * (a + b)), 1


class CountingPsi:
    """Wraps cf_ref.psi so every fill is counted, then puts it back."""

    def __init__(self):
        self.n = 0
        self._saved = None

    def __enter__(self):
        self._saved = cf_ref.psi
        outer = self

        def counted(*a, **kw):
            outer.n += 1
            return self._saved(*a, **kw)

        cf_ref.psi = counted
        return self

    def __exit__(self, *exc):
        cf_ref.psi = self._saved
        return False


def run_oracle(case, root_fn):
    """tau/flag/fills from the shipped oracle with its root_u swapped, so the
    30-step Newton, the bracket extension and the saturation branch are literally
    cf_ref.tau_policy on both arms and the only difference is the fallback."""
    saved = cf_ref.root_u
    lam, M, d, K, v = (case["lam"], case["M"], case["d"], case["K"], case["variant"])
    cf_ref.root_u = lambda l, m, dd, kk, vv=0: root_fn(l, m, dd, kk, vv)
    try:
        with CountingPsi() as cnt:
            tau, flag = cf_ref.tau_policy(lam, M, d, K, v)
        return dict(tau=float(tau), flag=int(flag), fills=cnt.n,
                    fills_u=cnt.n, fills_d=0, ms=float("nan"))
    finally:
        cf_ref.root_u = saved


# ---------------------------------------------------------------------------
# host side: compile a throwaway, instrumented fp64 ref
# ---------------------------------------------------------------------------
DRIVER = r"""
// GENERATED by tools/cf_scan_ab.py -- do not edit, do not commit.
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include "threshold_cf.h"
#ifdef CF_AB_DEVICE
#include <string>
#include "hpocl_runtime.h"
#endif
extern long hpocl_ab_psi_u, hpocl_ab_psi_d;   // in the instrumented ref copy

struct Case { const char* name; int M, d, K, variant, n; const double* lam; };
@CASES@
static const Case cases[] = { @CASETABLE@ };

int main(int argc, char** argv) {
  const int reps = argc > 1 ? atoi(argv[1]) : 21;
  for (unsigned ci = 0; ci < sizeof(cases) / sizeof(cases[0]); ++ci) {
    const Case& c = cases[ci];
    hpocl::CfConfig cfg;
    cfg.M = c.M; cfg.d = c.d; cfg.K = c.K;
    cfg.variant = c.variant ? hpocl::ThpVariant::kMMSE_UCD : hpocl::ThpVariant::kGMD;
    hpocl::CfResult warm = hpocl::threshold_closed_form(c.lam, cfg);  // warm caches
    (void)warm;
    std::vector<double> ms;
    long fu = -1, fd = -1;
    hpocl::CfResult r;
    for (int it = 0; it < reps; ++it) {
      hpocl_ab_psi_u = 0; hpocl_ab_psi_d = 0;
      auto t0 = std::chrono::steady_clock::now();
      r = hpocl::threshold_closed_form(c.lam, cfg);
      auto t1 = std::chrono::steady_clock::now();
      ms.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
      if (fu < 0) { fu = hpocl_ab_psi_u; fd = hpocl_ab_psi_d; }
      else if (fu != hpocl_ab_psi_u || fd != hpocl_ab_psi_d) {
        printf("c %u UNSTABLE_FILLS %ld/%ld vs %ld/%ld\n", ci, fu, fd,
               hpocl_ab_psi_u, hpocl_ab_psi_d);
        return 3;   // a fill count that depends on the repeat is not a count
      }
    }
    std::sort(ms.begin(), ms.end());
    printf("c %u %d %d %ld %ld %.17g %.17g %.9f\n", ci, r.flag, r.newton_used,
           fu, fd, r.tau_cf_dB, r.tau_dB, ms[ms.size() / 2]);
  }
#ifdef CF_AB_DEVICE
  // The DEVICE arm, which cf_bench cannot give for these spectra: its cl loop
  // skips every channel with pyflag != 0 or flat_ratio < 0.05, and this case list
  // is exactly those channels, so the kernel's fallback path has no comparison
  // anywhere in `make check`. kernels/threshold_cf.cl is loaded relative to the
  // CWD, so running this binary in two trees A/Bs two kernels.
  {
    hpocl::Runtime rt;
    if (!rt.init("cpu")) { printf("# device init failed\n"); return 2; }
    std::string kopts = rt.device_fp64() ? "-DHPOCL_FP64" : "";
    if (!rt.load_program({"kernels/threshold_cf.cl"}, kopts.c_str())) return 2;
    printf("# device %s fp64=%d opts '%s'\n", rt.device_name(),
           (int)rt.device_fp64(), kopts.c_str());
    for (unsigned ci = 0; ci < sizeof(cases) / sizeof(cases[0]); ++ci) {
      const Case& c = cases[ci];
      double tcf = 0, tau = 0; int fl = -1;
      double best = 1e30;
      for (int it = 0; it < reps; ++it) {
        rt.threshold_cf(c.lam, 1, c.K, c.d, c.M, c.variant, &tcf, &tau, &fl);
        best = std::min(best, rt.last_us());
      }
      printf("d %u %d %.17g %.17g %.3f\n", ci, fl, tcf, tau, best);
    }
  }
#endif
  return 0;
}
"""


def instrument(src_text, tag):
    """Insert two deficit-fill counters into a COPY of ref/threshold_cf.cpp."""
    out = ("// AUTO-INSTRUMENTED by tools/cf_scan_ab.py (%s): fill counters only.\n"
           "long hpocl_ab_psi_u = 0, hpocl_ab_psi_d = 0;\n" % tag) + src_text
    for sig, ctr in (("double psi_u(double u, const double* lam, const CfConfig& c,"
                      " const PsiPre* w) {", "hpocl_ab_psi_u"),
                     ("double psi_du(double u, const double* lam, const CfConfig& c,"
                      " const PsiPre* w) {", "hpocl_ab_psi_d")):
        if out.count(sig) != 1:
            raise SystemExit("[cf_scan_ab] cannot instrument: %r occurs %d times"
                             % (sig, out.count(sig)))
        out = out.replace(sig, sig + "\n  ++%s;" % ctr)
    return out


def build_host(source, workdir, tag, cases, device=False):
    with open(source, encoding="utf-8") as f:
        ref_src = f.read()
    ref_path = os.path.join(workdir, "threshold_cf_instr_%s.cpp" % tag)
    with open(ref_path, "w", encoding="utf-8") as f:
        f.write(instrument(ref_src, tag))
    decl, table = [], []
    for i, c in enumerate(cases):
        arr = "lam%d" % i
        decl.append("static const double %s[] = {%s};" %
                    (arr, ",".join("%.17g" % v for v in c["lam"])))
        table.append('{"%s",%d,%d,%d,%d,%d,%s}' %
                     (c["name"], c["M"], c["d"], c["K"], c["variant"],
                      len(c["lam"]), arr))
    drv_path = os.path.join(workdir, "driver_%s.cpp" % tag)
    with open(drv_path, "w", encoding="utf-8") as f:
        # @TOKEN@ substitution, not %: the driver is mostly printf formats.
        f.write(DRIVER.replace("@CASES@", "\n".join(decl))
                      .replace("@CASETABLE@", ",\n  ".join(table)))
    exe = os.path.join(workdir, "cf_ab_%s%s" % (tag, "_dev" if device else ""))
    cmd = ["c++", "-O2", "-std=c++17", "-w",
           "-I" + os.path.dirname(os.path.abspath(source)),  # a pristine .h wins
           "-I" + os.path.join(ROOT, "include"),
           "-I" + os.path.join(ROOT, "ref"),
           "-I" + os.path.join(ROOT, "runtime"),
           "-I" + os.path.join(ROOT, "kernels"),             # psi_scan.clh
           ]
    if device:
        cmd += ["-DCF_AB_DEVICE"]
    cmd += [drv_path, ref_path]
    if device:
        cmd += [os.path.join(ROOT, "runtime", "hpocl_runtime.cpp"),
                "-framework", "OpenCL" if sys.platform == "darwin" else "-lOpenCL"]
    cmd += ["-o", exe]
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        sys.stderr.write(p.stdout + p.stderr)
        raise SystemExit("[cf_scan_ab] host build failed (%s)" % tag)
    return exe


def run_host(exe, cases, reps, cwd=ROOT):
    p = subprocess.run([exe, str(reps)], capture_output=True, text=True, cwd=cwd)
    if p.returncode != 0:
        sys.stderr.write(p.stdout + p.stderr)
        raise SystemExit("[cf_scan_ab] host run failed (%d) in %s" % (p.returncode, cwd))
    out = {}
    dev = {}
    for line in p.stdout.strip().splitlines():
        f = line.split()
        if line.startswith("# device"):
            print("[cf_scan_ab] %s (cwd %s)" % (line[1:], cwd))
        if f[0] == "d":
            dev[int(f[1])] = dict(name=cases[int(f[1])]["name"], flag=int(f[2]),
                                  tau_cf=float(f[3]), tau=float(f[4]), us=float(f[5]),
                                  fills=0, fills_u=0, fills_d=0, nit=0, ms=float("nan"))
            continue
        if f[0] != "c":
            continue
        i = int(f[1])
        out[i] = dict(name=cases[i]["name"], flag=int(f[2]), nit=int(f[3]),
                      fills_u=int(f[4]), fills_d=int(f[5]),
                      fills=int(f[4]) + int(f[5]),
                      tau_cf=float(f[6]), tau=float(f[7]), ms=float(f[8]))
    if len(out) != len(cases):
        raise SystemExit("[cf_scan_ab] host printed %d of %d cases"
                         % (len(out), len(cases)))
    host = [out[i] for i in sorted(out)]
    if dev:
        return host, [dev[i] for i in sorted(dev)]
    return host


# ---------------------------------------------------------------------------
def dtau(a, b):
    if np.isinf(a) or np.isinf(b):
        return 0.0 if a == b else float("inf")
    if np.isnan(a) or np.isnan(b):
        return 0.0 if (np.isnan(a) and np.isnan(b)) else float("inf")
    return abs(a - b)


def class_summary(cases, res, arms):
    for cls in ("flat", "zero", "below", "ctrl", "pool"):
        idx = [i for i, c in enumerate(cases) if case_class(c["name"]) == cls]
        if not idx:
            continue
        line = "[cf_scan_ab]   %-5s n=%3d flags=%s" % (
            cls, len(idx), sorted({res[arms[0]][i]["flag"] for i in idx}))
        for a in arms:
            fills = [res[a][i]["fills"] for i in idx]
            line += "  %s %d..%d (avg %.1f)" % (a, min(fills), max(fills),
                                                float(np.mean(fills)))
        print(line)


def mode_sweep(cases, ns, tol):
    print("[cf_scan_ab] --sweep: whole tau_policy per arm, against the N=401 "
          "baseline (the pre-change scan_root), oracle side")
    base = [run_oracle(c, lambda l, m, d, k, v: root_scan_n(l, m, d, k, v, 401))
            for c in cases]
    print("%6s %9s %10s %10s %12s %s" %
          ("N", "spacing", "fills_avg", "flagmis", "max|dtau| dB", "worst"))
    bad = 0
    for n in ns:
        res = [run_oracle(c, lambda l, m, d, k, v, n=n: root_scan_n(l, m, d, k, v, n))
               for c in cases]
        nflag = sum(r["flag"] != b["flag"] for r, b in zip(res, base))
        dts = [(dtau(r["tau"], b["tau"]), cases[i]["name"], b["tau"], r["tau"])
               for i, (r, b) in enumerate(zip(res, base))]
        worst = max(dts)
        print("%6d %9.4f %10.1f %10d %12.3e %s" %
              (n, 12.0 / (n - 1), float(np.mean([r["fills"] for r in res])),
               nflag, worst[0],
               "" if (nflag == 0 and worst[0] <= tol) else
               "%s: %.4g -> %.4g" % (worst[1], worst[2], worst[3])))
        if nflag:
            bad += nflag
    return bad


def mode_countexamples():
    cases = {c["name"]: c for c in build_cases()}
    band = cf_ref.PSI_BAND
    print("[cf_scan_ab] Psi dead band = %g (fp64; the fp32 kernel uses 1e-3)" % band)

    print("\n(1) Psi -> 0 as u -> PSI_SCAN_LO, structurally: every deficit tends "
          "to log2(M), so\n    Psi -> ln(T*log2 M) - ln(K*d*log2 M) = 0 because "
          "T = K*d. The bottom endpoint is therefore\n    INSIDE the dead band "
          "for a shallow spectrum, and |Psi(LO)| scales with lam, so an\n    "
          "endpoint bracket test is not scale-covariant. zero1_M16_d4_UCD "
          "(lam[0] = 0), rescaled:")
    c = cases["zero1_M16_d4_UCD"]
    print("    %10s %14s %10s %10s %12s %s" %
          ("scale", "Psi(1e-6)", "endpoint", "scan flag", "scan tau dB", "Psi(1)"))
    for s in (1e-4, 1e-2, 1.0, 1e2, 1e4):
        lam = c["lam"] * s
        M, d, K, v = c["M"], c["d"], c["K"], c["variant"]
        flo = cf_ref.psi(lam, cf_ref.PSI_SCAN_LO, M, d, K, v)
        fhi = cf_ref.psi(lam, cf_ref.PSI_SCAN_HI, M, d, K, v)
        u0, fl = root_scan_n(lam, M, d, K, v, 401)
        saved = cf_ref.root_u
        cf_ref.root_u = lambda l, m, dd, kk, vv=0: root_scan_n(l, m, dd, kk, vv, 401)
        try:
            tau, _ = cf_ref.tau_policy(lam, M, d, K, v)
        finally:
            cf_ref.root_u = saved
        print("    %10g %+14.3e %10s %10d %12.4f %+.3e" %
              (s, flo, "bracket" if (flo <= -band and fhi >= band) else "NO",
               fl, tau, cf_ref.psi(lam, 1.0, M, d, K, v)))

    print("\n(2) Psi is not monotone, so the TOP endpoint can be inside the band "
          "too.\n    zero2_M64_d2_UCD (user 0 entirely dead):")
    c = cases["zero2_M64_d2_UCD"]
    lam, M, d, K, v = c["lam"], c["M"], c["d"], c["K"], c["variant"]
    print("    lam = %s" % np.array2string(lam, precision=6, max_line_width=200))
    for u in (1e-6, 31.0, 100.0, 320.0, 1000.0, 1e6):
        f = cf_ref.psi(lam, u, M, d, K, v)
        print("    Psi(%9.4g) = %+13.6e   %s" %
              (u, f, "in band" if abs(f) < band else ("<= -band" if f < 0 else ">= +band")))
    u0, fl = root_scan_n(lam, M, d, K, v, 401)
    saved = cf_ref.root_u
    cf_ref.root_u = lambda l, m, dd, kk, vv=0: root_scan_n(l, m, dd, kk, vv, 401)
    try:
        tau, _ = cf_ref.tau_policy(lam, M, d, K, v)
    finally:
        cf_ref.root_u = saved
    print("    scan: flag=%d u0=%.4f tau=%.4f dB  (BOTH endpoints are in band, so "
          "an endpoint test says flag 2, tau=+inf)" % (fl, u0, tau))

    print("\n(3) A dead-band plateau makes the bisection's u0 and the scan's u0 "
          "different\n    Newton basins. Same grid, only u0 differs.")
    # lam quoted verbatim from the adversarial sample described in
    # kernels/psi_scan.clh item 3, so this counterexample needs no sample file.
    prof = {"s_M16_d2_K4_v0_i11": (16, 2, 4, 0,
                                   np.array([0.0, 0.0, 2026.895778, 1573.899618,
                                             1450.712488, 927.661269, 446.939541,
                                             364.441077]))}
    for name, (M, d, K, v, lam) in prof.items():
        print("    %s  M=%d d=%d K=%d var=%d" % (name, M, d, K, v))
        for lu in (-6.0, -1.56, 0.0, 2.0, 4.0, 5.5545, 6.0):
            f = cf_ref.psi(lam, 10.0 ** lu, M, d, K, v)
            print("      Psi(10^%+7.4f) = %+13.6e  %s" %
                  (lu, f, "in band" if abs(f) < band else ("<= -band" if f < 0 else ">= +band")))
        for tag, fn in (("scan u0    ", lambda: root_scan_n(lam, M, d, K, v, 401)),
                        ("bisect u0  ", lambda: root_bisect(lam, M, d, K, v, 401))):
            u0, fl = fn()
            saved = cf_ref.root_u
            cf_ref.root_u = lambda l, m, dd, kk, vv=0: (u0, fl)
            try:
                tau, _ = cf_ref.tau_policy(lam, M, d, K, v)
            finally:
                cf_ref.root_u = saved
            print("      %s = %12.6g -> Newton tau = %.4f dB" % (tag, u0, tau))

    print("\n    The same failure without a plateau: Psi leaving the dead band "
          "MORE THAN ONCE, so\n    'the first transition' and 'a bracket' are not "
          "the same question. This one is in\n    --pool, so it is reproducible "
          "without quoting a spectrum.")
    pool = {c["name"]: c for c in random_pool()}
    c = pool["flat_M64_d4_K4_v1_i7"]
    lam, M, d, K, v = c["lam"], c["M"], c["d"], c["K"], c["variant"]
    print("    flat_M64_d4_K4_v1_i7  M=%d d=%d K=%d var=%d" % (M, d, K, v))
    print("    %10s %14s %s" % ("log10u", "Psi", "band state"))
    for lu10 in np.arange(-2.0, -1.0, 0.05):
        f = cf_ref.psi(lam, 10.0 ** lu10, M, d, K, v)
        state = "in band" if abs(f) < cf_ref.PSI_BAND else ("<= -band" if f < 0 else ">= +band")
        print("    %10.2f %+14.6e %s" % (lu10, f, state))
    for tag, fn in (("scan u0    ", lambda: root_scan_n(lam, M, d, K, v, cf_ref.PSI_SCAN_N)),
                    ("bisect u0  ", lambda: root_bisect(lam, M, d, K, v, cf_ref.PSI_SCAN_N))):
        u0, fl = fn()
        saved = cf_ref.root_u
        cf_ref.root_u = lambda l, m, dd, kk, vv=0: (u0, fl)
        try:
            tau, _ = cf_ref.tau_policy(lam, M, d, K, v)
        finally:
            cf_ref.root_u = saved
        print("      %s = %12.6g -> Newton tau = %.4f dB (same %d-point grid both "
              "arms)" % (tag, u0, tau, cf_ref.PSI_SCAN_N))
    return 0


def mode_bisect_check(cases, tol):
    """Scan u0 against bisection u0, on the same grid, through the same Newton.

    Reproduces counterexample (3) as a rate rather than as one channel: this is
    the measurement that rejected the fixed 40-step ln-u bisection README's PL
    list item 7 asks for.
    """
    bad = 0
    rows = []
    for c in cases:
        lam, M, d, K, v = (c["lam"], c["M"], c["d"], c["K"], c["variant"])
        out = {}
        for tag, fn in (("scan", root_scan_n), ("bisect", root_bisect)):
            u0, fl = fn(lam, M, d, K, v, cf_ref.PSI_SCAN_N)
            saved = cf_ref.root_u
            cf_ref.root_u = lambda l, m, dd, kk, vv=0: (u0, fl)
            try:
                tau, flag = cf_ref.tau_policy(lam, M, d, K, v)
            finally:
                cf_ref.root_u = saved
            out[tag] = (float(tau), int(flag), u0)
        dt = dtau(out["scan"][0], out["bisect"][0])
        nf = out["scan"][1] != out["bisect"][1]
        if nf or dt > tol:
            bad += 1
            rows.append((dt, nf, c["name"], out["scan"], out["bisect"]))
    rows.sort(reverse=True)
    print("[cf_scan_ab] scan u0 vs 40-step ln-u bisection u0, same %d-point grid, "
          "same Newton: %d of %d channels differ by more than %.1e dB or by a flag"
          % (cf_ref.PSI_SCAN_N, len(rows), len(cases), tol))
    for dt, nf, name, sc, bi in rows[:8]:
        print("   %-28s scan flag=%d tau=%12.4f (u0=%.6g) | bisect flag=%d "
              "tau=%12.4f (u0=%.6g) | dtau=%.3g%s" %
              (name, sc[1], sc[0], sc[2], bi[1], bi[0], bi[2], dt,
               "  FLAG FLIP" if nf else ""))
    return bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", action="store_true",
                    help="also measure the fp64 C++ ref (needs c++)")
    ap.add_argument("--new-source",
                    default=os.path.join(ROOT, "ref", "threshold_cf.cpp"))
    ap.add_argument("--old-source", default=None,
                    help="pristine pre-change threshold_cf.cpp for a C++ A/B")
    ap.add_argument("--pool", action="store_true",
                    help="add the scale-swept random pool (slower, more fallback cases)")
    ap.add_argument("--sweep", default=None,
                    help="comma-separated grid sizes: re-measure the PSI_SCAN_N table")
    ap.add_argument("--countexamples", action="store_true",
                    help="print the Psi profiles the psi_scan.clh comments quote")
    ap.add_argument("--bisect-check", action="store_true",
                    help="rate the rejected 40-step ln-u bisection against the scan")
    ap.add_argument("--device", action="store_true",
                    help="also run the fp32 kernel on the same cases (needs an "
                         "OpenCL device; cf_bench skips exactly these channels)")
    ap.add_argument("--device-cwd", action="append", default=[],
                    help="extra working directory to load kernels/threshold_cf.cl "
                         "from, for a device-side before/after A/B")
    ap.add_argument("--reps", type=int, default=21)
    ap.add_argument("--rounds", type=int, default=1,
                    help="alternating timing rounds; medians of medians")
    ap.add_argument("--tol", type=float, default=1e-3)
    ap.add_argument("--keep", default=None, help="keep the generated build dir here")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    if args.countexamples:
        return mode_countexamples()

    cases = build_cases()
    if args.pool:
        cases += random_pool()
    if args.sweep:
        ns = [int(x) for x in args.sweep.split(",")]
        return 1 if mode_sweep(cases, ns, args.tol) else 0
    if args.bisect_check:
        return 1 if mode_bisect_check(cases, args.tol) else 0

    res = {}
    res["legacy401"] = [run_oracle(c, lambda l, m, d, k, v: root_scan_n(l, m, d, k, v, 401))
                        for c in cases]
    res["shipped"] = [run_oracle(c, cf_ref.root_u) for c in cases]

    workdir = args.keep or tempfile.mkdtemp(prefix="cf_scan_ab_")
    os.makedirs(workdir, exist_ok=True)
    if args.host:
        for tag, src in (("old", args.old_source), ("new", args.new_source)):
            if src is None:
                continue
            exe = build_host(src, workdir, tag, cases)
            rounds = [run_host(exe, cases, args.reps) for _ in range(args.rounds)]
            merged = []
            for i in range(len(cases)):
                m = dict(rounds[0][i])
                m["ms"] = statistics.median(r[i]["ms"] for r in rounds)
                for k in ("flag", "fills_u", "fills_d", "tau"):
                    if any(r[i][k] != rounds[0][i][k] for r in rounds):
                        raise SystemExit("[cf_scan_ab] %s case %s: %s unstable across "
                                         "rounds" % (tag, m["name"], k))
                merged.append(m)
            res["host_" + tag] = merged
            print("[cf_scan_ab] host %s: %s (%d cases, %d reps x %d rounds)"
                  % (tag, src, len(cases), args.reps, args.rounds))

    arms = [k for k in ("legacy401", "shipped", "host_old", "host_new") if k in res]
    bad = 0
    print("[cf_scan_ab] %d cases: %s" % (len(cases), ", ".join(
        "%d %s" % (sum(case_class(c["name"]) == k for c in cases), k)
        for k in ("flat", "zero", "below", "ctrl", "pool")
        if any(case_class(c["name"]) == k for c in cases))))
    if not args.quiet:
        print("%-26s %-7s %6s %6s %9s %10s" %
              ("case", "flagL/S", "fillL", "fillS", "msL", "dtau_dB"))
    for i, c in enumerate(cases):
        L, N = res["legacy401"][i], res["shipped"][i]
        dt = dtau(L["tau"], N["tau"])
        mism = L["flag"] != N["flag"] or dt > args.tol
        if mism:
            bad += 1
        if not args.quiet or mism:
            print("%-26s %d/%-5d %6d %6s %9s %10.2e%s" %
                  (c["name"], L["flag"], N["flag"], L["fills"], N["fills"], "-", dt,
                   "   <-- MISMATCH" if mism else ""))
    class_summary(cases, res, [a for a in arms if not a.startswith("host")])

    for a, b in (("legacy401", "shipped"), ("host_old", "host_new"),
                 ("shipped", "host_new"), ("legacy401", "host_old")):
        if a not in res or b not in res:
            continue
        dt = max(dtau(x["tau"], y["tau"]) for x, y in zip(res[a], res[b]))
        nf = sum(x["flag"] != y["flag"] for x, y in zip(res[a], res[b]))
        fa = float(np.mean([x["fills"] for x in res[a]]))
        fb = float(np.mean([y["fills"] for y in res[b]]))
        print("[cf_scan_ab] %-10s vs %-10s max|dtau| = %.3e dB, flag mismatches = %d, "
              "avg fills %.1f -> %.1f (%.2fx)" % (a, b, dt, nf, fa, fb, fa / fb if fb else 0))
        if nf or dt > args.tol:
            bad += 1 + nf
        if a.startswith("host") and b.startswith("host"):
            for cls in ("flat", "zero", "below", "ctrl", "pool"):
                idx = [i for i, c in enumerate(cases) if case_class(c["name"]) == cls]
                if not idx:
                    continue
                ma = statistics.median(res[a][i]["ms"] for i in idx)
                mb = statistics.median(res[b][i]["ms"] for i in idx)
                print("[cf_scan_ab]   host ms median, %-5s %s %.4f -> %s %.4f  (ratio %.3f)"
                      % (cls, a, ma, b, mb, ma / mb if mb else float("nan")))
            class_summary(cases, res, [a, b])
    if args.device:
        exe = build_host(args.new_source, workdir, "new", cases, device=True)
        def dev_run(cwd):
            r = run_host(exe, cases, max(3, args.reps), cwd=cwd)
            return r[1] if isinstance(r, tuple) else None
        runs = [(ROOT, dev_run(ROOT))]
        for d in args.device_cwd:
            runs.append((d, dev_run(d)))
        if runs[0][1] is None:
            raise SystemExit("[cf_scan_ab] the device arm printed nothing -- no "
                             "OpenCL device, or the kernel failed to build")
        base_dir, base = runs[0]
        res["device"] = base
        print("[cf_scan_ab] device arm: %d cases on %s, %d reps (min of the event "
              "times reported)" % (len(base), base_dir, max(3, args.reps)))
        nflag = 0
        maxdt = 0.0
        for d, other in runs[1:]:
            nf = sum(a["flag"] != b["flag"] for a, b in zip(base, other))
            dt = max(dtau(a["tau"], b["tau"]) for a, b in zip(base, other))
            nflag += nf
            maxdt = max(maxdt, dt)
            print("[cf_scan_ab] device kernel %s vs %s: flag mismatches = %d, "
                  "max|dtau| = %.3e dB" % (base_dir, d, nf, dt))
            if nf or dt > 0.0:
                for a, b in zip(base, other):
                    if a["flag"] != b["flag"] or dtau(a["tau"], b["tau"]) > 0.0:
                        print("   %-28s %s flag=%d tau=%.6g | %s flag=%d tau=%.6g"
                              % (a["name"], base_dir, a["flag"], a["tau"], d,
                                 b["flag"], b["tau"]))
            if nf:
                bad += nf
        # fp64 reference arm for the device comparison: the C++ host if it was
        # measured, else the numpy oracle. Both run band = 1e-6 against the
        # device's 1e-3.
        fp64 = res.get("host_new", res["shipped"])
        fp64_name = "host C++ fp64" if "host_new" in res else "oracle numpy fp64"
        nfh = sum(a["flag"] != b["flag"] for a, b in zip(fp64, base))
        dh = max(dtau(a["tau"], b["tau"]) for a, b in zip(fp64, base))
        print("[cf_scan_ab] device(fp32) vs %s: flag mismatches = %d, max|dtau| = "
              "%.3e dB -- nonzero is EXPECTED and pre-existing: the fp32 dead band "
              "is 1e-3 against 1e-6, so the two sides can pick different "
              "transitions on a near-flat spectrum (README's fallback-scan record "
              "measures tau moving 0.0000 dB for that on its 18 shapes). What must "
              "hold is that the device arm is unchanged by THIS edit, which is the "
              "line above." % (fp64_name, nfh, dh))
        for i, (a, b) in enumerate(zip(fp64, base)):
            if a["flag"] != b["flag"] or dtau(a["tau"], b["tau"]) > args.tol:
                print("   %-28s %s flag=%d tau=%12.6g | device flag=%d tau=%12.6g"
                      % (cases[i]["name"], fp64_name, a["flag"], a["tau"],
                         b["flag"], b["tau"]))
        for cls in ("flat", "zero", "below", "ctrl", "pool"):
            idx = [i for i, c in enumerate(cases) if case_class(c["name"]) == cls]
            if not idx:
                continue
            print("[cf_scan_ab]   device %-5s flags=%s  us min %.1f median %.1f max %.1f"
                  % (cls, sorted({base[i]["flag"] for i in idx}),
                     min(base[i]["us"] for i in idx),
                     statistics.median(base[i]["us"] for i in idx),
                     max(base[i]["us"] for i in idx)))
    if args.keep:
        print("[cf_scan_ab] generated sources kept in %s" % workdir)
    print("[cf_scan_ab] %s" % ("FAIL" if bad else "OK"))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
