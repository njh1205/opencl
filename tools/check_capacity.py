#!/usr/bin/env python3
"""capacity_check: one constant, N copies, zero drift.

Several capacity/algorithm constants in this stack MUST have the same value in
more than one place, because the device kernel sizes a private/local array with
it while the host wrapper validates the caller's shape against it (or because
the fp32 kernel, the fp64 C++ reference and the numpy oracle have to agree on an
algorithm parameter). A mismatch does not fail loudly: the kernel returns early
and leaves the output buffers untouched, or the three implementations quietly
solve different problems.

History, i.e. why this gate exists: TQ (the Gauss-Hermite span) lived in three
files and was 10 in the kernel while the host and the oracle said 5, and
NEWTON_ITERS lived in three files too. Both were changed by hand, in sequence,
and the drift was only caught by an accuracy regression.

This script is pure text analysis: no build, no device. It parses each definition
site, prints the value it found and where, and fails on any disagreement.
"""
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def read(*parts):
    with open(os.path.join(ROOT, *parts), encoding="utf-8") as f:
        return f.read()


def num(pattern, text, where, cast=float):
    m = re.search(pattern, text, re.M)
    if not m:
        raise SystemExit("[capacity_check] FAIL cannot find %s in %s" % (pattern, where))
    return cast(m.group(1))


def main():
    rt = read("runtime", "hpocl_runtime.cpp")
    dp = read("kernels", "datapath.cl")
    dc = read("kernels", "decomp.cl")
    pi = read("kernels", "power_iter.cl")
    th = read("kernels", "thp_precoder.cl")
    mc = read("kernels", "mcs_table.clh")
    cf = read("kernels", "threshold_cf.cl")
    rr = read("tools", "rate_ref.py")
    cr = read("tools", "cf_ref.py")
    rc = read("ref", "threshold_cf.cpp")
    rh = read("ref", "threshold_cf.h")
    sh = read("opencl", "build", "build_xclbin.sh")

    # (label, [(value, where), ...]) -- every value in a row must be equal
    rows = []

    pmax = num(r"#define\s+PMAX\s+(\d+)", pi, "power_iter.cl", int)
    rows.append(("PMAX (power_iter local/private vector bound)", [
        (pmax, "kernels/power_iter.cl #define PMAX"),
        (num(r"kPowerIterMaxM\s*=\s*(\d+)", rt, "hpocl_runtime.cpp", int),
         "runtime/hpocl_runtime.cpp kPowerIterMaxM"),
    ]))

    # decomp.cl has ONE CEILING PAIR PER KERNEL, because a static __private
    # array is allocated at its ceiling for every work-item whether or not the
    # launch uses it: sharing jacobi_svd's 32x40 with gmd_k/ucd_k/cbd_k would
    # cost them 12.5x/21.7x/14.4x in private footprint for geometries they never
    # see. Each pair is mirrored by a k*Max* constant in the host wrapper that
    # refuses the launch (and NaN-fills) before the kernel can silently return
    # untouched buffers.
    for mac, host, who in (("DECOMP", "kJacobiMax", "jacobi_svd rows/cols"),
                           ("GMD", "kGmdMax", "gmd_k+ucd_k rows/streams"),
                           ("CBD", "kCbdMax", "cbd_k urows+vrows/streams")):
        for ax in ("M", "D"):
            rows.append(("%s_MAX%s (%s)" % (mac, ax, who), [
                (num(r"#define\s+%s_MAX%s\s+(\d+)" % (mac, ax), dc, "decomp.cl", int),
                 "kernels/decomp.cl #define %s_MAX%s" % (mac, ax)),
                (num(r"%s%s\s*=\s*(\d+)" % (host, ax), rt, "hpocl_runtime.cpp", int),
                 "runtime/hpocl_runtime.cpp %s%s" % (host, ax)),
            ]))

    # The ceilings are only useful if they actually COVER the geometries the
    # control plane feeds them. This is the gate that was missing when
    # jacobi_svd shipped at 16x8 and silently refused the deployed front end
    # (null_basis factorises A [(K-1)*Nr x NRF] = 28x32, analog_precoder
    # factorises H_u [Nr x Nt] = 4x16): nothing in `make check` exercised those
    # shapes, so a NaN-filled basis looked like a passing build. Every case in
    # tools/pipeline_ref.py and every case in runtime/sched_main.cpp is checked
    # against all four SVD/decomp call sites the pipeline can express.
    jacm = num(r"#define\s+DECOMP_MAXM\s+(\d+)", dc, "decomp.cl", int)
    jacd = num(r"#define\s+DECOMP_MAXD\s+(\d+)", dc, "decomp.cl", int)
    gmdm = num(r"#define\s+GMD_MAXM\s+(\d+)", dc, "decomp.cl", int)
    gmdd = num(r"#define\s+GMD_MAXD\s+(\d+)", dc, "decomp.cl", int)
    geoms = []          # (where, K, Nt, Nr, NRF, d)
    pl = read("tools", "pipeline_ref.py")
    m = re.search(r"^CASES\s*=\s*\[(.*?)\]", pl, re.M | re.S)
    if not m:
        raise SystemExit("[capacity_check] FAIL cannot find CASES in tools/pipeline_ref.py")
    for tup in re.findall(r"\(([^)]*)\)", m.group(1)):
        v = [int(x) for x in tup.replace(" ", "").split(",")]
        if len(v) >= 5:
            geoms.append(("tools/pipeline_ref.py CASES", v[0], v[1], v[2], v[3],
                          v[4], False))
    # SCHED_CASE(K, Nt, Nr, NRF, d, M, variant, gamma_dB): only the first five
    # are geometry, and gamma is a float, so slice before converting.
    sm = read("runtime", "sched_main.cpp")
    # A SCHED_CASE whose line is marked INFEASIBLE is a deliberate guard case --
    # it exists so sched_bench proves front_end()'s fe_flag fires -- and must be
    # allowed to fail feasibility. Requiring the marker in the source line is
    # what stops a genuinely broken config from being waved through as a "guard".
    for line in sm.splitlines():
        if "SCHED_CASE(" not in line:
            continue
        tup = line[line.index("SCHED_CASE(") + len("SCHED_CASE("):]
        tup = tup[:tup.index(")")] if ")" in tup else tup
        v = tup.replace(" ", "").split(",")[:5]
        if len(v) == 5 and all(x.lstrip("-").isdigit() for x in v):
            v = [int(x) for x in v]
            geoms.append(("runtime/sched_main.cpp SCHED_CASE",
                          v[0], v[1], v[2], v[3], v[4],
                          "INFEASIBLE" in line))
    if not geoms:
        raise SystemExit("[capacity_check] FAIL found no pipeline geometry to check")
    def check_geom(where, K, Nt, Nr, NRF, d, out, allow_infeasible=False):
        nul = NRF - (K - 1) * Nr
        rows = (K - 1) * Nr
        if nul < d:
            out.append("%s K=%d Nr=%d NRF=%d gives nul=%d < d=%d: the BD basis "
                       "cannot carry the streams" % (where, K, Nr, NRF, nul, d))
        # allow_infeasible is set ONLY for a sched_main case whose source line
        # is marked INFEASIBLE: that case exists so sched_bench proves
        # front_end()'s fe_flag fires. Everything else -- every real config in
        # every pool, every pipeline_ref case -- must be feasible, because an
        # infeasible geometry is not a hard configuration, it is one whose
        # schedule is numerical noise (F_RF*N_k == 0).
        if rows >= min(Nt, NRF) and not allow_infeasible:
            out.append("%s K=%d Nt=%d Nr=%d NRF=%d: RF-BD INFEASIBLE, "
                       "(K-1)*Nr=%d >= min(Nt,NRF)=%d, so rank(A)=%d and "
                       "F_RF*N_k==0 (fe_flag=1); this shape is a guard case, "
                       "not a configuration" % (where, K, Nt, Nr, NRF, rows,
                       min(Nt, NRF), min(Nt, NRF)))
        # The four decompositions the front end performs, priced by the
        # ALGORITHM each one actually gets (ref/pipeline.cpp's provider
        # dispatch): the analog step asks for take < n leading columns of a wide
        # H_u, so it is served by svd_wide()'s Gram form, which never builds an
        # Nt x Nt array and is bounded by Nr. The other three ask for the full
        # basis and go through one-sided Jacobi, so they ARE bounded by the
        # DECOMP_ ceilings.
        for name, nrow, ncol, lim_m, lim_d, mac in (
                ("analog svd_wide(H_u)", Nr, Nr, jacm, jacm, "DECOMP"),
                ("null_basis svd(A)", rows, NRF, jacm, jacd, "DECOMP"),
                ("power_whitening svd(C)", nul, nul, jacm, jacd, "DECOMP"),
                ("gbar svd(Gbar)", Nr, nul, jacm, jacd, "DECOMP"),
                ("gmd/ucd design", nul, d, gmdm, gmdd, "GMD")):
            if nrow > lim_m or ncol > lim_d:
                out.append("%s: %s is %dx%d, over %s_MAXM/MAXD = %dx%d"
                           % (where, name, nrow, ncol, mac, lim_m, lim_d))

    worst, pool_worst = [], []
    for where, K, Nt, Nr, NRF, d, guard in geoms:
        check_geom(where, K, Nt, Nr, NRF, d, worst, guard)
    rows.append(("front-end geometry fits the decomp ceilings (%d cases)" % len(geoms),
                 [(0, "no case over capacity")] + [(-1, w) for w in worst]))

    # The REAL envelope. Everything above checks the shapes the benches happen
    # to use; this checks the shapes the USER'S configurations actually have,
    # read straight out of mix_precode/{classical,ai}/config_pools/*.json. It
    # exists because a cost model and a capacity ceiling built from an invented
    # geometry are both worthless: the geometry the README used to call
    # "deployed" (K=8, Nt=16, Nr=4, NRF=32, d=4) is in NO pool and is RF-BD
    # INFEASIBLE ((K-1)*Nr = 28 >= min(Nt,NRF) = 16, so F_RF*N_k == 0 and the
    # whitened spectrum is numerical noise -- production python degenerates the
    # same way). Every config in every pool is feasible; this gate fails if a
    # future one is not, or if it stops fitting a ceiling.
    import glob, json
    pools = sorted(glob.glob(os.path.join(ROOT, "..", "mix_precode", "*",
                                          "config_pools", "*.json")))
    real = []
    for pf in pools:
        try:
            with open(pf, encoding="utf-8") as f:
                cfgs = json.load(f)
        except Exception as e:                       # unreadable pool: say so
            worst.append("%s: cannot parse (%s)" % (os.path.basename(pf), e))
            continue
        if isinstance(cfgs, dict):
            cfgs = cfgs.get("configs", cfgs.get("pool", []))
        for c in cfgs:
            if not isinstance(c, dict):
                continue
            try:
                t = (int(c["num_users"]), int(c["num_tx_antennas"]),
                     int(c["num_rx_antennas"]), int(c["num_rf_chains"]),
                     int(c["num_streams_per_user"]))
            except (KeyError, TypeError, ValueError):
                continue
            if t not in [r[1] for r in real]:
                real.append((os.path.relpath(pf, ROOT), t))
    if not real:
        raise SystemExit("[capacity_check] FAIL found no config pool under "
                         "mix_precode/*/config_pools -- the real envelope is "
                         "unchecked")
    maxt = num(r"#define\s+MAXT\s+(\d+)", cf, "threshold_cf.cl", int)
    for src, (K, Nt, Nr, NRF, d) in real:
        tag = "%s K=%d Nt=%d Nr=%d NRF=%d d=%d" % (src, K, Nt, Nr, NRF, d)
        check_geom(tag, K, Nt, Nr, NRF, d, pool_worst)
        if K * d > maxt:
            pool_worst.append("%s: T=K*d=%d > MAXT=%d" % (tag, K * d, maxt))
    rows.append(("real config pools fit the ceilings and RF-BD feasibility "
                 "(%d distinct configs)" % len(real),
                 [(0, "every config feasible and in capacity")] +
                 [(-1, w) for w in pool_worst]))

    kmax = num(r"#define\s+KMAX\s+(\d+)", th, "thp_precoder.cl", int)
    rows.append(("KMAX (THP feedback history)", [
        (kmax, "kernels/thp_precoder.cl #define KMAX"),
        (num(r"kThpMaxK\s*=\s*(\d+)", rt, "hpocl_runtime.cpp", int),
         "runtime/hpocl_runtime.cpp kThpMaxK"),
    ]))

    # MMAX is a constellation bound on the device and a bits-per-symbol bound on
    # the host: the host rejects m > 8, and 2^8 == MMAX must hold exactly.
    mmax = num(r"#define\s+MMAX\s+(\d+)", dp, "datapath.cl", int)
    mhost = num(r"\(m & 1\) \|\| m > (\d+)", rt, "hpocl_runtime.cpp", int)
    rows.append(("MMAX (constellation size vs host bits-per-symbol cap)", [
        (mmax, "kernels/datapath.cl #define MMAX"),
        (1 << mhost, "runtime/hpocl_runtime.cpp dp_dims_ok (1 << %d)" % mhost),
    ]))

    # The LLR candidate cache is bounded by LLR_CACHE_MAX; its default must stay
    # MMAX so the shipped build takes the cached path for every supported
    # constellation, and a PL build lowers it with -DLLR_CACHE_MAX (gated by
    # `make datapath_pl_bench`).
    default_cache = (mmax if re.search(r"#ifndef LLR_CACHE_MAX\s*\n"
                                       r"#define LLR_CACHE_MAX MMAX", dp) else -1)
    rows.append(("LLR_CACHE_MAX default (softdemod private cache)", [
        (mmax, "kernels/datapath.c…AX is"),
        (default_cache, "kernels/datapath.cl #define LLR_CACHE_MAX (default)"),
    ]))

    # threshold_cf has its own MAXT/MAXK/MAXD triple, unrelated to decomp.cl's
    # MAXM/MAXD: T = K*d streams and the per-LLR distance-candidate array.
    rows.append(("MAXT (threshold_cf streams per channel)", [
        (num(r"#define\s+MAXT\s+(\d+)", cf, "threshold_cf.cl", int),
         "kernels/threshold_cf.cl #define MAXT"),
        (num(r"kCfMaxT\s*=\s*(\d+)", rt, "hpocl_runtime.cpp", int),
         "runtime/hpocl_runtime.cpp kCfMaxT"),
    ]))
    rows.append(("MAXD (threshold_cf distance candidates)", [
        (num(r"#define\s+MAXD\s+(\d+)", cf, "threshold_cf.cl", int),
         "kernels/threshold_cf.cl #define MAXD"),
        (num(r"kCfMaxD\s*=\s*(\d+)", rt, "hpocl_runtime.cpp", int),
         "runtime/hpocl_runtime.cpp kCfMaxD"),
    ]))

    cqn = num(r"#define\s+HPOCL_CQI_N\s+(\d+)", mc, "mcs_table.clh", int)
    rows.append(("HPOCL_CQI_N (CQI table length)", [
        (cqn, "kernels/mcs_table.clh #define HPOCL_CQI_N"),
        (len(re.findall(r"\d+", re.search(r"CQI_M = np\.array\(\[(.*?)\]", rr, re.S).group(1))),
         "tools/rate_ref.py CQI_M"),
    ]))
    gmax = num(r"#define\s+HPOCL_GMI_MAX\s+([\d.]+)f?", mc, "mcs_table.clh")
    rows.append(("HPOCL_GMI_MAX (non-finite / out-of-range GMI refusal)", [
        (gmax, "kernels/mcs_table.clh #define HPOCL_GMI_MAX"),
        (num(r"& \(g <= ([\d.]+)\)", rr, "rate_ref.py"), "tools/rate_ref.py select()"),
    ]))

    tq = num(r"#define\s+TQ\s+\(\(real\)([\d.]+)\)", cf, "threshold_cf.cl")
    rows.append(("TQ (Gauss-Hermite span)", [
        (tq, "kernels/threshold_cf.cl #define TQ"),
        (num(r"kTQ\s*=\s*([\d.]+)", rc, "ref/threshold_cf.cpp"), "ref/threshold_cf.cpp kTQ"),
        (num(r"^TQ\s*=\s*([\d.]+)", cr, "tools/cf_ref.py"), "tools/cf_ref.py TQ"),
    ]))

    # The near-flat fallback grid. kernels/psi_scan.clh is ONE loop text compiled
    # into both sides (device via Runtime's .clh inliner, host via -Ikernels), and
    # tools/cf_ref.py root_u carries a verbatim port of it -- so the oracle is the
    # only site that can drift, which is exactly how TQ ended up 10 in the kernel
    # and 5 everywhere else. It drifted before this gate existed, in the other
    # direction and without anyone noticing: the device scanned 161 points and the
    # host plus the oracle 401, which agrees on tau to 0.0004 dB (README's
    # fallback-scan decision record) but can disagree on the FLAG -- 1 vs 2 is a
    # finite tau against +-inf, i.e. a different decision.
    # PSI_SCAN_N has three sites, not two: ref/threshold_cf.cpp carries a
    # static_assert that spells the number, so moving the resolution fails the PS
    # compile with the re-measurement procedure in its message.
    ps = read("kernels", "psi_scan.clh")
    rows.append(("PSI_SCAN_N (fallback grid points, incl. both endpoints)", [
        (num(r"#define\s+PSI_SCAN_N\s+(\d+)", ps, "psi_scan.clh", int),
         "kernels/psi_scan.clh #define PSI_SCAN_N"),
        (num(r"static_assert\(PSI_SCAN_N == (\d+)", rc, "ref/threshold_cf.cpp", int),
         "ref/threshold_cf.cpp static_assert(PSI_SCAN_N == ...)"),
        (num(r"^PSI_SCAN_N\s*=\s*(\d+)", cr, "tools/cf_ref.py", int),
         "tools/cf_ref.py PSI_SCAN_N"),
    ]))
    for mac in ("PSI_SCAN_LO", "PSI_SCAN_HI"):
        rows.append(("%s (fallback grid endpoint in u)" % mac, [
            (num(r"#define\s+%s\s+([\d.eE+-]+)" % mac, ps, "psi_scan.clh"),
             "kernels/psi_scan.clh #define %s" % mac),
            (num(r"^%s\s*=\s*([\d.eE+-]+)" % mac, cr, "tools/cf_ref.py"),
             "tools/cf_ref.py %s" % mac),
        ]))
    # The dead band is precision-keyed ON PURPOSE: 1e-3 in the fp32 kernel (Psi's
    # own fp32 round-off floor) against 1e-6 in fp64, and README's fallback-scan
    # decision record measures tau moving 0.0000 dB between the two. What must NOT
    # drift is the fp64 value, which three sites now share: the kernel's
    # #ifdef HPOCL_FP64 branch, ref/threshold_cf.cpp kPsiBand (it used to be a bare
    # `eps = 1e-6` local of scan_root) and the oracle's PSI_BAND.
    rows.append(("PSI_BAND fp64 (Psi dead band; fp32 kernel is 1e-3 by design)", [
        (num(r"#ifdef HPOCL_FP64[\s\S]*?#define PSI_BAND \(real\)([\d.eE+-]+)",
             cf, "threshold_cf.cl"),
         "kernels/threshold_cf.cl #define PSI_BAND (fp64 branch)"),
        (num(r"kPsiBand\s*=\s*([\d.eE+-]+)", rc, "ref/threshold_cf.cpp"),
         "ref/threshold_cf.cpp kPsiBand"),
        (num(r"^PSI_BAND\s*=\s*([\d.eE+-]+)", cr, "tools/cf_ref.py"),
         "tools/cf_ref.py PSI_BAND"),
    ]))

    ni = num(r"#define\s+NEWTON_ITERS\s+(\d+)", cf, "threshold_cf.cl", int)
    rows.append(("NEWTON_ITERS (safeguarded Newton budget)", [
        (ni, "kernels/threshold_cf.cl #define NEWTON_ITERS"),
        (num(r"newton_iters\s*=\s*(\d+)", rh, "ref/threshold_cf.h", int),
         "ref/threshold_cf.h CfConfig::newton_iters"),
        (num(r"-DNEWTON_ITERS=(\d+)", sh, "build_xclbin.sh", int),
         "opencl/build/build_xclbin.sh COMMON"),
    ]))

    bad = 0
    for label, sites in rows:
        vals = {v for v, _ in sites}
        if len(vals) == 1:
            v = sites[0][0]
            vs = ("%g" % v) if isinstance(v, float) else str(v)
            print("[capacity_check] %-52s = %-6s (%d sites agree)"
                  % (label, vs, len(sites)))
        else:
            bad += 1
            print("[capacity_check] FAIL %s disagrees:" % label)
            for v, where in sites:
                print("                 %-10s %s" % (v, where))
    if bad:
        print("[capacity_check] %d constant group(s) out of step" % bad)
        return 1
    print("[capacity_check] OK: %d shared constants, %d definition sites, all agree"
          % (len(rows), sum(len(s) for _, s in rows)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
