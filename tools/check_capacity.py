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

    rows.append(("DECOMP_MAXM (decomp private array rows)", [
        (num(r"#define\s+DECOMP_MAXM\s+(\d+)", dc, "decomp.cl", int),
         "kernels/decomp.cl #define DECOMP_MAXM"),
        (num(r"kDecompMaxM\s*=\s*(\d+)", rt, "hpocl_runtime.cpp", int),
         "runtime/hpocl_runtime.cpp kDecompMaxM"),
    ]))
    rows.append(("DECOMP_MAXD (decomp private array streams)", [
        (num(r"#define\s+DECOMP_MAXD\s+(\d+)", dc, "decomp.cl", int),
         "kernels/decomp.cl #define DECOMP_MAXD"),
        (num(r"kDecompMaxD\s*=\s*(\d+)", rt, "hpocl_runtime.cpp", int),
         "runtime/hpocl_runtime.cpp kDecompMaxD"),
    ]))

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
