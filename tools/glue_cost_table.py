#!/usr/bin/env python3
"""glue_cost_table: the glue-offload decision, re-derived at the REAL shapes.

`runtime/hpocl_schedule.cpp` decides whether each of the three Nt-class glue
GEMMs (H_k F_RF, F_RF N_k, F_RF F_BB) runs on the PS or as one wbb_apply launch,
with a cheap proxy rule: offload iff flops >= kGemmMinFlops. The proxy exists
because the real question is a TIME comparison

    F/kHostFlopRate  >  F/kDevFlopRate + LAUNCH + E*kConvFlopsPerElem/kHostFlopRate

with E = M*N + SC*N + SC*M the fp64<->fp32 staging the kernel interface forces on
the PS, and a proxy that quietly disagrees with it at some deployed shape would be
a cost model that is only true where nobody looks.

So this script is the check: it reads the constants and kGemmMinFlops out of
runtime/hpocl_schedule.h (one definition, no copy here), enumerates every distinct
config in mix_precode/*/config_pools/*.json -- the same envelope
tools/check_capacity.py gates -- and asserts the proxy agrees with the time model
evaluated at kLaunchWorstS, the TOP of the documented 10-50 us launch range. That
is the conservative direction: a site is offloaded only if offloading still wins
when a launch costs 50 us, so no decision can be inverted by the one constant in
the model that is a range instead of a measurement.

It then reports, informationally, the sites the mid-range kLaunchS WOULD offload
and the rule does not, with the microseconds at stake -- so the conservatism is a
number rather than a mood. `--launch <seconds>` re-derives everything for a
measured board launch overhead, which is the first thing to re-run once the XRT
number exists (docs/QX49DR_SV3_上板依赖清单.md).

Pure text + arithmetic: no build, no device. Run it after touching the cost rule.

    python3 tools/glue_cost_table.py                  # table + agreement check
    python3 tools/glue_cost_table.py --quiet          # agreement check only
    python3 tools/glue_cost_table.py --launch 12e-6   # with a measured launch
"""
import glob
import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def read(*parts):
    with open(os.path.join(ROOT, *parts), encoding="utf-8") as f:
        return f.read()


def konst(name, text):
    m = re.search(r"inline constexpr double\s+%s\s*=\s*([0-9.eE+-]+)" % name, text)
    if not m:
        raise SystemExit("[glue_cost] FAIL cannot find %s in hpocl_schedule.h" % name)
    return float(m.group(1))


def configs():
    """Every distinct (K, Nt, Nr, NRF, d) in the pools, with its source file."""
    out = []
    pools = sorted(glob.glob(os.path.join(ROOT, "..", "mix_precode", "*",
                                          "config_pools", "*.json")))
    if not pools:
        raise SystemExit("[glue_cost] FAIL no config pool under "
                         "mix_precode/*/config_pools -- nothing to cost")
    for pf in pools:
        with open(pf, encoding="utf-8") as f:
            cfgs = json.load(f)
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
            if t not in [r[1] for r in out]:
                out.append((os.path.basename(pf), t))
    return out


def sites(K, Nt, Nr, NRF, d):
    """(name, M, N, SC) per site, exactly as ref/pipeline.cpp calls gemm.run().

    eff   : W = F_RF^T [NRF x Nt],        X = H    [K*Nr x Nt]   -> [K*Nr x NRF]
    FN    : W = stack_k(N_k^T) [K*nul x NRF], X = F_RF [Nt x NRF] -> [Nt x K*nul]
    power : W = F_BB^T [T x NRF],         X = F_RF [Nt x NRF]    -> [Nt x T]
    `nul` is the BD nullity NRF - (K-1)*Nr, taken from the GEOMETRY the way
    front_end() takes it (see the comment there on why it is not a rank test).
    """
    nul = NRF - (K - 1) * Nr
    T = K * d
    return [("H*F_RF", NRF, Nt, K * Nr),
            ("F_RF*N_k", K * nul, NRF, Nt),
            ("F_RF*F_BB", T, NRF, Nt)]


def main():
    quiet = "--quiet" in sys.argv
    h = read("runtime", "hpocl_schedule.h")
    HOST = konst("kHostFlopRate", h)
    DEV = konst("kDevFlopRate", h)
    MID = konst("kLaunchS", h)
    CONV = konst("kConvFlopsPerElem", h)
    MINF = konst("kGemmMinFlops", h)
    LAUNCH = konst("kLaunchWorstS", h)
    if "--launch" in sys.argv:
        LAUNCH = float(sys.argv[sys.argv.index("--launch") + 1])
        print("[glue_cost] launch overridden on the command line: %.3g s" % LAUNCH)
    print("[glue_cost] constants from runtime/hpocl_schedule.h: host %.3g flop/s, "
          "device %.3g flop/s, launch %.3g s (worst; mid-range %.3g s), staging "
          "%.3g flop/elem, kGemmMinFlops %.3g" % (HOST, DEV, LAUNCH, MID, CONV, MINF))
    for nm, L in (("worst-case", LAUNCH), ("mid-range", MID)):
        print("[glue_cost] crossover ignoring staging, %s launch: F > %.3g flop"
              % (nm, HOST * L / (1.0 - HOST / DEV)))

    cfgs = configs()
    bad, conservative, noff, total = [], [], 0, 0
    if not quiet:
        print("\n%-6s %-2s %-4s %-3s %-4s %-2s %-4s | %-9s %9s %9s  rule" %
              ("pool", "K", "Nt", "Nr", "NRF", "d", "nul", "site",
               "host us", "dev us"))
    for src, (K, Nt, Nr, NRF, d) in sorted(cfgs, key=lambda r: -r[1][1]):
        if (K - 1) * Nr >= min(Nt, NRF):
            bad.append("%s K=%d Nt=%d: RF-BD infeasible, no glue to cost"
                       % (src, K, Nt))
            continue
        for nm, M, N, SC in sites(K, Nt, Nr, NRF, d):
            total += 1
            if M <= 0 or N <= 0 or SC <= 0:
                bad.append("%s K=%d Nt=%d site %s: non-positive shape %dx%dx%d"
                           % (src, K, Nt, nm, M, N, SC))
                continue
            F = 8.0 * M * N * SC                 # complex MAC = 8 flops
            E = M * N + SC * N + SC * M
            th = F / HOST
            td = F / DEV + LAUNCH + E * CONV / HOST
            td_mid = F / DEV + MID + E * CONV / HOST
            proxy = F >= MINF
            model = th > td                      # at the WORST launch
            if not quiet:
                print("%-6s %-2d %-4d %-3d %-4d %-2d %-4d | %-9s %9.1f %9.1f  %s%s"
                      % (src[:6], K, Nt, Nr, NRF, d, NRF - (K - 1) * Nr, nm,
                         th * 1e6, td * 1e6, "DEV" if proxy else "ps",
                         "" if proxy == model else "   <-- DISAGREE"))
            if proxy:
                noff += 1
            if proxy != model:
                bad.append("%s K=%d Nt=%d Nr=%d NRF=%d d=%d %s: proxy=%s "
                           "model@worst-launch=%s (host %.1f us vs device %.1f us)"
                           % (src, K, Nt, Nr, NRF, d, nm, proxy, model,
                              th * 1e6, td * 1e6))
            elif not proxy and th > td_mid:
                conservative.append(
                    "%s K=%d Nt=%d Nr=%d NRF=%d d=%d %s: F = %.3f Mflop, kept on "
                    "the PS; the mid-range launch would save %.1f us (%.1f%% of "
                    "the host cost)" % (src, K, Nt, Nr, NRF, d, nm, F / 1e6,
                                        (th - td_mid) * 1e6,
                                        100 * (th - td_mid) / th))
    print("\n[glue_cost] %d distinct configs, %d sites: %d offloaded, %d kept on "
          "the PS; %d site(s) the mid-range launch would also offload; %d "
          "disagreement(s) with the worst-case model"
          % (len(cfgs), total, noff, total - noff, len(conservative), len(bad)))
    for w in conservative:
        print("  conservative by design: %s" % w)
    if bad:
        for b in bad:
            print("  FAIL %s" % b)
        raise SystemExit("[glue_cost] FAIL: the flops proxy does not implement the "
                         "worst-case time model at every real shape")
    print("[glue_cost] OK: the proxy rule and the worst-case-launch model agree on "
          "all %d sites of all %d real configs" % (total, len(cfgs)))


if __name__ == "__main__":
    main()
