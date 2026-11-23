#!/usr/bin/env python3
"""fec_bler: the uncoded-vs-coded waterfall for the FEC layer. A MEASUREMENT, not
a gate -- `make check` does not run it, because a BER is a statistic and a gate has
to be a statement that can be false.

What it measures, and with what:
  TX   info[L] -> CRC-16 -> (133,171)_8 K=7 rate-1/2 conv encoder + zero tail
       -> puncture to 2/3 or 3/4 -> S-random interleave -> m-bit labels of a
       unit-power Gray square QAM (tools/modem_common.qam_map)
  CH   flat AWGN, n0 = E|noise|^2 per complex sample, Es = 1
  RX   exact logsumexp LLRs over the constellation (modem_common.soft_llr, the
       same estimator kernels/datapath.cl::softdemod_llr_k implements) ->
       deinterleave -> depuncture (erased positions get llr = 0) -> Viterbi
  The coded path IS tools/fec_ref.py's numpy oracle, which runtime/fec_main.cpp
  gates bit-exact against ref/fec.cpp, so this measures the decoder the C++
  reference implements rather than a third, unvetted one.

Eb/N0 convention (this repo's n0; not a claim about Sionna's normalisation):
  Es = 1 (unit average symbol power), n0 = E|noise|^2 per complex sample, so
  Es/N0 = 1/n0 and Eb/N0[dB] = -10log10(n0) - 10log10(m * R_eff), with
  R_eff = L/npunc for a coded frame and 1 for the uncoded one. sionna.phy.utils.
  ebnodb2no plays the same role; whether its `no` is per complex sample or per
  real dimension is a Sionna detail nothing here depends on.

Statistics, printed with every point, because a BER without an error bar is not a
number (README makes the same point about the GMI noise floor):
  * frames, info bits and symbols actually simulated, and the seed they came from
  * the two-sided 95% binomial half width 1.96*sqrt(p(1-p)/N); when zero block
    errors were seen, the rule of three instead (95% upper bound 3/N), because a
    zero has no half width
  * the resolution floor 1/N
  * early stop at --min-errors block errors, so every non-floored point carries at
    least that many errors and the relative CI is bounded by 1.96/sqrt(min-errors)
Each point derives its own stream from (seed, Eb/N0), so one row is reproducible
without re-running the sweep that found it.

Coding gain is quoted at BLER = --target (default 1e-2; Sionna's compute_bler
convention: a block error is any differing element of the block) by log-linear
interpolation between the two measured points that bracket it, with the CIs of both
endpoints propagated into the printed +-.

Usage:
  python3 tools/fec_bler.py                      # full sweep
  python3 tools/fec_bler.py --quick              # ~1/5 of the frames
  python3 tools/fec_bler.py --m 4 --rates 2/3 --frames 2000 --seed 7
"""
import argparse
import os
import re
import sys
import time

import numpy as np
from scipy.special import erfc

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fec_ref as fr
import modem_common as mc

K_CONSTRAINT = fr.K_CONSTRAINT
CID = 2                                  # CRC-16 on every coded frame
Z95 = 1.959963984540054                  # two-sided 95% normal quantile


# ---------------------------------------------------------------------------
# the CQI table, parsed out of kernels/mcs_table.clh rather than retyped: one
# definition of a standard table is the whole point of that header.
# ---------------------------------------------------------------------------
def load_cqi_table(path=None):
    if path is None:
        path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "kernels", "mcs_table.clh")
    txt = open(path).read()
    m = re.search(r"kCqiM\[[^\]]*\]\s*=\s*\{([^}]*)\}", txt)
    r = re.search(r"kCqiR\[[^\]]*\]\s*=\s*\{([^}]*)\}", txt)
    if not m or not r:
        raise SystemExit("[fec_bler] cannot parse kernels/mcs_table.clh")
    cm = np.array([int(x) for x in m.group(1).split(",")], np.int64)
    cr = np.array([int(x) for x in r.group(1).split(",")], np.int64)
    if len(cm) != 16 or len(cr) != 16 or cm[0] != 0 or cr[0] != 0:
        raise SystemExit("[fec_bler] mcs_table.clh did not parse to 16 (m,R) pairs")
    se = cm.astype(float) * cr.astype(float) / 1024.0
    return cm, cr, se


def nearest_cqi(cm, cr, se, m, r_eff):
    """The CQI entry with this modulation order whose m*R/1024 is closest to the
    m*R_eff this convolutional code actually delivers."""
    sel = np.flatnonzero(cm == m)
    if len(sel) == 0:
        return None
    j = int(sel[np.argmin(np.abs(se[sel] - m * r_eff))])
    return j, int(cm[j]), int(cr[j]), float(se[j])


# ---------------------------------------------------------------------------
# Eb/N0 <-> n0, and the exact uncoded BER this chain has to reproduce
# ---------------------------------------------------------------------------
def ebnodb2n0(ebno_db, m, r_eff):
    return 10.0 ** (-(ebno_db + 10.0 * np.log10(m * r_eff)) / 10.0)


def pam_gray_ber_exact(m, n0):
    """Exact per-bit error rate of Gray-labelled square QAM over flat AWGN.

    Built from the per-axis L-PAM decision regions rather than from a textbook
    approximation: P(decide i | sent j) is the Gaussian mass of level i's decision
    interval, and the bit-error probability sums over the levels whose Gray label
    disagrees in that bit position. This exists as an independent check of the whole
    normalisation chain -- an Es/N0-vs-Eb/N0 factor of two, a wrong n0/2 split or a
    mislabelled constellation all show up here as a measured-vs-analytic BER off by
    a constant, while the waterfall shape still looks plausible.
    """
    mh = m // 2
    L = 1 << mh
    d = mc.pam_delta(m)                    # unit average symbol power
    a = (2 * np.arange(L) - (L - 1)) * (d / 2.0)
    lab = mc.bit_cols(np.arange(L) ^ (np.arange(L) >> 1), mh)    # Gray, MSB first
    sig = np.sqrt(n0 / 2.0)                # noise std per real dimension
    thr = np.concatenate([[-np.inf], (a[:-1] + a[1:]) / 2.0, [np.inf]])
    phi = lambda x: 0.5 * erfc(-x / np.sqrt(2.0))
    P = (phi((thr[1:, None] - a[None, :]) / sig)
         - phi((thr[:-1, None] - a[None, :]) / sig))             # P[i, j]
    per_bit = []
    for b in range(mh):
        diff = lab[:, b][:, None] != lab[:, b][None, :]          # [i, j]
        per_bit.append((P * diff).sum(axis=0).mean())            # uniform over j
    return float(np.mean(per_bit))


# ---------------------------------------------------------------------------
# one waterfall curve
# ---------------------------------------------------------------------------
def pick_L(m, pid, target):
    """Smallest L >= target whose punctured length is a whole number of m-bit
    labels, so no padding bit is transmitted and R_eff = L/npunc is exact."""
    L = target
    while True:
        _, _, ncoded, npunc = fr.geom(L, CID, pid)
        if npunc % m == 0:
            return L, ncoded, npunc
        L += 1


class Curve:
    def __init__(self, m, pid, L_target, seed, batch, max_frames, min_errors,
                 coded=True, probe_frames=None):
        self.m, self.pid, self.seed = m, pid, seed
        self.coded = coded
        self.batch = batch
        self.max_frames = max_frames
        self.probe_frames = probe_frames or max(256, max_frames // 5)
        self.min_errors = min_errors
        if coded:
            self.L, self.ncoded, self.npunc = pick_L(m, pid, L_target)
            self.name = "coded %d-QAM, rate %d/%d" % (1 << m, *fr.puncture_rate(pid))
            self.perm, self.par_eff, self.att = fr.make_interleaver(
                2, self.npunc, 8, seed)
            self.crc_mat = fr.crc_matrix(self.L, fr.CRC[CID][1], fr.CRC[CID][2])
        else:
            # The uncoded reference carries the SAME information payload per frame,
            # so the two BLERs are comparable at equal information content and not
            # merely at equal symbol count. npunc is padded up to a whole number of
            # labels; the padding bits are transmitted and counted in R_eff, which
            # is the honest thing to do (they cost energy and carry nothing).
            self.L = L_target
            self.npunc = -(-self.L // m) * m
            self.pad = self.npunc - self.L
            self.ncoded = self.npunc
            self.name = "uncoded %d-QAM" % (1 << m)
            self.perm = np.arange(self.npunc, dtype=np.int32)
            self.par_eff, self.att = 0, 1
        self.r_eff = self.L / self.npunc

    # -- TX / RX ------------------------------------------------------------
    def encode(self, info):
        if not self.coded:
            return (np.concatenate([info, np.zeros((len(info), self.pad), np.uint8)],
                                   axis=1) if self.pad else info)
        coded = fr.conv_encode_batch(info, CID)
        return fr.interleave(coded[:, fr.puncture_mask(self.ncoded, self.pid)],
                             self.perm)

    def decode(self, llr):
        if not self.coded:
            # hard decision on the exact LLR == minimum-distance decision; llr > 0
            # favours bit 1 (modem_common.soft_llr's sign convention). Getting this
            # the other way round measures BER = 1 - BER, which at high SNR looks
            # like a catastrophic waterfall instead of an obviously wrong 1.0 -- the
            # first run of this file did exactly that and the analytic column caught
            # it immediately.
            return (llr > 0.0).astype(np.uint8)[:, :self.L], np.ones(len(llr), bool)
        dep = fr.depuncture(fr.deinterleave(llr, self.perm), self.ncoded, self.pid)
        a_hat, _ = fr.viterbi(dep)
        info = a_hat[:, :self.L]
        # CRC verdict, vectorised: recompute the parity of the decoded information
        # bits and compare with the decoded parity bits. A frame whose bits are
        # wrong but whose CRC still passes is an UNDETECTED error and is counted --
        # a 16-bit CRC on a ~100-bit frame is not a formality.
        par = (info.astype(np.int8) @ self.crc_mat.T) % 2
        return info, (par == a_hat[:, self.L:].astype(np.int8)).all(axis=1)

    # -- one measurement ----------------------------------------------------
    def run_point(self, ebno_db, probe=False):
        n0 = ebnodb2n0(ebno_db, self.m, self.r_eff)
        rng = np.random.default_rng((self.seed + int(round(ebno_db * 1000))) % (2 ** 32))
        budget = self.probe_frames if probe else self.max_frames
        frames = bit_err = blk_err = crc_bad = undetected = 0
        while frames < budget:
            B = min(self.batch, budget - frames)
            info = rng.integers(0, 2, (B, self.L)).astype(np.uint8)
            llr = fr.qam_llr(self.encode(info), 1 << self.m, n0, rng)
            hat, crc_ok = self.decode(llr)
            ne = hat != info
            bad = ne.any(axis=1)
            frames += B
            bit_err += int(ne.sum())
            blk_err += int(bad.sum())
            if self.coded:
                crc_bad += int((~crc_ok).sum())
                undetected += int((bad & crc_ok).sum())
            if blk_err >= self.min_errors:
                break
        return dict(ebno=float(ebno_db), probe=probe,
                    esno=ebno_db + 10 * np.log10(self.m * self.r_eff), n0=n0,
                    frames=frames, info_bits=frames * self.L,
                    symbols=frames * self.npunc // self.m, bit_err=bit_err,
                    blk_err=blk_err, bler=blk_err / frames, ber=bit_err / (frames * self.L),
                    crc_bad=crc_bad, undetected=undetected)


# ---------------------------------------------------------------------------
# statistics
# ---------------------------------------------------------------------------
def ci95(p, n):
    """Two-sided 95% binomial half width (normal approximation, within a few
    percent of Clopper-Pearson at the counts used here)."""
    return Z95 * np.sqrt(p * (1 - p) / n) if n and 0 < p < 1 else 0.0


def bler_ci_str(p):
    """What to print for a BLER: a half width when it was measured, the rule of
    three when zero block errors were seen (a zero has no half width), and nothing
    when p == 1."""
    n = p["frames"]
    if p["blk_err"] == 0:
        return "95%%< %.2e (0 errors)" % (3.0 / n)
    if p["blk_err"] == n:
        return "+-%.1e" % ci95(p["bler"], n)
    return "+-%.1e" % ci95(p["bler"], n)


def interp_ebn0(points, target, min_err=1):
    """Eb/N0 at BLER = target, log-linear between the two bracketing points, with
    both endpoints' binomial CIs propagated.

    log10(BLER) is close to linear in Eb/N0 across one decade of the waterfall,
    which is why the interpolation happens there and not in BLER; the two bracketing
    points are printed so the reader can see how far apart they are and how steep
    the segment is.
    """
    # Prefer endpoints that each carry >= min_err block errors: a point that hit
    # the frame cap with 1 error has a 200% CI and would dominate the +- for no
    # reason. If no such pair brackets the target, fall back to every measured
    # point and say so in the returned flag.
    usable = sorted([p for p in points if p["blk_err"] >= min_err and 0 < p["bler"] < 1],
                    key=lambda p: p["ebno"])
    allp = sorted([p for p in points if 0 < p["bler"] < 1], key=lambda p: p["ebno"])
    for pts, weak in ((usable, False), (allp, True)):
        r = _interp(pts, target)
        if r is not None:
            return r[0], r[1], r[2], r[3], weak
    return None, None, None, None, True


def _interp(pts, target):
    for a, b in zip(pts, pts[1:]):
        if (a["bler"] - target) * (b["bler"] - target) > 0:
            continue
        la, lb = np.log10(a["bler"]), np.log10(b["bler"])
        if la == lb:
            continue
        span = b["ebno"] - a["ebno"]
        lt = np.log10(target)
        x = a["ebno"] + (la - lt) / (la - lb) * span
        # d x / d la and d x / d lb, each times that point's CI in log10 units
        dxa = (lt - lb) / (la - lb) ** 2 * span
        dxb = (la - lt) / (la - lb) ** 2 * span
        ra = Z95 * np.sqrt((1 - a["bler"]) / a["blk_err"])
        rb = Z95 * np.sqrt((1 - b["bler"]) / b["blk_err"])
        return x, float(np.hypot(0.4343 * dxa * ra, 0.4343 * dxb * rb)), (a, b), \
            float(span / abs(la - lb))          # dB per decade of BLER
    return None


def locate_and_sweep(curve, args):
    """Coarse-to-fine: bracket BLER = target, narrow the bracket to <= --bracket dB,
    then measure a final grid that spans it. Returns (all_points, bracket)."""
    pts = []
    cache = {}

    def measure(e, probe):
        # Cached on Eb/N0 alone, and a probe is UPGRADED in place when the final
        # grid lands on the same point: keying on (Eb/N0, probe) measured such a
        # point twice and printed two rows for it, one of them with a fifth of the
        # frames, which reads like a contradiction.
        key = round(e, 6)
        have = cache.get(key)
        if have is not None and not (have["probe"] and not probe):
            return have
        p = curve.run_point(e, probe)
        if have is None:
            pts.append(p)
        else:
            pts[pts.index(have)] = p
        cache[key] = p
        return p

    step = args.probe_step
    e = args.probe_start
    lo = hi = None
    while e <= args.probe_end + 1e-9:
        p = measure(e, True)
        if p["bler"] < args.target:
            hi = p
            break
        lo = p
        e += step
    if hi is None:
        return pts, None                      # never crossed: no gain number
    if lo is None:                            # crossed at the very first probe
        e = hi["ebno"] - step
        while e >= args.probe_start - 4 * step:
            p = measure(e, True)
            if p["bler"] >= args.target:
                lo = p
                break
            hi = p
            e -= step
        if lo is None:
            return pts, None
    # refine to <= bracket dB with full-budget points, so the final interpolation
    # uses measurements rather than probes at its endpoints
    while hi["ebno"] - lo["ebno"] > args.bracket + 1e-9:
        mid = 0.5 * (lo["ebno"] + hi["ebno"])
        p = measure(mid, False)
        if p["bler"] < args.target:
            hi = p
        else:
            lo = p
    grid = np.linspace(lo["ebno"] - args.pad_db, hi["ebno"] + args.pad_db, args.npts)
    for e in grid:
        measure(float(e), False)
    return pts, (lo, hi)


def main():
    ap = argparse.ArgumentParser(description="FEC waterfall measurement")
    ap.add_argument("--m", default="2,4,6", help="bits per symbol to sweep")
    ap.add_argument("--rates", default="1/2,2/3,3/4", help="punctured rates")
    ap.add_argument("--L", type=int, default=96, help="information bits per frame")
    # 6000 frames x 15 errors: the numbers below were chosen from the measured
    # throughput on this host (~11 000 coded frames/s at m=4, ~7 300 at m=6, batch
    # 512), which puts a full 3 x 3 sweep at ~60 s, and from the statistics: at
    # BLER 4e-3 a point needs 3 750 frames to collect 15 errors, so the
    # below-target endpoint of the interpolation bracket is measured rather than
    # extrapolated from 1 or 2 errors.
    ap.add_argument("--frames", type=int, default=6000, help="max frames per point")
    ap.add_argument("--min-errors", type=int, default=15,
                    help="stop a point early once this many block errors are in")
    ap.add_argument("--batch", type=int, default=512, help="frames decoded at once")
    ap.add_argument("--target", type=float, default=1e-2, help="BLER to quote gain at")
    ap.add_argument("--probe-start", type=float, default=-2.0)
    ap.add_argument("--probe-end", type=float, default=20.0)
    ap.add_argument("--probe-step", type=float, default=2.0)
    ap.add_argument("--bracket", type=float, default=1.0,
                    help="dB width the final bracket is refined to")
    ap.add_argument("--pad-db", type=float, default=1.0,
                    help="dB added on each side of the bracket for the final grid")
    ap.add_argument("--npts", type=int, default=5, help="final grid points")
    ap.add_argument("--seed", type=int, default=20261026)
    ap.add_argument("--quick", action="store_true", help="~1/5 of the frames")
    args = ap.parse_args()
    if args.quick:
        args.frames, args.min_errors, args.batch = 600, 10, 256
        args.npts, args.probe_step = 3, 3.0
    ms = [int(x) for x in args.m.split(",")]
    pids = {"1/2": 0, "2/3": 1, "3/4": 2}
    rates = [pids[r.strip()] for r in args.rates.split(",")]
    cm, cr, se = load_cqi_table()

    t0 = time.time()
    print("[fec_bler] uncoded vs coded waterfall for the FEC layer (measurement, "
          "not a gate)")
    print("[fec_bler] decoder: tools/fec_ref.py numpy oracle, gated bit-exact vs "
          "ref/fec.cpp by build/fec_bench; modem: tools/modem_common.py")
    print("[fec_bler] code: K=%d (133,171)_8 zero-tailed, d_free computed = %d; "
          "CRC-16 (TS 38.212) per frame; S-random interleaver, S clamped to "
          "floor(sqrt(n)/3)" % (K_CONSTRAINT, fr.D_FREE_EXPECTED))
    print("[fec_bler] budget: <= %d frames/point (probe %d), early stop at %d block "
          "errors, batch %d, target BLER %.0e, bracket %.2f dB, seed %d"
          % (args.frames, max(256, args.frames // 5), args.min_errors, args.batch,
             args.target, args.bracket, args.seed))
    print("[fec_bler] noise floor: 1/%d = %.1e is the smallest measurable BLER, and "
          "the 95%% relative CI at the early-stop count (%d errors) is %.0f%%, i.e. "
          "+- %.2f dB on a waterfall of slope 1 decade/dB"
          % (args.frames, 1.0 / args.frames, args.min_errors,
             100 * Z95 / np.sqrt(args.min_errors), 0.4343 * Z95 / np.sqrt(args.min_errors)))

    gains = {}
    for m in ms:
        curves = [("uncoded", Curve(m, 0, args.L, args.seed, args.batch,
                                    args.frames * 2, args.min_errors, coded=False))]
        curves += [("rate %d/%d" % fr.puncture_rate(p),
                    Curve(m, p, args.L, args.seed + 1000 * (p + 1), args.batch,
                          args.frames, args.min_errors)) for p in rates]
        for label, cv in curves:
            pts, bracket = locate_and_sweep(cv, args)
            fin = [p for p in pts if not p["probe"]]
            print("\n=== m = %d, %s ===" % (m, cv.name))
            print("    frame: L = %d info bits -> npunc = %d channel bits -> %d "
                  "symbols/frame, R_eff = %.4f, interleaver S_eff = %d (%d restarts)"
                  % (cv.L, cv.npunc, cv.npunc // m, cv.r_eff, cv.par_eff, cv.att))
            j = nearest_cqi(cm, cr, se, m, cv.r_eff) if cv.coded else None
            if j:
                print("    mcs_table.clh: nearest entry CQI %d = (m=%d, R=%d/1024), "
                      "SE = %.4f bits/channel use; this code delivers m*R_eff = %.4f"
                      % (j[0], j[1], j[2], j[3], m * cv.r_eff))
            print("    %-7s %-7s %-9s %7s %9s %8s %7s %9s %8s %-18s %s"
                  % ("Eb/N0", "Es/N0", "n0", "frames", "info bits", "blk err",
                     "bit err", "BLER", "BER", "95% CI on BLER", "note"))
            for p in sorted(pts, key=lambda q: q["ebno"]):
                note = "probe" if p["probe"] else ""
                if p["blk_err"] == 0:
                    note += " FLOOR 1/%d" % p["frames"]
                if not cv.coded:
                    # The analytic column is the normalisation check. Its ratio is
                    # only meaningful when the bit count is: the +- is the Poisson
                    # relative CI 1.96/sqrt(bit errors), so a ratio of 1.6 +- 0.6 at
                    # 10 bit errors is agreement, not a discrepancy.
                    an = pam_gray_ber_exact(m, p["n0"])
                    rel = Z95 / np.sqrt(p["bit_err"]) if p["bit_err"] else np.inf
                    note += (" | analytic BER %.3e, measured/analytic %.3f +-%.3f"
                             % (an, p["ber"] / an if an > 0 else float("nan"), rel))
                elif p["undetected"]:
                    note += " | %d block errors the CRC MISSED" % p["undetected"]
                print("    %-7.2f %-7.2f %-9.2e %7d %9d %8d %7d %9.2e %8.2e %-18s %s"
                      % (p["ebno"], p["esno"], p["n0"], p["frames"], p["info_bits"],
                         p["blk_err"], p["bit_err"], p["bler"], p["ber"],
                         bler_ci_str(p), note.strip()))
            x, err, br, slope, weak = interp_ebn0(pts, args.target,
                                                 args.min_errors)
            if x is None:
                print("    BLER = %.0e not bracketed -- no gain number" % args.target)
                gains[(m, label)] = (None, None)
                continue
            print("    Eb/N0 at BLER = %.0e : %.3f +- %.3f dB   (waterfall slope "
                  "%.2f dB/decade there; bracket %.2f..%.2f dB%s)"
                  % (args.target, x, err, slope, br[0]["ebno"], br[1]["ebno"],
                     "; WEAK: no pair of points with >= %d errors brackets it, so "
                     "this leans on a low-count endpoint" % args.min_errors
                     if weak else ""))
            print("    the +- is the binomial CI of the two endpoints propagated "
                  "through the log-linear interpolation; it does NOT include the "
                  "curvature error of assuming log10(BLER) linear in Eb/N0, which "
                  "is why the bracket is refined to <= %.2f dB first" % args.bracket)
            print("    endpoints: %.2f dB -> BLER %.2e (%d frames, %d errors), "
                  "%.2f dB -> %.2e (%d frames, %d errors)"
                  % (br[0]["ebno"], br[0]["bler"], br[0]["frames"], br[0]["blk_err"],
                     br[1]["ebno"], br[1]["bler"], br[1]["frames"], br[1]["blk_err"]))
            print("    totals: %d frames, %d info bits, %d symbols, %d block errors "
                  "over %d points (%d of them probes)"
                  % (sum(p["frames"] for p in pts), sum(p["info_bits"] for p in pts),
                     sum(p["symbols"] for p in pts), sum(p["blk_err"] for p in pts),
                     len(pts), len(pts) - len(fin)))
            gains[(m, label)] = (x, err)

    print("\n=== coding gain at BLER = %.0e  (uncoded Eb/N0 minus coded Eb/N0) ==="
          % args.target)
    for m in ms:
        u, ue = gains.get((m, "uncoded"), (None, None))
        for (mm, label), (c, ce) in gains.items():
            if mm != m or label == "uncoded":
                continue
            if u is None or c is None:
                print("  m=%d %-12s : not bracketed" % (m, label))
            else:
                print("  m=%d %-12s : %+6.3f dB  +- %.3f dB   (uncoded %.3f +- %.3f, "
                      "coded %.3f +- %.3f)" % (m, label, u - c, np.hypot(ue or 0, ce or 0),
                                               u, ue or 0, c, ce or 0))
    print("[fec_bler] %.1f s wall" % (time.time() - t0))
    return 0


if __name__ == "__main__":
    sys.exit(main())
