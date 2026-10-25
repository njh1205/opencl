#!/usr/bin/env python3
"""Oracle for the SVD / GMD / UCD / CBD decomposition operators.

Ground truth is the production python:
  mix_precode/classical/digital_precoder.py        svd/gmd/ucd + thp_precoding
  mix_precode/full_digital_mu/digital_precoder.py  GP-CBD (design_cbd_..._from_svd)
Emits build/decomp_vectors.bin.

The per-case field order below is the binary contract with
runtime/decomp_main.cpp -- keep the two in sync. The layout is APPEND-ONLY:
  * header   : legacy "<5i2d" (strat, m, d, nsym, bits, snr, aux) followed by
               the extension "<2d3i" (waterfill, min_pl, active_count,
               degenerate, thp_ok).
  * payload  : the legacy 12 fields in their original order, then Q and u.
Cases 0..8 (strat 0/1/2) are generated exactly as before -- same rng draw
order, same legacy fields -- so their vectors are bit-identical to the
pre-CBD contract and the recorded errors stay comparable.
"""
import sys, struct, types, pathlib, warnings
import numpy as np
from scipy.linalg import svd

warnings.filterwarnings("ignore")           # macOS Accelerate matmul noise
ROOT = pathlib.Path(__file__).resolve().parents[2]


def _load(name, relpath):
    """Import a production module without touching it.

    full_digital_mu/digital_precoder.py uses PEP-604 annotations but no
    `from __future__ import annotations`, so it cannot be imported by the
    python 3.9 that ships on the dev host; injecting the future import is
    enough and leaves the reference file untouched.
    """
    path = ROOT / relpath
    sys.path.insert(0, str(path.parent))
    try:
        src = "from __future__ import annotations\n" + path.read_text()
    finally:
        sys.path.pop(0)
    mod = types.ModuleType(name)
    mod.__file__ = str(path)
    sys.modules[name] = mod                 # dataclasses resolves via sys.modules
    exec(compile(src, str(path), "exec"), mod.__dict__)
    return mod


_classical = _load("hpocl_dp_classical", "mix_precode/classical/digital_precoder.py")
_thp = _load("hpocl_thp_classical", "mix_precode/classical/thp_precoding.py")
_fdmu = _load("hpocl_dp_fdmu", "mix_precode/full_digital_mu/digital_precoder.py")
thp_transmit_from_upper = _thp.thp_transmit_from_upper

BITS, NSYM = 6, 16
CBD_PAIR_THRESHOLD = 1.7                    # production cbd_pair_threshold
STRAT = ("svd", "gmd", "ucd", "cbd")        # 0, 1, 2, 3


def case(strat, m, d, snr, wf=False, min_pl=0.0, spec=None, deg=False, note=""):
    return dict(strat=strat, m=m, d=d, snr=snr, wf=wf, min_pl=min_pl,
                spec=spec, deg=deg, note=note)


# ---- legacy block: unchanged, keeps cases 0..8 and their vectors identical --
CASES = [case(strat, m, d, snr, note="legacy")
         for strat in (0, 1, 2)              # 0 svd, 1 gmd, 2 ucd
         for (m, d, snr) in ((6, 4, 100.0), (8, 4, 30.0), (5, 3, 50.0))]

# ---- GP-CBD (strat 3): active-count selection, odd/even pairing, dimensions --
CASES += [
    case(3, 6, 4, 100.0, note="cbd even count, all streams paired"),
    case(3, 8, 4, 30.0, note="cbd"),
    case(3, 5, 3, 50.0, note="cbd odd count -> middle stream stays 1x1"),
    case(3, 7, 5, 20.0, note="cbd d=5 odd, m=7"),
    case(3, 6, 4, 0.005, note="cbd low snr -> active_count 1, R diagonal"),
    case(3, 4, 1, 40.0, note="cbd d=1 (no pair possible)"),
    case(3, 4, 4, 40.0, note="cbd square m=d"),
    case(3, 16, 8, 40.0, note="cbd MAXM x MAXD"),
    case(3, 8, 8, 40.0, note="cbd square at MAXD"),
    case(3, 6, 4, 1e6, note="cbd high snr: 1e-12 floor pulls dead streams in"),
    case(3, 6, 4, 40.0, spec=[3.0, 3.0, 1.0, 1.0], deg=True,
         note="cbd tied singular values -> zero coupling, 1x1 blocks"),
    case(3, 6, 4, 40.0, spec=[3.0, 2.0, 0.0, 0.0], deg=True,
         note="cbd rank deficient -> active_count 2"),
    case(3, 6, 4, 1e6, spec=[3.0, 2.0, 1e-8, 1e-9], deg=True,
         note="cbd ill-conditioned pair (d1/d2 = 3e9)"),
]

# ---- UCD waterfill=true (oracle -> C++ -> kernel), incl. min_pl > 0 ----------
CASES += [
    case(2, 6, 4, 100.0, wf=True, note="ucd waterfill, min_pl=0"),
    case(2, 8, 4, 30.0, wf=True, min_pl=0.05, note="ucd waterfill, min_pl>0"),
    case(2, 5, 3, 0.5, wf=True, min_pl=0.25, note="ucd waterfill, spread spectrum"),
    case(2, 6, 4, 0.1, wf=True, min_pl=0.3, spec=[4.0, 3.0, 2.0, 0.5],
         note="ucd waterfill: min_pl binds on the weak and on the cut stream"),
    case(2, 6, 4, 0.1, wf=True, min_pl=-0.5, spec=[4.0, 3.0, 2.0, 0.5],
         note="ucd waterfill, min_pl<0 (>=0 clamp must come first)"),
    case(2, 6, 4, 100.0, wf=False, min_pl=0.3, note="ucd no waterfill, min_pl>0"),
]

# ---- dimension coverage for the legacy strategies ---------------------------
CASES += [
    case(0, 16, 8, 40.0, note="svd MAXM x MAXD"),
    case(1, 16, 8, 40.0, note="gmd MAXM x MAXD"),
    case(2, 16, 8, 40.0, note="ucd MAXM x MAXD"),
    case(0, 4, 4, 40.0, note="svd square m=d"),
    case(1, 4, 4, 40.0, note="gmd square m=d"),
    case(2, 4, 4, 40.0, note="ucd square m=d"),
    case(0, 4, 1, 40.0, note="svd d=1"),
    case(1, 4, 1, 40.0, note="gmd d=1 (positive_count == 1 path)"),
    case(2, 4, 1, 40.0, note="ucd d=1"),
]

# ---- tied / rank-deficient spectra for the legacy strategies ----------------
CASES += [
    case(1, 6, 4, 40.0, spec=[3.0, 3.0, 1.0, 1.0], deg=True, note="gmd tied"),
    case(2, 6, 4, 40.0, spec=[3.0, 3.0, 1.0, 1.0], deg=True, note="ucd tied"),
    case(1, 6, 4, 40.0, spec=[3.0, 2.0, 0.0, 0.0], deg=True, note="gmd rank deficient"),
    case(2, 6, 4, 40.0, spec=[3.0, 2.0, 0.0, 0.0], deg=True, note="ucd rank deficient"),
    case(0, 6, 4, 40.0, spec=[3.0, 3.0, 1.0, 1.0], deg=True, note="svd tied"),
    case(0, 6, 4, 40.0, spec=[3.0, 2.0, 0.0, 0.0], deg=True, note="svd rank deficient"),
]

# ---- UCD waterfill: the fp32 cancellation band of the active-set scan ---------
# Production's scan tests `water_level > alpha/singular_power[cnt-1]` with
# water_level = (num_streams + inv)/cnt. At cnt == 1 that is `d + thr > thr`,
# unconditionally true in exact arithmetic -- but in fp32 the `+d` is absorbed
# once thr = alpha/sigma0^2 exceeds d/eps32 (~3.4e7 at d=4), no cnt breaks, and
# the kernel used to keep active = d with an all-zero loading. Measured:
# production gives active=1, pl=[4,0,0,0]; the un-re-associated fp32 form gave
# active=4, pl=[0,0,0,0], max|d pl| = 4.0 and rho_eq 9.0e-10 vs 0.0.
# Trigger is snr*sigma0^2 <~ 1.2e-7, so these two vectors sit inside it:
#   sigma0=1e-2, snr=1e-4 -> alpha/sp0 = 1.0e8   (-40 dB, weak channel)
#   sigma0=1e-3, snr=1e-2 -> alpha/sp0 = 1.0e8   (-20 dB, weaker channel)
# NOT degenerate: the spectra are distinct and well separated, so the e_cl gate
# applies to these rows and a regression here fails the bench.
# deg=True here does NOT mean the channel spectrum is tied -- [1e-2,8e-3,5e-3,
# 2e-3] is well separated. It means the AUGMENTED spectrum is tied in fp32: the
# bug fires only when alpha/sigma0^2 > d/eps32 ~ 3.4e7, i.e. when pl*sigma^2 is
# < 1.2e-7 of alpha, so saug = sqrt(pl*sigma^2 + alpha) rounds to sqrt(alpha) for
# EVERY stream and the fp32 GMD sees an exactly-tied spectrum while the fp64 ref
# resolves it to ~1e-8. Vd/upper/P/Q are then basis-dependent in the usual
# degenerate sense and print n/a. What stays meaningful, and what decomp_main
# gates on these rows, is the rotation-invariant set: S, diag(upper), the active
# count and -- the point of these vectors -- power_loading, which is a scalar per
# stream. Pre-fix the kernel returned pl=[0,0,0,0] against production's
# [4,0,0,0] (rel = 1.0); post-fix it is 0.0.
CASES += [
    case(2, 6, 4, 1e-4, wf=True, spec=[1e-2, 8e-3, 5e-3, 2e-3], deg=True,
         note="ucd waterfill: fp32 active-set cancellation band (alpha/sp0=1e8)"),
    case(2, 6, 4, 1e-2, wf=True, spec=[1e-3, 8e-4, 5e-4, 2e-4], deg=True,
         note="ucd waterfill: same band at sigma0=1e-3"),
]

# ---- tall right basis: rows > d, the shape the pipeline actually feeds -------
# ref/pipeline.cpp:136,146 and ref/rx.cpp:33,36 call gmd_factors/ucd_design with
# rows = o.nul > d (the BD null basis V is [nul x d]), and gmd_dev is written for
# that ("rows may exceed d"). The kernel path had only ever been driven with
# rows == d, so the tall loops and the `rows > MAXM` guard were unexercised while
# README claimed otherwise. m=16 with d=4 gives rows/d = 4 at rows = MAXM.
CASES += [
    case(1, 16, 4, 40.0, note="gmd tall basis: rows=16 > d=4"),
    case(2, 16, 4, 40.0, wf=True, note="ucd tall basis: rows=16 > d=4"),
    case(2, 16, 4, 40.0, wf=True, min_pl=0.05, note="ucd tall basis + min_pl>0"),
]


def _rand_unitary(rng, rows, cols):
    a = (rng.normal(0, 1, (rows, cols)) + 1j * rng.normal(0, 1, (rows, cols))) / np.sqrt(2)
    q, r = np.linalg.qr(a)
    diag = np.diag(r)
    return q * (diag / np.abs(diag))


def build_case(cs, rng):
    strat, m, d, snr = cs["strat"], cs["m"], cs["d"], cs["snr"]
    if cs["spec"] is None:
        h = (rng.normal(0, 1, (m, d)) + 1j * rng.normal(0, 1, (m, d))) / np.sqrt(2)
    else:
        spec = np.asarray(cs["spec"], dtype=float)
        assert spec.size == d, (cs, spec.size)
        h = (_rand_unitary(rng, m, d) @ np.diag(spec)
             @ _rand_unitary(rng, d, d).conj().T)
    # one SVD for everything: design_precoder() re-derives the same basis
    # internally, so vd and the transceiver factors stay in one basis (with a
    # tied spectrum two independent SVDs would not agree).
    u, s, vh = svd(h, full_matrices=False)
    v = vh.conj().T[:, :d]
    s = np.asarray(s[:d], dtype=float)

    if strat == 3:
        prec = _fdmu.DigitalStructuredPrecoder(num_rf_chains=m, num_streams=d)
        assert prec.cbd_pair_threshold == CBD_PAIR_THRESHOLD
    else:
        prec = _classical.DigitalStructuredPrecoder(
            num_rf_chains=m, num_streams=d, ucd_waterfill=cs["wf"],
            ucd_min_power_loading=cs["min_pl"])

    P = np.zeros((d, d), complex)
    R = np.zeros((d, d), complex)
    Q = np.zeros((m, d), complex)
    pl = np.zeros(d)
    if strat == 3:
        # design_precoder() has no "cbd" strategy: the chain is built by
        # design_cbd_transceiver_from_svd() (see multiuser_simulation_
        # environment.build_structured_digital_chain), so take the tx factor
        # p_cbd as v_d and the realised rho from compute_stream_snr().
        blk = prec.design_cbd_transceiver_from_svd(u, v, s, snr)
        P, R, Q = blk.p_cbd, blk.r_cbd, blk.q_cbd
        act = int(blk.active_count)
        vd = P
        rr = prec.compute_stream_snr(h, vd, snr)
        # target per-stream SNR of the CBD chain: every stream of a pair lands
        # on the geometric-mean pair SNR, i.e. the squared r_cbd diagonal.
        rt = np.abs(np.diag(R)) ** 2 * snr
    else:
        act = 0
        vd, rr, rt = prec.design_precoder(h, snr, STRAT[strat])
        if strat == 1:
            blk = prec.design_gmd_transceiver_from_svd(u, v, s)
            P, R, Q = blk.p_gmd, blk.r_gmd, blk.q_gmd
        elif strat == 2:
            blk = prec.design_ucd_transceiver_from_svd(u, v, s, snr)
            # UCDTransceiverBlock exposes the receive combiner w_ucd, not the
            # raw q_ucd: w_ucd = q_ucd @ diag(1/diag(r_aug_ucd)).
            P, R, Q = blk.p_ucd, blk.r_aug_ucd, blk.w_ucd
            pl = np.asarray(blk.power_loading, dtype=float)

    cod = ((rng.integers(0, 4, (d, NSYM)) - 1.5)
           + 1j * (rng.integers(0, 4, (d, NSYM)) - 1.5))
    # THP end-to-end oracle: only for gmd, and only when diag(R) is invertible
    # (thp_transmit_from_upper raises on a zero diagonal, which is exactly what
    # a rank-deficient spectrum produces).
    thp_ok = int(strat == 1 and not np.any(np.isclose(np.diag(R), 0.0)))
    if thp_ok:
        xtx, dp, b1 = thp_transmit_from_upper(cod, R, BITS, p=P)
    else:
        b1 = np.eye(d, dtype=complex)
        dp = cod.copy()
        xtx = P @ cod if strat in (2, 3) else vd @ cod

    out = dict(cs, h=h, S=s, u=u, v=v, vd=vd, rr=rr, rt=rt, P=P, R=R, Q=Q,
               pl=pl, b1=b1, cod=cod, xtx=xtx, act=act, thp=thp_ok)
    # no NaN/inf may ever reach the binary contract: the C++ side compares with
    # fmax/abs, where a single NaN would silently poison every threshold.
    for k in ("h", "S", "u", "v", "vd", "rr", "rt", "P", "R", "Q", "pl", "b1",
              "cod", "xtx"):
        assert np.all(np.isfinite(out[k])), (cs["note"], k)
    return out


def field_list(c):
    """(name, array, element_count) -- the binary contract.

    The first 12 entries are the legacy order and must not move; Q and u were
    appended for the CBD left factor and for phase-aligning the left basis.
    """
    m, d = c["m"], c["d"]
    return [("h", c["h"], m * d),      ("S", c["S"], d),
            ("v", c["v"], d * d),      ("vd", c["vd"], d * d),
            ("rr", c["rr"], d),        ("rt", c["rt"], d),
            ("P", c["P"], d * d),      ("R", c["R"], d * d),
            ("pl", c["pl"], d),        ("b1", c["b1"], d * d),
            ("cod", c["cod"], d * NSYM), ("xtx", c["xtx"], d * NSYM),
            # Q is the receive-side chain production hands to the detector:
            # q_gmd (strat 1), w_ucd = q_ucd diag(1/diag r_aug_ucd) (strat 2),
            # q_cbd (strat 3); zeros for strat 0. u is the oracle left singular
            # basis, needed to phase-align the C++/kernel left factor.
            ("Q", c["Q"], m * d),      ("u", c["u"], m * d)]


def main():
    rng = np.random.default_rng(5)
    out = [struct.pack("<i", len(CASES))]
    for spec in CASES:
        c = build_case(spec, rng)
        out.append(struct.pack("<5i2d", c["strat"], c["m"], c["d"], NSYM,
                               BITS, c["snr"], 0.0))
        out.append(struct.pack("<2d3i", float(c["wf"]), float(c["min_pl"]),
                               c["act"], int(c["deg"]), c["thp"]))
        for _, arr, n in field_list(c):
            a = np.ascontiguousarray(arr)
            assert a.size == n, (spec, a.size, n)
            dt = np.complex128 if np.iscomplexobj(a) else np.float64
            out.append(a.astype(dt).tobytes())
    pathlib.Path("build").mkdir(exist_ok=True)
    pathlib.Path("build/decomp_vectors.bin").write_bytes(b"".join(out))
    print(f"wrote build/decomp_vectors.bin ({len(CASES)} cases)")
    for i, c in enumerate(CASES):
        print(f"  case{i:<2d} {STRAT[c['strat']]:<3} m={c['m']:<2d} d={c['d']} "
              f"snr={c['snr']:<8g} wf={int(c['wf'])} min_pl={c['min_pl']:<5g} "
              f"deg={int(c['deg'])} :: {c['note']}")


if __name__ == "__main__":
    main()
