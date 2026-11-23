#!/usr/bin/env python3
"""The modem math shared by every numpy oracle in tools/.

tools/datapath_ref.py and tools/chain_ref.py each carried their own copy of this
file's twelve functions. An AST comparison showed ten of them byte-identical and
the other two differing only in generality (logsumexp's axis argument) and name
(gmi vs gmi_per_stream) -- but chain_ref.py had also deleted the docstrings, and
one of those was the single most load-bearing comment in the modem path:

    Production get_constellation labels level idx with idx ^ (idx >> 1) ... so
    label -> amplitude needs the full Gray *decode*: a single XOR is the encoder
    and only inverts itself for L <= 4 (it silently mislabels 64/256-QAM).

Two independent oracles for one convention means the two validation chains can
disagree and both still pass against their own copy, which is exactly what the
"three-way cross-check" claim cannot afford. So: one implementation, imported by
datapath_ref.py and chain_ref.py. (modem_ref.py needs none of it -- it calls the
production bicm_metrics/thp_precoding functions directly, which is why it is the
strongest of the three oracles and the one that can catch a bug *here*.)

The conventions are not invented here -- check_production() pins label->symbol,
the replica lattice and the centered modulo against the production modem
(mix_precode/classical/bicm_metrics.py, thp_precoding.py) on every run.
"""
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]
                       / "mix_precode" / "classical"))


def gray_decode(g):
    """Full binary-reflected Gray decode (XOR cascade): label -> level index."""
    v, s = np.array(g, copy=True), np.array(g) >> 1
    while s.any():
        v ^= s
        s >>= 1
    return v


def pam_delta(m):
    """PAM spacing of unit-power square QAM = production spacing map."""
    return float(np.sqrt(6.0 / ((1 << m) - 1)))


def pam_amps(m):
    """Per-axis amplitude of every PAM level, indexed by its m/2-bit label.

    Production get_constellation labels level idx with idx ^ (idx >> 1) and
    places it at (2*idx - (L-1)) * delta/2, so label -> amplitude needs the
    full Gray *decode*: a single XOR is the encoder and only inverts itself
    for L <= 4 (it silently mislabels 64/256-QAM).
    """
    L = 1 << (m // 2)
    return (2 * gray_decode(np.arange(L)) - (L - 1)) * (pam_delta(m) / 2)


def bit_cols(idx, nb):
    """MSB-first bit matrix of the integers in idx, nb bits each."""
    return np.stack([(idx >> (nb - 1 - b)) & 1 for b in range(nb)], axis=1)


def modulo_period(m):
    """THP receive-modulo period L*spacing, spelled the way production's RECEIVE
    fold spells it.

    Production has two algebraically equal spellings that are 1 ulp apart at
    m in {2, 8} (identical at m = 4, 6):

      thp_precoding.qam_axis_order_and_spacing  ->  L * (2/sqrt(2(L^2-1)/3))
          used by thp_receive_equalized, i.e. by the receive fold, and by
          ref/modem.cpp and tools/modem_ref.py
      multiuser_simulation_environment.qam_modulo_period  ->  L * sqrt(6/(M-1))
          which is what pam_delta() computes, and what this file used to use

    The receive fold is the authority for this chain, so this now matches it
    bit-for-bit and check_production's period assert can be an exact equality
    instead of a 1e-12 tolerance papering over the two. chain_bench is the one
    gate that computes the period independently on both sides (C++ = thp
    spelling, oracle = this file), so the 1 ulp was live exposure there the
    moment a case used m = 2 or 8 -- today chain_ref.CASES are M = 16 and 64,
    where the two spellings agree exactly, which is why nothing had caught it.
    Measured effect of the change: datapath_vectors.bin's period shifts 1 ulp at
    m = 2/8, moving zf/llr_fold by <= 9.9e-16 (the fold is continuous in period
    unless floor() flips, and the closest any probe gets to +-period/2 is 1e-5,
    about 2.2e10 ulps), against gates of 1e-4 / 3e-3.
    """
    L = 1 << (m // 2)
    return L * (2.0 / np.sqrt(2.0 * (L * L - 1) / 3.0))


def qam_map(bits, m):
    """bits[...,m] uint8 MSB first -> symbols; first m/2 bits drive the I axis."""
    mh, amp = m // 2, pam_amps(m)
    w = 1 << np.arange(mh - 1, -1, -1)
    return amp[(bits[..., :mh] * w).sum(-1)] + 1j * amp[(bits[..., mh:] * w).sum(-1)]


def constellation(m):
    """points[M] complex and labels[M,m] uint8, in production table order:
    level index ascending (I outer, Q inner), label = idx ^ (idx >> 1) per axis.
    qam_map/soft_llr only use the label -> symbol map, so the row order is free;
    matching get_constellation lets check_production compare element-wise."""
    mh, L = m // 2, 1 << (m // 2)
    idx = np.arange(L)
    amp = (2 * idx - (L - 1)) * (pam_delta(m) / 2)     # by level index
    lab = idx ^ (idx >> 1)                             # Gray encode
    iI, iQ = (v.ravel() for v in np.meshgrid(idx, idx, indexing="ij"))
    pts = amp[iI] + 1j * amp[iQ]
    bits = np.concatenate([bit_cols(lab[iI], mh), bit_cols(lab[iQ], mh)], axis=1)
    return pts, bits.astype(np.uint8)


def folded_candidates(m, radius):
    """Periodic replica set of a modulo receiver: (2r+1)^2 * M candidates.

    Order and labels of enumerate_thp_symbol_replicas: real shift outer,
    imaginary shift inner, base constellation innermost, labels copied.
    """
    pts, lab = constellation(m)
    sh = np.arange(-radius, radius + 1)
    off = (sh[:, None] + 1j * sh[None, :]).ravel() * modulo_period(m)
    return (pts[None, :] + off[:, None]).ravel(), np.tile(lab, (off.size, 1))


def mod_fold(z, period):
    """Centered modulo into [-period/2, period/2), per axis (floor form)."""
    z = np.asarray(z, dtype=np.complex128)
    fold = lambda v: v - np.floor((v + period / 2.0) / period) * period
    return fold(z.real) + 1j * fold(z.imag)


def logsumexp(e, axis=-1):
    """log sum exp, max subtracted (same stabilization as the kernels)."""
    mx = e.max(axis=axis, keepdims=True)
    s = np.log(np.exp(e - mx).sum(axis=axis, keepdims=True))
    return np.squeeze(mx + s, axis=axis)


def soft_llr(z, cand, clab, n0):
    """Exact log-MAP LLRs over an arbitrary candidate set, p(z|c) ~ e^{-d2/n0}.

    Sign convention llr = ln P(b=1|z) - ln P(b=0|z) (llr > 0 favours bit 1),
    which is the sign the gmi() penalty log2(1 + exp(-(2c-1) llr)) expects.
    cand = the M base points (parallel branch) or the replica set (folded).
    """
    diff = z[..., None] - cand
    e = -(diff.real ** 2 + diff.imag ** 2) / n0
    llr = np.empty(z.shape + (clab.shape[1],))
    for b in range(clab.shape[1]):
        one = clab[:, b] == 1
        llr[..., b] = logsumexp(e[..., one]) - logsumexp(e[..., ~one])
    return llr


def gmi(llr, bits, m):
    """m - mean_sym sum_b log2(1 + exp(-(2c_b - 1) llr_b)), softplus stabilized.

    Deliberately UNCLAMPED and per stream. Production's
    bicm_metrics.estimate_bicm_gmi_*_from_received clamp the AGGREGATE at zero
    (max(0.0, num_streams*bits_per_symbol - loss)), so a caller that compares
    against production must apply max(0.0, sum(per_stream)) itself -- see
    kernels/datapath.cl's gmi_k CLAMP CONTRACT and tools/chain_ref.py.
    """
    t = (1 - 2 * bits.astype(np.float64)) * llr
    sp = np.maximum(t, 0.0) + np.log1p(np.exp(-np.abs(t)))
    return m - sp.sum(-1).mean(-1) / np.log(2.0)


def labeled_points(cand, clab):
    """label tuple -> sorted list of the points carrying it: an order-free view.

    Needed because production lists its QPSK table in Gray-LABEL order
    (-1-1j, -1+1j, +1+1j, +1-1j) while constellation() lists every order in
    level-index order (I axis outer, amplitude ascending) -- the two agree for
    16/64/256-QAM, whose production branches are `for real in pam_gray: for imag
    in pam_gray`, and are a permutation of one another at m = 2. A logsumexp
    consumes the label -> point PAIRING and nothing else, so comparing pairings
    is the order-free (and for QPSK the only possible) element-wise check.
    """
    out = {}
    for c, l in zip(cand, clab):
        out.setdefault(tuple(int(v) for v in l), []).append((float(c.real),
                                                             float(c.imag)))
    return {k: sorted(v) for k, v in out.items()}


def check_production(extra_replica_cases=()):
    """Pin label -> symbol, the replica lattice and the modulo to production.

    `extra_replica_cases` adds (m, radius) pairs to the default replica check,
    so a caller can cover the exact radius its own vectors use. This is the
    union of the two checks datapath_ref.py and chain_ref.py used to carry
    separately (each covered a subset, so a convention could be pinned on one
    path and silently wrong on the other).
    """
    from bicm_metrics import (enumerate_thp_symbol_replicas, get_constellation,
                              modulate_bits_with_labeling)
    from thp_precoding import centered_modulo_complex, qam_axis_order_and_spacing

    rng = np.random.default_rng(7)
    for m in (2, 4, 6, 8):                     # QPSK / 16 / 64 / 256-QAM
        pts, lab = constellation(m)
        psym, pbits = get_constellation(m)
        mine = {tuple(int(v) for v in lab[q]): pts[q] for q in range(1 << m)}
        prod = {tuple(int(v) for v in row): sym for row, sym in zip(pbits, psym)}
        assert mine.keys() == prod.keys(), f"label set differs at m={m}"
        assert max(abs(mine[k] - prod[k]) for k in mine) < 1e-12, f"amps m={m}"
        bits = rng.integers(0, 2, (64, m)).astype(np.uint8)
        assert np.abs(qam_map(bits, m)
                      - modulate_bits_with_labeling(m, bits)).max() < 1e-12
        L, spacing = qam_axis_order_and_spacing(m)
        assert L == 1 << (m // 2) and abs(modulo_period(m) - L * spacing) < 1e-12
        # a single XOR is the Gray *encoder*; only the full cascade inverts it
        assert np.array_equal(gray_decode(np.arange(L) ^ (np.arange(L) >> 1)),
                              np.arange(L)), f"gray decode is not the inverse at m={m}"

    # (2, 2) first: QPSK at radius 2 is the one order/radius where the replica
    # lattice overlaps the base one (period 2*delta puts a replica exactly delta
    # from a base point, so the flipped-bit subset gains a second co-minimal
    # candidate and the folded LLR shifts by log 2). It was not covered here,
    # and it cannot be covered by the element-wise form: production's QPSK row
    # order differs from ours, so the lattice is compared as label -> points.
    for m, radius in ((2, 2), (4, 2), (6, 2), (8, 1)) + tuple(extra_replica_cases):
        cand, clab = folded_candidates(m, radius)
        pcand, pclab = enumerate_thp_symbol_replicas(m, radius)
        if m != 2:            # row order matches production for 16/64/256-QAM
            assert np.abs(cand - pcand).max() < 1e-12 and \
                np.array_equal(clab, pclab), \
                f"replica lattice differs at m={m} radius={radius}"
        mine, prod = labeled_points(cand, clab), labeled_points(pcand, pclab)
        assert mine.keys() == prod.keys(), \
            f"replica label set differs at m={m} radius={radius}"
        worst = 0.0
        for k, pts in mine.items():
            ppts = prod[k]
            assert len(pts) == len(ppts), f"replica multiplicity at m={m} label {k}"
            worst = max([worst] + [abs(a[0] - b[0]) + abs(a[1] - b[1])
                                   for a, b in zip(pts, ppts)])
        assert worst < 1e-12, \
            f"replica lattice differs at m={m} radius={radius} ({worst:.3e})"

    period = modulo_period(6)
    x = np.array([period / 2, -period / 2, period / 2 - 1e-3, -3.1 * period,
                  2.7j * period])
    # exact equality: the fold is the same floor expression on both sides
    assert np.abs(mod_fold(x, period) - centered_modulo_complex(x, period)).max() == 0.0
    print("[oracle] production cross-check OK (get_constellation / "
          "modulate_bits_with_labeling / enumerate_thp_symbol_replicas "
          "incl. QPSK order-free / centered_modulo_complex, M <= 256)")
