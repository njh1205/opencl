#!/usr/bin/env python3
"""Oracle for the receive-side control plane (ref/rx.cpp).

Ground truth is the production python in mix_precode/classical, i.e. the
per-scheduling-instant assembly of
MultiUserSimulationEnvironment.build_structured_digital_chain:
  svd : q, r, p = u_eff, diag(s), v_eff
  gmd : DigitalStructuredPrecoder.design_gmd_transceiver_from_svd
        -> q_gmd, r_gmd, p_gmd
  ucd : DigitalStructuredPrecoder.design_ucd_transceiver_from_svd(
        snr_per_stream=snr, alpha_override=1/max(snr, 1e-12))
        -> w_ucd, r_aug_ucd, p_ucd
  scale    = sqrt(digital_power_constraint / ||Frf Fbb||_F^2), 1.0 if power<=1e-12
  r_chain -> scale * r_chain,   diag_b = diag(scale * r_chain)
Production defaults confirmed in DigitalStructuredPrecoder.__init__ and in the
environment constructor that forwards them: ucd_waterfill=False,
ucd_min_power_loading=0.0 (so power_loading = ones(d)); digital_power_constraint
defaults to total_streams = K*d.

Emits build/rx_vectors.bin. Layout, little endian throughout:
  int32 ncase
  per case: int32 Nr, rows, d, Nt, T, strategy (ASCII of 's'/'g'/'u'),
            float64 snr, float64 p_constraint, then field_list()
  field dtype: complex128 (re, im) / float64 by numpy kind
The per-case field order in field_list() is the binary contract with
runtime/rx_main.cpp -- keep the two in sync.
"""
import pathlib
import struct
import sys

import numpy as np
from scipy.linalg import svd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]
                       / "mix_precode" / "classical"))
from digital_precoder import DigitalStructuredPrecoder
from multiuser_simulation_environment import MultiUserSimulationEnvironment
from thp_precoding import compute_effective_channel

SHAPES = ((4, 4, 2), (6, 4, 4), (2, 8, 2), (8, 8, 4))  # (Nr, rows, d), rows>Nr
SNRS = (25.0, 64.0, 100.0, 400.0)     # per shape; only ucd reads it (alpha=1/snr)
STRATS = ("svd", "gmd", "ucd")
NT = 12                               # tx antennas; NRF = rows (BD null-space dim)
USERS = 2                             # F_BB = [P_1, P_2], so T = USERS * d
SEED0 = 20261024
CASES = [(Nr, rows, d, strat, SNRS[i], SEED0 + 31 * k)
         for i, (Nr, rows, d) in enumerate(SHAPES)
         for k, strat in enumerate(STRATS)]


class PowerStub:
    """Only the attribute the unbound normalize_digital_precoder touches."""

    def __init__(self, constraint):
        self.digital_power_constraint = float(constraint)


def rand_cplx(rng, shape):
    return (rng.normal(0.0, 1.0, shape) + 1j * rng.normal(0.0, 1.0, shape)) / np.sqrt(2.0)


def reduced_svd(gbar, d):
    """Production split: u [Nr x d], s [d] descending, v [rows x d] of Gbar."""
    u, s, vh = svd(gbar, full_matrices=False)
    return u[:, :d], np.asarray(s[:d], dtype=float), vh.conj().T[:, :d]


def strategy_factors(prec, strat, snr, u, s, v):
    """(q_local, r_local, p) as build_structured_digital_chain assembles them."""
    if strat == "svd":
        return u, np.diag(s).astype(complex), v
    if strat == "gmd":
        blk = prec.design_gmd_transceiver_from_svd(u_eff=u, v_eff=v, singular_values=s)
        return blk.q_gmd, blk.r_gmd, blk.p_gmd
    blk = prec.design_ucd_transceiver_from_svd(
        u_eff=u, v_eff=v, singular_values=s, snr_per_stream=snr,
        alpha_override=1.0 / max(float(snr), 1e-12))
    return blk.w_ucd, blk.r_aug_ucd, blk.p_ucd


def check_chain(strat, gbar, u, s, v, Q, R, P, snr):
    """Definitional properties of the assembled chain.

    SVD/GMD export a unitary combiner, so Q^H Gbar P = R holds on the full
    channel (the dropped singular directions are orthogonal to Q and P). MMSE-UCD
    exports w_ucd = q_ucd diag(1/safe(diag(r_aug_ucd))), which is not unitary:
    the exact statement there is the factorization of the rank-d truncation
    Gbar_d = Qg R P^H with Qg = Q diag(diag(R)) = q_ucd, and the combiner then
    normalizes the effective diagonal only up to O(alpha), alpha = 1/snr:
      diag(Q^H Gbar P)_i = (Qg^H Qg)_ii = 1 - sum_k alpha/(s_k^2+alpha) |G_ki|^2.
    """
    d = len(s)
    B = Q.conj().T @ gbar @ P
    Qg = Q @ np.diag(np.diag(R)) if strat == "ucd" else Q
    assert np.abs(np.tril(R, -1)).max() < 1e-12                       # upper
    assert np.abs(Qg @ R @ P.conj().T - u @ np.diag(s) @ v.conj().T).max() < 1e-9
    if strat != "svd":                                                # equal |diag|
        diag = np.abs(np.diag(R))
        assert diag.std() <= 1e-12 * diag.mean(), (strat, diag)
    if strat != "ucd":
        assert np.abs(Q.conj().T @ Q - np.eye(d)).max() < 1e-9        # unitary
        assert np.abs(B - R).max() < 1e-9                             # Q^H Gbar P = R
        return
    alpha = 1.0 / max(float(snr), 1e-12)
    assert np.abs(np.diag(B).imag).max() < 1e-12
    dev = 1.0 - np.diag(B).real
    assert -1e-12 <= dev.min() and dev.max() <= alpha / s.min() ** 2 + 1e-12, dev


def power_scale(frf, fbb, constraint):
    """correction_scale, cross-checked against normalize_digital_precoder."""
    power = float(np.linalg.norm(frf @ fbb, "fro") ** 2)
    scale = float(np.sqrt(constraint / power)) if power > 1e-12 else 1.0
    normalized = MultiUserSimulationEnvironment.normalize_digital_precoder(
        PowerStub(constraint), f_rf=frf, f_bb=fbb)
    assert np.abs(normalized - fbb * scale).max() < 1e-12 * np.abs(fbb).max()
    return scale


def build_case(Nr, rows, d, strat, snr, seed):
    """One whitened channel -> combiner/r_chain -> power normalization -> diag_b."""
    rng = np.random.default_rng(seed)
    prec = DigitalStructuredPrecoder(rows, d)
    gbar = rand_cplx(rng, (Nr, rows))            # power-whitened BD channel
    u, s, v = reduced_svd(gbar, d)
    Q, R, P = strategy_factors(prec, strat, snr, u, s, v)
    check_chain(strat, gbar, u, s, v, Q, R, P, snr)

    # phase-only RF front end and the two user digital blocks F_BB = [P_1, P_2]
    frf = np.exp(1j * rng.uniform(0.0, 2.0 * np.pi, (NT, rows)))
    _, _, p2 = strategy_factors(prec, strat, snr,
                                *reduced_svd(rand_cplx(rng, (Nr, rows)), d))
    fbb = np.hstack([P, p2])
    constraint = float(USERS * d)                # env default: total_streams
    scale = power_scale(frf, fbb, constraint)

    r_norm = scale * R                           # correction_scale * r_local
    b_eff, _ = compute_effective_channel(np.eye(d, dtype=complex), r_norm,
                                         np.eye(d, dtype=complex))
    diag_b = np.diag(b_eff)                      # diag_b of thp_receive_equalized
    assert np.all(np.abs(diag_b) > 1e-12)

    return dict(Nr=Nr, rows=rows, d=d, Nt=NT, T=USERS * d, strat=strat[0],
                snr=snr, p_constraint=constraint, U=u, V=v, sigma=s, Q=Q, R=R,
                frf=frf, fbb=fbb, scale=np.array([scale]), r_norm=r_norm,
                diag_b=diag_b)


def field_list(c):
    """(name, array, element_count) -- the binary contract with rx_main.cpp."""
    Nr, rows, d, T = c["Nr"], c["rows"], c["d"], c["T"]
    return [("U", c["U"], Nr * d),        ("V", c["V"], rows * d),
            ("sigma", c["sigma"], d),     ("Q", c["Q"], Nr * d),
            ("R", c["R"], d * d),         ("frf", c["frf"], NT * rows),
            ("fbb", c["fbb"], rows * T),  ("scale", c["scale"], 1),
            ("r_norm", c["r_norm"], d * d), ("diag_b", c["diag_b"], d)]


def main():
    out = [struct.pack("<i", len(CASES))]
    for spec in CASES:
        c = build_case(*spec)
        out.append(struct.pack("<6i2d", c["Nr"], c["rows"], c["d"], c["Nt"],
                               c["T"], ord(c["strat"]), c["snr"],
                               c["p_constraint"]))
        for name, arr, count in field_list(c):
            a = np.ascontiguousarray(arr)
            assert a.size == count, (name, a.size, count)
            dtype = np.complex128 if np.iscomplexobj(a) else np.float64
            a = a.astype(dtype)
            assert np.all(np.isfinite(a.view(np.float64))), (name, a)
            out.append(a.tobytes())
    pathlib.Path("build").mkdir(exist_ok=True)
    blob = b"".join(out)
    pathlib.Path("build/rx_vectors.bin").write_bytes(blob)
    print(f"wrote build/rx_vectors.bin ({len(CASES)} cases, {len(blob)} bytes)")


if __name__ == "__main__":
    main()
