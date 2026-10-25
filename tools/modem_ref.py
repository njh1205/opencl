#!/usr/bin/env python3
"""Oracle for the receive-side modem reference (ref/modem.cpp).

Ground truth is the production python in mix_precode/classical:
  thp_precoding : qam_axis_order_and_spacing, centered_modulo_complex
  bicm_metrics  : get_constellation, modulate_bits_with_labeling,
                  bit_llr_from_distances, enumerate_thp_symbol_replicas,
                  stable_binary_logloss_from_signed_llr,
                  estimate_bicm_gmi_parallel_from_received,
                  estimate_bicm_gmi_thp_from_received
Scope: gray_standard labeling, square QAM M in {16, 64, 256}.

Metric convention being pinned down (n0 = E|noise|^2 per complex sample):
  d2      = |y - candidate|^2,   metric = -d2 / n0
  llr_b   = logsumexp_{label bit b = 1}(metric)
          - logsumexp_{label bit b = 0}(metric)          (positive -> bit 1)
  folded  : candidates = constellation + period*(lr + i*li), |lr|,|li| <= 2
  GMI     = m - mean_i sum_b log2(1 + exp(-(2 c_b - 1) llr_b))
The production *_from_received estimators reach the same metric through a unit
diagonal channel and snr_per_stream = 1/n0; the parallel one scales only the
constellation (so it takes y/sqrt(n0)), the THP one scales received sample and
replicas alike (so it takes the folded y itself). Both are called here and must
agree with the GMI computed from the LLRs above -- that equivalence is what
runtime/modem_main.cpp then re-checks against the C++ port.

Emits build/modem_vectors.bin. Layout, little endian throughout:
  int32 ncase
  per case: int32 M, int32 n, float64 n0, float64 period, then field_list()
  field dtype: complex128 (re, im) / float64 / uint8 according to the array
The per-case field order in field_list() is the binary contract with
runtime/modem_main.cpp -- keep the two in sync.
"""
import struct
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]
                       / "mix_precode" / "classical"))
from bicm_metrics import (bit_llr_from_distances, enumerate_thp_symbol_replicas,
                          estimate_bicm_gmi_parallel_from_received,
                          estimate_bicm_gmi_thp_from_received, get_constellation,
                          modulate_bits_with_labeling,
                          stable_binary_logloss_from_signed_llr)
from thp_precoding import centered_modulo_complex, qam_axis_order_and_spacing

NSYM = 512            # symbols per case
REPLICA_RADIUS = 2    # |l| <= 2 shifts per axis in the folded branch
ORDERS = (16, 64, 256)
NOISE = (0.5, 0.1, 0.02)          # n0: 3 dB, 10 dB, 17 dB per complex symbol
SEED0 = 20261024
CASES = [(M, n0, SEED0 + 97 * k)
         for k, (M, n0) in enumerate((M, n0) for M in ORDERS for n0 in NOISE)]


def deficit(bits, llr):
    """mean_i sum_b log2(1 + exp(-(2 c_b - 1) llr_b)) via the production loss."""
    per_bit = [[stable_binary_logloss_from_signed_llr((2 * int(c) - 1) * value)
                for c, value in zip(row_c, row_l)]
               for row_c, row_l in zip(bits, llr)]
    return float(np.sum(np.asarray(per_bit), axis=1).mean())


def branch_llr(y, cand, cand_bits, m, n0):
    """Per-bit logsumexp LLRs [n, m] using bicm_metrics.bit_llr_from_distances."""
    llr = np.zeros((y.shape[0], m))
    zero = [cand_bits[:, b] == 0 for b in range(m)]
    for i in range(y.shape[0]):
        d2 = np.abs(y[i] - cand) ** 2 / n0
        for b in range(m):
            llr[i, b] = bit_llr_from_distances(d2, zero[b], ~zero[b])
    return llr


def build_case(M, n0, seed):
    """Bits -> gray_standard symbols -> AWGN -> parallel and THP-folded metrics."""
    m = int(round(np.log2(M)))
    rng = np.random.default_rng(seed)
    symbols, bit_table = get_constellation(m, labeling="gray_standard")
    m_axis, spacing = qam_axis_order_and_spacing(m)
    period = m_axis * spacing

    idx = rng.integers(0, M, NSYM)                       # constellation index
    bits = bit_table[idx].astype(np.uint8)               # [n, m]
    sym = modulate_bits_with_labeling(m, bits.reshape(-1), "gray_standard")
    assert np.array_equal(sym, symbols[idx]), "bits -> symbol round trip"

    def awgn():
        return np.sqrt(n0 / 2.0) * (rng.standard_normal(NSYM)
                                    + 1j * rng.standard_normal(NSYM))

    y_par = sym + awgn()                       # combined received, parallel
    y_thp = sym + awgn()                       # equalized THP received, pre-modulo
    z_fold = centered_modulo_complex(y_thp, period)   # receive modulo fold

    rep_sym, rep_bits = enumerate_thp_symbol_replicas(m, REPLICA_RADIUS,
                                                      "gray_standard")
    assert rep_sym.size == (2 * REPLICA_RADIUS + 1) ** 2 * M
    llr_par = branch_llr(y_par, symbols, bit_table, m, n0)
    llr_fold = branch_llr(z_fold, rep_sym, rep_bits, m, n0)

    # GMI straight from the LLRs (the formula ref/modem.cpp implements) ...
    mine = np.array([m - deficit(bits, llr_par), m - deficit(bits, llr_fold)])
    # ... must equal the production end-to-end estimators, whose values are the
    # ones written out as the ground truth.
    prod = np.array([
        estimate_bicm_gmi_parallel_from_received(
            np.ones((1, 1), complex), 1.0 / n0, m, idx.reshape(-1, 1),
            (y_par / np.sqrt(n0)).reshape(-1, 1)),
        estimate_bicm_gmi_thp_from_received(
            np.ones((1, 1), complex), 1.0 / n0, m, idx.reshape(-1, 1),
            z_fold.reshape(-1, 1), replica_radius=REPLICA_RADIUS),
    ])
    # Production clamps the aggregate at zero; with the single stream this case
    # uses, the aggregate IS the per-stream value, so the clamp is modelled
    # explicitly rather than avoided by an assertion that the operating point
    # stays above it. (Measured: a matched demodulator keeps the per-stream GMI
    # positive down to -20 dB, so `clamped == mine` on every case shipped here --
    # the max() is the contract, not an active correction.)
    clamped = np.maximum(0.0, mine)
    assert np.all(np.abs(prod - clamped) < 1e-9), (M, n0, prod, clamped)

    return dict(M=M, m=m, n0=n0, period=period, bits=bits.reshape(-1),
                sym=sym, y_par=y_par, y_thp=y_thp, z_fold=z_fold,
                llr_par=llr_par.reshape(-1), llr_fold=llr_fold.reshape(-1),
                gmi=prod)


def field_list(c):
    """(name, array, element_count) -- the binary contract with modem_main.cpp."""
    n, m = NSYM, c["m"]
    return [("bits", c["bits"], n * m),     ("sym", c["sym"], n),
            ("y_par", c["y_par"], n),       ("y_thp", c["y_thp"], n),
            ("z_fold", c["z_fold"], n),     ("llr_par", c["llr_par"], n * m),
            ("llr_fold", c["llr_fold"], n * m), ("gmi", c["gmi"], 2)]


def main():
    out = [struct.pack("<i", len(CASES))]
    for spec in CASES:
        c = build_case(*spec)
        out.append(struct.pack("<2i2d", c["M"], NSYM, c["n0"], c["period"]))
        for name, arr, count in field_list(c):
            a = np.ascontiguousarray(arr)
            assert a.size == count, (name, a.size, count)
            dtype = (np.complex128 if np.iscomplexobj(a)
                     else np.uint8 if a.dtype.kind == "u" else np.float64)
            out.append(a.astype(dtype).tobytes())
    pathlib.Path("build").mkdir(exist_ok=True)
    blob = b"".join(out)
    pathlib.Path("build/modem_vectors.bin").write_bytes(blob)
    print(f"wrote build/modem_vectors.bin ({len(CASES)} cases x {NSYM} symbols, "
          f"{len(blob)} bytes)")


if __name__ == "__main__":
    main()
