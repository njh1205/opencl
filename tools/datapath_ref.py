#!/usr/bin/env python3
"""fp64 numpy oracle for the per-symbol PL datapath (kernels/datapath.cl).

Standalone on purpose: the datapath math below is written out from the formulas
so the oracle cannot inherit a bug from the code under test. The conventions
themselves are *not* invented here -- check_production() pins them against the
production modem (mix_precode/classical/bicm_metrics.py, thp_precoding.py) on
every run, and ref/modem.cpp is the same port bit-for-bit (read-only, not
included). Emits build/datapath_vectors.bin.

Per case (T = U*d streams over one shared channel, all fp64 complex):
  bits[T,nsym,m] --gray QAM--> s[T,nsym] --Frf@Fbb--> x[Nt,nsym]
    --H@x+n--> y[Nr,nsym] --W_u^H / diag_b--> z[T,nsym]
      parallel branch : z ----------------------------> llr  -> gmi
      THP folded branch: z + period*shifts --modulo--> zf --> llr_fold -> gmi_fold
W_u holds the d columns of the zero-forcing combiner of user u's streams scaled
so that W^H G = diag(diag_b); dividing by diag_b (thp_receive_equalized) then
restores unit gain per stream and the soft demodulators can use one scalar noise
variance n0 = 1/snr_lin (the mean post-equalization variance; a production
datapath would carry a per-stream vector).

The field_list() order below is the binary contract with
runtime/datapath_main.cpp -- keep the two in sync.
"""
import struct, pathlib, sys
import numpy as np
from modem_common import (bit_cols, check_production, constellation, folded_candidates,
                              gmi, gray_decode, logsumexp, mod_fold, modulo_period,
                              pam_amps, pam_delta, qam_map, soft_llr)

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]
                       / "mix_precode" / "classical"))

NEDGE = 8          # fold-edge probes appended to the THP branch input
CASES = [          # Nr >= 2T keeps ZF noise sane; diag=None -> NULL diag_b
    dict(Nt=8,  Nr=8,  NRF=4, U=2, d=2, nsym=64, m=4, snr_db=10.0, radius=2,
         diag=(1.3 + 0.20j, 0.8 - 0.15j, 1.15 + 0.05j, 0.9 - 0.25j)),
    dict(Nt=16, Nr=12, NRF=6, U=2, d=3, nsym=48, m=6, snr_db=16.0, radius=2,
         diag=None),
    dict(Nt=4,  Nr=6,  NRF=2, U=1, d=2, nsym=32, m=4, snr_db=4.0, radius=1,
         diag=(1.3, 0.8)),
    dict(Nt=8,  Nr=6,  NRF=2, U=1, d=2, nsym=16, m=8, snr_db=12.0, radius=1,
         diag=None),
]


def edge_probes(period):
    """NEDGE probes hugging the fold edges on both axes, >=1e-5 (~100 fp32 ulp)
    away from +-period/2: the fold is discontinuous there, so a closer point
    could round onto the other side in fp32 and jump by a whole period."""
    off = np.array([1e-5, 1e-3, -1e-5, -1e-3])
    re = np.concatenate([period / 2 + off, -period / 2 + off])
    return re + 1j * re[::-1]


def build_case(spec, rng):
    Nt, Nr, NRF, U, d, nsym, m = (spec[k] for k in
                                  ("Nt", "Nr", "NRF", "U", "d", "nsym", "m"))
    T, radius, period = U * d, spec["radius"], modulo_period(spec["m"])
    assert m % 2 == 0 and Nr >= T and NRF >= T, "square QAM, Nr/NRF >= T"
    bits = rng.integers(0, 2, (T, nsym, m)).astype(np.uint8)
    pts, lab = constellation(m)
    s = qam_map(bits, m)

    def cn(shape):                     # circularly symmetric CN(0, 1)
        return (rng.normal(size=shape) + 1j * rng.normal(size=shape)) / np.sqrt(2)

    Frf = np.exp(1j * rng.uniform(0, 2 * np.pi, (Nt, NRF))) / np.sqrt(NRF)
    Fbb, _ = np.linalg.qr(cn((NRF, T)))               # orthonormal columns
    Fbb = Fbb * np.sqrt(T) / np.linalg.norm(Frf @ Fbb)  # ||Frf Fbb||_F^2 = T
    F = Frf @ Fbb

    H = cn((Nr, Nt))
    x = F @ s
    G = H @ F                                         # [Nr,T] effective channel
    Gi = np.linalg.inv(G.conj().T @ G)
    # production derives diag_b = diag(W^H H P); here it is prescribed and the
    # ZF combiner is scaled per stream so that W^H G = diag(diag_b) exactly.
    diag_b = (np.ones(T, complex) if spec["diag"] is None
              else np.asarray(spec["diag"], complex))
    assert diag_b.size == T
    Wzf = (G @ Gi) * diag_b.conj()[None, :]
    W = np.stack([Wzf[:, u * d:(u + 1) * d] for u in range(U)])   # [U,Nr,d]

    n0 = 10.0 ** (-spec["snr_db"] / 10.0)             # post-equalization var
    n0_ch = n0 / (Gi.real.diagonal() / np.abs(diag_b) ** 2).mean()
    n = np.sqrt(n0_ch) * cn((Nr, nsym))
    y = H @ x + n
    z = (np.stack([W[u].conj().T @ y for u in range(U)]).reshape(T, nsym)
         / diag_b[:, None])

    # THP branch: residual replica offsets the receive modulo removes, then the
    # fold-edge probes (they are demodulated and compared, but not part of any
    # stream, so gmi_fold only sees the first T*nsym folded symbols).
    off = rng.integers(-radius, radius + 1, (2, T, nsym))
    zf_in = np.concatenate([(z + period * (off[0] + 1j * off[1])).ravel(),
                            edge_probes(period)])
    zf = mod_fold(zf_in, period)

    llr = soft_llr(z, pts, lab, n0)                   # parallel branch
    llr_fold = soft_llr(zf, *folded_candidates(m, radius), n0)
    return dict(spec, T=T, period=period, n0=n0, diag_b=diag_b, bits=bits,
                Frf=Frf, Fbb=Fbb, H=H, n=n, W=W, s=s, x=x, y=y, z=z,
                zf_in=zf_in, zf=zf, llr=llr, llr_fold=llr_fold,
                gmi=gmi(llr, bits, m),
                gmi_fold=gmi(llr_fold[:T * nsym].reshape(T, nsym, m), bits, m))


def field_list(c):
    """(name, array, element_count) -- the binary contract."""
    Nt, Nr, NRF, U, d, nsym, m, T = (c[k] for k in
                                     ("Nt", "Nr", "NRF", "U", "d", "nsym", "m", "T"))
    nf = T * nsym + NEDGE
    return [("bits", c["bits"], T * nsym * m), ("Frf", c["Frf"], Nt * NRF),
            ("Fbb", c["Fbb"], NRF * T),        ("H", c["H"], Nr * Nt),
            ("n", c["n"], Nr * nsym),          ("W", c["W"], U * Nr * d),
            ("diag_b", c["diag_b"], T),        ("s", c["s"], T * nsym),
            ("x", c["x"], Nt * nsym),          ("y", c["y"], Nr * nsym),
            ("z", c["z"], T * nsym),           ("zf_in", c["zf_in"], nf),
            ("zf", c["zf"], nf),               ("llr", c["llr"], T * nsym * m),
            ("llr_fold", c["llr_fold"], nf * m), ("gmi", c["gmi"], T),
            ("gmi_fold", c["gmi_fold"], T)]


def main():
    np.seterr(all="ignore")   # macOS Accelerate raises spurious FP flags in @
    check_production()
    rng = np.random.default_rng(20261024)
    out = [struct.pack("<i", len(CASES))]
    for spec in CASES:
        c = build_case(spec, rng)
        out.append(struct.pack("<10i2d", c["Nt"], c["Nr"], c["NRF"], c["U"],
                               c["d"], c["nsym"], c["m"], c["radius"],
                               int(spec["diag"] is not None), NEDGE,
                               c["n0"], c["period"]))
        for name, arr, n in field_list(c):
            a = np.ascontiguousarray(arr)
            assert a.size == n and np.isfinite(a).all(), (spec, name, a.size, n)
            dt = (np.complex128 if np.iscomplexobj(a)
                  else np.uint8 if a.dtype == np.uint8 else np.float64)
            out.append(a.astype(dt).tobytes())
        print(f"  case Nt={c['Nt']} Nr={c['Nr']} U={c['U']} d={c['d']} m={c['m']} "
              f"r={c['radius']} diag={'yes' if spec['diag'] else 'NULL'} "
              f"snr={c['snr_db']:.0f}dB: GMI {c['gmi'].mean():.3f} / folded "
              f"{c['gmi_fold'].mean():.3f} of {c['m']}, max |LLR| "
              f"{np.abs(c['llr_fold']).max():.1f}")
    pathlib.Path("build").mkdir(exist_ok=True)
    pathlib.Path("build/datapath_vectors.bin").write_bytes(b"".join(out))
    print(f"wrote build/datapath_vectors.bin ({len(CASES)} cases)")


if __name__ == "__main__":
    main()
