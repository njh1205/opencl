#!/usr/bin/env python3
"""End-to-end chain oracle (modulation/demodulation full loop).

Ground truth is the production python in mix_precode/classical. The control
plane reuses the validated pipeline_ref pattern (AnalogPrecoder, BD basis,
power whitening, gauged SVD, cf_ref.tau_policy, DigitalStructuredPrecoder,
rx_ref.power_scale) -- nothing is re-implemented here. The datapath follows
the same formulas as tools/datapath_ref.py (gray_standard QAM, log-MAP LLR,
modulo receiver) and the THP feedback uses thp_precoding.thp_precoder_from_b1.
The modem stage is cross-checked against the production GMI estimators on
every run (assert < 1e-6). Emits build/chain_vectors.bin.

Binary layout, little endian throughout:
  int32 ncase
  per case: int32 K,Nt,Nr,NRF,d,M,variant,nul,nsym, float64 s_db,tau_db,
            float64 pscale, int32 mode   (pscale = power normalisation factor)
  then field_list(), in this exact order (runtime/chain_main.cpp mirrors it):
    bits  u8    [T*nsym*m ]  transmitted bits, stream-major (stream = k*d+j)
    H     c128  [K*Nr*Nt  ]  per-user channels
    Frf   c128  [Nt*NRF   ]  phase-only RF precoder
    Fbb   c128  [NRF*T    ]  digital blocks N_k C_k^{-1/2} V_D,k, *pscale
    Q     c128  [K*Nr*d   ]  receive combiners: u_eff / q_gmd / w_ucd
    R     c128  [K*d*d    ]  equalized upper factors, *pscale (diag_b diag)
    diagb c128  [K*d      ]  pscale * diag(R): receiver divide (GMD/SVD only)
    B     c128  [K*d*d    ]  THP feedback upper: I (SVD) / R (GMD/UCD)
    n     c128  [K*Nr*nsym]  AWGN (seed-fixed; reused by C++ and kernels)
    s0    c128  [T*nsym   ]  mapped constellation symbols
    sfull c128  [T*nsym   ]  after per-user THP feedback (SVD: = s0)
    x     c128  [Nt*nsym  ]  Frf Fbb sfull
    y     c128  [K*Nr*nsym]  H_k x + n_k
    z     c128  [T*nsym   ]  Q_k^H y_k, / diagb for SVD and GMD (UCD: raw)
    zf    c128  [T*nsym   ]  centered modulo of z (THP receive)
    llr   f64   [T*nsym*m ]  logsumexp LLRs, positive favours bit 1
    gmi   f64   [T        ]  per-stream GMI in bits/symbol
    ber   f64   [T        ]  per-stream hard-decision bit error rate
"""
import sys, struct, pathlib
import numpy as np
from modem_common import (bit_cols, check_production, constellation, folded_candidates,
                              gray_decode, logsumexp, mod_fold, modulo_period, pam_amps,
                              pam_delta, qam_map, soft_llr)
from modem_common import gmi as gmi_per_stream   # same function, one name here
from scipy.linalg import svd

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "mix_precode" / "classical"))
sys.path.insert(0, str(HERE))
np.seterr(all="ignore")        # macOS Accelerate BLAS FP warnings in @

from multiuser_simulation_environment import MultiUserSimulationEnvironment as Env
from analog_precoder import AnalogPrecoder
from digital_precoder import DigitalStructuredPrecoder
from thp_precoding import thp_precoder_from_b1
from bicm_metrics import (get_constellation, modulate_bits_with_labeling,
                          enumerate_thp_symbol_replicas,
                          estimate_bicm_gmi_parallel_from_received,
                          estimate_bicm_gmi_thp_from_received)
import cf_ref
from env_common import dummy_env          # shared dimension shell (was: pipeline_ref)
from rx_ref import power_scale

NSYM = 128
RADIUS = 2                     # enumerate_thp_symbol_replicas ring count
SEED = 20261024
# (K, Nt, Nr, NRF, d, M, variant, s_db). Nt >= NRF keeps the BD power
# whitening C = (F_rf N)^H (F_rf N) full rank (the suggested Nt < NRF shapes
# make C singular, so Fbb/R/pscale become numerically unstable and cannot be
# compared at 1e-9). s_db is chosen away from the seed's switching threshold
# to pin each branch: 13 < tau(16.8) -> SVD (parallel), 18 > tau(13.6) ->
# GMD/THP (folded), 19 > tau(15.0) -> MMSE-UCD/THP.
CASES = [
    (2, 8, 2, 4, 2, 64, 0, 13.0),
    (3, 12, 2, 6, 2, 16, 0, 18.0),
    (2, 12, 2, 6, 2, 64, 1, 19.0),
]


def ber_per_stream(llr, bits):
    return ((llr > 0) != bits).reshape(bits.shape[0], -1).mean(axis=1)

def build_case(spec, rng):
    K, Nt, Nr, NRF, d, M, variant, s_db = spec
    T, m = K * d, int(round(np.log2(M)))
    env = dummy_env(K, Nt, Nr, NRF, d)
    prec = DigitalStructuredPrecoder(NRF, d)
    snr_ps = 10 ** (s_db / 10) / T
    n0 = 1.0 / snr_ps
    period = modulo_period(m)

    # ---- control plane (pipeline_ref pattern) ----------------------------
    H = (rng.normal(0, 1, (K, Nr, Nt)) + 1j * rng.normal(0, 1, (K, Nr, Nt))) / np.sqrt(2)
    Frf = AnalogPrecoder(Nt, NRF).build_precoder(H, d)
    eff = env.build_effective_channels(H, Frf)
    nul = NRF - (K - 1) * Nr
    lam, gbar, uu, ss, vv = [], [], [], [], []
    for k in range(K):
        N = env.build_bd_digital_basis(eff, k)
        _, _, G = env.build_power_whitened_bd_channel(eff, Frf, k, bd_basis=N)
        u, s, vh = svd(G, full_matrices=False)
        lam += list(s[:d] ** 2)
        gbar.append(G); uu.append(u); ss.append(s); vv.append(vh.conj().T)
    tau, _ = cf_ref.tau_policy(np.array(lam), M, d, K, variant)
    mode = 1 if s_db >= tau else 0                    # 0 SVD, 1 THP
    strat = ("svd", "gmd", "ucd")[0 if mode == 0 else 1 + variant]

    Fbb_blocks, Qs, R_un, Bs = [], [], [], []
    for k in range(K):
        G, u, s, v = gbar[k], uu[k], ss[k], vv[k]
        v_eff, u_eff = v[:, :d].copy(), u[:, :d].copy()
        # canonical SVD gauge (ref/pipeline.cpp): largest entry of G V real
        gm = G @ v_eff
        for j in range(d):
            z = gm[np.argmax(np.abs(gm[:, j])), j]
            ph = np.conj(z / abs(z))
            v_eff[:, j] *= ph
            u_eff[:, j] *= ph
        vd, _ = prec.design_from_svd(v_eff=v_eff, singular_values=s[:d],
                                     snr_per_stream=snr_ps, strategy=strat)
        if strat == "svd":
            q, r, b = u_eff, np.diag(s[:d]).astype(complex), np.eye(d, dtype=complex)
        elif strat == "gmd":
            blk = prec.design_gmd_transceiver_from_svd(u_eff, v_eff, s[:d])
            q, r, b = blk.q_gmd, blk.r_gmd, blk.r_gmd
        else:
            blk = prec.design_ucd_transceiver_from_svd(
                u_eff, v_eff, s[:d], snr_per_stream=snr_ps,
                alpha_override=1.0 / max(float(snr_ps), 1e-12))
            q, r, b = blk.w_ucd, blk.r_aug_ucd, blk.r_aug_ucd
        N = env.build_bd_digital_basis(eff, k)
        _, whitening, _ = env.build_power_whitened_bd_channel(eff, Frf, k, bd_basis=N)
        Fbb_blocks.append(N @ whitening.inverse_sqrt @ vd)   # f_bb, user block
        Qs.append(q); R_un.append(r); Bs.append(b)
    Fbb = np.hstack(Fbb_blocks)                        # [NRF x T], unnormalised
    pscale = power_scale(Frf, Fbb, float(T))           # rx_ref normalisation
    Fbb = Fbb * pscale
    R = np.stack([p * r for p, r in zip(np.full(K, pscale), R_un)])
    Q = np.stack(Qs)
    diagb = np.stack([np.diag(r) for r in R])
    B = np.stack([np.eye(d, dtype=complex) if mode == 0 else r for r in R])

    # ---- transmit datapath ------------------------------------------------
    bits = rng.integers(0, 2, (T, nsym := NSYM, m), dtype=np.uint8)
    s0 = qam_map(bits, m)                              # [T, nsym]
    if mode == 0:
        sfull = s0
    else:
        dp = []
        for k in range(K):
            cod = s0[k * d:(k + 1) * d]
            b1 = R[k] / np.diag(R[k])[:, None]         # row / diag (scale-free)
            dp.append(thp_precoder_from_b1(cod, b1, m))
        sfull = np.vstack(dp)                          # [T, nsym], original order
    x = Frf @ (Fbb @ sfull)                            # [Nt, nsym]
    n = np.sqrt(n0 / 2.0) * (rng.standard_normal((K, Nr, nsym))
                             + 1j * rng.standard_normal((K, Nr, nsym)))
    y = np.einsum("kra,as->krs", H, x) + n             # [K, Nr, nsym]

    # ---- receive datapath --------------------------------------------------
    z = np.vstack([Q[k].conj().T @ y[k] for k in range(K)])   # [T, nsym]
    if mode == 0 or variant == 0:                      # UCD: w_ucd pre-normalised
        z = z / diagb.reshape(-1, 1)
    zf = mod_fold(z.reshape(-1), period).reshape(T, nsym)

    if mode == 0:
        cand, clab = constellation(m)
        llr = soft_llr(z.reshape(-1), cand, clab, n0).reshape(T, nsym, m)
    else:
        cand, clab = folded_candidates(m, RADIUS)
        if variant == 0:
            llr = soft_llr(zf.reshape(-1), cand, clab, n0).reshape(T, nsym, m)
        else:                                          # per-stream noise floor
            qn = (np.abs(Q) ** 2).sum(axis=1)          # [K,d] = ||Q_k col j||^2
            llr = np.empty((T, nsym, m))
            for st in range(T):
                k, j = st // d, st % d
                llr[st] = soft_llr(zf[st], cand, clab, n0 * qn[k, j])
    gmi = gmi_per_stream(llr, bits, m)                 # [T]
    ber = ber_per_stream(llr, bits)                    # [T]

    # ---- production GMI cross-check (modem part; scalar-n0 branches) ------
    if variant == 0:
        mh = m // 2
        w = 1 << np.arange(mh - 1, -1, -1)
        lvl = (gray_decode((bits[..., :mh] * w).sum(-1)) * (1 << mh)
               + gray_decode((bits[..., mh:] * w).sum(-1)))
        if mode == 0:
            prod = estimate_bicm_gmi_parallel_from_received(
                np.ones(T, complex), 1.0 / n0, m, lvl.T, (z.T) / np.sqrt(n0))
        else:
            prod = estimate_bicm_gmi_thp_from_received(
                np.ones(T, complex), 1.0 / n0, m, lvl.T, zf.T,
                replica_radius=RADIUS)
        assert prod > 0.05, (spec, prod)               # production clamps at 0
        # the estimator returns total bits over the T streams (ideal_rate =
        # T*m), i.e. the sum of the per-stream GMIs
        # production clamps the AGGREGATE at zero (bicm_metrics returns
        # max(0.0, num_streams*m - loss)); our per-stream gmi is unclamped by
        # contract (see kernels/datapath.cl gmi_k), so the clamp belongs here.
        agg = max(0.0, float(gmi.sum()))
        assert abs(prod - agg) < 1e-6, (spec, prod, agg)

    return dict(K=K, Nt=Nt, Nr=Nr, NRF=NRF, d=d, M=M, variant=variant,
                nul=nul, nsym=nsym, s_db=s_db, tau=float(tau), pscale=pscale,
                mode=mode,
                bits=bits, H=H, Frf=Frf, Fbb=Fbb, Q=Q, R=R, diagb=diagb, B=B,
                n=n, s0=s0, sfull=sfull, x=x, y=y, z=z, zf=zf,
                llr=llr.reshape(-1), gmi=gmi, ber=ber)


def field_list(c):
    """(name, array, element_count) -- the binary contract with chain_main.cpp."""
    K, Nr, Nt, NRF, d = (c[k] for k in ("K", "Nr", "Nt", "NRF", "d"))
    T, m, nsym = K * d, int(round(np.log2(c["M"]))), c["nsym"]
    return [("bits", c["bits"], T * nsym * m), ("H", c["H"], K * Nr * Nt),
            ("Frf", c["Frf"], Nt * NRF), ("Fbb", c["Fbb"], NRF * T),
            ("Q", c["Q"], K * Nr * d), ("R", c["R"], K * d * d),
            ("diagb", c["diagb"], K * d), ("B", c["B"], K * d * d),
            ("n", c["n"], K * Nr * nsym), ("s0", c["s0"], T * nsym),
            ("sfull", c["sfull"], T * nsym), ("x", c["x"], Nt * nsym),
            ("y", c["y"], K * Nr * nsym), ("z", c["z"], T * nsym),
            ("zf", c["zf"], T * nsym), ("llr", c["llr"], T * nsym * m),
            ("gmi", c["gmi"], T), ("ber", c["ber"], T)]


def main():
    # one shared check (tools/modem_common.py) covering every modulation order
    # and replica radius this oracle uses; GMI itself is asserted against the
    # production estimators per case in build_case().
    check_production(tuple((int(round(np.log2(M))), RADIUS) for M in (16, 64)))
    rng = np.random.default_rng(SEED)
    out = [struct.pack("<i", len(CASES))]
    for spec in CASES:
        c = build_case(spec, rng)
        out.append(struct.pack("<9i3di", *(c[k] for k in
                       ("K", "Nt", "Nr", "NRF", "d", "M", "variant", "nul",
                        "nsym", "s_db", "tau", "pscale", "mode"))))
        for name, arr, count in field_list(c):
            a = np.ascontiguousarray(arr)
            assert a.size == count, (spec, name, a.size, count)
            dtype = (np.complex128 if np.iscomplexobj(a)
                     else np.uint8 if a.dtype.kind == "u" else np.float64)
            a = a.astype(dtype)
            if a.dtype.kind != "u":                    # bits are 0/1 by construction
                assert np.all(np.isfinite(a.view(np.float64))), (spec, name)
            out.append(a.tobytes())
        print(f"  case K={c['K']} var={c['variant']} s={c['s_db']}dB "
              f"tau={c['tau']:.3f}dB mode={c['mode']} "
              f"GMI={c['gmi'].mean():.4f}/{c['M']}")
    pathlib.Path("build").mkdir(exist_ok=True)
    blob = b"".join(out)
    pathlib.Path("build/chain_vectors.bin").write_bytes(blob)
    print(f"wrote build/chain_vectors.bin ({len(CASES)} cases, {len(blob)} bytes)")


if __name__ == "__main__":
    main()
