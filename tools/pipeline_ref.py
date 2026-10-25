#!/usr/bin/env python3
"""Oracle for the per-scheduling-instant control-plane pipeline.

Exported quantities are gauge-invariant: LAPACK and the deployed Jacobi SVD
pick different singular-vector phases, and the BD basis / analog precoder
carry their own unitary gauges, so the physical precoder
P_k = F_RF N_k V_D,k is exported instead of N_k and V_D,k separately.
Composes the production python (AnalogPrecoder, BD/whitening helpers,
DigitalStructuredPrecoder) with the independently validated threshold
policy of tools/cf_ref.py (closed form + converged Newton, +-inf when the
branch ordering is uniform). Emits build/pipeline_vectors.bin."""
import sys, struct, pathlib
import numpy as np
from scipy.linalg import svd
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "mix_precode" / "classical"))
sys.path.insert(0, str(HERE))
from multiuser_simulation_environment import MultiUserSimulationEnvironment as Env
from env_common import dummy_env   # shared dimension shell
from analog_precoder import AnalogPrecoder
from digital_precoder import DigitalStructuredPrecoder
import cf_ref

CASES = [(2, 4, 2, 8, 2, 64, 0), (2, 4, 2, 8, 2, 64, 1), (3, 6, 2, 12, 2, 16, 0)]
SNR_DB = (18.0, 30.0, 22.0)     # one below-ish, one above-ish threshold region


def main():
    rng = np.random.default_rng(33)
    out, ncase = [], []
    for (K, Nt, Nr, NRF, d, M, variant), s_db in zip(CASES, SNR_DB):
        env = dummy_env(K, Nt, Nr, NRF, d)
        H = (rng.normal(0, 1, (K, Nr, Nt)) + 1j * rng.normal(0, 1, (K, Nr, Nt))) / np.sqrt(2)
        Frf = AnalogPrecoder(Nt, NRF).build_precoder(H, d)
        eff = env.build_effective_channels(H, Frf)
        nul = NRF - (K - 1) * Nr
        sig, Vd, B, rr, rt, Nk, Pk = [], [], [], [], [], [], []
        lam = []
        for k in range(K):
            N = env.build_bd_digital_basis(eff, k)
            Nk.append(N)
            _, _, G = env.build_power_whitened_bd_channel(eff, Frf, k, bd_basis=N)
            u, s, vh = svd(G, full_matrices=False)
            sig.append(s[:d]); lam += list(s[:d] ** 2)
        tau, flag = cf_ref.tau_policy(np.array(lam), M, d, K, variant)
        print(f"  case K={K} var={variant} s={s_db}dB: tau={tau:.6f} dB (flag {flag})")
        mode = 1 if s_db >= tau else 0
        prec = DigitalStructuredPrecoder(NRF, d)
        snr_ps = 10 ** (s_db / 10) / (K * d)
        for k in range(K):
            N = Nk[k]
            _, _, G = env.build_power_whitened_bd_channel(eff, Frf, k, bd_basis=N)
            u, s, vh = svd(G, full_matrices=False)
            v_eff = vh.conj().T[:, :d]
            # canonical SVD gauge, same rule as ref/pipeline.cpp: make the
            # largest entry of G V (= sigma_j u_j) real positive, so that the
            # exported physical precoder is gauge-free
            gm = G @ v_eff
            for j in range(d):
                z = gm[np.argmax(np.abs(gm[:, j])), j]
                v_eff[:, j] = v_eff[:, j] * np.conj(z / abs(z))
            strat = ("svd", "gmd", "ucd")[0 if mode == 0 else (1 + variant)]
            # design_precoder() would redo the SVD internally and lose the gauge
            vd, r_tgt = prec.design_from_svd(v_eff=v_eff, singular_values=s[:d],
                                             snr_per_stream=snr_ps, strategy=strat)
            r_real = prec.compute_stream_snr(G, vd, snr_ps)
            if mode == 0:
                b = np.eye(d, dtype=complex)
            elif variant == 0:
                b = prec.design_gmd_transceiver_from_svd(u, vh.conj().T, s[:d]).r_gmd
            else:
                b = prec.design_ucd_transceiver_from_svd(u, vh.conj().T, s[:d], snr_ps).r_aug_ucd
            Vd.append(vd); B.append(b); rr.append(r_real); rt.append(r_tgt)
            Pk.append(Frf @ N @ vd)             # physical precoder, gauge-free
        ncase.append(1)
        out.append(struct.pack("<8i2d", K, Nt, Nr, NRF, d, M, variant, nul,
                               s_db, float(tau)))
        out.append(np.ascontiguousarray(H).astype(np.complex128).tobytes())
        out.append(np.ascontiguousarray(np.stack(Pk)).astype(np.complex128).tobytes())
        out.append(np.asarray(sig, np.float64).tobytes())          # [K][d]
        out.append(np.ascontiguousarray(np.stack(B)).astype(np.complex128).tobytes())
        out.append(np.asarray(rr, np.float64).tobytes())           # [K][d]
        out.append(np.asarray(rt, np.float64).tobytes())           # [K][d]
        out.append(struct.pack("<i", mode))
    pathlib.Path("build").mkdir(exist_ok=True)
    pathlib.Path("build/pipeline_vectors.bin").write_bytes(
        struct.pack("<i", sum(ncase)) + b"".join(out))
    print(f"wrote build/pipeline_vectors.bin ({sum(ncase)} cases)")


if __name__ == "__main__":
    main()
