#!/usr/bin/env python3
"""Oracle for the multi-user RF-BD front end (ref/frontend.cpp).
Ground truth: MultiUserSimulationEnvironment.build_effective_channels /
build_bd_digital_basis / build_power_whitened_bd_channel, instantiated
without the channel model (the three functions only need K, N_RF, d).
Basis comparisons are subspace-level (null space has rotational freedom),
so the bench checks projectors and sigma(Gbar), not elements.
"""
import sys, struct, pathlib
import numpy as np
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "mix_precode" / "classical"))
from multiuser_simulation_environment import MultiUserSimulationEnvironment as Env
from env_common import dummy_env   # shared dimension shell
from analog_precoder import AnalogPrecoder

# non-degenerate: Nt > (K-1)*Nr so F_RF N has full column rank beyond
# the trivial null(F_RF) directions
CASES = [(2, 4, 2, 8, 2), (3, 6, 2, 12, 2), (2, 3, 2, 6, 2)]  # K,Nt,Nr,NRF,d


def main():
    rng = np.random.default_rng(21)
    out = [struct.pack("<i", len(CASES))]
    for (K, Nt, Nr, NRF, d) in CASES:
        env = dummy_env(K, Nt, Nr, NRF, d)
        H = (rng.normal(0, 1, (K, Nr, Nt)) + 1j * rng.normal(0, 1, (K, Nr, Nt))) / np.sqrt(2)
        Frf = (rng.normal(0, 1, (Nt, NRF)) + 1j * rng.normal(0, 1, (Nt, NRF))) / np.sqrt(2 * Nt)
        eff = env.build_effective_channels(H, Frf)
        k = 0
        N = env.build_bd_digital_basis(eff, k)
        _, wh, Gbar = env.build_power_whitened_bd_channel(eff, Frf, k, bd_basis=N)
        nul = N.shape[1]
        out.append(struct.pack("<6i", K, Nt, Nr, NRF, d, nul))
        for a in (H, Frf, np.stack(eff), N, Gbar):
            out.append(np.ascontiguousarray(a).astype(np.complex128).tobytes())
        out.append(wh.eigenvalues.astype(np.float64).tobytes())
        out.append(struct.pack("<d", wh.eigenvalue_floor))
        Frf_oracle = AnalogPrecoder(Nt, NRF).build_precoder(H, d)
        out.append(np.ascontiguousarray(Frf_oracle).astype(np.complex128).tobytes())
    pathlib.Path("build").mkdir(exist_ok=True)
    pathlib.Path("build/frontend_vectors.bin").write_bytes(b"".join(out))
    print(f"wrote build/frontend_vectors.bin ({len(CASES)} cases)")


if __name__ == "__main__":
    main()
