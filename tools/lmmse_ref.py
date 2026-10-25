#!/usr/bin/env python3
"""Oracle for the LMMSE-CSI deployment path of the closed-form threshold.

Ground truth for `eps_s` is the production full-covariance LMMSE pilot
estimator (mix_precode/classical/channel_estimation_mmse.py): with an identity
pilot, the posterior error covariance eigen-modes are  lambda*nv/(lambda+nv),
so the scalar this stack deploys is exactly
    eps_s = tr(Re)/tr(Rh) = sum lambda*nv/(lambda+nv) / sum lambda
which ref/threshold_cf.cpp::lmmse_eps_s reproduces and this script asserts
against `prepare_full_covariance_mmse_pilot` to ~1e-12.

The deployed threshold then runs the 4-step fixed point of
ref/threshold_cf.cpp::threshold_deploy_lmmse (Wiener-shrink the prior spectrum
with the current rho, re-evaluate the closed-form threshold, re-derive rho),
mirrored here with a converged Newton (cf_ref.tau_policy, iters=30). Emits
build/lmmse_vectors.bin consumed by runtime/lmmse_main.cpp.
"""
import sys, struct, pathlib
import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "mix_precode" / "classical"))
sys.path.insert(0, str(HERE))
from channel_estimation_mmse import prepare_full_covariance_mmse_pilot
import cf_ref

# mild-shrink cases keep a finite fixed point; strong-shrink cases (nv ~ 1)
# collapse the spectrum and saturate the threshold to +inf (always SVD), which
# is the legitimate deployed outcome -- the bench verifies the semantics.
CASES = [  # (M, d, K, variant, neig, nv)
    (16, 2, 4, 0, 8, 1e-4), (64, 4, 8, 0, 16, 1e-3), (64, 4, 8, 1, 16, 1e-3),
    (256, 4, 8, 0, 32, 1e-4), (64, 2, 4, 0, 8, 1e-3),
    (16, 2, 4, 0, 8, 1.0), (64, 2, 4, 0, 8, 0.1),
]


def eps_s_production(eig, nv):
    """tr(posterior)/tr(prior) of the production LMMSE pilot estimator."""
    n = len(eig)
    prior = np.diag(np.asarray(eig, dtype=complex))
    prepared = prepare_full_covariance_mmse_pilot(
        channel_covariance=prior,
        channel_mean=np.zeros(n, complex),
        num_rx_antennas=1,
        num_tx_antennas=n,
        pilot_length=n,
        pilot_snr_db=10.0 * np.log10(1.0 / nv),
    )
    return float(np.trace(prepared.posterior_covariance).real
                 / max(np.trace(prior).real, 1e-12))


def eps_s_formula(eig, nv):
    num = sum(l * nv / (l + nv) for l in eig)
    den = sum(eig)
    return num / den if den > 0 else 0.0


def main():
    rng = np.random.default_rng(20261024)
    out = []
    for (M, d, K, variant, neig, nv) in CASES:
        eig = np.sort(np.exp(rng.normal(0.0, 0.8, neig)))[::-1]  # prior spectrum
        lam_hat = np.sort(np.exp(rng.normal(0.0, 0.8, K * d)))[::-1]
        eps = eps_s_production(eig, nv)
        assert abs(eps - eps_s_formula(eig, nv)) < 1e-12, (eps, eps_s_formula(eig, nv))

        # 4-step fixed point, mirror of threshold_deploy_lmmse (shrink_iters=4)
        T = K * d
        lam_t = lam_hat.copy()
        tau, flag = cf_ref.tau_policy(lam_t, M, d, K, variant)
        rho = 10.0 ** (tau / 10.0) / T
        for _ in range(4):
            lam_t = lam_hat / (1.0 + rho * lam_hat * eps)
            tau, flag = cf_ref.tau_policy(lam_t, M, d, K, variant)
            rho = 10.0 ** (tau / 10.0) / T
        assert np.isfinite(lam_t).all()

        out.append(struct.pack("<5i2d", M, d, K, variant, neig, nv, 0.0))
        out.append(np.asarray(eig, np.float64).tobytes())
        out.append(np.asarray(lam_hat, np.float64).tobytes())
        out.append(np.asarray([eps], np.float64).tobytes())
        out.append(np.asarray(lam_t, np.float64).tobytes())
        out.append(np.asarray([tau, float(flag)], np.float64).tobytes())

    pathlib.Path("build").mkdir(exist_ok=True)
    pathlib.Path("build/lmmse_vectors.bin").write_bytes(
        struct.pack("<i", len(CASES)) + b"".join(out))
    print(f"wrote build/lmmse_vectors.bin ({len(CASES)} cases) "
          f"| eps production cross-check OK")


if __name__ == "__main__":
    main()
