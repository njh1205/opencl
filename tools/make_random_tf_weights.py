#!/usr/bin/env python3
"""Random-weight model in the HPOCL flat layout + numpy reference outputs.

Layout, header and normalization block all come from tools/ckpt_io.py -- the
same single source tools/export_transformer.py uses for real checkpoints, so
this file cannot drift from the deployed format. Random weights use identity
normalization (mean 0 / std 1), which keeps the harness exercising the same
code path a real export does.
"""
from __future__ import annotations

import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import ckpt_io                                                        # noqa: E402
import transformer_ref as ref                                         # noqa: E402

DIMS = dict(token_dim=5, config_dim=10, d_model=32, heads=4, layers=2)
T, P = 32, 4


def random_weights(rng: np.random.Generator) -> dict:
    D, L = DIMS["d_model"], DIMS["layers"]

    def r(*sh, s=0.3):
        return rng.normal(0, s, sh).astype(np.float32)

    W = {}
    W["tp_W1"] = r(D, DIMS["token_dim"]); W["tp_b1"] = r(D)
    W["tp_W2"] = r(D, D);                 W["tp_b2"] = r(D)
    W["cp_W1"] = r(D, DIMS["config_dim"]); W["cp_b1"] = r(D)
    W["cp_W2"] = r(D, D);                 W["cp_b2"] = r(D)
    for l in range(L):
        p = f"l{l}_"
        for nm, sh in (("qW", (D, D)), ("qb", (D,)), ("kW", (D, D)), ("kb", (D,)),
                       ("vW", (D, D)), ("vb", (D,)), ("oW", (D, D)), ("ob", (D,))):
            W[p + nm] = r(*sh)
        W[p + "ln_g"] = np.ones(D, np.float32)
        W[p + "ln_b"] = r(D, s=0.05)
    W["pool_w"] = r(D);    W["pool_b"] = r(1)
    W["sh_W"] = r(D, D);   W["sh_b"] = r(D)
    W["th_W"] = r(1, D);   W["th_b"] = r(1)
    W["sg_W1"] = r(D, D + ckpt_io.SIGN_EXTRA_FEATURES); W["sg_b1"] = r(D)
    W["sg_W2"] = r(1, D);  W["sg_b2"] = r(1)
    return W


def main() -> int:
    rng = np.random.default_rng(11)
    W = random_weights(rng)
    norm = ckpt_io.identity_norm(DIMS)

    pathlib.Path("build").mkdir(exist_ok=True)
    blob = ckpt_io.pack_weights(W, DIMS, T, P, norm)
    pathlib.Path("build/tf_weights.bin").write_bytes(blob)

    tokens = rng.normal(0, 1, (T, DIMS["token_dim"])).astype(np.float32)
    mask = np.zeros(T, np.float32)
    mask[:24] = 1.0
    config = rng.normal(0, 1, DIMS["config_dim"]).astype(np.float32)
    snr = np.array([5.0, 10.0, 15.0, 20.0], np.float32)
    # identity normalization: raw inputs are already what the graph consumes,
    # and the reference tau is already in the units forward() returns.
    tau, sign = ref.forward(W, dict(DIMS, T=T, P=P), tokens, mask, config, snr)
    pathlib.Path("build/tf_expected.bin").write_bytes(
        ckpt_io.expected_bytes(DIMS, T, P, tokens, mask, config, snr, tau, sign))
    print(f"wrote build/tf_weights.bin ({len(blob)} B, schema {ckpt_io.SCHEMA}) "
          f"+ tf_expected.bin (tau_ref={tau:.6f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
