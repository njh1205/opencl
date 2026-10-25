#!/usr/bin/env python3
"""PyTorch checkpoint -> HPOCL flat fp32 weight binary (no torch needed).

    python3 tools/export_transformer.py CKPT build/tf_weights.bin [--t-max 32]

Reads a SpectralSetTransformerThresholdRegressor checkpoint
(mix_precode/ai/learn_svd_ucd_threshold.py) through tools/ckpt_io.py, which is
the single source of truth for the container keys, the state_dict -> weight
mapping and the tensor order. d_model / heads / layers come from the
checkpoint's own `args`; token_dim / config_dim are read off the projection
weights. Nothing is guessed: the previous version derived `heads` by scanning
divisors of d_model (256 -> 16, while the trained model has 8) and hardcoded
`token_projection.2`, which does not exist in these checkpoints (it is `.3`),
so it raised KeyError and had never been run against a real model.

Every tensor is shape-checked against the layout before writing, and
--t-max / --p-sign are validated against the device scratch limits
(kernels/tf_ops.cl TF_TMAX). Add --emit-expected to also write
build/tf_expected.bin from tools/transformer_ref.py, which lets
`./build/tf_bench build/tf_weights.bin` validate this exact model on an
OpenCL device (that is what `make tf_export_check` does).
"""
from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import ckpt_io                                                       # noqa: E402
import transformer_ref as ref                                        # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ckpt", help="torch.save() checkpoint (.pt)")
    ap.add_argument("out", help="output flat fp32 weight binary")
    ap.add_argument("--t-max", type=int, default=32,
                    help="max spectral tokens the deployed graph will be fed "
                         f"(1..{ckpt_io.TF_TMAX})")
    ap.add_argument("--p-sign", type=int, default=4,
                    help="number of probe SNR points for the sign head (>=1)")
    ap.add_argument("--emit-expected", metavar="DIR", nargs="?", const="build",
                    help="also write DIR/tf_expected.bin (inputs + numpy "
                         "reference tau/sign) so tf_bench can check this model")
    ap.add_argument("--dataset", help="training .npz (tokens/token_masks/"
                    "config_features); with it the expected file uses a real "
                    "in-distribution sample instead of synthetic noise")
    ap.add_argument("--sample", type=int, default=0, help="dataset row for --dataset")
    ap.add_argument("--seed", type=int, default=0, help="RNG seed for --emit-expected")
    a = ap.parse_args()

    ck = ckpt_io.load_ckpt_numpy(a.ckpt)
    W, dims = ckpt_io.state_dict_to_weights(ck)      # validates shapes + heads|d_model
    art = ck.get("artifacts") or {}
    norm = ckpt_io.norm_from_artifacts(art, dims) if art else None
    if norm is None:
        print("WARNING: checkpoint has no artifacts -- exporting without the "
              "normalization block; the deployed graph will consume already-"
              "normalized inputs and emit a normalized tau", file=sys.stderr)

    T = a.t_max
    tokens = mask = config = None
    if a.dataset:
        z = np.load(a.dataset, allow_pickle=True)
        for key in ("tokens", "token_masks", "config_features"):
            if key not in z.files:
                sys.exit(f"--dataset {a.dataset} has no '{key}' (keys: {z.files})")
        tokens = np.asarray(z["tokens"][a.sample], np.float32)
        mask = np.asarray(z["token_masks"][a.sample], np.float32)
        config = np.asarray(z["config_features"][a.sample], np.float32)
        T = int(tokens.shape[0])
        if tokens.shape[1] != dims["token_dim"] or config.size != dims["config_dim"]:
            sys.exit(f"dataset shapes {tokens.shape}/{config.shape} disagree with "
                     f"token_dim={dims['token_dim']}/config_dim={dims['config_dim']}")

    blob = ckpt_io.pack_weights(W, dims, T, a.p_sign, norm)   # validates T/P ranges
    out = pathlib.Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(blob)
    print(f"wrote {out} ({len(blob)} bytes): " +
          " ".join(f"{k}={v}" for k, v in dims.items()) +
          f" T={T} P={a.p_sign} norm={'yes' if norm else 'no'}")

    if a.emit_expected:
        d = pathlib.Path(a.emit_expected)
        d.mkdir(parents=True, exist_ok=True)
        P = a.p_sign
        if tokens is None:
            rng = np.random.default_rng(a.seed)
            tokens = rng.normal(0, 1, (T, dims["token_dim"])).astype(np.float32)
            mask = np.zeros(T, np.float32)
            mask[: max(1, T - T // 4)] = 1.0        # exercise a partially masked set
            config = rng.normal(0, 1, dims["config_dim"]).astype(np.float32)
        snr = np.linspace(5.0, 20.0, P).astype(np.float32)

        # expected.bin carries the RAW inputs and the tau in dB: the runtime
        # normalizes internally and denormalizes its output, so the harness
        # must not pre-normalize. The numpy reference does it explicitly here.
        if norm is not None:
            tn = np.where(mask[:, None].astype(bool),
                          (tokens - norm["feature_mean"]) / norm["feature_std"],
                          0.0).astype(np.float32)
            cn = ((config - norm["config_mean"]) / norm["config_std"]).astype(np.float32)
        else:
            tn, cn = tokens, config
        tau_n, sign = ref.forward(W, dict(dims, T=T, P=P), tn, mask, cn, snr)
        tau_db = tau_n * norm["target_std"] + norm["target_mean"] if norm else float(tau_n)
        (d / "tf_expected.bin").write_bytes(
            ckpt_io.expected_bytes(dims, T, P, tokens, mask, config, snr, tau_db, sign))
        src = "dataset sample" if a.dataset else "synthetic"
        print(f"wrote {d / 'tf_expected.bin'} ({src}, numpy tau_ref={tau_db:.6f} dB); "
              f"now run ./build/tf_bench {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
