#!/usr/bin/env python3
"""End-to-end gate for the *deployed* transformer path (not random weights).

tf_bench validates the inference graph against numpy with random weights, which
proves the operators but not the exporter: the real checkpoint path had a
state_dict mapping that raised KeyError (`token_projection.2` vs the actual
`.3`) and derived `heads` by scanning divisors of d_model (256 -> 16, the trained
model has 8), and none of that was covered by any test.

This script closes that gap. It
  1. exports the real checkpoint through tools/export_transformer.py,
  2. checks the exported model reproduces the dataset's own threshold labels,
  3. runs ./build/tf_bench on the exported binary, i.e. numpy reference vs the
     OpenCL graph, on the real d_model/heads/layers.

Skips (exit 0) when the archived checkpoint or dataset is absent, so `make check`
stays usable outside this workspace. Point TF_CKPT / TF_DATASET at another
checkpoint to test a different model.
"""
from __future__ import annotations

import os
import subprocess
import sys
import pathlib
import warnings

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import ckpt_io                                                        # noqa: E402
import transformer_ref as ref                                         # noqa: E402

HERE = pathlib.Path(__file__).resolve().parents[1]
DEFAULT_CKPT = (HERE.parent / "mix_precode/_archive_20261018/bulk/results/"
                "svd_ucd_threshold_poolv2_3x1000_fast/poolv2_3x1000_fast_model.pt")
N_LABEL = 200          # dataset rows used for the label-agreement check
MAE_TOL_DB = 2.0       # the trained student's own MAE is ~0.6 dB on this pool


def fail(msg: str) -> int:
    print(f"[tf_export_check] FAIL: {msg}")
    return 1


def main() -> int:
    ckpt = pathlib.Path(os.environ.get("TF_CKPT", sys.argv[1] if len(sys.argv) > 1 else DEFAULT_CKPT))
    ds = pathlib.Path(os.environ.get("TF_DATASET", str(ckpt).replace("_model.pt", "_dataset.npz")))
    if not ckpt.is_file():
        print(f"[tf_export_check] SKIP: no checkpoint at {ckpt}")
        return 0
    if not ds.is_file():
        print(f"[tf_export_check] SKIP: no dataset at {ds} (needed for in-distribution inputs)")
        return 0

    ck = ckpt_io.load_ckpt_numpy(ckpt)
    W, dims = ckpt_io.state_dict_to_weights(ck)
    norm = ckpt_io.norm_from_artifacts(ck["artifacts"], dims)
    z = np.load(ds, allow_pickle=True)
    for key in ("tokens", "token_masks", "config_features", "thresholds_db"):
        if key not in z.files:
            return fail(f"{ds.name} has no '{key}'")
    tok, msk, cfg, lab = z["tokens"], z["token_masks"], z["config_features"], z["thresholds_db"]
    T = int(tok.shape[1])
    if T > ckpt_io.TF_TMAX:
        return fail(f"dataset T={T} exceeds the device scratch TF_TMAX={ckpt_io.TF_TMAX}")
    P = 4
    print(f"[tf_export_check] {ckpt.name}: " + " ".join(f"{k}={v}" for k, v in dims.items()) +
          f" T={T} P={P} | dataset {tok.shape[0]} rows")

    # ---- 2) does the exported model reproduce the dataset's own labels? ----
    def predict(i: int) -> float:
        tn = np.where(msk[i][:, None].astype(bool),
                      (tok[i] - norm["feature_mean"]) / norm["feature_std"], 0.0).astype(np.float32)
        cn = ((cfg[i] - norm["config_mean"]) / norm["config_std"]).astype(np.float32)
        t, _ = ref.forward(W, dict(dims, T=T, P=1), tn, msk[i], cn, np.array([20.0], np.float32))
        return float(t) * norm["target_std"] + norm["target_mean"]

    n = min(N_LABEL, tok.shape[0])
    with warnings.catch_warnings():
        # masked token rows make the pooling score NaN before softmax_masked
        # discards them; the prediction itself is finite (asserted below).
        warnings.simplefilter("ignore", RuntimeWarning)
        err = np.array([predict(i) - float(lab[i]) for i in range(n)])
    if not np.isfinite(err).all():
        return fail(f"{int((~np.isfinite(err)).sum())} non-finite predictions")
    mae = float(np.mean(np.abs(err)))
    print(f"[tf_export_check] numpy vs dataset labels ({n} rows): MAE={mae:.4f} dB "
          f"bias={float(np.median(err)):+.4f} dB max={float(np.max(np.abs(err))):.4f} dB")
    if mae > MAE_TOL_DB:
        return fail(f"exported model disagrees with its own training labels "
                    f"(MAE {mae:.3f} dB > {MAE_TOL_DB}); the state_dict mapping, the "
                    f"head count or the normalization block is wrong")

    # ---- 1)+3) export, then run the OpenCL graph on it ----
    out = HERE / "build" / "tf_real.bin"
    cmd = [sys.executable, str(HERE / "tools/export_transformer.py"), str(ckpt), str(out),
           "--dataset", str(ds), "--p-sign", str(P), "--emit-expected", str(out.parent)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        return fail(f"export_transformer.py failed:\n{r.stdout}{r.stderr}")
    print(" ".join(l for l in r.stdout.splitlines() if l.startswith("wrote")))

    bench = HERE / "build" / "tf_bench"
    if not bench.is_file():
        return fail(f"{bench} not built -- run `make build/tf_bench` first")
    r = subprocess.run([str(bench), str(out), os.environ.get("DEV", "cpu")],
                       capture_output=True, text=True)
    sys.stdout.write(r.stdout)
    if r.returncode != 0:
        return fail(f"tf_bench on the exported real model failed:\n{r.stderr}")
    print("[tf_export_check] OK: real checkpoint -> flat binary -> OpenCL graph matches numpy")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
