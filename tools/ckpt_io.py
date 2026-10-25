#!/usr/bin/env python3
"""Single source of truth for the HPOCL transformer weight binary.

Three things used to be encoded independently and had already drifted apart:

  * how a torch.save() checkpoint is read (export_transformer.py required torch
    and looked for the wrong container keys; exp_threshold_policy.py had a
    working torch-free reader),
  * how a state_dict maps onto the flat weight names (export_transformer.py
    hardcoded `token_projection.2` / `sign_condition_projection.2`, but the real
    checkpoints use index 3, so the exporter raised KeyError and had never been
    run against a real model),
  * the tensor order / header of build/tf_weights.bin (written twice, in
    export_transformer.py and make_random_tf_weights.py, and mirrored a third
    time by the `sizes` list in runtime/transformer_predictor.cpp).

Everything below is shared by tools/export_transformer.py,
tools/make_random_tf_weights.py and tools/exp_threshold_policy.py. The device
side is kernels/tf_ops.cl; the host side is runtime/transformer_predictor.cpp,
whose `sizes` list must stay in sync with tensor_order() -- `make tf_export_check`
exercises that path against a real checkpoint so drift fails loudly.

Model: SpectralSetTransformerThresholdRegressor
(mix_precode/ai/learn_svd_ucd_threshold.py), eval-mode semantics mirrored by
tools/transformer_ref.py.
"""
from __future__ import annotations

import io
import pickle
import struct
from pathlib import Path

import numpy as np

MAGIC = 0x54464857          # "TFHW" -- schema 1: header is exactly 8 ints
MAGIC_V2 = 0x54464858       # "TFHX" -- schema 2: 10-int header + norm block.
# A distinct magic (rather than a version field inside the old layout) keeps the
# two schemas unambiguous: in v1 the second int is token_dim, in v2 it is the
# schema, and token_dim==2 would be indistinguishable.
SCHEMA = 2
HEADER = "<10i"   # MAGIC_V2 schema token_dim config_dim d_model heads layers T P has_norm
HEADER_FIELDS = ("magic", "schema", "token_dim", "config_dim", "d_model", "heads",
                 "layers", "T", "P", "has_norm")
LEGACY_HEADER = "<8i"       # schema 1: magic token_dim config_dim d_model heads layers T P

# Sign head input is [shared(D) || snr_norm, snr_norm^2, 1] -> D + 3 columns.
SIGN_EXTRA_FEATURES = 3

# kernels/tf_ops.cl sizes its attention/pooling scratch by token count; a larger
# T would overflow those private arrays. Keep in sync with TF_TMAX in tf_ops.cl.
TF_TMAX = 64


# ---------------------------------------------------------------------------
# checkpoint loading (no torch required: torch.save() is a zip + pickled
# FloatStorage views, which is all this model's checkpoint contains)
# ---------------------------------------------------------------------------
def load_ckpt_numpy(path: str | Path) -> dict:
    """torch.save() checkpoint -> dict of numpy arrays (FloatStorage -> float32)."""
    path = Path(path)
    import zipfile

    z = zipfile.ZipFile(path)
    prefix = path.stem
    storages: dict[str, "_Storage"] = {}

    class _Storage:
        def __init__(self, key: str, numel: int) -> None:
            self.arr = np.frombuffer(z.read(f"{prefix}/data/{key}"), np.float32).copy()

    def rebuild(storage, offset, size, stride, req, hooks):
        return np.asarray(storage.arr)[offset:offset + int(np.prod(size))].reshape(size)

    class _Unpickler(pickle.Unpickler):
        def find_class(self, module, name):
            if module == "torch" and name == "FloatStorage":
                return _Storage
            if module == "torch._utils" and name == "_rebuild_tensor_v2":
                return rebuild
            return super().find_class(module, name)

        def persistent_load(self, pid):
            if pid[0] == "storage":
                key, numel = pid[2], pid[4]
                storages.setdefault(key, _Storage(key, numel))
                return storages[key]
            raise ValueError(f"unsupported persistent id {pid!r}")

    return _Unpickler(io.BytesIO(z.read(f"{prefix}/data.pkl"))).load()


def _container(ck: dict) -> tuple[dict, dict, dict]:
    """Accept the project's checkpoint layout, and plain state_dicts too."""
    if "model_state_dict" in ck:
        return (ck["model_state_dict"], ck.get("artifacts") or {}, ck.get("args") or {})
    for key in ("state_dict", "model"):
        if key in ck and isinstance(ck[key], dict):
            return ck[key], {}, {}
    # maybe the caller already passed a bare state_dict
    if all(hasattr(v, "shape") for v in ck.values()):
        return ck, {}, {}
    raise KeyError(
        "checkpoint has neither 'model_state_dict' nor 'state_dict'/'model'; "
        f"top-level keys are {sorted(ck)}"
    )


def _linear_pair(md: dict, prefix: str) -> tuple[int, int]:
    """First and last Linear index of an nn.Sequential, discovered not assumed.

    Real checkpoints use (0, 3) for token_projection and (0, 2) for
    config_projection -- i.e. the index depends on how many activations sit
    between the two Linears, so it must be read from the keys.
    """
    idx = sorted(
        int(k[len(prefix):].split(".")[0])
        for k in md
        if k.startswith(prefix) and k.endswith(".weight") and "in_proj" not in k
    )
    if len(idx) != 2:
        raise KeyError(
            f"{prefix}* expected exactly 2 Linear layers, found indices {idx}; "
            "this exporter mirrors Linear,GELU[,Dropout],Linear and cannot map "
            "a deeper head -- extend tensor_order()/state_dict_to_weights() first"
        )
    return idx[0], idx[-1]


def state_dict_to_weights(ck: dict) -> tuple[dict, dict]:
    """checkpoint -> (W in transformer_ref naming, dims).

    Raises with the offending key name instead of silently dropping layers.
    """
    md, _art, args = _container(ck)
    f32 = lambda a: np.ascontiguousarray(a, np.float32)

    D = int(args["d_model"]) if "d_model" in args else None
    H = int(args["num_heads"]) if "num_heads" in args else None
    L = int(args["num_layers"]) if "num_layers" in args else None
    if None in (D, H, L):
        raise KeyError(
            "checkpoint args must carry d_model/num_heads/num_layers; "
            f"found {sorted(args)}. Guessing heads from d_model is what produced "
            "the original bug (d_model=256 -> heads=16 instead of 8)."
        )

    W: dict[str, np.ndarray] = {}

    def lin2(prefix: str, w1: str, b1: str, w2: str, b2: str) -> None:
        i, j = _linear_pair(md, prefix)
        W[w1] = f32(md[f"{prefix}{i}.weight"]); W[b1] = f32(md[f"{prefix}{i}.bias"])
        W[w2] = f32(md[f"{prefix}{j}.weight"]); W[b2] = f32(md[f"{prefix}{j}.bias"])

    lin2("token_projection.", "tp_W1", "tp_b1", "tp_W2", "tp_b2")
    lin2("config_projection.", "cp_W1", "cp_b1", "cp_W2", "cp_b2")

    n_blk = sum(1 for k in md
                if k.startswith("encoder_blocks.") and k.endswith("attention.in_proj_weight"))
    if n_blk != L:
        raise ValueError(f"args num_layers={L} but state_dict has {n_blk} encoder blocks")

    for l in range(L):
        p, P = f"l{l}_", f"encoder_blocks.{l}.attention."
        ipw = f32(md[P + "in_proj_weight"]); ipb = f32(md[P + "in_proj_bias"])
        if ipw.shape != (3 * D, D):
            raise ValueError(f"{P}in_proj_weight is {ipw.shape}, expected {(3 * D, D)}")
        # nn.MultiheadAttention packs the fused in_proj as [q; k; v]
        W.update({p + "qW": ipw[:D], p + "qb": ipb[:D],
                  p + "kW": ipw[D:2 * D], p + "kb": ipb[D:2 * D],
                  p + "vW": ipw[2 * D:], p + "vb": ipb[2 * D:],
                  p + "oW": f32(md[P + "out_proj.weight"]),
                  p + "ob": f32(md[P + "out_proj.bias"]),
                  p + "ln_g": f32(md[f"encoder_blocks.{l}.norm.weight"]),
                  p + "ln_b": f32(md[f"encoder_blocks.{l}.norm.bias"])})

    W["pool_w"] = f32(md["pool_score.weight"]).reshape(-1)
    W["pool_b"] = f32(md["pool_score.bias"]).reshape(-1)

    sh = [k for k in ("shared_head.0.weight", "shared_head.2.weight") if k in md]
    if not sh:
        raise KeyError("no shared_head.*.weight in checkpoint")
    W["sh_W"] = f32(md[sh[0]]); W["sh_b"] = f32(md[sh[0].replace("weight", "bias")])

    # (1, D): transformer_ref does linear(x, th_W, th_b)[0, 0]
    W["th_W"] = f32(md["threshold_head.weight"]).reshape(1, -1)
    W["th_b"] = f32(md["threshold_head.bias"]).reshape(-1)

    sg = [k for k in ("sign_condition_projection.0.weight", "sign_condition_projection.2.weight",
                      "sign_condition_projection.3.weight") if k in md]
    if len(sg) < 2:
        raise KeyError(f"sign_condition_projection needs 2 Linears, found {sg}")
    W["sg_W1"] = f32(md[sg[0]]); W["sg_b1"] = f32(md[sg[0].replace("weight", "bias")])
    W["sg_W2"] = f32(md[sg[-1]]).reshape(1, -1)          # (1, D), see th_W
    W["sg_b2"] = f32(md[sg[-1].replace("weight", "bias")]).reshape(-1)

    dims = dict(token_dim=int(W["tp_W1"].shape[1]), config_dim=int(W["cp_W1"].shape[1]),
                d_model=D, heads=H, layers=L)
    validate(W, dims)
    return W, dims


# ---------------------------------------------------------------------------
# flat binary layout -- the one place the order is written down
# ---------------------------------------------------------------------------
def tensor_order(dims: dict) -> list[tuple[str, tuple[int, ...]]]:
    """(name, shape) in file order. Must match runtime/transformer_predictor.cpp."""
    D, L = dims["d_model"], dims["layers"]
    out = [("tp_W1", (D, dims["token_dim"])), ("tp_b1", (D,)),
           ("tp_W2", (D, D)), ("tp_b2", (D,)),
           ("cp_W1", (D, dims["config_dim"])), ("cp_b1", (D,)),
           ("cp_W2", (D, D)), ("cp_b2", (D,))]
    for l in range(L):
        out += [(f"l{l}_{n}", (D,) if n.endswith("b") or n.startswith("ln_") else (D, D))
                for n in ("qW", "qb", "kW", "kb", "vW", "vb", "oW", "ob", "ln_g", "ln_b")]
    out += [("pool_w", (D,)), ("pool_b", (1,)),
            ("sh_W", (D, D)), ("sh_b", (D,)),
            ("th_W", (1, D)), ("th_b", (1,)),
            ("sg_W1", (D, D + SIGN_EXTRA_FEATURES)), ("sg_b1", (D,)),
            ("sg_W2", (1, D)), ("sg_b2", (1,))]
    return out


def validate(W: dict, dims: dict) -> None:
    """Shape-check every tensor against tensor_order(); report all mismatches."""
    D = dims["d_model"]
    if D % dims["heads"]:
        raise ValueError(f"d_model={D} not divisible by heads={dims['heads']}")
    if not 0 < dims["heads"] <= 64:
        raise ValueError(f"heads={dims['heads']} out of range")
    bad = []
    for name, shape in tensor_order(dims):
        if name not in W:
            bad.append(f"{name}: missing")
        elif tuple(W[name].shape) != shape:
            bad.append(f"{name}: got {tuple(W[name].shape)}, layout wants {shape}")
    if bad:
        raise ValueError("weight/layout mismatch:\n  " + "\n  ".join(bad))


# ---------------------------------------------------------------------------
# normalization: the trained model consumes (x - mean)/std and emits a
# normalized tau. Without these the deployed graph cannot reproduce training
# semantics, so they travel in the same binary as the weights.
#   feature_mean/std [token_dim], config_mean/std [config_dim], target_mean/std
# ---------------------------------------------------------------------------
NORM_FIELDS = ("feature_mean", "feature_std", "config_mean", "config_std",
               "target_mean", "target_std")


def identity_norm(dims: dict) -> dict:
    """No-op normalization (random-weight validation uses this)."""
    return dict(feature_mean=np.zeros(dims["token_dim"], np.float32),
                feature_std=np.ones(dims["token_dim"], np.float32),
                config_mean=np.zeros(dims["config_dim"], np.float32),
                config_std=np.ones(dims["config_dim"], np.float32),
                target_mean=0.0, target_std=1.0)


def norm_from_artifacts(art: dict, dims: dict) -> dict:
    """checkpoint['artifacts'] -> normalization block (mirrors exp_threshold_policy)."""
    missing = [k for k in ("feature_mean", "feature_std", "config_mean", "config_std",
                           "target_mean", "target_std") if k not in art]
    if missing:
        raise KeyError(f"checkpoint artifacts lack {missing}; cannot export a "
                       "deployable binary (the model needs its normalization)")
    fm = np.asarray(art["feature_mean"], np.float32).reshape(-1)
    fs = np.asarray(art["feature_std"], np.float32).reshape(-1)
    cm = np.asarray(art["config_mean"], np.float32).reshape(-1)
    cs = np.asarray(art["config_std"], np.float32).reshape(-1)
    if fm.size != dims["token_dim"] or cm.size != dims["config_dim"]:
        raise ValueError(f"artifacts dims {fm.size}/{cm.size} disagree with the "
                         f"projection weights {dims['token_dim']}/{dims['config_dim']}")
    # std floors: a zero std would divide by zero at inference (see exp_threshold_policy)
    fs = np.maximum(fs, 1e-6).astype(np.float32)
    cs = np.where(cs <= 1e-6, 1.0, cs).astype(np.float32)
    return dict(feature_mean=fm, feature_std=fs, config_mean=cm, config_std=cs,
                target_mean=float(art["target_mean"]), target_std=float(art["target_std"]))


def norm_bytes(norm: dict, dims: dict) -> bytes:
    parts = [np.ascontiguousarray(norm[k], np.float32).tobytes()
             for k in ("feature_mean", "feature_std")]
    parts += [np.ascontiguousarray(norm[k], np.float32).tobytes()
              for k in ("config_mean", "config_std")]
    parts.append(struct.pack("<2f", float(norm["target_mean"]), float(norm["target_std"])))
    return b"".join(parts)


def norm_size(dims: dict) -> int:
    return 4 * (2 * dims["token_dim"] + 2 * dims["config_dim"] + 2)


def header_bytes(dims: dict, T: int, P: int, has_norm: bool = True) -> bytes:
    if not 0 < T <= TF_TMAX:
        raise ValueError(f"T={T} out of range (kernels/tf_ops.cl scratch is TF_TMAX={TF_TMAX})")
    if P < 1:
        raise ValueError(f"P={P} must be >= 1")
    return struct.pack(HEADER, MAGIC_V2, SCHEMA, dims["token_dim"], dims["config_dim"],
                       dims["d_model"], dims["heads"], dims["layers"], T, P,
                       1 if has_norm else 0)


def pack_weights(W: dict, dims: dict, T: int, P: int, norm: dict | None = None) -> bytes:
    """Flat binary: header, tensors in tensor_order(), then the norm block."""
    validate(W, dims)
    blobs = [header_bytes(dims, T, P, has_norm=norm is not None)]
    for name, shape in tensor_order(dims):
        arr = np.ascontiguousarray(W[name], np.float32)
        if arr.size != int(np.prod(shape)):
            raise ValueError(f"{name}: {arr.size} floats, layout wants {int(np.prod(shape))}")
        blobs.append(arr.tobytes())
    if norm is not None:
        blobs.append(norm_bytes(norm, dims))
    return b"".join(blobs)


def expected_bytes(dims: dict, T: int, P: int, tokens, mask, config, snr,
                   tau: float, sign) -> bytes:
    """build/tf_expected.bin: inputs + numpy reference outputs for tf_bench."""
    return b"".join([
        struct.pack("<4i", T, P, dims["d_model"], dims["heads"]),
        np.ascontiguousarray(tokens, np.float32).tobytes(),
        np.ascontiguousarray(mask, np.float32).tobytes(),
        np.ascontiguousarray(config, np.float32).tobytes(),
        np.ascontiguousarray(snr, np.float32).tobytes(),
        struct.pack("<f", float(tau)),
        np.ascontiguousarray(sign, np.float32).tobytes(),
    ])
