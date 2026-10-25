"""numpy mirror of SpectralSetTransformerThresholdRegressor (eval mode).
Semantics follow mix_precode/ai/learn_svd_ucd_threshold.py exactly:
  token_proj / config_proj = Linear,GELU,Linear ; broadcast add ; mask
  blocks: MHA(batch_first) -> +residual -> LayerNorm -> mask
  pool: Linear(d,1) score, masked softmax over tokens, weighted sum
  shared_head = Linear,GELU ; threshold_head = Linear
  sign head: [shared || snr_norm, snr_norm^2, 1] -> Linear,GELU,Linear
"""
import numpy as np

def linear(x, W, b): return x @ W.T + b

def gelu(x):
    from math import sqrt
    from scipy.special import erf
    return 0.5 * x * (1 + erf(x / sqrt(2)))

def layernorm(x, g, b, eps=1e-5):
    mu = x.mean(-1, keepdims=True)
    va = x.var(-1, keepdims=True)
    return (x - mu) / np.sqrt(va + eps) * g + b

def softmax_masked(s, mask):
    s = np.where(mask.astype(bool), s, -np.inf)
    s = s - s.max(-1, keepdims=True)
    e = np.exp(s)
    return e / e.sum(-1, keepdims=True)

def forward(W, dims, tokens, mask, config, snr):
    T, H, L = dims["T"], dims["heads"], dims["layers"]
    D = dims["d_model"]
    dh = D // H
    valid = mask.astype(bool)
    h = linear(tokens, W["tp_W1"], W["tp_b1"]); h = gelu(h)
    h = linear(h, W["tp_W2"], W["tp_b2"])
    c = linear(config[None, :], W["cp_W1"], W["cp_b1"]); c = gelu(c)
    c = linear(c, W["cp_W2"], W["cp_b2"])
    h = (h + c) * mask[:, None]
    for l in range(L):
        p = f"l{l}_"
        q = linear(h, W[p + "qW"], W[p + "qb"])
        k = linear(h, W[p + "kW"], W[p + "kb"])
        v = linear(h, W[p + "vW"], W[p + "vb"])
        qr = q.reshape(T, H, dh).transpose(1, 0, 2)
        kr = k.reshape(T, H, dh).transpose(1, 0, 2)
        vr = v.reshape(T, H, dh).transpose(1, 0, 2)
        sc = qr @ kr.transpose(0, 2, 1) / np.sqrt(dh)
        sc = np.where(valid[None, None, :], sc, -np.inf)
        sc = sc - sc.max(-1, keepdims=True)
        e = np.exp(sc); e /= e.sum(-1, keepdims=True)
        a = (e @ vr).transpose(1, 0, 2).reshape(T, D)
        o = linear(a, W[p + "oW"], W[p + "ob"])
        h = layernorm(h + o, W[p + "ln_g"], W[p + "ln_b"]) * mask[:, None]
    s = h @ W["pool_w"] + W["pool_b"]
    w = softmax_masked(s[None, :], mask)[0]
    g = (w[:, None] * h).sum(0)
    sh = gelu(linear(g[None, :], W["sh_W"], W["sh_b"]))[0]
    tau = float(linear(sh[None, :], W["th_W"], W["th_b"])[0, 0])
    scale = max(1.0, float(np.max(np.abs(snr))))
    sn = snr / scale
    feats = np.stack([sn, sn ** 2, np.ones_like(sn)], -1)
    si = np.concatenate([np.broadcast_to(sh, (len(snr), D)), feats], -1)
    z = gelu(linear(si, W["sg_W1"], W["sg_b1"]))
    sign = linear(z, W["sg_W2"], W["sg_b2"])[:, 0]
    return tau, sign
