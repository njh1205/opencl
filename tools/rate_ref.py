#!/usr/bin/env python3
"""rate_ref: numpy oracle for the code-rate / MCS layer (kernels/rate.cl).

Independent implementation of the selection rule on the 3GPP TS 38.214 Table
5.2.2.1-2 CQI table -- the table is retyped here from the specification on
purpose, so a typo in kernels/mcs_table.clh shows up as a bench failure rather
than as a self-consistent wrong answer. Two self-checks run first:

  * every SE = m*R/1024 equals the published spectral-efficiency column to 4
    decimals (0.1523, 0.2344, ... 5.5547), which is what catches a transposed
    or mis-typed rate;
  * SE is strictly increasing in the index, which is what makes "the largest
    supportable index" well defined.

Rule (see yang_unified_transceiver: the MART paper designs the transceiver for a
*given* modulation and coding scheme, so the scheme is an outer parameter):
with GMI the mutual information per coded symbol, (m, R) is supportable iff
R*m <= GMI, i.e. SE <= GMI. Pick the largest such index; none -> outage (-1).

Writes build/rate_vectors.bin, little endian:
  int32   ncase
  per case: int32 K, float64 eps, float32 gmi[K],
            int32 idx[K], int32 m[K], int32 r[K],
            float64 se[K], float64 marg[K], float64 info[K], uint8 out[K]
The expected values are float64 computed from the float32-rounded GMI, i.e.
from exactly the number the kernel receives. SE is exact in both precisions
(m*R is an integer < 2^24 and 1024 is a power of two), so the boundary cases
below are not rounding-sensitive.
"""
import os
import struct
import sys

import numpy as np

CQI_M = np.array([0, 2, 2, 2, 2, 2, 2, 4, 4, 4, 6, 6, 6, 6, 6, 6], dtype=np.int32)
CQI_R = np.array([0, 78, 120, 193, 308, 449, 602, 378, 490, 616,
                  466, 567, 666, 772, 873, 948], dtype=np.int32)
# published spectral-efficiency column of TS 38.214 Table 5.2.2.1-2
CQI_SE_PUBLISHED = np.array([0.0, 0.1523, 0.2344, 0.3770, 0.6016, 0.8770,
                             1.1758, 1.4766, 1.9141, 2.4063, 2.7305, 3.3223,
                             3.9023, 4.5234, 5.1152, 5.5547])
SE = (CQI_M.astype(np.float64) * CQI_R.astype(np.float64) / 1024.0)


def selfcheck():
    d = np.abs(SE[1:] - CQI_SE_PUBLISHED[1:])
    if d.max() > 5e-5:                       # published column is 4 decimals
        raise SystemExit("[rate_ref] table mismatch vs TS 38.214: max %.2e" % d.max())
    if np.any(np.diff(SE[1:]) <= 0):
        raise SystemExit("[rate_ref] SE is not strictly increasing in the index")
    # SE must be exactly representable in fp32, or the boundary cases below
    # would compare a rounded device value against an exact oracle value.
    if np.any(SE.astype(np.float32).astype(np.float64) != SE):
        raise SystemExit("[rate_ref] SE is not exactly representable in fp32")
    print("[rate_ref] CQI table OK: 15 entries, SE %.4f..%.4f, fp32-exact"
          % (SE[1], SE[-1]))


def select(gmi, eps):
    """numpy transcription of cqi_select / cqi_se / cqi_info_rate."""
    g = np.asarray(gmi, dtype=np.float64)
    idx = np.full(g.shape, -1, dtype=np.int32)
    # Non-finite GMI is an upstream fault, not a measurement: outage (the same
    # rule as cqi_select, which also caps at HPOCL_GMI_MAX = 8 bits/symbol).
    ok = np.isfinite(g) & (g > 0.0) & (g <= 8.0)
    for j in range(1, len(SE)):
        idx = np.where(ok & (SE[j] <= g + eps), j, idx)
    se = np.where(idx >= 0, SE[np.maximum(idx, 0)], 0.0)
    m = np.where(idx >= 0, CQI_M[np.maximum(idx, 0)].astype(np.float64), 0.0)
    r = np.where(idx >= 0, CQI_R[np.maximum(idx, 0)].astype(np.float64), 0.0)
    marg = np.where(idx >= 0, g - se, 0.0)     # outage margin is 0, never inf
    info = np.where(idx >= 0, np.minimum(g, se), 0.0)
    out = (idx < 0).astype(np.uint8)
    return idx, m.astype(np.int32), r.astype(np.int32), se, marg, info, out


def cases():
    """Deliberately adversarial: exact table boundaries, one ulp either side,
    the eps slack, the whole special-value set, saturation and full outage."""
    j = np.arange(1, 16)
    exact = SE[1:16]
    c = []
    # 0: exactly on every table entry, no slack -> must select that entry
    c.append((exact.astype(np.float32), 0.0, "on-boundary eps=0"))
    # 1: one fp32 ulp below every entry -> must fall back to the previous one
    below = (exact - np.spacing(exact.astype(np.float32)).astype(np.float64))
    c.append((below.astype(np.float32), 0.0, "1 ulp below eps=0"))
    # 2: 1e-6 below with a 1e-5 slack -> slack must rescue the entry
    c.append(((exact - 1e-6).astype(np.float32), 1e-5, "1e-6 below eps=1e-5"))
    # 3: specials, incl. NaN / +-inf / just under the first entry
    c.append((np.array([0.0, -1.0, 0.1, float(SE[1]) - 1e-7, np.nan,
                        np.inf, -np.inf, float(SE[1])], dtype=np.float32),
              0.0, "specials"))
    # 4: saturation at and above the top entry
    c.append((np.array([SE[-1], SE[-1] + 1e-7, 6.0, 100.0], dtype=np.float32),
              0.0, "saturate"))
    # 5/6: random realistic GMI (0.1..6 bits) at K=8 and K=64
    rng = np.random.default_rng(20261025)
    for k, eps, tag in ((8, 1e-5, "random K=8"), (64, 0.0, "random K=64")):
        c.append(((0.1 + 5.9 * rng.random(k)).astype(np.float32), eps, tag))
    # 7: full outage
    c.append((np.zeros(4, dtype=np.float32), 0.0, "all outage"))
    # 8: a monotone staircase across the whole table
    c.append((np.linspace(0.0, 6.0, 13).astype(np.float32), 1e-5, "staircase"))
    # 9: single stream (K=1, the smallest legal launch)
    c.append((np.array([2.4063], dtype=np.float32), 0.0, "K=1"))
    _ = j
    return c


def main():
    selfcheck()
    out_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "build", "rate_vectors.bin")
    cs = cases()
    blob = [struct.pack("<i", len(cs))]
    for gmi, eps, tag in cs:
        idx, m, r, se, marg, info, out = select(gmi, eps)
        blob.append(struct.pack("<i", len(gmi)))
        blob.append(struct.pack("<d", eps))
        blob.append(gmi.astype("<f4").tobytes())
        blob.append(idx.astype("<i4").tobytes())
        blob.append(m.astype("<i4").tobytes())
        blob.append(r.astype("<i4").tobytes())
        blob.append(se.astype("<f8").tobytes())
        blob.append(marg.astype("<f8").tobytes())
        blob.append(info.astype("<f8").tobytes())
        blob.append(out.astype(np.uint8).tobytes())
        nout = int(out.sum())
        print("  case K=%-3d eps=%-6.0e %-20s: idx %d..%d, SE %.4f..%.4f, "
              "outage %d" % (len(gmi), eps, tag, idx.min(), idx.max(),
                             se.min(), se.max(), nout))
    with open(out_path, "wb") as f:
        f.write(b"".join(blob))
    print("wrote %s (%d cases)" % (out_path, len(cs)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
