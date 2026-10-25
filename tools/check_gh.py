#!/usr/bin/env python3
"""Consistency gate for the duplicated Gauss-Hermite table.

The quadrature table exists twice, and GH_N is defined in both places:

  include/gh_table.h    fp64, consumed by ref/threshold_cf.cpp   (host path)
  kernels/gh_table.clh  fp64 + fp32 blocks, consumed by
                        kernels/threshold_cf.cl                  (device path)

Nothing in the build compares them, so editing one generator (or one GH_N)
silently makes the host and the device integrate at different orders -- a
deviation small enough to be swallowed by cf_bench's fp32 tolerance. This
script is that comparison; `make check` runs it first.

Checks, in order of severity:
  1. GH_N identical in both files.
  2. fp64 kGH_Z / kGH_W bit-identical between the two files.
  3. fp32 block in the device header is the correctly-rounded narrowing of
     the fp64 block, to within 1 ulp (the generator emits %.9e, which can be
     up to ~0.5 ulp away from the shortest round-trip form).
  4. sum(kGH_W) == sqrt(pi) and the nodes are antisymmetric -- i.e. the table
     really is a Gauss-Hermite rule of that order, not just two equal files.
  5. the table matches an INDEPENDENT generator, numpy.polynomial.hermite
     .hermgauss(GH_N). Checks 1-4 are all self-consistency: two files produced
     by the same wrong generator, or by a right generator at the wrong order
     with a matching GH_N, pass every one of them. numpy's hermgauss uses Newton
     iteration from an asymptotic initial guess, a different algorithm from the
     Golub-Welsch eigensolve, so agreement to 1e-12 pins the nodes down.
"""
from __future__ import annotations

import math
import re
import struct
import sys
from pathlib import Path

try:
    import numpy as np
except ImportError:          # gh_check is the first gate in `make check`; on a
    np = None                # numpy-free host it should still run checks 1-4


ROOT = Path(__file__).resolve().parents[1]
HOST = ROOT / "include" / "gh_table.h"
DEVICE = ROOT / "kernels" / "gh_table.clh"

# 1 ulp of tolerance for the fp32 block (see check 3 above).
FP32_ULP_TOL = 1.0

# Tolerances for check 5 (independent generator). Measured at GH_N = 48, the
# shipped order: max |dkGH_Z| = 0.0 and max relative |dkGH_W| = 0.0, i.e. the
# committed tables are bit-identical to hermgauss(48). The tolerances below are
# loose enough that a different but correct generator still passes, and tight
# enough that a wrong order, a wrong weight convention (probabilists' e^{-x^2/2}
# instead of physicists' e^{-x^2}) or a transposed table all fail.
NODE_ABS_TOL = 1e-12
WEIGHT_REL_TOL = 1e-10


def read(path: Path) -> str:
    if not path.is_file():
        fail(f"missing {path}")
    return path.read_text()


def fail(msg: str) -> None:
    print(f"[gh_check] FAIL: {msg}")
    sys.exit(1)


def gh_n(text: str, path: Path) -> int:
    m = re.search(r"#define\s+GH_N\s+(\d+)", text)
    if not m:
        fail(f"{path}: no '#define GH_N'")
    return int(m.group(1))


def blocks(text: str, name: str) -> list[list[float]]:
    """Every `<name>[GH_N] = { ... };` initializer, in file order."""
    out: list[list[float]] = []
    for m in re.finditer(re.escape(name) + r"\s*\[\s*GH_N\s*\]\s*=\s*\{(.*?)\}", text,
                         re.S):
        vals = [float(tok) for tok in m.group(1).replace("\n", " ").split(",") if tok.strip()]
        out.append(vals)
    return out


def bits(x: float) -> int:
    return struct.unpack("<Q", struct.pack("<d", x))[0]


def fp32_ulp_distance(got: float, want: float) -> float:
    """Distance in fp32 ulps between `got` and the fp32 narrowing of `want`."""
    g = struct.unpack("<I", struct.pack("<f", got))[0]
    w = struct.unpack("<I", struct.pack("<f", want))[0]
    return float(abs(int(g) - int(w)))


def main() -> None:
    host, device = read(HOST), read(DEVICE)

    n_host, n_dev = gh_n(host, HOST), gh_n(device, DEVICE)
    if n_host != n_dev:
        fail(f"GH_N differs: {HOST.name}={n_host} vs {DEVICE.name}={n_dev} "
             f"-- host and device would integrate at different orders")
    n = n_host

    # The device header carries two blocks per array: fp64 (#ifndef HPOCL_FP32)
    # then fp32 (#else). The host header carries one (fp64).
    for name in ("kGH_Z", "kGH_W"):
        hb = blocks(host, name)
        db = blocks(device, name)
        if len(hb) != 1:
            fail(f"{HOST.name}: expected 1 {name} block, found {len(hb)}")
        if len(db) != 2:
            fail(f"{DEVICE.name}: expected 2 {name} blocks (fp64 + fp32), found {len(db)}")
        for arr, tag in ((hb[0], f"{HOST.name} {name}"), (db[0], f"{DEVICE.name} {name} fp64"),
                         (db[1], f"{DEVICE.name} {name} fp32")):
            if len(arr) != n:
                fail(f"{tag}: {len(arr)} values but GH_N={n}")

        # check 2: fp64 blocks bit-identical
        for i, (a, b) in enumerate(zip(hb[0], db[0])):
            if bits(a) != bits(b):
                fail(f"{name}[{i}] fp64 differs: host {a!r} vs device {b!r}")

        # check 3: fp32 block is the narrowing of the fp64 block, within 1 ulp
        worst_i, worst = 0, 0.0
        for i, (f32, f64) in enumerate(zip(db[1], db[0])):
            dist = fp32_ulp_distance(f32, f64)
            if dist > worst:
                worst_i, worst = i, dist
        if worst > FP32_ULP_TOL:
            fail(f"{name}[{worst_i}] fp32 block is {worst:.1f} ulp from the fp64 "
                 f"value (tolerance {FP32_ULP_TOL})")

    # check 4: the table is really a Gauss-Hermite rule of order n
    z, w = blocks(host, "kGH_Z")[0], blocks(host, "kGH_W")[0]
    total = math.fsum(w)
    if abs(total - math.sqrt(math.pi)) > 1e-12 * math.sqrt(math.pi):
        fail(f"sum(kGH_W) = {total!r}, expected sqrt(pi) = {math.sqrt(math.pi)!r}")
    for i in range(n // 2):
        if abs(z[i] + z[n - 1 - i]) > 1e-12 * max(1.0, abs(z[i])):
            fail(f"nodes not antisymmetric at i={i}: {z[i]!r} vs {z[n - 1 - i]!r}")
        if abs(w[i] - w[n - 1 - i]) > 1e-12 * max(1.0, abs(w[i])):
            fail(f"weights not symmetric at i={i}: {w[i]!r} vs {w[n - 1 - i]!r}")
    if any(v <= 0.0 for v in w):
        fail("non-positive quadrature weight")

    # check 5: independent generator. numpy's hermgauss overflows and returns
    # non-finite weights for large N (measured: fine at 48, 96, 192; NaN weights
    # from 384 up), so a non-finite reference is reported and skipped rather than
    # failed -- checks 1-4 still apply, and GH_N would have to be raised far
    # beyond the shipped 48 for that to matter.
    dz = dw = float("nan")
    z_ref = w_ref = None
    if np is None:
        print("[gh_check] NOTE: numpy unavailable -- check 5 skipped, "
              "checks 1-4 still enforced")
    else:
        z_ref, w_ref = np.polynomial.hermite.hermgauss(n)
    if z_ref is not None and not (np.isfinite(z_ref).all() and np.isfinite(w_ref).all()):
        print(f"[gh_check] NOTE: numpy hermgauss({n}) is not finite on this "
              f"numpy build -- check 5 skipped, checks 1-4 still enforced")
        z_ref = None
    if z_ref is not None:
        dz = float(np.max(np.abs(np.asarray(z) - z_ref)))
        dw = float(np.max(np.abs(np.asarray(w) - w_ref) / w_ref))
        if dz > NODE_ABS_TOL:
            fail(f"kGH_Z disagrees with numpy hermgauss({n}) by {dz:.3e} "
                 f"(tolerance {NODE_ABS_TOL:g}) -- wrong order or wrong convention?")
        if dw > WEIGHT_REL_TOL:
            fail(f"kGH_W disagrees with numpy hermgauss({n}) by {dw:.3e} relative "
                 f"(tolerance {WEIGHT_REL_TOL:g})")
        if np is not None and any(np.diff(np.asarray(z)) <= 0):
            fail("kGH_Z is not strictly ascending")

    fp32_worst = max(
        max(fp32_ulp_distance(a, b) for a, b in zip(blocks(device, nm)[1], blocks(device, nm)[0]))
        for nm in ("kGH_Z", "kGH_W")
    )
    indep = (f", matches numpy hermgauss({n}) to dz<={dz:.1e} dw<={dw:.1e} rel"
             if dz == dz else "")          # dz is NaN when check 5 was skipped
    print(f"[gh_check] OK: GH_N={n}, fp64 blocks bit-identical, "
          f"fp32 block <= {fp32_worst:.0f} ulp, sum(kGH_W)=sqrt(pi){indep}")


if __name__ == "__main__":
    main()
