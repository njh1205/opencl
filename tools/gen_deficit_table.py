#!/usr/bin/env python3
"""Emit the ln-deficit lookup table that main.tex:169-171 claims is deployed.

main.tex says the online refinement evaluates "log D on a uniform log-gamma grid
of 1600 points per modulation and branch, with the analytic tail C*Q(t) beyond
it". A repo-wide grep for `1600`, `log_grid`, `loggamma`, `log_gamma` found no
such table (README.md:189 records the claim as WRONG -- not implemented): the
deployed path evaluated log D by on-the-fly Gauss-Hermite-48 quadrature on every
call, at 2.65 Mflop per Psi and 212 Mflop for the deployed 20 Newton steps at
(M,d,K) = (64,4,8). This generator makes the sentence true. It writes BOTH
copies of the table, exactly like gen_gh_table.py + gen_gh_cl.py write the two
copies of the Gauss-Hermite rule:

  include/deficit_table.h    fp64, consumed by ref/threshold_cf.cpp   (PS path)
  kernels/deficit_table.clh  fp64 + fp32 blocks, consumed by
                             kernels/threshold_cf.cl                  (PL path)

tools/check_deficit_table.py is the gate that proves the two agree and that the
numbers are right; `make deficit_check` runs it.

-----------------------------------------------------------------------------
WHAT IS TABULATED, AND WHY t IS THE GRID VARIABLE
-----------------------------------------------------------------------------
The per-axis bit-metric deficit of ref/threshold_cf.cpp axis_deficit() is, after
the substitution s = n/sigma (n ~ N(0,sigma^2), sigma = delta/(2t)),

    dy/sigma = 2t(j-q) + s - 2lLt,      d2 = (dy/sigma)^2 / 2
    LLR_b(s) = ln sum_{q,l: bit_b(q)=1} e^-d2 - ln sum_{q,l: bit_b(q)=0} e^-d2
    axis(t)  = (1/L) sum_j E_s[ sum_b softplus_bits((2 bit_b(j)-1) LLR_b(s)) ]

so it depends on (t, L, folded) and on NOTHING else -- the M and gamma a caller
passes in only ever enter through t = delta*sqrt(gamma/2), delta = sqrt(6/(M-1)).
Inside psi_u() that t is t = sqrt(2*u*lambda): no delta at all. (Verified
numerically: axis_deficit at equal t is identical to 1e-15 for M = 16/64/256 for
t >= 2.5, where the error-event term dominates; below that the t -> 0 limit
ma = log2 L bits per axis makes the deficit genuinely L-dependent, which is why
the table is per-M and not one curve.)

Because t^2 = gamma*(M-1)/12, a grid uniform in ln gamma IS a grid uniform in
ln t with twice the spacing; main.tex's "uniform log-gamma grid" is therefore
exactly what is emitted here, and DFT_DLNG = 2*DFT_DLNT records both spellings.

-----------------------------------------------------------------------------
WHICH FUNCTION IS TABULATED: converged quadrature, not the live GH-48
-----------------------------------------------------------------------------
GH-48 is only trustworthy up to t ~ 3: measured against the converged reference
below, its error is -0.0318 dB at t = 4, -0.1205 dB at t = 5, -0.6685 dB at
t = 7, +1.6161 dB at t = 9.5, -7.53 dB at t = 12, and it UNDERFLOWS TO ZERO at
t = 20 (the outermost GH-48 node is 8.975, but the failure is resolution, not
truncation -- see kTQ in ref/threshold_cf.cpp). Tabulating GH-48 out to a high
handover point would tabulate a wrong function, so the table holds the
CONVERGED deficit, computed here by composite Gauss-Legendre in s on a mesh
graded towards |s| ~ t (where the dominant error event sits, at Gaussian weight
e^-t^2/2), with a mesh-refinement self-convergence check.

Two consequences, both measured and both improvements:
  * the quadrature->tail step at TQ = 5 (+0.3046 dB in D, which makes Psi a
    staircase: max step 0.0709, and 11 sign changes on one cf_bench channel at
    the old TQ = 10) moves to DFT_THI, where the plain tail is worth 5.21/t^2 dB
    -> 0.0051 dB at t = 32, i.e. 60x smaller, AND it moves out of the region the
    dominant (weakest) stream occupies: over the cf_bench pool that stream
    reaches t = 1.9..16.0 at the root, entirely inside the grid.
  * the deficit error the deployed function carries drops from 0.184 dB (the
    analytic tail just above TQ = 5) to 0.0051 dB (the analytic tail just above
    DFT_THI = 32).
No fitted constant is introduced: above DFT_THI the deployed form stays exactly
the C*Q(t) tail with the 3-term lnQ_asym that ref/threshold_cf.cpp already had.

-----------------------------------------------------------------------------
COST
-----------------------------------------------------------------------------
~250 s of CPU for the shipped grid (5 constellations x 2 branches x 1600 points)
on an Apple M4, parallel over grid points (--jobs). Run it only when the grid
changes; the headers are committed. --jobs 1 makes it reproducible on a
single core (it is deterministic either way: Pool.map preserves order).
"""
from __future__ import annotations

import argparse
import os
import pathlib
import sys
import time

import numpy as np
from numpy.polynomial.hermite import hermgauss
from numpy.polynomial.legendre import leggauss

ROOT = pathlib.Path(__file__).resolve().parents[1]
HOST = ROOT / "include" / "deficit_table.h"
DEVICE = ROOT / "kernels" / "deficit_table.clh"

# --------------------------------------------------------------------------
# the shipped grid. DFT_N = 1600 is main.tex:170's number, per modulation and
# branch. Constellations: the power-of-four MCS set the benches and the PL build
# deploy (cf_bench/lmmse_bench/pipeline_bench use 16/64/256; 4 and 1024 are the
# two capacity edges the entry guards admit, M >= 4 and 5*ceil(L/2) <= MAXD).
# Any other M -- a non-square M, or a square one outside this set, e.g. M = 144
# -- falls back to the compiled-in quadrature, which is kept reachable for
# exactly that reason (and as the reference the accuracy gate checks against).
DFT_M = (4, 16, 64, 256, 1024)
DFT_N = 1600
# DFT_TLO: below it the deployed path falls back to GH-48 quadrature, which is
# essentially exact there (measured error vs the converged reference: 1.3e-7 dB
# at t = 0.1, 1e-15 dB at t = 0.05), so the low handover costs 1.3e-7 dB
# against the 0.3046 dB the high handover costs today. 1e-4 also covers the
# deepest point the near-flat scan reaches (t = 3.9e-4 on the cf_bench pool), so
# the scan fallback is table-served too.
DFT_TLO = 1e-4
# DFT_THI: the tail handover. Plain C*Q(t) sits 5.21/t^2 dB above the converged
# deficit (measured: 0.0361 dB at t = 12, 0.0133 at 20, 0.0092 at 24, 0.0051 at
# 32, 0.0033 at 40), and the residual is a clean O(t^-2) law, so this is a
# one-parameter trade against grid density: the span ln(THI/TLO) sets the step
# h and the cubic interpolation error grows like h^4. At N = 1600, TLO = 1e-4,
# THI = 32 the measured worst interpolation error is 5.4e-6 dB -- four orders
# below the handover step it is being traded against.
DFT_THI = 32.0

LN2 = np.log(2.0)
KCMOD = 4.0 * np.pi / LN2                  # Lemma 4 constant, 4*pi/ln2 = 18.13
CUTOFF = 60.0                              # candidate pruning margin (see below)
BIG = 1e18                                 # padding sentinel: exp(-BIG) == 0.0
_GL, _GH, _BITS, _CLS = {}, {}, {}, {}


# ==========================================================================
# constellation helpers (identical to ref/threshold_cf.cpp make_axis /
# tools/cf_ref.py gray_bits: reflected Gray label, L = sqrt(M) levels per axis)
# ==========================================================================
def gray_bits(L: int):
    if L not in _BITS:
        j = np.arange(L)
        g = j ^ (j >> 1)
        ma = int(round(np.log2(L)))
        _BITS[L] = ((g[:, None] >> np.arange(ma)[None, :]) & 1, ma)
    return _BITS[L]


def softplus_bits(T):
    """log2(1+exp(-T)), stable form: ref/threshold_cf.cpp softplus_bits."""
    return np.log2(1.0 + np.exp(-np.abs(T))) + np.maximum(-T, 0.0) / LN2


def c_const(folded: bool, L: int) -> float:
    """Lemma 4 tail constant C: kCmod for the folded (mod) axis,
    kCmod*(L-1)/L for the unfolded (fin) axis. Ratio L/(L-1) exactly."""
    return KCMOD if folded else KCMOD * (L - 1) / L


def classes(j: int, L: int, folded: bool):
    """Per label bit b: the integer offsets m = j-q-lL of the candidates whose
    Gray bit b is 1, and of those whose bit b is 0. Depends on (j,L,folded)
    only, so it is cached; the offset in the quadrature variable is 2t*m."""
    key = (j, L, folded)
    if key in _CLS:
        return _CLS[key]
    bits, ma = gray_bits(L)
    nimg = 2 if folded else 0
    out = []
    for b in range(ma):
        c0, c1 = [], []
        for q in range(L):
            for l in range(-nimg, nimg + 1):
                (c1 if bits[q][b] else c0).append(j - q - l * L)
        out.append((np.array(sorted(c0)), np.array(sorted(c1))))
    _CLS[key] = out
    return out


def window(A: np.ndarray, X: float, t: float) -> np.ndarray:
    """The members of one bit class the integral actually needs.

    Pruning is PER CLASS, which is what makes it safe: a class keeps its own
    members around the window centre plus a sqrt(2*CUTOFF) margin in the
    quadrature variable, so every dropped term is at least CUTOFF below THAT
    CLASS'S OWN minimum and the log-sum-exp changes by <= n*exp(-CUTOFF) ~ 1e-24
    (measured: pruning changes ln D by exactly 0.0 ulp at every t tested,
    t = 0.5..20, L = 2..32, both branches). Pruning against the GLOBAL minimum
    instead -- the obvious shortcut -- is NOT safe: a class whose own minimum
    happens to sit near the cutoff then loses terms level with that minimum, and
    the resulting LLR error is O(1) in a regime where the softplus is linear,
    i.e. an O(1) bit error in the deficit.
    """
    if A.size == 0:
        return np.array([0])
    a0 = float(np.abs(A).min())
    lim = a0 + 2.0 * X + np.sqrt(2.0 * CUTOFF) / (2.0 * t) + 1.0
    sel = A[np.abs(A) <= lim]
    return sel if sel.size else np.array([A[int(np.argmin(np.abs(A)))]])


def g_of_s(S, t: float, L: int, folded: bool, prune: bool = True):
    """(1/L) sum_j sum_b softplus_bits(...) at every s in S."""
    bits, ma = gray_bits(L)
    S = np.asarray(S, dtype=np.float64)
    X = float(np.abs(S).max()) / (2.0 * t)
    tot = np.zeros_like(S)
    for j in range(L):
        gj = bits[j]
        for b, (c0, c1) in enumerate(classes(j, L, folded)):
            if prune:
                o0 = window(c0, X, t) * (2.0 * t)
                o1 = window(c1, X, t) * (2.0 * t)
            else:
                o0, o1 = c0 * (2.0 * t), c1 * (2.0 * t)
            n = max(o0.size, o1.size)
            p0 = np.full(n, BIG); p0[:o0.size] = o0
            p1 = np.full(n, BIG); p1[:o1.size] = o1
            D0 = 0.5 * (p0[None, :] + S[:, None]) ** 2
            D1 = 0.5 * (p1[None, :] + S[:, None]) ** 2
            m0, m1 = D0.min(1), D1.min(1)
            l0 = -m0 + np.log(np.exp(-(D0 - m0[:, None])).sum(1))
            l1 = -m1 + np.log(np.exp(-(D1 - m1[:, None])).sum(1))
            tot += softplus_bits((2.0 * gj[b] - 1.0) * (l1 - l0))
    return tot / L


def graded_mesh(t: float, npg: int = 8, scale: float = 1.0,
                strunc: float = 1.0):
    """Composite Gauss-Legendre mesh in s over [-S,S], refined inside
    ||s|-t| <= R. The integrand phi(s)*g(s) is concentrated there: the dominant
    error event is the nearest-neighbour crossing at s = +-t, of width ~1/(2t)
    in s and Gaussian weight e^{-t^2/2}. Outside R the integrand is < e^-40 of
    its peak, so a 0.5-wide panel with GL-8 is already exact to the last bit.
    This is why the reference converges where GH-48 does not: GH puts its nodes
    where e^{-z^2} is large (|z| < 9), which at t = 9.5 is nowhere near the
    feature, and its local spacing there (0.43) is 11.6x the feature width.

    Panel widths are set by the two scales the integrand has: the hinge width
    1/(2t) inside the refined zone (GL-8 on a panel one hinge wide is ~1e-10
    relative) and the Gaussian scale 1 outside it. Both caps matter for cost:
    without them the low-t end (t <= 0.1, where 2t*m spacing collapses and every
    candidate has to be kept) runs 4x the nodes for identical digits. Measured
    against the uncapped mesh over t = 1e-3..32, L = 8 and L = 32, both branches:
    max difference 3.2e-11 nats, and 0.0 for 15 of the 22 probes.
    """
    R = min(max(45.0 / t, 2.0), 12.0) * strunc
    S = (t + min(max(45.0 / t, 9.0), 10.0)) * strunc
    wf = min(max(0.5 / t, 1.0 / 64.0), 1.0) * scale
    wc = 0.5 * scale
    A, B = min(S, max(0.0, t - R)), min(S, t + R)
    segs = [(-S, -B, wc), (-B, -A, wf), (-A, A, wc), (A, B, wf), (B, S, wc)]
    if 8 not in _GL:
        _GL.update({n: leggauss(n) for n in (8, 10, 12)})
    gx, gw = _GL[npg]
    xs, ws = [], []
    for a, b, w in segs:
        if b <= a:
            continue
        n = max(1, int(np.ceil((b - a) / w)))
        e = np.linspace(a, b, n + 1)
        h = (e[1:] - e[:-1]) / 2.0
        c = (e[1:] + e[:-1]) / 2.0
        xs.append(c[:, None] + h[:, None] * gx[None, :])
        ws.append(h[:, None] * gw[None, :])
    return np.concatenate(xs).ravel(), np.concatenate(ws).ravel()


def ln_axis_ref(t: float, L: int, folded: bool, scale: float = 1.0,
                npg: int = 8, strunc: float = 1.0) -> float:
    """ln(per-axis deficit) from the converged reference. Log domain, with
    phi(t) factored out: at t = 32 the integral itself is ~e^{-512}, far below
    the fp64 range, while ln D is what the solver consumes.

    `scale` halves the PANEL WIDTHS only and `strunc` widens the truncation
    window; they are separate knobs because the two errors are separate. An
    earlier version scaled the window too, which made the low-t refinement check
    measure its own truncation (5.7e-7 nats at t = 1e-3, from dropping |s| > 4.5
    where the softplus has grown to ma bits against phi(s) = 1.5e-5) and hid the
    real panel error, which is 3e-13."""
    S, w = graded_mesh(t, npg=npg, scale=scale, strunc=strunc)
    g = g_of_s(S, t, L, folded)
    tot = float(np.sum(w * np.exp(-0.5 * (S * S - t * t)) * g))
    return -0.5 * t * t - 0.5 * np.log(2.0 * np.pi) + np.log(tot)


def ln_deficit_ref(t: float, M: int, folded: bool, scale: float = 1.0,
                   npg: int = 8, strunc: float = 1.0) -> float:
    """ln D(t) for the shipped deficit definition: D = 2 * per-axis deficit."""
    L = int(round(np.sqrt(M)))
    return ln_axis_ref(float(t), L, folded, scale=scale, npg=npg,
                       strunc=strunc) + np.log(2.0)


def axis_gh(t: float, L: int, folded: bool, N: int = 48) -> float:
    """The LIVE deployed rule (ref/threshold_cf.cpp axis_deficit /
    kernels/threshold_cf.cl axis_deficit / tools/cf_ref.py axis_deficit), for
    the cross-checks in tools/check_deficit_table.py. Unpruned on purpose: the
    GH nodes are fixed, so there is nothing to gain and something to lose."""
    if N not in _GH:
        _GH[N] = hermgauss(N)
    z, w = _GH[N]
    g = g_of_s(np.sqrt(2.0) * z, t, L, folded, prune=False)
    return float(np.sum(w * g) / np.sqrt(np.pi))


def lnQ_asym(t):
    """ln Q(t) as deployed: ref/threshold_cf.cpp lnQ_asym, verbatim (3-term
    asymptotic series, clamped at c <= 0). Not the exact ln(0.5*erfc): the two
    differ by +4.6e-4 dB at t = 5.5 and +1e-6 dB at t = 20, which is included
    in the measured handover step below."""
    t = np.asarray(t, dtype=np.float64)
    t2 = t * t
    c = 1.0 - 1.0 / t2 + 3.0 / (t2 * t2)
    return -0.5 * t2 - 0.5 * np.log(2.0 * np.pi) - np.log(t) + np.log(np.where(c > 0.0, c, 1.0))


def grid(n: int = DFT_N, tlo: float = DFT_TLO, thi: float = DFT_THI):
    """the uniform ln-gamma (= ln t) grid: t_k = TLO*(THI/TLO)^(k/(n-1))."""
    return np.exp(np.linspace(np.log(tlo), np.log(thi), n))


# ==========================================================================
# table construction
# ==========================================================================
def _one(job):
    t, M, folded, scale, npg, strunc = job
    return ln_deficit_ref(t, M, folded, scale=scale, npg=npg, strunc=strunc)


def build_table(M: int, folded: bool, n=DFT_N, tlo=DFT_TLO, thi=DFT_THI,
                jobs: int = 1, pool=None) -> np.ndarray:
    ts = grid(n, tlo, thi)
    jobs = [(float(t), M, folded, 1.0, 8, 1.0) for t in ts]
    if pool is not None:
        return np.array(pool.map(_one, jobs, chunksize=8), dtype=np.float64)
    return np.array([_one(j) for j in jobs], dtype=np.float64)


def interp_ln_deficit(row, t, n=DFT_N, tlo=DFT_TLO, thi=DFT_THI):
    """4-point Lagrange lookup of ln D at t, on the uniform ln-t grid.

    This is the REFERENCE COPY of the arithmetic that three deployed sites
    implement character-for-character: ref/threshold_cf.cpp dft_lookup(),
    kernels/threshold_cf.cl dft_lookup() and tools/cf_ref.py _dft_lookup().
    cf_bench's maxtab column compares C++ against the oracle and its maxcl
    column the kernel against C++, so a change here is a change in four places.

    Indexed by log(t/DFT_TLO)/DFT_DLNT rather than (log t - DFT_LNTLO)/DFT_DLNT
    on purpose: on the fp32 PL datapath the second form carries log(t)'s ~1e-6
    absolute error into a subtraction whose result is only 12.66, i.e. 1.3e-4
    grid steps = 2.3e-3 dB of ln D at t = 32, while the ratio form logs a
    well-conditioned number and lands at 7.6e-6 steps = 1.3e-4 dB. Both are
    ~1e-16 in fp64. The weights sum to 1 and reduce to a single node at
    f = 0,1,2,3, so the two clamped end intervals interpolate, never
    extrapolate.
    """
    dlnt = np.log(thi / tlo) / (n - 1)
    x = np.log(np.asarray(t, dtype=np.float64) / tlo) / dlnt
    x = np.clip(x, 0.0, n - 1.0)
    i0 = np.clip(np.floor(x).astype(int) - 1, 0, n - 4)
    f = x - i0
    y = np.asarray(row)
    # Lagrange basis for nodes at 0,1,2,3 evaluated at f = x - i0 (so f is in
    # [1,2) in the interior, [0,1) on the first interval and [2,3] on the last,
    # where i0 is clamped): w_k(f) = prod_{m != k} (f-m)/(k-m). Each w_k is 1 at
    # f = k and 0 at the other three nodes, so the stencil interpolates, never
    # extrapolates. Written out and checked node-by-node against the product form
    # -- the first version of this used the basis for nodes at -1,0,1,2, which
    # evaluated the whole table one grid step low (2e-2 dB at t = 0.5 rising to
    # 3.4e-1 dB at t = 3) and was caught by check_deficit_table.py's GH-48 probe.
    w0 = -(f - 1.0) * (f - 2.0) * (f - 3.0) / 6.0
    w1 = f * (f - 2.0) * (f - 3.0) / 2.0
    w2 = -f * (f - 1.0) * (f - 3.0) / 2.0
    w3 = f * (f - 1.0) * (f - 2.0) / 6.0
    return w0 * y[i0] + w1 * y[i0 + 1] + w2 * y[i0 + 2] + w3 * y[i0 + 3]


def self_convergence(M: int, folded: bool, ts, jobs=1, pool=None) -> float:
    """max |ln D(base mesh) - ln D(refined mesh)| over the probe set ts.
    The reference has to be shown to be converged before it is allowed to
    define the deployed function; measured 1e-11..4e-11 over t = 0.5..20."""
    a = [(float(t), M, folded, 1.0, 8, 1.0) for t in ts]
    b = [(float(t), M, folded, 0.5, 8, 1.0) for t in ts]
    c = [(float(t), M, folded, 1.0, 8, 1.5) for t in ts]
    if pool is not None:
        va = np.array(pool.map(_one, a, chunksize=1))
        vb = np.array(pool.map(_one, b, chunksize=1))
        vc = np.array(pool.map(_one, c, chunksize=1))
    else:
        va = np.array([_one(j) for j in a])
        vb = np.array([_one(j) for j in b])
        vc = np.array([_one(j) for j in c])
    return max(float(np.abs(va - vb).max()), float(np.abs(va - vc).max()))


# ==========================================================================
# header emission
# ==========================================================================
HDR_COMMENT = """\
// Auto-generated by tools/gen_deficit_table.py -- DO NOT EDIT.
//
// ln D(t) of the bit-metric deficit, on a uniform ln-gamma grid, for the
// constellations in kDFT_M and for both branches:
//   row 2*mi + 0  = D_fin (unfolded PAM axis, c_const(mod=0))
//   row 2*mi + 1  = D_mod (folded / modulo axis, c_const(mod=1))
//   column k      = t_k = DFT_TLO * (DFT_THI/DFT_TLO)^(k/(DFT_N-1))
// with t = delta*sqrt(gamma/2), delta = sqrt(6/(M-1)); inside psi that is
// t = sqrt(2*u*lambda), and ln gamma = 2*ln t + ln((M-1)/3), so a grid uniform
// in ln t with step DFT_DLNT is a grid uniform in ln gamma with step DFT_DLNG.
//
// Values are the CONVERGED deficit (composite Gauss-Legendre in the
// standard-normal variable, mesh graded towards |s| = t, refinement-checked to
// 1e-11), NOT the GH-48 rule the fallback path uses: GH-48 is worth -0.12 dB at
// t = 5 and underflows to zero at t = 20. See the generator docstring.
//
// Consumer contract (identical text in ref/threshold_cf.cpp,
// kernels/threshold_cf.cl and tools/cf_ref.py):
//   t <  DFT_TLO  -> GH-48 quadrature (compiled in, kept reachable)
//   t <= DFT_THI  -> 4-point Lagrange on this grid, in ln t
//   t >  DFT_THI  -> ln C + lnQ_asym(t), the analytic tail
// DFT_THI is the tail handover: the step it leaves in D is 0.0051 dB, against
// 0.3046 dB for the TQ = 5 quadrature/tail switch it replaces.
"""


def emit(host_path: pathlib.Path, dev_path: pathlib.Path, tab: np.ndarray,
         Ms, n: int, tlo: float, thi: float) -> None:
    rows = tab.shape[0]
    dlnt = float(np.log(thi / tlo) / (n - 1))
    defines = [
        f"#define DFT_N {n}",
        f"#define DFT_NM {len(Ms)}",
        f"#define DFT_ROWS {rows}",
        f"#define DFT_TLO {tlo:.17e}",
        f"#define DFT_THI {thi:.17e}",
        f"#define DFT_LNTLO {np.log(tlo):.17e}",
        f"#define DFT_DLNT {dlnt:.17e}",
        f"#define DFT_DLNG {2.0 * dlnt:.17e}",
    ]
    mlist = ", ".join(str(m) for m in Ms)

    def body(fmt, vals):
        out = []
        for r in range(rows):
            mi, br = r // 2, r % 2
            out.append("  {" + f" /* M={Ms[mi]} {'mod' if br else 'fin'} */")
            out += [", ".join(fmt % v for v in vals[r][i:i + 4]) + ","
                    for i in range(0, n, 4)]
            out.append("  },")
        return out

    # ---- host header (fp64 only, `static const` like include/gh_table.h) ----
    h = ["#ifndef HPOCL_DEFICIT_TABLE_H", "#define HPOCL_DEFICIT_TABLE_H",
         HDR_COMMENT, "// fp64 host copy; kernels/deficit_table.clh carries the same numbers",
         "// (fp64 and fp32) for the device. tools/check_deficit_table.py fails the",
         "// build if the two drift, so edit neither: edit the generator."]
    h += defines
    h += [f"static const int kDFT_M[DFT_NM] = {{{mlist}}};",
          f"static const double kDFT_LND[DFT_ROWS][DFT_N] = {{"]
    h += body("%.17e", tab)
    h += ["};", "#endif  // HPOCL_DEFICIT_TABLE_H"]
    host_path.write_text("\n".join(h) + "\n")

    # ---- device header (fp64 + fp32 blocks, like kernels/gh_table.clh) ----
    # The include guard is mandatory: Runtime::build() inlines #include textually
    # and concatenates every .cl into ONE program source, so `#pragma once`
    # would be seen twice. The fp64 extension pragma must precede the first
    # `double`, exactly as in kernels/gh_table.clh -- threshold_cf.cl includes
    # this header before its own pragma block. tools/check_registry.py check E
    # forbids a macro name shared with another kernels/*.cl(h), which is why the
    # storage class is spelled __constant here instead of hidden behind a
    # DFT_TBL macro of the kind kernels/mcs_table.clh uses.
    d = ["#ifndef HPOCL_DEFICIT_TABLE_CLH", "#define HPOCL_DEFICIT_TABLE_CLH",
         HDR_COMMENT,
         "// Device copy. Same numbers as include/deficit_table.h; the fp32 block is",
         "// the correctly-rounded narrowing of the fp64 one (check_deficit_table.py).",
         "// __constant, not __global: the table has to travel inside the program",
         "// image (runtime/hpocl_runtime.cpp passes no such argument), so the PL",
         "// build maps it to on-chip read-only memory. Shipped-grid capacity:",
         "// fp32 64000 B = 500 Kibit = 14 BRAM36 or 2 URAM288 by CAPACITY; a",
         "// 1600x32b row does not fit one 1024x36 BRAM36, so depth granularity",
         "// can cost 2 per row (20 total). fp64 is 125.0 KB. Arithmetic on the",
         "// standard 36 Kib / 288 Kib block sizes -- NOT a Vitis synthesis",
         "// report, and not checked against the XCZU49DR resource totals."]
    d += defines
    d += [f"__constant int kDFT_M[DFT_NM] = {{{mlist}}};",
          "#ifndef HPOCL_FP32",
          "#pragma OPENCL EXTENSION cl_khr_fp64 : enable",
          "__constant double kDFT_LND[DFT_ROWS][DFT_N] = {"]
    d += body("%.17e", tab)
    d += ["};", "#else", "__constant float kDFT_LND[DFT_ROWS][DFT_N] = {"]
    # The `f` suffix matters: an unsuffixed 9-digit literal is a DOUBLE constant in
    # OpenCL C, so a PL build narrows 8000 of them and warns on every one
    # (measured: 16000 warnings from this header with %.9e, 0 with %.9ef -- and
    # kernels/gh_table.clh still carries 96 of them for the same reason). 9
    # significant digits round-trip fp32, so the value is unchanged: the check
    # gate measures the fp32 block against the fp64 narrowing in ulps.
    d += body("%.9ef", tab)
    d += ["};", "#endif", "#endif  // HPOCL_DEFICIT_TABLE_CLH"]
    dev_path.write_text("\n".join(d) + "\n")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--N", type=int, default=DFT_N)
    ap.add_argument("--tlo", type=float, default=DFT_TLO)
    ap.add_argument("--thi", type=float, default=DFT_THI)
    ap.add_argument("--ms", type=str, default=",".join(str(m) for m in DFT_M))
    ap.add_argument("--jobs", type=int, default=min(10, os.cpu_count() or 1))
    ap.add_argument("--host", type=str, default=str(HOST))
    ap.add_argument("--device", type=str, default=str(DEVICE))
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--cache", type=str, default="",
                    help="npz to load the quadrature from / save it to, so that a "
                         "header-format change does not cost another quadrature "
                         "pass. Keyed by the grid parameters and refused if they "
                         "differ. No make target uses it; keep it out of the repo "
                         "(it duplicates the headers).")
    a = ap.parse_args(argv)
    Ms = tuple(int(x) for x in a.ms.split(",") if x.strip())

    t0 = time.time()
    tab = np.zeros((2 * len(Ms), a.N), dtype=np.float64)
    key = f"{a.N}|{a.tlo!r}|{a.thi!r}|{','.join(map(str, Ms))}"
    hit = False
    if a.cache and pathlib.Path(a.cache).is_file():
        z = np.load(a.cache, allow_pickle=False)
        if str(z["key"]) == key and z["tab"].shape == tab.shape:
            tab, hit = z["tab"], True
            if not a.quiet:
                print(f"[deficit_gen] reusing the quadrature cached in {a.cache}",
                      flush=True)
    if not hit:
        pool = None
        if a.jobs > 1:
            from multiprocessing import Pool
            pool = Pool(a.jobs)
        for mi, M in enumerate(Ms):
            for br, folded in enumerate((False, True)):
                tab[2 * mi + br] = build_table(M, folded, a.N, a.tlo, a.thi, pool=pool)
                if not a.quiet:
                    print(f"[deficit_gen] M={M:5d} {'mod' if folded else 'fin'}: "
                          f"ln D in [{tab[2*mi+br].min():.4f}, {tab[2*mi+br].max():.4f}]"
                          f"  ({time.time()-t0:.0f}s)", flush=True)
        if pool is not None:
            pool.close(); pool.join()
        if a.cache:
            np.savez_compressed(a.cache, key=np.array(key), tab=tab)

    # measured handover step: table top vs the analytic tail that takes over.
    steps = []
    for mi, M in enumerate(Ms):
        L = int(round(np.sqrt(M)))
        for br, folded in enumerate((False, True)):
            lnD = tab[2 * mi + br][-1]
            tail = np.log(c_const(folded, L)) + lnQ_asym(a.thi)
            steps.append(4.3429448 * (lnD - tail))
    host_path, dev_path = pathlib.Path(a.host), pathlib.Path(a.device)
    emit(host_path, dev_path, tab, Ms, a.N, a.tlo, a.thi)
    kb64 = tab.size * 8 / 1024.0
    kb32 = tab.size * 4 / 1024.0
    if not a.quiet:
        print(f"[deficit_gen] wrote {host_path} ({host_path.stat().st_size/1024:.0f} KB text)"
              f" and {dev_path} ({dev_path.stat().st_size/1024:.0f} KB text)")
        print(f"[deficit_gen] grid: N={a.N} t=[{a.tlo:g},{a.thi:g}] "
              f"h(ln t)={np.log(a.thi/a.tlo)/(a.N-1):.6e} h(ln gamma)={2*np.log(a.thi/a.tlo)/(a.N-1):.6e}")
        print(f"[deficit_gen] arrays: {len(Ms)} M x 2 branches x {a.N} = {tab.size} values; "
              f"PL footprint fp32 {kb32:.1f} KB, PS/host fp64 {kb64:.1f} KB")
        print(f"[deficit_gen] tail handover step at t={a.thi:g}: "
              f"max {max(abs(s) for s in steps):.5f} dB (was 0.3046 dB at TQ=5)")
        print(f"[deficit_gen] {time.time()-t0:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
