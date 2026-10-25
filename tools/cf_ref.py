#!/usr/bin/env python3
"""Independent numpy reference + test-vector generator for the closed-form
switching threshold. Emits build/cf_vectors.bin consumed by cf_bench.
Checks Lemma 4 (C_mod/C_fin = L/(L-1)) via the quadrature deficits."""
import numpy as np, os, pathlib, re, struct
from numpy.polynomial.hermite import hermgauss

Z, W = hermgauss(48)
RSQPI = 1.0 / np.sqrt(np.pi)

def gray_bits(L):
    j = np.arange(L)
    g = j ^ (j >> 1)
    ma = int(round(np.log2(L)))
    return (g[:, None] >> np.arange(ma)[None, :]) & 1, ma

def axis_deficit(folded, gamma, M):
    L = int(round(np.sqrt(M))); delta = np.sqrt(6.0 / (M - 1))
    bits, ma = gray_bits(L)
    x = (2.0 * np.arange(L) - (L - 1)) * delta / 2
    sig2 = 1.0 / (2.0 * gamma); sig = np.sqrt(sig2)
    rng = L * delta; nimg = 2 if folded else 0
    imgs = np.arange(-nimg, nimg + 1) * rng
    acc = 0.0
    for j in range(L):
        y = x[j] + np.sqrt(2) * sig * Z                      # [GH]
        for b in range(ma):
            m0 = bits[:, b] == 0
            d2 = ((y[:, None, None] - x[None, :, None] - imgs[None, None, :]) ** 2
                  ) / (2 * sig2)                            # [GH,L,img]
            f0 = d2[:, m0, :].reshape(d2.shape[0], -1)
            f1 = d2[:, ~m0, :].reshape(d2.shape[0], -1)
            l0 = -f0.min(axis=1) + np.log(np.exp(-(f0 - f0.min(axis=1)[:, None])).sum(1))
            l1 = -f1.min(axis=1) + np.log(np.exp(-(f1 - f1.min(axis=1)[:, None])).sum(1))
            llr = l1 - l0                                    # log p0-denom convention
            t = (2 * bits[j, b] - 1) * llr
            loss = np.log2(1 + np.exp(-np.abs(t))) + np.maximum(-t, 0) / np.log(2)
            acc += RSQPI * np.sum(W * loss)
    return acc / L

from scipy.special import erfc
# GH-48 quadrature band; beyond it the analytic tail C*Q(t) is used. Must equal
# kTQ in ref/threshold_cf.cpp and TQ in kernels/threshold_cf.cl --
# tools/check_capacity.py fails the build on drift. Measured against a converged
# reference for the SAME deficit definition (composite Gauss-Legendre with
# Richardson extrapolation, self-convergence <= 2e-11 relative, cross-checked
# against GH-N with N = 192/384), M = 16/64/256 all identical:
#      t      GH-48 err     C*Q(t) err
#      4.0    -0.0318 dB    +0.2684 dB
#      5.0    -0.1205       +0.1841
#      7.0    -0.6685       +0.1006
#      9.5    +1.6161       +0.0566
# The branches cross at t ~ 5.3, so this value caps the worst-case deficit error
# at 0.184 dB. GH-48's outermost node is 8.975315 (n-span 12.69 sigma) and the
# dominant error event sits at z = t/sqrt(2) = 6.72 for t = 9.5, i.e. INSIDE the
# span: the rule fails on RESOLUTION, not truncation (local node spacing 0.43
# there vs a softplus transition width of 1/(2*sqrt(2)*t) = 0.037). Raising the
# order fixes it monotonically, which a truncation story cannot explain: at
# t = 9.5, GH-96 +0.2319 dB, GH-192 +0.0337 dB, GH-384 -0.0072 dB. Full table
# and the Psi-staircase consequence: kTQ in ref/threshold_cf.cpp.
TQ = 5.0
# Lemma-4 ratio tolerance for the in-band probe: measured rel err of
# D_mod/D_fin vs L/(L-1) at t = 3.5 is <= 1e-14 for M = 16/64/256 (the ratio is
# exact to ~1e-15 for t >= 3 and only degrades below t ~ 2.3, where it is
# 5.6e-7 at t = 2). 1e-9 leaves >5 orders of margin and still fails if the
# probe ever lands on the tail branch by accident.
QUAD_RATIO_TOL = 1e-9

# ---------------------------------------------------------------------------
# The generated ln-D table of main.tex:169-171 ("log D on a uniform log-gamma
# grid of 1600 points per modulation and branch, with the analytic tail C*Q(t)
# beyond it"). HPOCL_DEFICIT_TABLE is the SAME switch as in
# ref/threshold_cf.cpp and kernels/threshold_cf.cl, spelled here as an
# environment variable because this is python:
#     HPOCL_DEFICIT_TABLE=0 python3 tools/cf_ref.py     # pre-table oracle
# main() writes the setting into build/cf_vectors.bin and cf_bench refuses a
# vectors file whose setting differs from the one it was compiled with, so the
# two cannot be mixed by accident.
#
# The table is PARSED from include/deficit_table.h rather than recomputed by
# importing tools/gen_deficit_table.py: the oracle must consume the same
# numbers the C++ includes, so that a stale or hand-edited header fails
# cf_bench instead of being silently regenerated. tools/check_deficit_table.py
# is what proves those numbers equal a fresh converged quadrature.
# ---------------------------------------------------------------------------
_ROOT = pathlib.Path(__file__).resolve().parents[1]
DEFICIT_TABLE = os.environ.get("HPOCL_DEFICIT_TABLE", "1") != "0"


def _load_deficit_table():
    path = _ROOT / "include" / "deficit_table.h"
    if not path.is_file():
        return None
    txt = path.read_text()

    def define(name, cast=float):
        m = re.search(r"#define\s+%s\s+([-+0-9.eE]+)" % name, txt)
        if not m:
            raise SystemExit("[cf_ref] %s: no '#define %s'" % (path, name))
        return cast(m.group(1))

    n, nm = define("DFT_N", int), define("DFT_NM", int)
    rows = define("DFT_ROWS", int)
    if rows != 2 * nm:
        raise SystemExit("[cf_ref] %s: DFT_ROWS != 2*DFT_NM" % path)
    ms = [int(x) for x in re.search(r"kDFT_M\[DFT_NM\]\s*=\s*\{([^}]*)\}",
                                    txt).group(1).split(",") if x.strip()]
    m = re.search(r"kDFT_LND\[DFT_ROWS\]\[DFT_N\]\s*=\s*\{", txt)
    i, depth, start, tab = m.end() - 1, 0, None, []
    while i < len(txt):
        ch = txt[i]
        if ch == "{":
            depth += 1
            if depth == 2:
                start = i + 1
        elif ch == "}":
            depth -= 1
            if depth == 1 and start is not None:
                body = re.sub(r"/\*.*?\*/", " ", txt[start:i], flags=re.S)
                tab.append([float(x) for x in body.split(",") if x.strip()])
                start = None
            if depth == 0:
                break
        i += 1
    if len(tab) != rows or any(len(r) != n for r in tab):
        raise SystemExit("[cf_ref] %s: malformed kDFT_LND" % path)
    return {"M": ms, "index": {M: k for k, M in enumerate(ms)}, "n": n,
            "tlo": define("DFT_TLO"), "thi": define("DFT_THI"),
            "dlnt": define("DFT_DLNT"), "tab": np.array(tab)}


DFT = _load_deficit_table() if DEFICIT_TABLE else None


def _dft_lookup(row, t):
    """4-point Lagrange on the uniform ln-t grid. IDENTICAL arithmetic, in the
    same order, as ref/threshold_cf.cpp dft_lookup() and
    kernels/threshold_cf.cl dft_lookup(); tools/gen_deficit_table.py
    interp_ln_deficit() is the reference copy. cf_bench's maxtab column compares
    this against the C++ and maxcl against the kernel, so a change here is a
    change in four places."""
    tab, n = DFT["tab"], DFT["n"]
    x = np.log(np.asarray(t, dtype=np.float64) / DFT["tlo"]) / DFT["dlnt"]
    x = np.clip(x, 0.0, n - 1.0)
    i0 = int(np.clip(np.floor(x) - 1, 0, n - 4))
    f = x - i0
    w0 = -(f - 1.0) * (f - 2.0) * (f - 3.0) / 6.0
    w1 = f * (f - 2.0) * (f - 3.0) / 2.0
    w2 = -f * (f - 1.0) * (f - 3.0) / 2.0
    w3 = f * (f - 1.0) * (f - 2.0) / 6.0
    y = tab[row]
    return w0 * y[i0] + w1 * y[i0 + 1] + w2 * y[i0 + 2] + w3 * y[i0 + 3]


def _t(g, M): return np.sqrt(6.0 / (M - 1)) * np.sqrt(g / 2)
def _c(mod, M):
    L = np.sqrt(M); cmod = 4 * np.pi / np.log(2)
    return cmod if mod else cmod * (L - 1) / L
def _lnQ_asym(t):
    """ln Q(t), t > TQ. Verbatim copy of ref/threshold_cf.cpp lnQ_asym (3-term
    asymptotic series, clamped at c <= 0). Measured against exact ln(0.5*erfc):
    +4.6e-4 at t = 5.5, +5.3e-5 at t = 8, +5.3e-7 at t = 17.4."""
    l = -0.5 * t * t - 0.5 * np.log(2.0 * np.pi) - np.log(t)
    t2 = t * t
    c = 1.0 - 1.0 / t2 + 3.0 / (t2 * t2)
    return l + np.log(c if c > 0.0 else 1.0)
def _branch(g, M):
    """(evaluator, table row group, t). 0 = ln-D table, 1 = analytic tail,
    2 = the live GH-48 quadrature. Mirrors ref/threshold_cf.cpp
    deficit_branch() and kernels/threshold_cf.cl deficit_branch() exactly,
    including the NaN case: g <= 0 makes t NaN, every comparison fails, and the
    quadrature's own guard saturates the deficit at log2(M) -- which is why
    ln_deficit_* needs no gamma guard on any of the three sides."""
    t = _t(g, M)
    if DFT is not None:
        mi = DFT["index"].get(M, -1)
        if mi >= 0:
            if DFT["tlo"] <= t <= DFT["thi"]: return 0, mi, t
            if t > DFT["thi"]: return 1, -1, t
            return 2, -1, t          # below the grid: GH-48, exact to 1e-9 dB there
    return (1, -1, t) if t > TQ else (2, -1, t)


def deficit_fin_quad(g, M):
    """the LIVE GH-48 rule alone: no table, no analytic-tail branch. This is the
    reference the deployed evaluator is gated against (cf_bench's maxq column)
    and the fallback for an untabulated M or a t below DFT_TLO."""
    # gamma <= 0 (or NaN): mirrors ref/threshold_cf.cpp / kernels/threshold_cf.cl.
    # Without this guard 1.0/(2.0*0.0) raises ZeroDivisionError here, while the
    # C++ and the kernel both saturate the deficit at the label entropy log2(M).
    if not (g > 0.0): return float(np.log2(M))
    return 2 * axis_deficit(False, g, M)


def deficit_mod_quad(g, M):
    if not (g > 0.0): return float(np.log2(M))
    return 2 * axis_deficit(True, g, M)


def _deficit(mod, g, M):
    b, mi, t = _branch(g, M)
    if b == 0: return float(np.exp(_dft_lookup(2 * mi + mod, t)))
    if b == 1: return float(_c(bool(mod), M) * 0.5 * erfc(t / np.sqrt(2)))
    return deficit_mod_quad(g, M) if mod else deficit_fin_quad(g, M)


def deficit_fin(g, M): return _deficit(0, g, M)
def deficit_mod(g, M): return _deficit(1, g, M)
def _ln_deficit(mod, g, M):
    """ln D in the log domain -- the form psi() must use, identical to
    ref/threshold_cf.cpp ln_deficit_impl. Above the table this is
    log(C) + _lnQ_asym(t), NOT log(C*0.5*erfc): the linear form underflows to
    exactly 0.0 at t >= 38 in fp64 (t >= 13.5 in fp32), so log() returned -inf,
    psi() returned nan, and the oracle's safeguarded Newton silently degraded to
    pure bisection. The cf_bench pool reaches t = 143 at its bracket edge, so
    this was live."""
    b, mi, t = _branch(g, M)
    if b == 0: return float(_dft_lookup(2 * mi + mod, t))
    if b == 1: return float(np.log(_c(bool(mod), M)) + _lnQ_asym(t))
    return float(np.log(deficit_mod_quad(g, M) if mod else deficit_fin_quad(g, M)))


def ln_deficit_fin(g, M): return _ln_deficit(0, g, M)
def ln_deficit_mod(g, M): return _ln_deficit(1, g, M)

def _gamma_star(lu, rho, variant, eta):
    if variant == 0:
        g = np.exp(np.log(np.maximum(lu, 1e-12)).mean())
        return rho * g / eta
    p = np.prod(1 + rho * lu)
    return (p ** (1.0 / len(lu)) - 1.0) / eta


def psi(lam, u, M, d, K, variant):
    """log-domain balance, mirror of ref/threshold_cf.cpp psi_u."""
    d2 = 6.0 / (M - 1); rho = 4.0 * u / d2
    eta = 1.0 + (d - 1) / (d * (M - 1.0))
    lam = np.asarray(lam).reshape(K, d)
    # ln_deficit_*, not np.log(deficit_*): the deployed Psi is built in the log
    # domain (see ln_deficit_fin) and the two differ above TQ, so the old form
    # made the oracle a *different function* from the C++/kernel -- measured
    # 2.78e-5 dB of systematic tau offset on the cf_bench M=64/d=4/K=8 UCD
    # config, which was the whole of cf_bench's maxtau budget.
    lf = np.array([ln_deficit_fin(4 * u * l / d2, M) for l in lam.ravel()])
    lm = np.array([np.log(float(d)) +
                   ln_deficit_mod(_gamma_star(lam[k], rho, variant, eta), M)
                   for k in range(K)])
    return (lf.max() + np.log(np.exp(lf - lf.max()).sum()) -
            (lm.max() + np.log(np.exp(lm - lm.max()).sum())))


def root_u(lam, M, d, K, variant=0):
    """Newton start point + method flag, mirror of ref/threshold_cf.cpp:
    0 = closed form (eq. cf), 1 = geometric scan of psi, 2 = no crossing."""
    u0, _ = tau_closed_form(lam, M, d, K)
    if u0 is not None:
        return u0, 0
    lo, ratio, eps = 1e-6, 1e12 ** (1.0 / 400), 1e-6
    u_neg = None
    with np.errstate(divide="ignore", invalid="ignore"):
        for _ in range(401):
            f = psi(lam, lo, M, d, K, variant)
            if f <= -eps:                       # dead band: near-flat spectra
                u_neg = lo                      # leave psi at round-off level
            elif f >= eps and u_neg is not None:
                return 0.5 * (u_neg + lo), 1
            lo *= ratio
    return None, 2


def tau_policy(lam, M, d, K, variant=0, iters=30):
    """Deployed switching threshold tau [dB] and method flag.

    Closed form (or scan fallback) + safeguarded Newton on psi. The default
    iters=30 is a converged oracle; the deployed budget is CfConfig::newton_iters
    = 20 (ref/threshold_cf.h) and cf_bench gates the deployed budget against an
    8-step run of the same code path. Flag 2 means psi never changes sign: the
    branch ordering is uniform and tau saturates to -inf (always THP) or +inf
    (always SVD); flag 4 (capacity) exists only in the kernel and the C++ ref.
    """
    u0, flag = root_u(lam, M, d, K, variant)
    T = K * d
    if u0 is None:
        sign = psi(lam, 1.0, M, d, K, variant)
        return (-np.inf if sign > 0 else np.inf), 2
    lo, hi = u0 / 4.0, u0 * 4.0
    for _ in range(6):
        if psi(lam, lo, M, d, K, variant) > 0: lo /= 4.0
        else: break
    for _ in range(6):
        if psi(lam, hi, M, d, K, variant) < 0: hi *= 4.0
        else: break
    u = u0
    for _ in range(iters):
        f = psi(lam, u, M, d, K, variant)
        if f <= 0: lo = u
        else: hi = u
        h = 1e-5
        fp = (psi(lam, u * (1 + h), M, d, K, variant) -
              psi(lam, u * (1 - h), M, d, K, variant)) / (2 * h * u)
        un = u - f / fp
        if not (lo < un < hi): un = 0.5 * (lo + hi)
        u = un
    return 10 * np.log10(4 * T * (M - 1) * u / 6.0), flag


def tau_closed_form(lam, M, d, K):
    T = K * d; eta = 1 + (d - 1) / (d * (M - 1))
    lam_min = lam.min(); mult = int(np.sum(np.abs(lam - lam_min) <= 1e-9 * max(1.0, lam_min))) or 1
    gk = np.exp(np.log(np.maximum(lam.reshape(K, d), 1e-12)).mean(1))
    g_min = gk.min()
    L = int(round(np.sqrt(M)))
    cmod = 4 * np.pi / np.log(2); cfin = cmod * (L - 1) / L
    den = g_min / eta - lam_min
    if den <= 1e-12: return None, None
    num = np.log(d * cmod / (mult * cfin)) - 0.5 * np.log(g_min / (eta * lam_min))
    u0 = num / den
    if u0 <= 0: return None, None
    return u0, 10 * np.log10(4 * T * (M - 1) * u0 / 6)

# ln-D probe set for cf_bench: below the grid (quadrature branch), on both grid
# edges (handover steps), across the grid (table branch) and far above it (tail).
TP = np.array([1e-5, 5e-5, 9.999e-5, 1e-4, 1.0001e-4, 3e-4, 1e-3, 1e-2, 0.1,
               0.5, 1.0, 1.5, 2.0, 3.0, 3.5, 5.0, 7.0, 10.0, 14.0, 20.0, 25.0,
               31.999, 32.0, 32.001, 40.0, 100.0, 300.0])
CF_MAGIC = 0x32544644          # 'DFT2' little-endian: vectors-file format tag
                               # (bumped from 'DFT1': the ln-D probe block now
                               #  carries gammas, not t -- a 'DFT1' file would
                               #  be silently misread, so cf_main refuses it)


def main():
    rng = np.random.default_rng(7)
    cfgs = [(16, 2, 4, 0), (64, 4, 8, 0), (64, 4, 8, 1), (256, 4, 8, 0),
            (64, 2, 4, 0)]                       # last: flat -> scan/saturation
    C = 8
    pathlib.Path("build").mkdir(exist_ok=True)
    # magic, config count, and the evaluator this oracle was run with, so that
    # cf_bench can refuse a vectors file generated under the other setting of
    # HPOCL_DEFICIT_TABLE instead of reporting it as an accuracy failure.
    blobs = [struct.pack("<3i", CF_MAGIC, len(cfgs), 1 if DFT is not None else 0)]
    report = []
    for ci, (M, d, K, var) in enumerate(cfgs):
        if ci + 1 == len(cfgs):                  # near-flat: closed form degenerates
            lam = 1.0 + 0.02 * rng.normal(0.0, 1.0, (C, K * d))
            lam = np.abs(lam)
        else:
            lam = np.sort(np.exp(rng.normal(0.0, 0.8, (C, K * d))), axis=1)[:, ::-1]
        lam = np.ascontiguousarray(lam, dtype=np.float64)
        tcf = np.zeros(C); flag = np.zeros(C, np.int32); tau = np.zeros(C)
        for c in range(C):
            u0, t = tau_closed_form(lam[c], M, d, K)
            if t is not None: tcf[c] = t
            tau[c], flag[c] = tau_policy(lam[c], M, d, K, var)
            if flag[c] == 2: assert np.isinf(tau[c]), (M, c)
        # Lemma-4 ratio checks: one probe on the LIVE QUADRATURE and one in the
        # analytic tail. The quadrature probe used to sit at t = 8, which is
        # above TQ = 5, so both operands took the C*Q(t) branch and the ratio was
        # L/(L-1) by construction -- an assertion that could not fail. It now
        # calls deficit_*_quad explicitly, so it tests the quadrature whatever
        # the deployed evaluator is, and it stays inside the band if TQ moves.
        # Not deficit_*(t_quad): at t = 3.5 that is the TABLE, whose own Lemma-4
        # error is 6.9e-7 at M = 4 (the model's low-SNR drift, measured) and so
        # would need a 1e-6 tolerance here instead of 1e-9.
        L = int(round(np.sqrt(M)))
        # t -> gamma in EXACTLY the spelling runtime/cf_main.cpp reconstructs
        # (`2.0 * t * t * (M - 1) / 6.0`) and ref/threshold_cf.cpp t_of()
        # inverts (`sqrt(6/(M-1)) * sqrt(gamma*0.5)`). The algebraically
        # equivalent `2.0 * (t / delta) ** 2` is NOT the same float: it differs
        # by 1 ulp, and because TP deliberately probes the two grid edges, a
        # 1-ulp gamma can land the recovered t on the other side of DFT_THI. It
        # did, at M = 256 / t = 32.0: the oracle took the table branch and the
        # C++ took the tail, so cf_bench's maxtab measured the handover step
        # itself -- 1.20e-3 nats = 0.00521 dB -- and failed a 1e-9 gate. The
        # gammas are now emitted directly (below), so no consumer reconstructs
        # one from a t at all.
        def g_of_t(t): return 2.0 * t * t * (M - 1) / 6.0
        t_hi = DFT["thi"] if DFT is not None else TQ
        t_quad, t_tail = 3.5, 2.0 * max(TQ, t_hi)
        assert _t(g_of_t(t_quad), M) <= TQ, (M, "Lemma-4 probe left the quadrature band")
        assert _t(g_of_t(t_tail), M) > max(TQ, t_hi), (M, "tail probe left the tail")
        rq = deficit_mod_quad(g_of_t(t_quad), M) / deficit_fin_quad(g_of_t(t_quad), M)
        # The tail probe is taken in the LOG domain, which is what psi() uses and
        # the only form that survives out here: 0.5*erfc(t/sqrt(2)) underflows to
        # exactly 0.0 in fp64 above t = 37.6 (fp32 above 13.5), so a linear-domain
        # ratio at t_tail = 2*max(TQ, DFT_THI) = 64 is 0/0. That is not
        # hypothetical -- it is what the first version of this probe did, and it
        # raised ZeroDivisionError as soon as DFT_THI moved t_tail past 37.6.
        ratio = float(np.exp(ln_deficit_mod(g_of_t(t_tail), M) -
                             ln_deficit_fin(g_of_t(t_tail), M)))
        assert abs(rq - L / (L - 1)) < QUAD_RATIO_TOL, (M, rq)
        assert abs(ratio - L / (L - 1)) < 1e-12, (M, ratio)
        report.append((M, rq, ratio, L / (L - 1), t_tail))
        blobs.append(struct.pack("<5i", M, d, K, var, C))
        blobs.append(lam.tobytes()); blobs.append(tcf.tobytes())
        blobs.append(flag.tobytes()); blobs.append(tau.astype(np.float64).tobytes())
        # deficit spot values at three gammas for cross-checking the C++/CL
        # QUADRATURE (cf_bench's maxq), emitted from deficit_*_quad so the
        # column keeps testing axis_deficit and not the table. t = 2.0 and 4.5
        # are inside the band where GH-48 is good to 1.1e-4 dB; t = 25 is beyond
        # GH-48's reach entirely (it underflows to zero at t = 20), so the third
        # probe is what pins the guard behaviour rather than the rule. All three
        # used to be t >= 11.2, i.e. tail-only, so maxq validated erfc and never
        # the GH rule.
        gs = np.array([g_of_t(2.0), g_of_t(4.5), g_of_t(t_tail)])
        blobs.append(struct.pack("<3d", *gs))
        blobs.append(np.array([deficit_fin_quad(g, M) for g in gs]).tobytes())
        blobs.append(np.array([deficit_mod_quad(g, M) for g in gs]).tobytes())
        # 4) ln D probes for cf_bench's maxtab (C++ vs oracle on the SAME
        #    evaluator: table, tail and quadrature branches all represented) and
        #    maxl4 (Lemma-4 ratio straight off the C++ values, which needs no
        #    oracle at all and so catches a wrong/transposed table even if both
        #    sides were wrong the same way). Deliberately straddling both grid
        #    edges -- 1e-5/5e-5/9.999e-5 below DFT_TLO and 1.0001e-4 above it,
        #    31.999 below DFT_THI and 32.0/32.001/40 at-or-above it -- so a
        #    branch-selection off-by-one cannot hide.
        #    The first array is the GAMMA, not the t (same 3 x nt doubles as
        #    before, so the layout size is unchanged): cf_bench then calls
        #    ln_deficit_*(g, M) on the identical bit pattern this oracle used,
        #    instead of rebuilding a gamma from a t and hoping the two spellings
        #    round-trip to the same branch. That is what the maxq block above
        #    has always done, which is why maxq agrees to 1e-16 while the
        #    t-based maxtab probe did not. cf_main recovers t from g with t_of's
        #    spelling for the `t >= 5` Lemma-4 filter only.
        GP = np.array([g_of_t(t) for t in TP], dtype=np.float64)
        blobs.append(struct.pack("<i", TP.size))
        blobs.append(GP.tobytes())
        blobs.append(np.array([ln_deficit_fin(g, M) for g in GP]).tobytes())
        blobs.append(np.array([ln_deficit_mod(g, M) for g in GP]).tobytes())
    pathlib.Path("build/cf_vectors.bin").write_bytes(b"".join(blobs))
    if DFT is not None:
        print("deficit evaluator: ln-D table (M=%s, N=%d, t in [%g,%g], tail above)"
              % (",".join(str(m) for m in DFT["M"]), DFT["n"], DFT["tlo"], DFT["thi"]))
    else:
        print("deficit evaluator: GH-48 quadrature up to TQ=%g, analytic tail above"
              " (HPOCL_DEFICIT_TABLE=0)" % TQ)
    for M, rq, rt, want, tt in report:
        print(f"M={M:4d}  Dmod/Dfin t=3.5 (quadrature)={rq:.12f}  t={tt:g} (tail)={rt:.12f}"
              f"  expected L/(L-1)={want:.12f}  rel err {abs(rq - want) / want:.2e} / "
              f"{abs(rt - want) / want:.2e}  {'OK' if abs(rq - want) < QUAD_RATIO_TOL else 'FAIL'}")
    print("wrote build/cf_vectors.bin")

if __name__ == "__main__":
    main()
