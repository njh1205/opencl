# HPOCL — Heterogeneous Hybrid-Precoding Operator Stack (AI-RAN flavor)

One OpenCL operator library + one host runtime, three execution targets, for the
SVD/THP hybrid-precoding switching law (closed-form threshold, main.tex) and its
learned surrogate (spectral set transformer), deployed on the QX_49DR_SV3
(Zynq UltraScale+ RFSoC, XCZU49DR-2FFVF1760) and validated today on any OpenCL
device (x86 CPU / Apple GPU / later: Zynq PS CPU and PL accelerator).

## Why this shape (the AI-infra story, honestly)

* **Operators, not monoliths.** Every numerical object of the paper is an
  OpenCL kernel with a stable ABI: covariance accumulation, iterative
  eigensolver (power iteration + deflation), baseband precoder GEMV, THP
  feedback/modulo datapath, the closed-form *decision operator*, and the
  transformer predictor's inference operators (dense/GELU/LayerNorm/masked
  self-attention/attention-pooling).
* **A runtime, not a script.** `Runtime` owns device policy (`cpu|gpu|acc` +
  fallback), buffer lifecycle, kernel registry, event profiling — the same
  vocabulary as inference runtimes (execution provider, operator dispatch,
  fallback chain), applied to a physical-layer workload.
* **Teacher / student policy pair.** The closed form (eq. cf + safeguarded
  Newton on the exact crossing equation) is the *teacher*; the spectral set
  transformer is the *student* that predicts the same threshold from BD
  spectral tokens. Both ship as operators; the runtime can run either, or
  cross-check them online (shadow mode).
* **Portability is the feature.** Same `.cl` sources and same host C++ build
  on: dev laptop (algorithm bring-up), cluster x86 CPU (current production),
  Zynq PS Cortex-A53 (control plane), Zynq PL via Vitis `.xclbin`
  (accelerated datapath). No hand-written Verilog anywhere: PL RTL is
  tool-synthesised from OpenCL C; RF/DDR/100G glue is vendor IP in the
  platform block design.

## Device / precision matrix

| target | OpenCL device | precision | role |
|---|---|---|---|
| laptop / cluster CPU | `CL_DEVICE_TYPE_CPU` | fp64 or fp32 | bring-up, regression |
| Zynq PS (A53) | host C++ (no OpenCL needed) or CPU device | fp64 | per-scheduling control: BD/whitening, GMD/UCD, closed form, Newton, LMMSE shrink |
| Zynq PL | `CL_DEVICE_TYPE_ACCELERATOR` (XRT) | **fp32** (`-DHPOCL_FP32`) | batched operators, per-symbol datapath |

fp64 is *not* assumed on PL: the runtime probes
`CL_DEVICE_DOUBLE_FP_CONFIG` and recompiles the datapath fp32 automatically.
The Newton solve runs on the **log-domain balance**
`Psi(u) = ln S_fin(u) - ln S_mod(u)` with log-sum-exp over per-stream
log-deficits — cancellation-free *and* underflow-free — which is what makes
an fp32 fabric datapath viable. Safeguarded Newton uses a bracket that is
geometrically extended until it provably contains the sign change, and a
**fixed 20-step budget** (`CfConfig::newton_iters`, `NEWTON_ITERS`,
`-DNEWTON_ITERS=20` — `tools/check_capacity.py` fails the build on drift).
Why 20 and not 8 is measured per regime in the comment on `newton_iters` in
`ref/threshold_cf.h`; the device-measured short version (`make cf_bench`, Apple
M4, **on the pre-table quadrature path**) is that 8 steps cost **8.7e-2 dB** on
one channel of the `M=64 d=4 K=8` UCD config (`ch3`, `flat_ratio = 0.162` — not
near-flat enough to be skipped) and 0.000 dB on the other four configs, i.e.
8 steps would violate the 0.05 dB accuracy target this README quotes. **That
argument no longer holds on the deployed table path**: the table's `Psi` is smooth
(converged deficit + 4-point Lagrange, no `TQ` staircase), and `dep` (20 vs 8) is
now **0.000 dB on every config**, so 20 -> 8 is affordable and worth 2.5x — see
the `NEWTON_ITERS` note in the Zynq section for what must be re-measured first. The deeply near-flat regime that `cf_bench`
skips (`flat_ratio < 0.05`) is worse still: only 2 of 20 Newton steps are
accepted there and the rest bisect. main.tex reports 5 steps for a CDL-A pool
that is not in this repo.

## Validation status (make check, OpenCL CPU/GPU device)

Every number in the *measured* column below was produced by `make check` on an
Apple M4 (`CL_DEVICE_DOUBLE_FP_CONFIG = 0`, so `threshold_cf` compiles
`-DHPOCL_FP32`) — not by a model of the formulas. Where a row is a relative
error it says so; the absolute rows state their units.

| check | bound | measured (device) |
|---|---|---|
| `gh_check` Gauss-Hermite table: host header vs device header vs `numpy.hermgauss(48)` | bit-identical / 1 ulp | `dz <= 0.0e+00`, `dw <= 0.0e+00`, `sum(w) = sqrt(pi)` |
| `registry_check` every kernel is registered, PL-linked and called; no macro or helper name shared by two sources | exact | 29 kernels / 10 sources, 27 macros + 18 helpers collision-free |
| `capacity_check` shared constants agree across kernel / host / oracle | exact | 12 constant groups, 26 definition sites |
| `allprog_check` all 10 sources build as ONE program, every kernel acquired | 29/29 | 29/29, twice (device-default opts and forced `-DHPOCL_FP32`) |
| `deficit_check` generated ln D table: host vs device header, fp32 narrowing, Lemma-4, tail handover, 90 values recomputed from scratch | bit-identical / 1 ulp / 1e-9 / 6e-3 dB / 1e-9 nats | 16 000 values bit-identical; worst 1 ulp; 5.50e-14; **0.00521 dB**; 1.36e-12 nats |
| quadrature (GH-48 fallback path) C++ vs numpy, both branches | 1e-9 rel | <=6.27e-16 |
| **table** ln D C++ vs numpy (`tab` column) | 1e-9 | 1.14e-13 |
| Lemma-4 ratio `D_mod/D_fin = L/(L-1)`, quadrature + table + tail (`l4`) | 1e-9 | <=5.7e-12 |
| tail handover step at `DFT_THI` (`hnd`) | 6e-3 dB | **5.20e-03 dB** (the `TQ=5` quadrature/tail switch it replaces was 0.3046 dB) |
| closed form `u*0` C++ vs numpy | 1e-9 dB | 0.00e+00 (40/40 channels string-identical) |
| Newton-root tau vs converged numpy oracle (30 steps) | 1e-4 dB | **<=2.49e-13** |
| method flag 0/1/2/4 vs numpy, exact | exact | 0 mismatches / 40 channels |
| fp32 OpenCL kernel vs fp64 host (tau, 5 configs x 8 ch) | 5e-2 dB | <=9.14e-05 |
| exact-crossing residual (sign change of Psi at u*) | all channels | 8/8 per config |
| deployed 20-step vs 8-step Newton (`dep`) — *evidence for the budget, not a tolerance* | 1.0 dB (divergence catch) | **0.000 dB on every config** with the table; it was **8.7e-2 dB** on one UCD channel on the quadrature path, which is what justified 20 |
| `op_bench` channel_cov vs fp64 golden ref, 7 shapes incl. 1x8 / 8x1 / 16x8 | **1e-6 relative** | 4.54e-07 |
| `op_bench` wbb_apply vs fp64 golden ref, same 7 shapes | **1e-6 relative** | 3.05e-08 |
| `op_bench` power_iter / deflate / switch_threshold / thp_precoder | 1e-3 / 1e-5 / 1e-4 | 1.91e-06 / 9.54e-07 / 1.91e-06 / 3.87e-07 |
| `op_bench` switch law with `nt/nrf = 8`, `eta = 1.0119`, `g_min = 0`, and `metric == tau` | mode mismatch 0, boundary exact | 0 mismatches, `>=` boundary pinned |
| `op_pl_bench` the same with `-DWBB_LOCAL_MAX=4096` (the tiled path the PL uses) | identical rel_err | 3.05e-08 — bit-identical to the untiled run |
| `op_bench` power_iter capacity guard (`M = 65 > PMAX`) | refuse + NaN | refused, outputs NaN |
| transformer predictor vs numpy eval semantics | 2e-3 | 2.4e-07 |
| `tf_export_check` real checkpoint -> flat binary -> OpenCL graph | exact tau | `err = 0.00e+00`. The `MAE 0.571 dB` this prints is **export fidelity over the first 200 of 3000 rows, all from config 0** (`N_LABEL = 200`), i.e. largely in-sample — it is NOT the predictor's accuracy. The checkpoint's own held-out metrics, shipped beside it in `mix_precode/_archive_20261018/bulk/results/svd_ucd_threshold_poolv2_3x1000_fast/poolv2_3x1000_fast_metrics.json`, are **test MAE 1.0722 dB, RMSE 2.3010 dB, decision accuracy 92.98 %**, on a pool of `K in {2,3,4}`, `T <= 9`, `M = 64` only — so neither number applies at the deployed `(64,4,8)`/`T = 32`, and none applies at `M = 256` |
| SVD sigma / V_D vs production python (scipy) | 1e-9 | 1.3e-15 / 3.0e-14 |
| GMD P,R / UCD V_D,upper,rho / CBD Q,R,P,active_count vs production python | 1e-9 | <=2.6e-14 (Q <=9.8e-14) |
| THP end-to-end x_tx vs production python (fp32 datapath) | 1e-5 | 5.0e-07 |
| fp32 PL kernels svd/gmd/ucd/cbd **incl. tall `V` rows>d and `m=16,d=8`** vs fp64 C++ | 3e-4 | <=3.2e-05 (48 vectors, worst `gmd m=4 d=4`) |
| degenerate vectors (9 of 48): rotation-invariant kernel outputs only | 3e-4 / 1e-5 | `cli <= 3.6e-07`; basis-dependent columns reported `n/a`, not silently skipped |
| CBD off-block spur vs production's re-detection gate (`spur32` column) | reported, >1 expected | 56x .. 134x — block structure must come from `active_count` |
| front end eff/N(projector)/eig/sigma(Gbar) vs production python | 1e-9 | <=2e-11 |
| pipeline: scheduling-instant control plane vs production python | 1e-9 | tau<=2e-10 dB, sigma<=1.5e-11, P<=6.9e-12, B<=2.2e-11, rho<=6.4e-10 |
| modem (QAM map, modulo, parallel/folded LLR, GMI) vs production python | 1e-9 | constellation/modulo bit-exact; LLR<=4.3e-14; GMI<=1.8e-15 |
| `datapath_bench` fp32 kernels vs fp64 numpy oracle (4 cases, m = 4..8) | 1e-4 linear / 1e-3 LLR / 3e-3 folded | qam<=1.3e-07, tx<=2.7e-07, ch<=1.5e-06, rx<=5.8e-07, fold<=6.4e-07, llr<=3.9e-05, folded<=2.8e-05, gmi<=1.3e-06 |
| `datapath_bench` radius-0 folded == parallel LLR (fp32 vs fp32) | 1e-5 | **0.0e+00** on all 4 cases |
| `datapath_pl_bench` the same with `-DLLR_CACHE_MAX=4` (recompute fallback for m>=6) | identical | every column identical to the cached run |
| `rate_bench` code-rate/MCS layer: numpy oracle vs host `.clh` vs fp32 kernel, 10 cases | index/m/R/outage exact, 1e-5 continuous | **0.00e+00** on all columns, all 10 cases |
| `datapath_bench` GMI -> MCS integration (device selection vs host `cqi_select` on the same GMI) | exact | 0 mismatches; sum SE 10.70 / 27.12 / 3.39 / 7.22 bpcu over the 4 cases |
| `chain_bench` end-to-end kernel chain vs fp64 C++ vs production oracle | 2e-3 | <=2e-06 (llr 1e-05), `thp32 = 0.0e+00` |
| receive control plane (q_gmd/w_ucd/R/diag/power scale) vs production python | 1e-9 | Q<=1.7e-16, R=0, scale<=5.6e-17 |
| LMMSE-CSI deployment (eps_s vs full-covariance LMMSE, shrink, fixed point) | 1e-12 / 1e-3 dB | eps<=1.7e-16; tau<=1.8e-4 dB |
| `asan_check` the 5 host-only benches rebuilt under ASan + UBSan | clean | clean |

**What changed to make the table above trustworthy.** Three defects used to make
parts of it meaningless, and each is now fixed *and* gated:

1. `TQ` (the quadrature -> analytic-tail crossover) lives in three files and had
   drifted to 10 in the kernel while the host and the oracle said 5. All three
   are 5, and `tools/check_capacity.py` fails the build on drift. The reason for
   5 is *resolution*, not truncation: GH-48's outermost node is **8.975315** (a
   12.69 sigma span) and the dominant error event sits at `z = t/sqrt(2)`, i.e.
   6.72 at `t = 9.5` — inside the span. What fails is that the softplus hinge is
   `1/(2*sqrt(2)*t) = 0.037` wide there against a local node spacing of 0.43.
   Measured GH-48 deficit error vs a converged reference: -0.0318 dB at `t = 4`,
   -0.1205 at `t = 5`, -0.6685 at `t = 7`, +1.6161 at `t = 9.5`, +1.9202 at
   `t = 10`; the analytic tail is +0.2684 / +0.1841 / +0.1006 / +0.0566 /
   +0.0513 dB at the same `t`. The branches cross at `t ~ 5.3`.
2. Above `TQ` the oracle used `log(C * 0.5 * erfc(t/sqrt(2)))` while the C++ ref
   and the kernel used `log(C) + lnQ_asym(t)` — two different functions, hence
   two different roots. `tools/cf_ref.py` now ports `ln_deficit_fin` /
   `ln_deficit_mod` verbatim. On the device this moved 39 of the 40 compared
   channels from 4.4e-5 / 5.1e-7 / 2.8e-5 / 2.2e-7 dB per config down to
   **2.1e-14 .. 3.6e-14 dB**. It also removed a live `-inf`/`nan` hazard: the
   linear form underflows to exactly 0.0 at `t >= 38` in fp64 (`t >= 13.5` in
   fp32) and this pool reaches `t = 71` at its root. The one channel that still
   differs (2.16e-5 dB) is a Newton-convergence residual of the fixed 20-step
   budget on a flat `Psi`, not a formula difference — the ladder is
   8 -> 20 steps = 8.7e-2 dB, 20 -> 30 steps = 2.2e-5 dB (**quadrature path**;
   on the deployed table path both rungs are 0.000 dB, because the table's `Psi`
   has no `TQ` staircase).
3. `CfResult::flag` now carries the kernel's full 0/1/2/4 encoding and saturated
   thresholds are `+-INFINITY` rather than `+-HUGE_VAL` (`HUGE_VAL` is
   `DBL_MAX`, so `std::isinf()` was false and `err_tau()` / the `(sat)`
   annotation mis-classified it). The host also refuses over-capacity shapes
   before launching (`Runtime::threshold_cf`), so the PS path cannot return a
   number the fabric could not have produced.

`ref/threshold_cf.cpp` (`kTQ`) carries the full measured tables.

Gauge policy: the SVD column phase is pinned canonically (largest entry of
Gbar V = sigma U real positive), so V_D, the physical precoder
P_k = F_RF N_k V_D,k and the receive combiner Q_k are deterministic across the
LAPACK-based production reference, the fp64 PS path and the fp32 PL kernels
(no hidden diagonal phase). The BD null basis carries a residual unitary
gauge, which cancels in every physical quantity above.

Decomposition operators are validated against the production python in
`mix_precode/classical` (DigitalStructuredPrecoder / thp_precoding) as oracle.
THP modulo uses the production floor-form centered modulo onto [-p/2, p/2).
Singular-vector comparisons are phase-aligned per column (unique up to phase).

## Paper claims vs implementation (findings for the paper authors)

`阈值闭式解推导/main.tex` derives the law this stack deploys. **main.tex is not
edited from this repo** — the table below is a findings list for its authors.
Line numbers are main.tex as of 2026-10-18. Verdicts: **CONFIRMED** (code
matches and the number was re-measured here), **STALE** (was true, superseded by
the code), **WRONG** (code or measurement contradicts it), **UNTESTED** (needs a
pool that is not in this repo). Every number in the evidence column was measured
on this machine against the shipped sources — an fp64 model of
`ref/threshold_cf.cpp` for the algorithmic rows, and two independent converged
references for the quadrature rows (composite Gauss–Legendre in the
standard-normal variable with Richardson extrapolation over h, h/2, h/4,
self-convergence ≤ 2e-11 relative; and raised-order Gauss–Hermite at
N = 96/192/384 via a Golub–Welsch tridiagonal eigensolve, because
`numpy.hermgauss` overflows and returns NaN weights for N ≥ 384).

| main.tex | claim | verdict | measured evidence |
|---|---|---|---|
| :22, :45-46 | `η = 1+(d−1)/(d(M−1))`; `γ*ₖ = ρgₖ/η` (GMD), `[(∏(1+ρλ))^{1/d}−1]/η` (UCD) | **CONFIRMED** | `thp_point` in `ref/threshold_cf.cpp`, `kernels/threshold_cf.cl` and `_gamma_star` in `tools/cf_ref.py` agree term for term |
| :55-58 | `Φ(u) = Σ D_fin(4uλ/δ²) − d Σ D_mod(4u g̃ₖ/δ²)`, `u = δ²ρ/4` | **CONFIRMED** | `psi_u` uses `4.0*u*lu[i]/d2` with `d2 = 6/(M−1)` and `ρ = 4u/d2`; `g̃ₖ(u) ≡ γ*ₖ/ρ` |
| :62, :128 | `τ = 10log10(4T(M−1)u*/6)` | **CONFIRMED** | identical expression in all three implementations |
| :123-127 | boxed `u*₀` | **CONFIRMED** | `threshold_closed_form` matches symbol for symbol, incl. the μ_λ tie test |
| :75-77 | `C_mod = 4π/ln2 ≈ 18.13`, `C_fin = C_mod(L−1)/L` | **CONFIRMED** | 4π/ln2 = 18.129440567308777; no factor missing in any implementation |
| :92-93 | `J(α) = (π/α)/sin(πα)`, `J(½) = 2π`, "verified numerically to eight digits" | **CONFIRMED** | `scipy.integrate.quad`: J(½) = 6.283185307202 vs 2π = 6.283185307180, rel err **3.5e-12** |
| :97 | `π/ln2 ≈ 4.532` bits per neighbour event | **CONFIRMED** | 4.532360141827194 |
| :103 | "the quadrature reproduces `C_mod/C_fin = L/(L−1)` to eight digits at M=16,64,256" | **CONFIRMED, but unqualified** | rel err of `D_mod/D_fin` vs `L/(L−1)`: ≤ **1.8e-15** for t ≥ 3, < 5e-9 for **t ≥ 2.25–2.35**, but 2.5e-7 at t = 2 and 4.2e-2 at t = 0.5. The claim needs "for t ≳ 2.3". It also stopped being *tested*: the probe sat at t = 8, above `TQ = 5`, so both operands took the `C·Q(t)` branch and the ratio was `L/(L−1)` by construction. Now probed at t = 3.5 (in band) and t = 25 (tail), with an assertion that the in-band probe really is in band |
| :130-135 | the `½ln(4πu)` prefactors cancel ⇒ closed form | **CONFIRMED** | both sides carry `(4πu)^{−1/2}`; the code implements the cancelled result |
| :136-138 | fallback scans "for a sign change of **Φ**"; "happened in **7 of 100** channels, all at d=2" | **WRONG (Φ) / UNTESTED (7 of 100)** | the scan is on **Ψ**, not Φ (`scan_root`, `psi_u`). The 7/100 rate reproduces nowhere here: the cf_bench pool is 38 flag-0 + 2 flag-2 + **0 flag-1** of 40, and `build/experiments/summary_all_configs.csv` reports `finite_frac 1.0`, `flag2_frac 0.0` for all 5 configs (2000 channels). Note "0.05 bit/s/Hz" there is a *rate*, not a threshold error |
| :140-145 | "Φ is strictly increasing on the operating interval … **the root is therefore unique**" | **WRONG for the implementation** | the lemma holds for Φ with the *exact* deficit. The deployed Ψ uses a piecewise deficit whose quadrature↔tail switch is a hard step, so Ψ is a staircase: measured max step **0.0709** (`TQ=5`) / **0.2584** (`TQ=10`), and **11** sign changes over `[u₀/4,4u₀]` on `M=16/d=2/K=4` ch4 at `TQ=10` vs **exactly 1** on all 39 measurable channels at `TQ=5`. 30/39 channels are locally non-monotone even at `TQ=5`. Uniqueness at `TQ=5` is empirical, not a consequence of the derivation |
| :147-150 | Corollary 1: `10log10(M−1)` in M, `10log10 T` in T, "depends on N_t,N_RF … **only** through λ_min and g_min" | **CONFIRMED for the closed form; CONTRADICTED by a shipped kernel** | leading-term steps `10log10(63/15) = 6.23` and `10log10(255/63) = 6.07` dB reproduce exactly. But `kernels/switch_law.clh` multiplies by an explicit `(nt/nrf)` and has no M or T dependence — see that file's header |
| :151-152 | "Measured: 15.66, 20.14, 24.76 dB at M=16,64,256 (steps +4.48, +4.62)" | **UNTESTED (internally consistent)** | 20.14−15.66 = 4.48 ✓, 24.76−20.14 = 4.62 ✓, both below the leading term ✓. The absolute values need a pool that is not in this repo. On one fixed spectrum (`λ = [[4,2,1,.5],[3,1.5,.9,.4]]`, K=2,d=4) the shipped code gives `τ_cf` = 21.460 / 26.752 / 32.436 dB (steps +5.29, +5.68) |
| :157-165 | `Φ′(u)` with `D′(γ) = (D/γ) dlogD/dlogγ`, UCD feedback `(1+u_eq)·mean(λ/(1+ρλ))/η` | **CONFIRMED as code, but it is the DEAD path** | `phi_sums` implements exactly this and `thp_point`'s `dgdrho = (1+geq)·(msum/d)/η` matches the UCD term symbol for symbol. Nothing calls `phi_u`/`phi_du`/`phi_sums` on either side; the deployed solver is Ψ-space (`psi_u`/`psi_du`), which main.tex never mentions. Same root set, different iteration |
| :165-166 | "a step leaving the bracket is replaced by bisection, so the iteration cannot diverge" | **CONFIRMED** | identical safeguard in all three; the bracket is additionally extended geometrically (6× per side), which the paper does not describe |
| :167-168 | "1.5e-3 dB maximum over **200 channels** at (64,4,8) with five iterations; **five is the deployed count**" | **"five is the deployed count" = STALE/WRONG; the rest UNTESTED** | the deployed count is **20** in all three sites. No 200-channel CDL-A pool exists here. On the available pool 5 iterations give ≤ **2.13e-4 dB** vs a Brent root — same order as 1.5e-3, so the figure is plausible but unverifiable |
| :169 | "three give 0.077–0.083 dB versus 0.063–0.068 dB [for five]" | **UNTESTED / inconsistent in magnitude** | same pool: 3 → ≤ **1.91e-1 dB**, 5 → ≤ **2.13e-4 dB**. The ordering (3 ≫ 5 worse) is confirmed; the magnitudes differ by 2.5× (3 steps) and 300× (5 steps) |
| — | `ref/threshold_cf.h` used to cite "main.tex's own **<0.05 dB** claim" | **MISQUOTE (now removed)** | main.tex contains no 0.05 dB claim; its only "0.05" is "0.05 bit/s/Hz" at :138. The 0.05 dB figure was `cf_bench`'s own `maxdep` tolerance |
| :169-171 | "Online evaluation uses `log D` on a uniform `log γ` grid of **1600 points** per modulation and branch, with the analytic tail `C Q(t)` beyond it" | **was WRONG — not implemented; now IMPLEMENTED** | Until this commit a repo-wide grep for `1600`, `log_grid`, `loggamma`, `log_gamma` returned **zero hits** and the deployed path evaluated `log D` by on-the-fly GH-48 quadrature per call. `kernels/deficit_table.clh` + `include/deficit_table.h` now carry `DFT_N = 1600` points x 10 rows (5 constellations x 2 branches) with the analytic tail `ln C + lnQ_asym(t)` above `DFT_THI = 32`. Two qualifications the paper should carry: the grid is uniform in **ln t**, not ln γ (equivalent up to the affine map `ln γ = 2 ln t + ln((M-1)/3)`, stated in the header), and the interpolation is **4-point Lagrange**, not the linear interpolation "a grid" usually implies |
| :179-194 | LMMSE `R_e = V diag(eig·nv/(eig+nv)) Vᴴ`, `ε_s = tr(R_e)/tr(R_h)`, "deterministic, deployable" | **CONFIRMED** | `lmmse_eps_s` is the trace ratio verbatim; `tools/lmmse_ref.py` cross-checks it against the production full-covariance estimator `prepare_full_covariance_mmse_pilot` and asserts 1e-12 |
| :200-205 | `γ_eff,i = ρλ̂/(1+ρλ̂ε_s)`, `λ̃ = λ̂/(1+ρλ̂ε_s)` | **CONFIRMED** | `shrink_spectrum` is character for character the formula; `lmmse_main.cpp` and `lmmse_ref.py` iterate the same 4-step ρ fixed point. The fixed point itself is not described in the paper |
| :206-211 | "the closed form is evaluated **verbatim** on λ̃" | **CONFIRMED** | `threshold_deploy_lmmse` calls the same `threshold_closed_form` on the shrunk spectrum; no second code path |
| :211 | "**Complexity remains O(T) scalar operations**" | **was WRONG on the quadrature path; TRUE now the table is in** | Counted exactly, by instrumenting every arithmetic operator and every transcendental in a `double`->counting-`double` transform of `ref/threshold_cf.cpp` that reproduces the shipped `tau` to 1e-10 dB (so it is the same program). **Quadrature path**, (M,d,K) = (64,4,8) GMD, 20 Newton: `psi_u` = **4 236 340 flop** and 600 240 `exp` (an earlier revision of this row said 2.65 Mflop — it applied the `(2n_img+1)` factor uniformly, but the folded deficit carries 5 images and the finite one 1, so it was low by 1.6x); one Newton iteration = **16 945 430 flop**; the whole solve = **340 275 085 flop + 48 210 888 `exp` + 6 125 633 `log` + 3 061 215 `log2`**, and **1 500 478 144 flop** at M = 256. The transcendentals, not the flops, set the A53 time: **~5.5 s per channel at M=64 and ~22.9 s at M=256** on a 1.4 GHz A53 (an earlier revision said 0.21 s / 1.1 s, which counted flops only and was low by 20-50x). **Table path**: ~**106 500 flop + ~13 000 transcendentals for the whole 20-step solve, independent of M** — so :211 became true *because of* the table |
| :212 | "κ=1 … is the no-fitting operating point" | **CONFIRMED** | `shrink_spectrum` hardwires κ=1; grep for `kappa`/`κ` across all sources: zero hits, so no tuning knob exists |
| :220-222 | MAE table 2.397 (κ=0) / **0.763** (κ=1) / 0.629 (κ=2), bias, std | **UNTESTED** | needs the 200 archived CDL-A channels (absent) and κ=0/κ=2, which the shipped code cannot express |
| :34 | "evaluated by quadrature with **no fitted constants**" | **CONFIRMED today** | GH-48 nodes/weights plus `4π/ln2`, `L/(L−1)`, `η`, `δ² = 6/(M−1)` — all exact. This is the reason the tail-correction option below was **not** taken |

### Decision record: the quadrature ↔ analytic-tail switch (measured, NOT implemented)

The switch at `t = TQ` is a **hard discontinuity** in `D(γ)`, because the two
branches do not agree where they meet. Its size is
`(tail err − GH-48 err)` evaluated at `t = TQ`:

| option | step in `D` at the switch | worst-case deficit error | max \|step in Ψ\| | Ψ sign changes on the pool | worst \|τ − τ(converged)\| | cost |
|---|---|---|---|---|---|---|
| `TQ = 10` (previous) | −1.8689 dB | 1.9202 dB | 0.2584 | up to **11** on one channel | 0.0945 dB | GH-48 |
| **`TQ = 5` (shipped)** | **+0.3046 dB** | **0.1841 dB** | **0.0709** | **1** on all 39 measurable | **0.0961 dB** | GH-48 |
| corrected tail `C·Q(t)(1 − 1.20/t²)`, `TQ = 4.5` | +0.0236 dB | 0.0683 dB | — | — | 0.0085 dB on the channel where `TQ=5` costs 0.0876 | GH-48 + 1 multiply |
| `GH_N = 192`, no switch | none (continuous) | 0.0337 dB at t = 9.5 | — | — | — | **4× quadrature** |

The measured coefficient of the O(t⁻²) term main.tex:68 already postulates:
`D_ref/(C·Q(t)) = 1 − 1.20/t²` with residual ≤ 0.0002 dB for t ≥ 12 (the raw
leading term is +0.1841 dB at t = 5, +0.1006 at t = 7, +0.0513 at t = 10; the
corrected form is −0.0295, −0.0071, −0.0011 dB at the same t).

**Rejected, with reasons.** The corrected tail would cut the step 13× and the
worst deficit error 2.7×, and on the pool it moves `τ` by ≤ 0.096 dB (0.0876 →
0.0085 dB on `M=16/d=2/K=4` ch1, ≤ 0.014 dB elsewhere) — but 1.20 is an
*identified* coefficient, which collides with main.tex:34 "no fitted constants"
unless it is derived from the same integration by parts that yields `J(½) = 2π`.
`GH_N = 192` needs no new constant and is continuous, but costs 4× the
quadrature on a fixed-size PL datapath. **Neither is implemented.** If the
0.3046 dB step ever has to go, derive the coefficient first and take the
corrected tail; do not raise `GH_N` on the PL.

Also measured and deliberately left alone: at `denom = g_min/η − λ_min` exactly
1e-2 — the operating point an earlier comment blamed for a "0.55 dB τ error at
`TQ=10`, 0.000 dB at `TQ=5`" — both values give the **same** τ and both match
the converged reference to ≤ 0.0001 dB. At `denom = 3e-3` both are 2.27 dB off
the converged reference, identically. That residual is the ill-conditioning of
the crossing itself (`dΨ/dlnu ≈ denom·u → 0`), not the quadrature band, so no
`TQ` fixes it; `cf_bench`'s `flat_ratio < 0.05` skip is what keeps it out of the
gates.

### Decision record: fallback-scan resolution and root uniqueness (NOT implemented)

The kernel scans 160 points (`kernels/threshold_cf.cl`) while `scan_root` in
`ref/threshold_cf.cpp` and `root_u` in `tools/cf_ref.py` scan 400, and the fp32
kernel's dead band is `PSI_BAND = 1e-3` against `1e-6` in fp64. Measured over 18
degenerate γ=0 spectra (1 and 2 zero streams, and 1 zero + flat rest, for
M ∈ {16,64}, d ∈ {2,4}) at **both** `TQ` values and **both** dead bands:

* max \|τ(400 pts) − τ(160 pts)\| = **0.0004 dB** (typically 0.0000), even
  though the raw `u₀` from the two grids differs by up to **9.3%**
  (2.115e5 vs 2.313e5);
* `PSI_BAND` 1e-6 → 1e-3 changes τ by **0.0000 dB**;
* Ψ has **exactly 1** sign change over the whole scan range [1e-6, 1e6] on every
  one of those cases.

So the 0.69 dB disagreement an earlier note attributed to the 160-vs-400 gap
**does not reproduce**; the scan only supplies a Newton start point, and Newton
lands on the same root. Unifying the grids would cost 2.5× the fallback scan on
the PL for ≤ 0.0004 dB, so it is **not done**.

A sign-change-count uniqueness assertion (report `flags |= 8` when Ψ crosses more
than once) was considered and **rejected**: it would need a fourth site to agree
on a new flag bit (kernel, C++ ref, oracle, bench) for a diagnostic; it would
stop the scan from early-exiting; and at `TQ = 5` there is nothing for it to
catch — 1 sign change on all 39 measurable pool channels and on all 18
degenerate γ=0 shapes. The cheap half of the win is already in: `cf_bench` now
compares flags for exact 0/1/2/4 equality, and `CfResult::flag` mirrors the
kernel encoding, so a method divergence between the three implementations fails
loudly instead of being folded into `fell_back`. If `TQ` ever moves up, or
`GH_N` moves down, re-measure the sign-change count before trusting uniqueness.


## Layout

```
kernels/            OpenCL C operator library (portable subset)
  threshold_cf.cl     closed form + safeguarded Newton (per-channel work-item)
  decomp.cl           jacobi_svd / gmd_k / ucd_k / cbd_k (fp32, per-user work-item)
  datapath.cl         per-symbol chain: qam_map / tx_combine / channel_apply /
                      rx_combine(+diag_b) / modulo_fold / softdemod (parallel +
                      folded replica) / gmi
  rate.cl             mcs_select_k: per-stream GMI -> CQI index, m, R, SE, margin
  channel_cov.cl power_iter.cl wbb_apply.cl thp_precoder.cl switch_threshold.cl
  tf_ops.cl           transformer predictor inference operators
  cplx.clh            the ONE device definition of complex arithmetic + the THP
                      centered modulo (guarded; several .cl land in one TU)
  mcs_table.clh       CQI table + code-rate selection rule, compiled into the
                      HOST as well (runtime/rate_main.cpp) as into rate.cl
  switch_law.clh      surrogate switching law, host (ref/hp_ref.cpp) + device
  gh_table.clh        Gauss-Hermite nodes/weights, fp64 + fp32 blocks
include/            hpocl_types.h (cplx, PrecoderCfg, McsSel), gh_table.h (the
                    host twin of gh_table.clh), hpocl_modulo.h (host twin of
                    cmod_axis), hpocl_clsrc.h (read_cl_source: the one .cl
                    #include inliner), hpocl_vecio.h (bench vector file reader)
ref/                CPU golden references (PS-role code, zero dependencies):
                    threshold_cf (closed form + Psi-Newton + LMMSE), decomp
                    (svd_cplx / gmd_factors(left+right) / ucd_design /
                    cbd_factors / stream_snr), modem (QAM/LLR/GMI), rx
                    (combiner / power scale), hp_ref (data-path), pipeline (full
                    scheduling instant), frontend (RF-BD front end: effective
                    channels, BD null basis on the scipy null_space convention,
                    floored C^{-1/2} power whitening, whitened Gbar_k)
runtime/            host runtime (plain OpenCL 1.2 C API => XRT-compatible)
  hpocl_runtime.*     device policy, the ONE kernel registry, profiling, and a
                      wrapper per operator group (control plane, decompositions,
                      datapath, code rate)
  transformer_predictor.*  inference graph for the spectral set transformer
  cf_main.cpp         three-way bench: numpy vs C++ PS path vs OpenCL kernel
  pipeline_main.cpp   scheduling-instant control plane vs production python
  decomp_main.cpp     svd/gmd/ucd/cbd: production python vs C++ vs fp32 kernel
  frontend_main.cpp   RF-BD front end vs production python
  bench_main.cpp      data-path operator bench vs CPU golden reference
  modem_main.cpp      receive modem (QAM/LLR/GMI) vs production python
  datapath_main.cpp   fp32 per-symbol datapath kernels vs numpy reference
  rate_main.cpp       code-rate/MCS layer: numpy oracle vs host .clh vs kernel
  chain_main.cpp      end-to-end kernel chain vs fp64 C++ vs production oracle
  rx_main.cpp         receive control plane vs production python
  lmmse_main.cpp      LMMSE-CSI threshold deployment vs numpy oracle
  tf_main.cpp         transformer predictor bench vs numpy reference
  allprog_main.cpp    allprog_check: every source in ONE program, all kernels
tools/              oracles + generators + static gates
  cf_ref.py decomp_ref.py datapath_ref.py chain_ref.py modem_ref.py rx_ref.py
  pipeline_ref.py frontend_ref.py lmmse_ref.py rate_ref.py transformer_ref.py
  modem_common.py env_common.py ckpt_io.py   (shared oracle pieces)
  check_gh.py check_registry.py check_capacity.py   (static gates, no device)
  gen_gh_table.py gen_gh_cl.py make_random_tf_weights.py export_transformer.py
  tf_export_check.py exp_threshold_policy.py
opencl/build/       build_xclbin.sh (Vitis flow for xczu49dr: compiles all ten
                    kernels/*.cl sources into one .xclbin), link.ini
docs/               exp_threshold_policy.md, vitis_flow_notes.md
```

### What goes into the PL `.xclbin`

`opencl/build/build_xclbin.sh` compiles **every** operator source with one
shared flag set (`-DHPOCL_FP32 -DNEWTON_ITERS=20 -I$ROOT/kernels`), 29 kernels
in 10 `.xo` objects, then links `"$OUT"/*.xo`:

| source | kernels in the xclbin |
|---|---|
| `threshold_cf.cl` | `threshold_cf` |
| `power_iter.cl` | `power_iter` `deflate` |
| `decomp.cl` | `jacobi_svd` `gmd_k` `ucd_k` `cbd_k` |
| `switch_threshold.cl` | `switch_threshold` |
| `wbb_apply.cl` | `wbb_apply` |
| `thp_precoder.cl` | `thp_precoder` |
| `channel_cov.cl` | `channel_cov` |
| `datapath.cl` | `qam_map_k` `tx_combine_k` `channel_apply_k` `rx_combine_k` `modulo_fold_k` `softdemod_llr_k` `softdemod_folded_k` `gmi_k` |
| `rate.cl` | `mcs_select_k` |
| `tf_ops.cl` | `tf_dense` `tf_gelu` `tf_add_broadcast` `tf_mul_mask` `tf_layernorm` `tf_attn` `tf_pool` `tf_sign_pack` `tf_add_elem` |

`gh_table.clh`, `cplx.clh`, `mcs_table.clh` and `switch_law.clh` are headers
(inlined into the sources that include them), not kernels. `make registry_check`
proves this table, the `__kernel void` names in `kernels/*.cl`, the host
registry in `runtime/hpocl_runtime.cpp` and the set of names some host file
actually calls are the same 29 names — a kernel that exists but is not
registered is source-JIT-only and therefore undeployable, which is exactly how
the eight datapath kernels were once lost.
`-DHPOCL_FP32` only changes `threshold_cf.cl` (the single source with an fp64
default); the other eight are fp32-only already. This table is checked
statically against the `__kernel void` names in `kernels/*.cl` — Vitis is not
available off-board, so the script itself is validated with `bash -n` plus that
cross-check, not by a real `v++` run.

### Compiled-in capacity limits (PL private/local memory is fixed-size)

| limit | value | meaning | out-of-capacity behaviour |
|---|---|---|---|
| `MAXT` (`threshold_cf.cl`) | 64 | max streams per channel, `T = K*d` | `flags[c] = 4`, `tau = NaN` |
| `MAXK` (`threshold_cf.cl`) | `MAXT` (64), `-DMAXK=n` to shrink | max users `K`; `K <= K*d <= MAXT` since `d >= 1` | `flags[c] = 4`, `tau = NaN` |
| `MAXD` (`threshold_cf.cl`) | 128 | max distance candidates in **one** of `d0[]`/`d1[]` per label bit. Folded quadrature emits `5·⌈L/2⌉` per array (the reflected Gray code splits the `L` levels exactly in half), so `5·⌈L/2⌉ <= 128` ⇒ `L <= 50` ⇒ `M <= 2500`; `M ∈ {4,16,64,256,1024}` fit, `M = 4096` does not | `flags[c] = 4`, `tau = NaN` |
| `kCapMaxT`/`kCapMaxK`/`kCapMaxD` (`ref/threshold_cf.h`) | 64 / 64 / 128 | **host mirror** of the three above, so the PS path refuses exactly the shapes the PL kernel refuses instead of returning a number the fabric cannot produce | `CfResult::flag = 4`, `tau = u_star = NaN` |
| `DECOMP_MAXM` / `DECOMP_MAXD` (`decomp.cl`) | 16 / 8 | rows and streams per user matrix | kernel writes `NaN`/`act = -1` sentinels, and `Runtime::jacobi_svd/gmd_op/ucd_op/cbd_op` refuse host-side and `NaN`-fill before launching |
| `PMAX` (`power_iter.cl`) | 64 | max `M`; the runtime launches with **local size == M**, so `M <= PMAX` and local size `>= M` are both required | `v_out`/`lambda_out` = `NaN` sentinel (written by *global* id, so it covers every row for any local size); `Runtime::power_iter` also refuses host-side and `NaN`-fills |
| `KMAX` (`thp_precoder.cl`) | 16 | feedback history `b[KMAX]` per work-item | kernel writes `NaN` into `Y`; `Runtime::thp_precoder` refuses host-side and `NaN`-fills |
| `MMAX` (`datapath.cl`) | 256 | largest constellation (`m <= 8`); `softdemod_llr_k` caches all `M` squared distances | `Runtime::qam_map/softdemod_*/gmi` refuse host-side; the kernels write `NaN` |
| `LLR_CACHE_MAX` (`datapath.cl`) | `MMAX` by default | private bytes per work-item in `softdemod_llr_k` = `4*LLR_CACHE_MAX` | a constellation above the bound takes the recompute fallback (two sweeps, `O(1)` memory) instead of failing |
| `WBB_LOCAL_MAX` (`wbb_apply.cl`) | 128 (= 1 KB) | local memory per work-group for the `wbb_apply` tile | a group whose tile would not fit takes the direct global-memory path |

Two of these were renamed to stop a silent collision: `decomp.cl` used to define
`MAXD 8` while `threshold_cf.cl` defines `MAXD 128`, and `Runtime::build()`
concatenates every `.cl` it is given into a **single** translation unit — so a
bench that loaded both would have resized `threshold_cf`'s `d0[]`/`d1[]` from 128
to 8 and overflowed it 16x. `make registry_check` now fails on any macro or
non-static helper name shared by two kernel sources, and `make allprog_check`
builds all ten sources as one program on a real device to prove the point.

Every host mirror above is checked against its `#define` by
`tools/check_capacity.py` (`make capacity_check`), which is text-only and runs
anywhere: 12 constant groups over 26 definition sites.

`flags[c]` (device), `CfResult::flag` (host) and the second element of
`cf_ref.tau_policy()`'s return (oracle) are **one encoding**:

| flag | meaning | `tau` |
|---|---|---|
| 0 | closed form (main.tex eq. cf) + safeguarded Newton | finite |
| 1 | near-flat spectrum: the geometric scan of `Psi` supplied the Newton start point | finite |
| 2 | saturated: `Psi` keeps one sign over the whole scan, so the branch ordering is uniform | `±INFINITY` (`-inf` = always THP, `+inf` = always SVD). `INFINITY`, not `HUGE_VAL`: `HUGE_VAL` is `DBL_MAX`, so `std::isinf()` was false and `err_tau()` / the `(sat)` annotation mis-classified it |
| 4 | outside the compiled capacity (`T > MAXT`, `K > MAXK`, `5·⌈L/2⌉ > MAXD`) **or `M` is not a perfect square** | `NaN`, host must not use it |

`cf_bench` tests flag 4 on **both** sides *before* its `flat_ratio < 0.05` skip,
so a near-flat over-capacity channel cannot slip through, and it compares the
remaining flags for exact equality rather than the old `(flag != 0) != fell_back`
(which could not tell 1 from 2). `CfResult::fell_back` is now exactly
`flag == 1`.

## Zynq control plane: what is actually optimised, and what is not

Asked directly — *is the control plane optimised for the Zynq?* — the honest
answer as of this commit is **no: it is PL-compatible and fully validated, but
it is neither costed nor wired**. The three gaps, in the order they bite:

**1. Nothing outside the benches calls the device operators.** `grep` for
`rt.threshold_cf(`, `rt.gmd_op(`, `rt.ucd_op(`, `rt.cbd_op(`, `rt.jacobi_svd(`
outside `runtime/*_main.cpp` returns nothing. `ref/pipeline.cpp` — the
scheduling-instant control plane, and the code a PS image would actually run —
includes no OpenCL header at all and calls the fp64 host references
(`svd_cplx`, `gmd_factors`, `ucd_design`, `threshold_closed_form`). So on the
board today the control plane is **scalar A53 C++**, and the 29 validated fp32
kernels are exercised only by `make check`. The kernels being correct is
necessary and not sufficient: there is no offload path.

**2. It is now costed — and the answer changed while it was being measured.**
Counts below are exact, from instrumenting every arithmetic operator and every
transcendental in a `double`-to-counting-`double` transform of
`ref/threshold_cf.cpp` that reproduces the shipped `tau` to 1e-10 dB; A53
projections use an explicit cycle model (1.4 GHz, in-order, 1 fp64 flop/cycle,
`exp` 80 / `log` 70 / `log2` 55 / `div` 29 / `sqrt` 33 cycles, x1.5 structural
overhead calibrated against a measured host run). They are **estimates**, and
the host they were timed on is a fanless M4 that throttles 3-3.7x under sustained
load, so only the counts are clock-invariant.

Per channel, `(M,d,K) = (64,4,8)` GMD, 20 Newton steps:

| | quadrature path (`-DHPOCL_DEFICIT_TABLE=0`) | deficit-table path (default) |
|---|---|---|
| flop | **340 275 085** | **~106 500** |
| `exp` / `log` / `log2` | **48 210 888 / 6 125 633 / 3 061 215** | ~3 300 / ~6 000 / **0** |
| dependence on `M` | 1 500 478 144 flop at `M = 256` (4.4x worse) | **none** (measured 199.7 us at `M=64` vs 209.3 us at `M=256`) |
| this host, per channel | 317.4 ms | **51.3 us** |
| A53 projection (**estimate**, not measurement) | **~5.5 s** (22.9 s at `M = 256`) | **~1.2 ms** |
| speedup, measured back-to-back in one process on a quiet machine | — | **6 187x at `M=64`, 28 290x at `M=256`**, 1 012x at `(16,2,4)` |
| `make cf_bench` wall clock | ~13 min | **3.2 s** |
| `make check` wall clock (20 gates) | ~25 min | **37 s** |

Both host columns were taken on the same core in the same process; the M4 is
fanless and throttles 3-3.7x under sustained load, so an earlier revision of this
table quoted 30 us / 10 697x from a throttled-state ratio. 51.3 us / 6 187x is
the conservative number and the one to quote. The *ratio* is the defensible
quantity; the absolute A53 times are projections from an explicit cycle model.

The per-channel spread on the quadrature path is **98-343 Mflop** across the
8-channel pool, tracking how many streams fall inside the quadrature band; an
all-tail channel costs 27 959 flop, i.e. **12 000x less**. The cost was bimodal
over four orders of magnitude, which is why a single "it costs X" statement was
never possible.

**Does it fit a scheduling interval?** (K = 8 users, one instant)

| | quadrature path | table path as landed | table + the hoists in item 6 |
|---|---|---|---|
| PS (A53), 1 ms | **no — short by ~5 500x** | **no — short by ~20x** | **no — short by ~14x** |
| PS (A53), 10 ms | **no — short by ~550x** | **no — short by ~1.5-2.7x** | **yes, ~1.5x margin** |
| PL, 1 ms | **no — short by >=290x.** 57.4 M transcendentals per channel; one II=1 `exp`/`log` unit at the 205 MHz this repo measured for a minimal kernel is a **287 ms hard floor**, and 20 serially-dependent Newton steps give almost no parallelism to spend | **yes — 10-50x headroom** (~106 k flop + 15 k transcendentals; `g[1]={C}` already batches) | **yes — 50-150x** |

**3. The table the paper claimed and the code lacked is now in.** main.tex
(~:169) says the online evaluation uses "`log D` on a uniform `log gamma` grid of
1600 points per modulation and branch, with the analytic tail `C Q(t)` beyond
it"; until this commit a repo-wide grep for `1600`/`log_grid`/`loggamma` returned
**zero hits**. `kernels/deficit_table.clh` + `include/deficit_table.h` now carry
`DFT_N = 1600` x 10 rows (5 constellations x 2 branches), generated by
`tools/gen_deficit_table.py` and gated by `tools/check_deficit_table.py`, with
4-point Lagrange interpolation in `ln t` and the analytic tail above
`DFT_THI = 32`. Two consequences worth stating plainly:

* **It also removes the quadrature/tail discontinuity.** The values are the
  *converged* deficit (composite Gauss-Legendre, refinement-checked to 1e-11),
  not GH-48 — which is worth -0.12 dB at `t = 5` and underflows to zero at
  `t = 20`. The handover step is now **0.0051 dB against 0.3046 dB** for the
  `TQ = 5` switch it replaces. That is the fix the "quadrature <-> analytic-tail
  switch" decision record below was looking for, arrived at from the cost side.
* **It changes deployed numbers.** Against the old GH-48 path the threshold moves
  by **<=0.096 dB** (0.027 dB at `(64,4,8)` GMD, 0.0037 dB at `M = 256`) — that
  gap is the GH-48 *model error being removed*, not interpolation error, which
  the generator measures at **5.7e-6 dB worst**. `-DHPOCL_DEFICIT_TABLE=0`
  restores the quadrature path, which stays compiled in and reachable as the
  fallback for `t < DFT_TLO` and for an untabulated `M`.
* **PL footprint, and the one real deployment risk.** 62.5 KB of fp32
  `__constant` = 500 Kibit = **14 BRAM36 by capacity, up to 20 with depth
  granularity** (a 1600x32b row does not fit one 1024x36 BRAM36), or 2 URAM288.
  That is capacity arithmetic on standard block sizes, **not a Vitis synthesis
  report**. It is now the memory item to budget — 2.4x the per-work-item scratch
  (3 328 B fp32 at `MAXT = MAXK = 64`, `MAXD = 128`), which at the `C = 1..8`
  this workload needs is 0.04-0.34% of on-chip memory and was never the binding
  constraint. The risk is `CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE`: 62.5 KB is
  **97.7% of the 64 KB OpenCL spec minimum**. The Apple M4 reports 1 GB so it is
  a non-issue here; if XRT reports 64 KB the fp32 `clBuildProgram` will fail.
  Mitigations, each one generator invocation plus `make deficit_check`:
  `--N 800` (31.25 KB, 8.4e-5 dB interpolation error), `--ms 64,256` (4 rows =
  25 KB, other `M` fall back to quadrature), or `--ms 64` (12.5 KB).
* **`NEWTON_ITERS` stays at 20**, but the evidence for it has changed: with the
  smooth table `Psi`, `dep` (20 steps vs 8) is **0.000 dB on every config**,
  where the quadrature path needed 20 to hold one UCD channel to 8.7e-2 dB. So
  20 -> 8 is now worth 2.5x and is affordable — but the near-flat population that
  `cf_bench` skips (`flat_ratio < 0.05`) is exactly the one that justified 20 and
  is still not measured on the table `Psi`. Re-measure that population first;
  the change has to move four sites together (`threshold_cf.cl`,
  `ref/threshold_cf.h`, `build_xclbin.sh`, and `tools/check_capacity.py` will
  fail until they agree).
* **Format tag**: `build/cf_vectors.bin` starts with magic `'DFT2'` and carries
  the evaluator flag it was generated under, so a stale file — or one generated
  under the other `HPOCL_DEFICIT_TABLE` setting — is refused with a message
  naming the fix instead of being silently misread.

**4. The threshold is no longer the bottleneck — the BD front end is, and the
shipped kernel refuses its geometry.** With the table landed the threshold drops
from **99.4%** of the PS control plane to **1.4%**. What replaces it: at
`K=8, Nt=16, Nr=4, NRF=32, d=4` (⇒ `rows = (K-1)*Nr = 28`, `nul = 4`),
`ref/pipeline.cpp::schedule()` spends **~2.05 ms of ~2.08 ms** in the front end,
and **91% of that is eight `null_basis` calls, each a 28x32 one-sided Jacobi
SVD** — projecting to **~19 ms on an A53**, over both a 1 ms and a 10 ms budget.
Moving it to the PL is the obvious answer and it is blocked: `jacobi_svd` is
called with `rows = 28 > DECOMP_MAXM = 16` and `n = NRF = 32 > DECOMP_MAXD = 8`,
so `Runtime::jacobi_svd` **refuses and NaN-fills without launching**. Same for
`analog_precoder`'s `svd_cplx(4x16)`. Raising the two macros is the single
highest-value remaining change (BRAM: `W[MAXM*MAXD] + Vv[MAXD*MAXD]` float2 =
23 KB/work-item at 32x40, x8 work-items = 184 KB, affordable against ~7.5 MB),
and it must be paired with cutting sweeps: the fp64 host exits after 5-6 sweeps
but the fp32 kernel's relative off-diagonal bottoms out above `tol`, so it runs
the full **16** — 2.7x wasted. Also note `null_basis` calls
`svd_cplx(A, rows=28, n=32)` with **m < n**, which contradicts `ref/decomp.h`'s
documented `m >= n` contract; it works for every geometry the pipeline can
express (`rows < NRF` is exactly what `nul > 0` requires), but the contract
comment is wrong and any optimisation that assumes `m >= n` will silently break.

**5. The transformer predictor is not the fast path.** It was the obvious
candidate and it is dominated on every axis. Cost: **21.22 Mflop** but
**5.05 MB of fp32 weights streamed per forward** — 70% of the ZU49DR's entire
on-chip memory, so it cannot be cached and runs **bandwidth-bound at 0.5-2.1 ms
on the PL** (not compute-bound at the 12.4 us the flop count implies) and
**7-15 ms on the A53**. Accuracy: **1.0722 dB held-out MAE, 2.3010 dB RMSE,
92.98% decision accuracy** against a control law whose target is **0.05 dB** and
whose exact solver now delivers **4.97e-14 dB** on 39 of 40 pool channels. Its
training pool is `K in {2,3,4}`, `T <= 9`, `M = 64`, labels on a 1 dB grid over
`s in [0,30] dB` — while the deployed shape is `K=8`, `T=32`, pool `tau`
38.8-47.4 dB, and `M = 256` was never trained. So it is outside its support on
every axis. Seeding does not rescue it either: the closed form's own seed `u*_0`
costs **~10 CU (~0.05 ms on A53)** and lands within 1.46 dB of the refined root,
i.e. comparable quality to the predictor's 1.07 dB for **200x less**. Use it for
shadow-mode monitoring (README's Zynq checklist step 4), not on the critical
path. If it ever is deployed, cache the 48 `cl_mem` across `forward()` calls and
drop the mid-graph single-scalar blocking readback: today it is **42 launches +
48 buffer create/destroy + 3 blocking reads per forward**, which on XRT is 1-5 ms
of pure driver overhead.

**6. The ordered plan from here**, ranked by (gain x confidence) / effort.
*DHV* = validatable on the dev host; *BOARD* = needs Vitis or the board.

| # | item | gain | numerics? | where |
|---|---|---|---|---|
| 1 | ~~land the deficit table~~ | **done** — measured 6 187x at `M=64`, 28 290x at `M=256`; `make cf_bench` 13 min -> 3.2 s, `make check` ~25 min -> 37 s | yes, <=0.096 dB toward the converged reference (that gap is GH-48 model error being removed, not interpolation error, which is 5.4e-6 dB) | DHV |
| 2 | raise `DECOMP_MAXM` 16->32 / `DECOMP_MAXD` 8->40 and batch the 8 `null_basis` SVDs into one `jacobi_svd` launch (`C = 8`); cut sweeps 16 -> 6 | front end **~19 ms -> ~0.6 ms on the PL** | no (same arithmetic) | BOARD for BRAM/Fmax, DHV for the guards |
| 3 | three hoists in the table path: precompute `log lambda_i` and `g_k`, make the grid index affine in `log u`, branch-test on `t^2 = 2*u*lambda` instead of `sqrt` | threshold **1.21 -> 0.35 ms** on A53; removes ~5 900 `log` + 6 400 `sqrt` + 1 300 `div` per channel | yes, at the 1e-16 level (measured <=2.2e-5 dB well-conditioned, <=9.1e-3 dB on one ill-conditioned channel — the same magnitude as `-ffp-contract=off`, which moves 17 of 24 channels by 2.16e-5 dB) | DHV |
| 4 | Newton 20 -> 8 | another **2.5x** (0.35 -> 0.15 ms) | yes — the ladder must be re-measured on the *table* `Psi`, which is smooth (no `TQ` staircase), so 8 should behave like the well-conditioned regime; re-run the near-flat channel that justified 20 | DHV |
| 5 | kill the PS allocator traffic: `psi_u`/`psi_du` do **164 `operator new` / 26 240 B per channel** under clang, which elides `make_axis`'s and the `d0/d1` `push_back` allocations — **GCC/libstdc++ on aarch64 does not**, where the structural count is ~2.07 M malloc/free pairs per channel. Replace with fixed-size stack arrays, as the `.cl` already does | up to **~90 ms/channel** on the quadrature path with g++; ~23 us on the table path | no, if summation order is preserved | DHV but only *measurable* with the real aarch64 g++ |
| 6 | analytic `psi_du`: add `d ln D / d ln t` as an 11th table row instead of the 3-point central difference | **2.5-2.8x on the whole Newton loop** (`psi_du` = 2.82-2.98x `psi_u`), frees 4 of the 6 private arrays | yes — needs a gate against a central difference, which already exists | DHV for accuracy, BOARD for the +6.25 KB |
| 7 | PL restructure of `threshold_cf.cl`: `-DHPOCL_L/-MA/-K/-D` so every trip count is a literal; delete the quadrature from the PL build (`t < DFT_TLO` -> `log2(M)`, worth <=1e-9 dB); constant-latency bracket expansion (6 unconditional `psi_u` + priority mux) instead of the data-dependent `for (e<6 && psi_u(lo)>0)`; fallback scan -> a fixed 40-step bisection on `ln u`, same grid and band on PS and PL (kills the live 160-point/`PSI_BAND=1e-3` vs 401-point/`1e-6` divergence); single exit; `flags |= 8` when the safeguard rejects too many steps | **~10-30 us/channel** at 200 MHz with 4-8 lanes, ~40-80 DSP48E2, constant latency | R1/R3/R6/R8 no; R2 <=1e-9 dB; R7 <=0.0004 dB | BOARD |
| 8 | make `threshold_cf.cl` **fp32 by default** (invert the `#ifdef`) | removes a footgun: `build_xclbin.sh` passes `-DHPOCL_FP32` but the one `v++ -c -t hw` run this repo ever did (`docs/vitis_flow_notes.md`) did **not**, so any new build script that forgets the flag silently tries to synthesise fp64 on a fabric whose DSP48E2 has no floating point at all | yes — `PSI_BAND` 1e-6 -> 1e-3, `cl_tol` 1e-6 -> 5e-2; the kernel already documents 1.3e-4 dB of fp32 index error, 350x inside budget | DHV |
| 9 | `exp`/`log` -> `exp2`/`log2` | PL only (~30-40% fewer LUTs/cycles per unit). On the A53: **zero** — ARMv8.0-A has no hardware `exp2`/`log2`, and `log2` measured *slower* than `log` on this host | yes, ~1 ulp | BOARD |
| 10 | a device-side `schedule()` (item 2 below) and a batch builder | turns "the kernels are correct" into "the board runs them" | no | DHV + BOARD |

Two things the cost model corrected elsewhere in this file: `channel_cov`'s
binding constraint on the PL is **traffic, not DSPs** — it reads **1.11 MB of
global memory for 65.5 KB of data (17x amplification)**, ~0.5 ms per covariance
update at ~2.4 GB/s per AXI port, which is a scheduling-instant-scale number for
an operator usually assumed free; and the ZU49DR's slice is **DSP48E2, not
DSP58** (DSP58 is Versal).

What *is* done for the fabric, and is not nothing:

| property | state |
|---|---|
| fp32 path | `-DHPOCL_FP32` gives `threshold_cf.cl` an all-fp32 build; the runtime probes `CL_DEVICE_DOUBLE_FP_CONFIG` and marshals fp32 automatically. Every other source is fp32-only already |
| fixed on-chip footprint | all private/local arrays are compile-time sized (`MAXT`/`MAXK`/`MAXD`, `DECOMP_MAXM`/`DECOMP_MAXD`, `PMAX`, `KMAX`, `MMAX`, `LLR_CACHE_MAX`, `WBB_LOCAL_MAX`), so Vitis gets a constant register/BRAM budget; `make capacity_check` keeps the host mirrors equal |
| out-of-capacity behaviour | loud, on both sides: the kernel writes `NaN`/`flag = 4`/`act = -1` sentinels and the host wrapper refuses *before* launching and `NaN`-fills, so the PS cannot return a number the fabric could not have produced |
| barriers | only `power_iter` has them (two per iteration, proven minimal for its decomposition — see its BARRIER BUDGET comment). The decomp family, `threshold_cf`, the datapath and the rate layer are barrier-free, one work-item per channel/matrix/stream |
| whole-library build | `make allprog_check` compiles all ten sources into ONE program on a real device and acquires all 29 kernels, which is what the `.xclbin` link does across `.xo` files |
| fp64 kernel path | **still never executed anywhere.** The Apple M4 reports `CL_DEVICE_DOUBLE_FP_CONFIG = 0`, so `Runtime::init` sets `dev_fp64_ = false` and `allprog_check` / `cf_bench` both compile `-DHPOCL_FP32`. The fp64 branch of `threshold_cf.cl` is syntax-checked only (and Apple's `clang -x cl` rejects `double` outright because it ignores the `cl_khr_fp64` pragma). It needs an fp64 CPU runtime — PoCL on x86 — to be more than a claim |
| PL configurations gated | `op_pl_bench` (`-DWBB_LOCAL_MAX=4096`, the tiled `wbb_apply`) and `datapath_pl_bench` (`-DLLR_CACHE_MAX=4`, the LLR recompute fallback) run the *fabric's* code paths, bit-identically, on the dev host |
| xclbin flow | `opencl/build/build_xclbin.sh` compiles all ten sources with one shared flag set and links them; `make registry_check` proves the kernel list, the host registry and the call sites are the same 29 names |

What is still missing for a real PS/PL split, in dependency order:

1. ~~**The deficit table**~~ — landed; see (3) above.
2. **A device-side `schedule()`** — the front end needs no new kernels: `null_basis`
   and `power_whitening` are `svd_cplx` (= `jacobi_svd`) plus a trailing-column
   selection and a floored `V diag(1/sqrt(S)) V^H` (= two `wbb_apply`), the
   effective/whitened channels are batched GEMMs (= `wbb_apply`), and
   `gmd_factors`/`ucd_design`/`threshold_closed_form` are `gmd_k`/`ucd_k`/
   `threshold_cf`. What does not exist is the composition, its SVD gauge
   canonicalisation in fp32, and a bench that gates device-vs-host per field the
   way `chain_bench` gates the datapath.
3. **The receive combiner cannot be offloaded yet.** `ucd_k` produces `Vd`,
   `rho_eq`, `upper` and `pl` but no left basis, so production's
   `q_ucd = u_eff diag(left_scale)` and `w_ucd = q_ucd diag(1/safe_diagonal)`
   (with the `1e-12` diagonal floor) — and `ref/rx.cpp`'s `q_gmd` — exist only on
   the PS. `tools/decomp_ref.py` ships a `Q` field for the UCD strategy that
   `runtime/decomp_main.cpp` reads and discards, so the gap is currently invisible
   in the report. Either add a `Q` output to `ucd_k` (a second `gmd_dev`-style
   left-basis rotation, mirroring `gmd_factors`' `U`/`urows`/`Q` arguments) or
   stop shipping the field and say the combiner is PS-only.
4. **Batching.** `Runtime::threshold_cf` already takes `C` channels and launches
   `C` work-items, and the decomp kernels take `C` matrices, so the fabric can be
   fed a whole scheduling batch — but the PS path solves one channel at a time,
   so nothing currently builds that batch.
5. **A synthesis report.** There is none in-tree (`docs/vitis_flow_notes.md`
   records what is and is not available off-board). Every PL claim above is about
   *structure* — footprint, barriers, trip counts — not about achieved Fmax,
   LUT/FF/DSP or power.

## Code-rate / MCS layer (`kernels/rate.cl`, `kernels/mcs_table.clh`)

The paper this stack implements (`mix_precode/_archive_20261018/root_files/yang_unified_transceiver.txt`,
the MART unified transceiver) is a **MIMO-BICM** model: information bits are
FEC-encoded at a code rate `R`, interleaved, then grouped into `q`-bit labels of
an `M = 2^q` QAM constellation. Its achievable-rate bound is evaluated for a
*given* modulation and coding scheme — "`a^2` is a constant, as data packets are
typically transmitted under a given modulation and coding scheme" — so the code
rate is an **outer parameter** of the transceiver design, not something the
precoder optimises. This layer is that outer parameter.

Rule (the standard GMI-based AMC test): with `GMI` the mutual information per
*coded* symbol — exactly what `kernels/datapath.cl::gmi_k` emits — a pair
`(m, R)` is supportable iff

```
R * m  <=  GMI        <=>        SE = m * R / 1024  <=  GMI
```

`mcs_select_k` picks the largest such entry of the 3GPP TS 38.214 Table
5.2.2.1-2 CQI table per stream and returns the index, `m`, `R`, the spectral
efficiency, the leftover rate margin, the deliverable information rate and an
outage flag. It is wired into the end of the datapath chain, not left as a
standalone operator: `datapath_bench` runs
`bits -> qam -> tx combine -> channel -> rx combine -> LLR -> GMI -> MCS` and
asserts that the device selection equals the host side of the same
`mcs_table.clh` fed the GMI the kernel just produced, reporting the resulting
sum spectral efficiency in bits/s/Hz. One work-item per stream; the table is `__constant` on the device
and `static const` on the host, from **one** header, so `rate_bench` is a
three-way gate (numpy oracle with the table retyped from the spec, host C++,
fp32 kernel) rather than a self-comparison.

Contracts worth knowing before you use it:

* **Non-finite or out-of-range GMI is an outage, not a rate.** `GMI <= m <= 8`,
  so `+inf` means an upstream fault (`n0 == 0` makes the LLR infinite).
  Selecting the top MCS from it would schedule a rate the link cannot carry, and
  silently selecting the bottom one would hide the fault, so `cqi_select` refuses
  both and the caller sees it in the outage counter.
* **GMI is NOT clamped, per stream.** Production's
  `bicm_metrics.estimate_bicm_gmi_*_from_received` clamps the **aggregate**
  (`max(0.0, num_streams*m - loss)`). `gmi_k` and `ref/modem.cpp` deliberately do
  not clamp, because a negative per-stream GMI is precisely the information a
  rate adapter needs; whoever sums the streams owns the clamp
  (`tools/chain_ref.py` does `max(0.0, gmi.sum())`). Measured so this is not a
  hypothetical: with a matched demodulator the per-stream GMI stays positive down
  to -20 dB (+0.0098 at -20 dB, +0.1014 at -10 dB, +0.2443 at -6 dB, m=8, 20000
  symbols), as `E[GMI] = I >= 0` requires.
* `rate_bench` measures **0.00e+00** on every column of all 10 cases: `SE` is
  `integer/1024`, hence exactly representable in fp32, so the discrete and the
  continuous outputs agree bit for bit across all three implementations. The
  cases include every table boundary, one fp32 ulp either side of it, the `eps`
  slack, `NaN`, `+-inf`, saturation, `K=1` and full outage.

**This is not a channel codec.** There is no LDPC/polar/turbo encoder or decoder
anywhere in the stack, and adding one is out of scope for a control-plane
operator library: what the threshold law, the scheduler and the paper's Problem 1
need is the (m, R) *pair* and the rate it implies, not the coded bits. If you
need real FEC, NVIDIA Sionna has it (`sionna.phy.fec`: 5G LDPC and polar
encoder/decoder pairs, convolutional + BCJR/Viterbi, CRC, interleaver) and its
`ebnodb2no` / `compute_ber` / `compute_bler` utilities; neither Sionna nor
TensorFlow is installed in this tree, so nothing here is cross-checked against
it.

## PL friendliness: measured decisions and deliberate non-changes

The dev host is an Apple M4 GPU-backed OpenCL device; the target is a Zynq
XCZU49DR fabric compiled by Vitis, which is **not available here**. So every
change below was either proven bit-exact (and therefore safe to make blind) or
gated in both configurations. Nothing here is justified by a synthesis report
that does not exist — see `docs/vitis_flow_notes.md`.

| kernel | change | evidence |
|---|---|---|
| `channel_cov` | the `i < j` work-item now writes **both** `R[i][j]` and `R[j][i]` and the `i > j` work-items exit, halving the multiply count and the global traffic (`2*S*N*M^2` -> `S*N*M(M+1)`) with no local memory and no barrier | **bit-exact**: `cmacr` accumulates `re += a.x*b.x + a.y*b.y`, `im += a.y*b.x - a.x*b.y`, and swapping `(i,j)` gives the same products in the same order with `im` exactly negated. `op_bench` rel_err unchanged at 4.54e-07 |
| `wbb_apply` | local-memory tile for `W`/`X`, cutting global traffic by `2*LM*LS/(LM+LS)` (16x at 16x16); direct path kept as a fallback | **bit-exact** (same `n = 0..N-1` order, operands moved from global to local): rel_err 3.05e-08 in both the default and the `-DWBB_LOCAL_MAX=4096` build. But it is **slower on a cached device** — see the occupancy table below, which is why the default is 1 KB and the PL build opts in |
| `power_iter` | the initial vector's `cos`/`sin` are now evaluated by all `M` work-items instead of serially in work-item 0 (~15% of a 24-iteration launch at `M=16`); the loop keeps exactly two barriers | **bit-exact** (same per-element expressions, same `0..M-1` sum order): `op_bench` err unchanged at 1.91e-06. The barrier count is already minimal for this decomposition — see the BARRIER BUDGET note in the file |
| `softdemod_llr_k` | `LLR_CACHE_MAX` bounds the private candidate cache; above it the kernel takes the two-sweep recompute path (`O(1)` memory) instead of allocating `4*MMAX = 1 KB` per work-item | the fallback **is** the folded kernel's `replica_sweep` at radius 0, whose identity with the cached path `datapath_bench` already gates at 1e-5 — measured `0.0e+00`. `make datapath_pl_bench` runs `-DLLR_CACHE_MAX=4`, so every `m >= 6` case takes the fallback and produces **identical** numbers to the cached run |
| `threshold_cf` | unchanged | the fp32 build (`-DHPOCL_FP32`, what the PL uses) has no fp64 private arrays at all; the fp64 default is a dev-host convenience. `psi_du`'s six `real[MAXK]` arrays are ~4.5 KB per work-item **only** in the fp64 build |

`wbb_apply` local-array size vs throughput. **These are Apple M4 GPU numbers and
only Apple M4 GPU numbers**: at the reference shape (`M = N = 16`, `SC = 64`),
complex MAC/s — OpenCL C 1.2 has no
variable-length `__local` array, so the tile is declared at `WBB_LOCAL_MAX`
slots and the runtime allocates that much **per work-group whether or not the
group uses it**, which is an occupancy limit rather than a hint:

| `WBB_LOCAL_MAX` | local memory / group | throughput | time |
|---|---|---|---|
| 4096 | 32 KB | 58 GFLOP/s | 1.7 us |
| 512 | 4 KB | 142 GFLOP/s | 0.7 us |
| **128 (default)** | **1 KB** | **338 GFLOP/s** | **0.3 us** |
| untiled (the old kernel) | 0 | 364 GFLOP/s | 0.3 us |

Do not read the GFLOP/s column as a device-independent property. On the ZU49DR
PL this kernel structure cannot reach anything like 338 GFLOP/s: the whole
device's fp32 FMA ceiling is 4 272 DSP48E2 x 2 x 200 MHz = 1.71 Tflop/s, and
`wbb_apply` is one work-item per `(m, sc)` with a serial `for n` reduction, i.e.
~1 complex MAC per iteration per work-item ≈ 2 DSPs ≈ **~0.8 Gflop/s** — about
400x below the figure in the table. The table's purpose is the *relative*
occupancy effect of a static `__local` array, which is device-independent; the
absolute throughput is not.

On a device with a cache hierarchy the re-reads the tile avoids were already
cheap, so a large static local array costs more than it saves (the table shows
the tiling not even paying for itself on the device it was measured on: untiled
364 > tiled 338 GFLOP/s). The PL is the opposite case — no cache, every re-read
is a DDR4 round trip over the AXI interconnect — which is where the traffic
reduction is supposed to pay, and which cannot be confirmed without Vitis. Hence:
default small, `v++ ... -DWBB_LOCAL_MAX=<BRAM budget>` for the fabric, and
`make op_pl_bench` gates the large-tile configuration so the path the PL
actually runs is not a path nobody tests.

**Deliberately not done**, with the reason:

* *Removing `power_iter`'s barriers by going to one work-item per matrix.* It
  would trade `M`-way intra-matrix parallelism for `nmat`-way inter-matrix
  parallelism, and the only in-tree caller launches `nmat = 1`, so it would be a
  strict loss here. The arithmetic is in the kernel's BARRIER BUDGET comment; if a
  batched caller appears, revisit.
* *Unroll / pipeline pragmas and `-D` compile-time dimensions.* Unverifiable
  without Vitis, and a pragma that cannot be measured is a comment that lies. The
  one compile-time specialisation that *is* measurable (`LLR_CACHE_MAX`,
  `WBB_LOCAL_MAX`) is in, and both settings are gated.
* *The quadrature -> analytic-tail discontinuity* (0.3046 dB at `TQ = 5`). Two
  measured fixes exist — a `1 - 1.20/t^2` tail correction at `TQ ~ 4.5` (jump
  0.0236 dB, worst deficit error 0.068 dB, one extra multiply, zero PL area) or
  `GH_N = 192` with no switch at all (0.034 dB at `t = 9.5`, 4x the quadrature).
  Neither is implemented: the first introduces a numerically-identified
  coefficient that collides with main.tex's "no fitted constants" unless it is
  derived as the next term of the same integration-by-parts expansion, and the
  second costs 4x on the fabric. Decision record with the full tables is in
  "Paper claims vs implementation" below.

## Weight binary (transformer predictor)

`tools/export_transformer.py` maps a PyTorch
`SpectralSetTransformerThresholdRegressor` state_dict (fused MHA in_proj split
into q/k/v) to a flat fp32 blob; header
`magic,token_dim,config_dim,d_model,heads,layers,T,P`; tensor order matches
`runtime/transformer_predictor.cpp`. `make_random_tf_weights.py` produces the
same layout with random weights plus numpy reference outputs for CI.

## Build & check (any OpenCL host)

```
make help            # every target
make check           # all 21 gates in ~37 s (it was ~25 min before the deficit
                     # table; cf_bench alone was ~13 min of that)
```

`make gh_gen` regenerates both Gauss-Hermite headers and then runs `gh_check`;
verified byte-for-byte reproducible (`diff` clean on `include/gh_table.h` and
`kernels/gh_table.clh`), so the generators are live tools and the headers are
not hand-maintained.

Static gates (no device, no build — they run anywhere and take seconds):
`gh_check` (Gauss-Hermite tables agree, host vs device vs `numpy.hermgauss`),
`deficit_check` (the generated ln D table: host vs device header bit-identical,
fp32 block a correctly-rounded narrowing, monotone, Lemma-4 ratio, tail-handover
step, 4th-difference smoothness, and 90 of the 16 000 values recomputed from
scratch), `registry_check` (29 kernels: registered, PL-linked, called, and no
macro or helper name shared by two sources), `capacity_check` (12 shared
constants over 26 definition sites). `deficit_full_check` is the strong form
(regenerate both headers and byte-compare, ~25 s) and `deficit_gen` / `gh_gen`
regenerate; both generators are verified byte-for-byte reproducible.

Device gates: `cf_bench` `op_bench` `op_pl_bench` `tf_bench` `tf_export_check`
`decomp_bench` `frontend_bench` `pipeline_bench` `modem_bench` `datapath_bench`
`datapath_pl_bench` `rate_bench` `rx_bench` `lmmse_bench` `chain_bench`
`allprog_check`.

`asan_check` rebuilds the five host-only benches with
`-fsanitize=address,undefined` and re-runs them: the fp64 references in `ref/`
are the oracles every kernel is judged against, so undefined behaviour there
invalidates the comparison rather than merely the run, and a plain `make check`
cannot see it because an out-of-bounds read usually returns a plausible number.

Two debug hooks, both env-driven so no source edit is needed to test a PL-style
build: `HPOCL_CLOPTS` forwards extra `-D` macros to the device compiler
(`datapath_bench`, `op_bench`, `allprog_check`), and `HPOCL_XCLBIN=/path.xclbin`
makes `Runtime::load_program` load a pre-synthesised binary instead of JIT
compiling. `CF_VERBOSE=1` prints the per-channel threshold trace.

## Zynq bring-up checklist (board side, not runnable remotely)

1. Vendor/Vivado platform for QX_49DR_SV3: PS + PL DDR4 x4 banks + RFDC +
   CMAC 100G as needed; export `.xsa`, build `.xpfm`.
2. `PLATFORM=... opencl/build/build_xclbin.sh hw` (fp32, `NEWTON_ITERS=20` —
   the deployed budget. All three sites default to **20**:
   `kernels/threshold_cf.cl` `#define NEWTON_ITERS`, `ref/threshold_cf.h`
   `CfConfig::newton_iters`, and the `-DNEWTON_ITERS=20` this script pins;
   `tools/check_capacity.py` fails the build if they drift. An earlier revision
   of this note said the first two default to 8 — they did not. main.tex reports
   5 steps only for its CDL-A pool, which is not in this repo.)
3. PS image with XRT; `Runtime::init("acc")` enumerates the PL device;
   host binary is the same source cross-compiled aarch64.
4. HIL loop per paper appendix: laptop emulator streams Ĝ_k / λ̂ + symbols
   over Ethernet; PS runs control plane; PL runs per-symbol datapath;
   cross-check τ against host teacher (shadow mode) and log MAE.
5. Fill the appendix table: latency / throughput / LUT-FF-DSP / power at
   (M,d,K)=(64,4,8) from the Vitis synthesis report + on-board timers.

## Vitis/Zynq 状态

> 中文文档：见 [`docs/vitis_flow_notes.md`](docs/vitis_flow_notes.md)（结论速览 / 证据链 / Windows vs Linux 流程支持矩阵 / 缺失清单 / 下一步建议）。

- **Windows（Vitis 2022.2）可用**：`v++ -c -k ... -t hw`（OpenCL kernel → RTL IP，.xo）。
  实测最小 kernel 约 1 分钟、Fmax ≈ 205 MHz、.xo 107 KB；必须给完整 .xpfm 路径
  （实际平台在 `base_platforms\`，默认 platforms 仓库为空）。这是"同一份 kernel 既能
  CPU/GPU 验证（make check）又能过 Vitis HLS 映射到 Zynq PL"的可移植性证据。
- **Windows 被硬拒**：`-t sw_emu` / `-t hw_emu` 立即报
  `ERROR: [v++ 82-987] Sorry, emulation flows are not yet supported on this operating system.`
  ——emulation 为 Linux 独占，非 license/安装问题。
- **缺口**：Linux Vitis（约 40–60 GB，WSL Ubuntu 22.04 或 Linux 机）+ XRT、
  ZCU49DR 平台 .xsa/.xpfm（现仅有 zcu102/zcu104）、Xilinx 许可（ZCU49DR 是否额外授权
  待确认）、上板硬件/SD 卡镜像流程。详见缺失清单。

## 部署架构与后续射频通路（2026-10 固化决策）

- **当前分工（hpocl 覆盖的负载）**：
  - kernel（29 个 .cl）→ **PL**（FPGA 编织，综合成硬件），与 ARM 核无关；
  - host（runtime/*.cpp + bench）→ **PS 侧 Linux（4× Cortex-A53 的任意核）**，代码不依赖核数
    （无 pthread/OpenMP/亲和性代码，Linux 调度器决定实际核）；
  - **2× Cortex-R5F 当前闲置**——hpocl 无硬实时控制任务，不"为用而用"。
- **后续（16T16R 射频数据通路）才需要 R5F**：
  - 多通道 RF-DAC/RF-ADC 需要 µs 级实时同步/触发/帧定时，A53 Linux 非实时（ms 级抖动）；
  - 届时 R5F（裸机/FreeRTOS + RPMsg）或 PL 状态机上场，同时需要板厂 RFDC + JESD204 参考设计；
  - 已同步至 `docs/QX49DR_SV3_上板依赖清单.md`。
- **信号源模式（约束）**：无外部射频信号源；基带 IQ 波形由**本地电脑软件生成**（与
  tools/ numpy oracle 同源），灌给板上 PS/PL；RFSoC 片内 RF-DAC 负责数模 + 上变频直接
  发射频（DC–6GHz），RF-ADC 下变频回基带；发射-接收可用板上回环（Loopback）自测，
  无需天线/外部仪器。
