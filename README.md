# HPOCL — Heterogeneous Hybrid-Precoding Operator Stack (AI-RAN flavor)

One OpenCL operator library + one host runtime, three execution targets, for the
SVD/THP hybrid-precoding switching law (closed-form threshold, main.tex) and its
learned surrogate (spectral set transformer), deployed on the QX_49DR_SV3
(Zynq UltraScale+ RFSoC, XCZU49DR-2FFVF1760) and validated today on any OpenCL
device (x86 CPU / Apple GPU / later: Zynq PS CPU and PL accelerator).

## Why this shape (the HPC infra story, honestly)

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

## System architecture — three layers, two cross-cutting

```
L2  schedule    runtime/hpocl_schedule.cpp   cost model -> per-operator PS|PL placement; C-channel batching
L1  operators   kernels/*.cl  (29 kernels)   decomposition | solvers | datapath | NN + rate
L0  backends    runtime/hpocl_runtime.cpp    device policy cpu|gpu|acc, kernel registry, event profiling,
                                             fallback chain, CL_DEVICE_DOUBLE_FP_CONFIG probe -> fp32/fp64 marshalling
---------------------------------------------------------------------------------------------
X1  numerics    tools/*_ref.py (12 oracles)  gates at bit-identical / 1 ulp / dB, + static capacity|registry|glue-cost gates
X2  portability opencl/build/build_xclbin.sh one .cl source -> x86 CPU | Apple GPU | A53 PS | PL .xclbin (no hand-written RTL)
```

**Workload profile: this is not big-FLOPs HPC.** One scheduling instant at the
`paper_sweep` shape (K=8, Nt=128, Nr=4, NRF=32, d=4, null batch 8x28x32) carries
four *different* bottlenecks at once, so it cannot be offloaded wholesale to any
one backend — per-operator placement is the point of the runtime:

| ingredient | arithmetic | host, measured (throttled M4) | bound by |
|---|---|---|---|
| null-basis Jacobi batch (fp64, 9 sweeps) | **22.86 Mflop** (40.63 at the 16-sweep cap) | 4.19 ms = **94% of `front_end`** | compute + serial sweep chain |
| glue GEMMs (eff / F_RF N / gram(FN) / reduced / whiten / Fbb / norm) | 3.44 Mflop | 0.21 ms (4.7%; **~15% on an A53**) | compute, glue-dominated |
| `F_RF` constant-modulus projection | ~0 flop | 5.86 us for 4 096 entries (measured, this host) | **transcendental**: 4 096 `std::abs` (=hypot) + 4 096 mul. The shipped code spells it `x/|x|`, **not** production's `exp(i*angle(x))`, and is **8.83x cheaper** (51.75 -> 5.86 us) at 2.2 ulp of the entry, with `frontend_bench` reading the same before and after (`ref/frontend.cpp:149-178`). The `16 384 (atan2+cos+sin+div)` figure still quoted by `docs/cost_model_real_shapes.md` §3 prices the *production* form and is stale for this tree |
| threshold solve (table path, 42 fills) | ~45 kflop | 0.022 ms at M=64 | **transcendental**: 1 680 log + 1 930 exp |
| *(contrast)* transformer student | 21.22 Mflop — less than the null batch | — | **bandwidth**: 5.05 MB of fp32 weights per forward = 70% of the ZU49DR's whole on-chip memory |

Numbers: `docs/cost_model_real_shapes.md` sections 2-3.

**Four magnitudes that must not be conflated.** Quoting any one of these as
"the performance" is wrong; the honest ledger is:

| quantity | value | status |
|---|---|---|
| PL fp32 ceiling | 4 272 DSP48E2 x 2 flop x 200 MHz = **1.71 Tflop/s** | theoretical |
| ~~PL measured anchor~~ **anchor invalidated 2026-11-06** | was: 28x32 C=8 null batch 1.18-1.21 ms / 40.63 Mflop ~ 34 Gflop/s. That 1.2 ms is an **Apple OpenCL `CL_PROFILING` event time, not device time**: the same launch's pipelined wall clock is **51.3 ms** (42x), so the batch really runs at **0.40-0.79 GFLOP/s** on this device -- *slower* than the 4.19 ms fp64 host path. See `docs/hpc_roofline_measured.md` §2 | **refuted**; the PL cost model needs re-baselining on wall clock (that doc, §10 item 1) |
| M4 GPU measured roofs (what this host can actually do) | compute **547.7 GFLOP/s** (`wbb_apply` M=2048 N=64 SC=512, exact flop count) · memory **88.7 GB/s** (`modulo_fold_k` over 134.2 MB) · ridge **6.18 flop/byte** · `exp` **22.2 Gexp/s** · dispatch floor **14.85 us** pipelined / **247 us** synchronous | **measured**, `docs/hpc_roofline_measured.md` §3 |
| A53 | **0.933 GFLOP/s** | **PROJECTION**, not measured |
| host M4 (fanless, throttles 3-3.7x) | `front_end` 4.45 ms, `schedule` 4.67 ms | measured, but **only ratios and counts are portable across machines** |
| the one real synthesis | `v++ -c -t hw` on 4 kernels: **Fmax ~205 MHz, 107 KB** | measured; contains **none** of the DSP-hungry sources, and no LUT/FF/DSP figures were recorded |

The ~50x gap between the PL anchor and the ceiling is structural and diagnosed,
not mysterious: serial iteration chains, inner trip counts that are runtime
kernel arguments (so no aggressive unroll), eight `#pragma`s in the whole tree
of which **zero** are synthesis hints, zero `fma`/`mad` (deliberate — `cmac`
keeps the replaced loops' op order to stay bit-identical with the host
reference), and AXI/DDR traffic amplification (`channel_cov` reads **1.11 MB
for 65.5 KB of data, 17x**). Knowing *where* each operator sits on the roofline
and placing it accordingly is the infra claim; raw FLOPs is not.

Packaging, the per-block hardware-utilisation map, the latency budget and the
full evidence ledger: **`docs/hpc_infra_positioning.md`**.

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
| `capacity_check` shared constants agree across kernel / host / oracle; every bench geometry AND every config in `mix_precode/*/config_pools` fits the ceilings and is RF-BD feasible | exact | 18 constant groups, 36 definition sites; 14 bench cases + 21 pool configs |
| `allprog_check` all 10 sources build as ONE program, every kernel acquired | 29/29 | 29/29, twice (device-default opts and forced `-DHPOCL_FP32`) |
| `sched_bench` **the device role of one scheduling instant vs the PS role** -- same `front_end()`/`design_instant()` bodies, decompositions and threshold as batched fp32 kernels and the three Nt-class glue GEMMs as `wbb_apply` launches; 11 cases from the real config pools, each run in **four roles** (`costed`, `force`, `glue`, `glue!`) plus a per-site sweep that offloads one GEMM at a time | mode/cf_flag/nul/fe_flag exact; tau 5e-2 dB; sigma 1e-4; `F_RF F_BB` 1e-3; pscale/diagb 1e-5 (SVD-only roles) and 1e-4 (glue roles, `tol_glue`) | **all 11 PASS in all four roles**: mode/flags exact, tau <= 5.7e-04 dB, sigma <= 1.3e-05, `F_RF` **0.0** (both roles run the Gram analog step on the PS), transmitted `F_RF F_BB` <= 5.8e-05, B/R <= 6.0e-05, Q <= 6.9e-05, pscale <= 5.7e-07, diagb <= 1.5e-05 (glue roles; 5.5e-06 SVD-only), constant-modulus 2.8e-17 |
| `glue_cost_check` the glue-offload cost rule re-derived from its own constants | the one-line flops proxy must equal the four-constant time model at the **worst-case** launch, on all 3 sites x 21 pool configs | **63/63 agree**, 16 sites offloaded / 47 on the PS, and the 14 sites the mid-range launch would also offload are printed with the <= 10.7 us at stake instead of being silently dropped |
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
| `fec_bench` the FEC layer: numpy oracle vs fp64 C++ vs exhaustive enumeration, **41 gates**, host-only (no OpenCL, no device) | 0 bit diffs; 1e-12 relative on the Viterbi path metric | **0 bit-level mismatches on all 12 cases**, worst metric 2.7e-16 rel; `d_free` = **10** three independent ways (Dijkstra on the 64-state graph, layered BFS at two branch budgets, exhaustive 2^16 enumeration in the oracle); Viterbi == brute force over all 2^L codewords to 5.4e-16 rel; CRC check values 0x31C3 / 0x29B1 / 0x21CF02 exact; noiseless BLER **0/1440** frames; all 52 000 single-bit errors detected; 0.05 s |
| `asan_check` the 6 host-only benches rebuilt under ASan + UBSan | clean | clean |

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

### Decision record: fallback-scan resolution and root uniqueness (resolution UNIFIED, bisection REJECTED)

**Partly superseded, and the measurements below are corrected where they were
too strong.** This record was written when `kernels/threshold_cf.cl` scanned 160
points while `scan_root` in `ref/threshold_cf.cpp` and `root_u` in
`tools/cf_ref.py` scanned 400, and it declined to unify them because "unifying
the grids would cost 2.5× the fallback scan on the PL for ≤ 0.0004 dB". That
trade was mispriced: the PL already had the cheap grid, and the 2.5× was being
paid by the **PS** — 402 Ψ fills on a near-flat channel (401 grid points plus the
one `psi_u(1.0)` that picks the saturation sign) against the 42 a whole flag-0
solve needs, ~2.0 ms on a 1.4 GHz A53 by `docs/cost_model_real_shapes.md` §4's
own model, for a τ the saturated branch then throws away. (§7.3 says 481 fills
and ~2.4 ms; 481 does not reproduce under an instrumented counter — see its
correction 3b.) It is now **implemented**: all three sites run
`PSI_SCAN_N = 161` from **one definition**,
`kernels/psi_scan.clh`, compiled into the device (Runtime's `.clh` inliner) and
the host (`-Ikernels`), ported verbatim into the oracle, and gated by
`tools/check_capacity.py` (3 sites for `PSI_SCAN_N` — the third is a
`static_assert` in `ref/threshold_cf.cpp` whose message is the re-measurement
procedure — 2 for each endpoint, 3 for the fp64 dead band). 161 is the grid the
device already shipped, so the fp32 kernel is **bit-identical**: `(real)1e6 /
(real)1e-6` is exactly `(real)1e12` in fp32 and in fp64, so the macro's ratio
reproduces the old `pow((real)1e12, (real)1/(real)160)` bit for bit and all 161
grid points match. The second half of PL-list item 7 — "fallback scan → a fixed
40-step bisection on `ln u`" — was **implemented, measured, and rejected**; see
the last paragraph.

The original measurement (18 degenerate γ=0 spectra: 1 and 2 zero streams, and 1
zero + flat rest, for M ∈ {16,64}, d ∈ {2,4}, at both `TQ` values and both dead
bands) still stands as far as it goes — max \|τ(400 pts) − τ(160 pts)\| = 0.0004
dB, typically 0.0000, with raw `u₀` differing by up to 9.3% (2.115e5 vs 2.313e5);
the 0.69 dB disagreement an earlier note blamed on the 160-vs-400 gap does not
reproduce. Re-measured on a 46-case set (`python3 tools/cf_scan_ab.py`, plus
`--sweep`, `--countexamples`, `--bisect-check`, `--bisect-check --pool`; 16 flat,
24 with 1-2 dead streams, 3 below the ln-D grid, 3 well-conditioned controls),
three qualifications:

* **the 0.0004 dB is Newton's residual, not the grid's.** The converged 30-step
  oracle moves **3.783e-7 dB** between 401 and 161 points; the deployed
  **20-step** budget moves **3.789e-4 dB** on the same cases. So the published
  figure is what an under-converged Newton does with a 9.3%-different `u₀`, and
  the grid resolution itself is worth three orders of magnitude less.
* **`PSI_BAND` 1e-6 → 1e-3 does not always leave τ alone.** 0.0000 dB on those 18
  shapes, but `zero2_M64_d2_UCD` (λ = [0, 0, 0.072, 0.110, 0.370, 1.163, 4.434,
  11.159], user 0 entirely dead) has one positive Ψ excursion peaking at
  **+5.4e-5** — above the fp64 band, 19× below the fp32 one. The fp64 host
  returns flag 1 / τ = 40.50 dB, the fp32 kernel flag 2 / τ = +inf: a live PS/PL
  **decision** difference on a rank-deficient spectrum, predating this change.
  `cf_bench` cannot see it (it skips every near-flat channel *before* comparing
  kernel flags — the gap is documented at the skip in `runtime/cf_main.cpp`);
  `tools/cf_scan_ab.py --device` can and does.
* **"exactly 1 sign change" is about zero crossings, and a bisection needs
  something stronger.** `flat_M64_d4_K4_v1_i7` crosses zero once but leaves the
  dead band three times (+ over 0.12 decades, − over 0.25, + again), and Ψ → 0
  *structurally* as u → 0 (every deficit → log2 M, so Ψ → ln(T·log2M) −
  ln(K·d·log2M) = 0 because T = K·d), which puts the bottom endpoint **inside**
  the band on exactly the shallow spectra the fallback exists for.

**Rejected: the fixed 40-step bisection on ln u** (PL-list item 7, and the reason
this record is not a straight "done"). Bisection needs a monotone predicate;
Ψ's dead-band excursions are not monotone, and no fixed-step bisection is
equivalent to a first-transition scan. Measured, all three reproducible from
`tools/cf_scan_ab.py`:

1. *the bottom endpoint cannot anchor a bracket*, and an endpoint test against an
   **absolute** band is not scale-covariant. `zero1_M16_d4_UCD` rescaled — the
   same channel at a different gain reference — gives Ψ(1e-6) = +9.0e-10,
   −1.1e-8, −7.8e-7, −1.4e-4, −1.0e-1 at λ·10^(−4,−2,0,+2,+4): "no bracket", so
   flag 2 and τ = ±inf, at the first three, while the scan finds the same root at
   all five (τ = 59.63, 39.63, 19.63, −0.37, −20.37 dB, exactly the −20 dB per
   100× that `τ = 10log10(4T(M−1)u/6)` with u ~ 1/λ must give). At λ·1 the flip
   even reverses the saturation **sign** (Ψ(1) = +1.9e-1 there against −1.4e-4 at
   λ·10⁻⁴), i.e. "always THP" becomes "always SVD" because the channel was
   expressed in different units. 14 of the 46 cases flip flag 1 → 2.
2. *the top endpoint cannot anchor one either*: `zero2_M64_d2_UCD` has Ψ(1e-6) =
   −1.3e-7 (in band), Ψ(100) = +5.4e-5 (the scan's transition, u₀ = 32.75,
   τ = 40.50 dB), Ψ(320) = +6.7e-12 and Ψ(u ≥ 1000) = **+0.0 exactly**, both
   log-sums collapsing onto the same dead-user terms. Both halves of an endpoint
   test fail at once on a root Newton confirms.
3. *even with a bracket, the bisection's `u₀` is not the scan's*, and Newton then
   lands in a different basin. Same grid on both arms, so `u₀` is the only
   difference: `--bisect-check --pool` moves 1 of 442 channels
   (`flat_M64_d4_K4_v1_i7`, τ = 11.7841 → 15.5407 dB), and a wider 720-channel
   adversarial sample (6 geometries × 2 variants × 60 draws, 6 spectrum kinds,
   0-2 dead streams, scale 1e-6..1e4) moves 3 by up to **30.6 dB**. The mechanism
   is a dead-band plateau: `s_M16_d2_K4_v0_i11` has Ψ ≤ −band only below
   log₁₀u = −1.56, then sits at +1.4e-8 — inside the 1e-6 band — over a
   **7-decade** plateau out to +5.5545. The scan returns u₀ = 0.5·(10^−1.575 +
   10^5.5545) = 1.9e5 → τ = 29.68 dB; the bisection's midpoint lands in the
   plateau, \|Ψ\| < band, its bracket collapses there, u₀ = 98.9 → τ = 4.27 dB.
   Both are ill-conditioned (`dΨ/dlnu ≈ 0` is the regime `cf_bench` skips at
   `flat_ratio < 0.05`), but 30 dB in a deployed number is not a trade to make
   for a start point Newton was already happy with.

**What the shipped unification measured** (instrumented Ψ-fill counters in a
*copy* of `ref/threshold_cf.cpp`; `tools/cf_scan_ab.py --host --old-source
<pristine>` for the C++ A/B, 51 reps × 5 alternating rounds, medians — this host
throttles 3-3.7× under load, so only ratios are quoted, and the two "after" ms
values are two independent runs of the same command):

| | before (401 pts) | after (161 pts) | |
|---|---|---|---|
| fills, flag 2 (saturated / near-flat) | 402 | **162** | 2.48× |
| fills, flag 1 (scan finds a root), avg over 24 | 305.8 (223..435) | **148.4** (115..200) | 2.06× |
| fills, flag 0 control (closed form + 20 Newton) | 42..43 | 42..43 | **1.00×** |
| fills, below-grid short circuit | 1 | 1 | unchanged |
| host ms median, flag 2 class | 0.0847 | 0.0344 / 0.0330 | **2.46× / 2.57×** |
| host ms median, flag 1 class | 0.0673 | 0.0328 / 0.0328 | **2.05× / 2.05×** |
| host ms median, flag 0 control | 0.0116 | 0.0116 | **1.00×** |
| max \|Δτ\|, host fp64, 46 cases | — | 3.789e-4 dB | 0 flag flips |
| max \|Δτ\|, `cf_bench` 40 channels, host **and** kernel | — | **0.000e+00 dB** | 0 flag flips |
| kernel vs pre-change kernel, 46 degenerate cases | — | **0.000e+00 dB**, 0 flag flips | bit-identical |
| `build/cf_vectors.bin`, 401-pt oracle vs 161-pt oracle | — | **byte-identical** | |
| `make lmmse_bench` (7 cases, 2 saturated) | ALL PASS | ALL PASS, output identical | |
| device event time, `cf_bench` 5 configs, 5 alternating rounds | 409 µs total | 408 µs total | ratio 1.002 (per config 1.000-1.011) |

The device time does not move because the device grid did not move; the win is
the PS's 2.5× and the removal of a PS/PL divergence whose failure mode was a
decision, not a rounding. `PSI_SCAN_N` stays one `#define`: `--sweep` measured
41 points (0.30 decades) as equally clean on these 46 spectra at 4.2× fewer
fills, and 21 points as clean too, but "clean on 46" is not a bound for an unseen
spectrum and at 11 points the flag flips (`zero2_M64_d2_UCD`, 40.50 dB → +inf) —
a silent decision change, which is the class of error this repo refuses to ship.

**The `flags |= 8` uniqueness assertion stays rejected**, but the reason changes:
it is no longer "there is nothing for it to catch". There now is — one channel in
442 leaves the dead band three times — and it is still not worth a fourth site
agreeing on a new flag bit for a diagnostic, *provided* nobody replaces the
first-transition scan with a method that assumes uniqueness (a bisection, a
secant step, a Newton-from-the-endpoints start). If such a change is ever
proposed, this assertion becomes the precondition, not a nicety. As before: if
`TQ` moves up or `GH_N` moves down, re-measure.


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
  hpocl_executor.*    bounded host task executor: up to four Linux/A53 workers
                      by default, configurable with HPOCL_HOST_WORKERS
  hpocl_schedule.*    hybrid device role plus schedule_host_batch(), which runs
                      independent PS scheduling instants in parallel without
                      sharing OpenCL queues or mutable kernel handles
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
docs/               hpc_infra_positioning.md, hpc_roofline_measured.md,
                    portability_matrix.md, ps_neon_audit.md,
                    exp_threshold_policy.md, vitis_flow_notes.md
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
| `make check` wall clock (24 gates) | ~25 min | **~60 s** |

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

**Does it fit a scheduling interval?** Whole instant at the REAL shapes
(measured with the current code; the derivation is in
`docs/cost_model_real_shapes.md`). A53 = cycle-model projection at
0.933 GFLOP/s; PL = one measured anchor, the 28x32 C=8 kernel event of 1.20 ms
at `paper_sweep` (K=8, Nt=128, the largest shape in any pool).

> **⚠️ Correction (2026-11-06).** That anchor was a `CL_PROFILING` **event** time on an
> Apple OpenCL 1.2 GPU, and on this device the event column under-reports device time by
> **42-490x (median 63.9x over 130 measured launches)**: it implies 3.9 TB/s for a 134 MB
> kernel and 22.1 Tflop/s for a 1.07 Gflop GEMM, i.e. ~44x this machine's DRAM rate and
> ~4-5x the M4 GPU's whole fp32 peak. The pipelined wall clock implies 88.7 GB/s and
> 517.8 GFLOP/s, which is where physics says they should land. So every **absolute** PL
> rate derived from that anchor (including the `0.0295 ms/Mflop` PL projection model in
> `docs/cost_model_real_shapes.md`) is not defensible; the 28x32 C=8 batch's real device
> time on this host is **51.3 ms**, not 1.20 ms. What survives: every host wall-clock
> number, every correctness gate, and every **ratio of event times for the same kernel at
> the same shape** (the bias cancels) -- e.g. cap-8/cap-16 = 0.98-1.006 and the
> `PSI_SCAN_N` 409/408 us = 1.002. Evidence and the re-measured roofline:
> `docs/hpc_roofline_measured.md` §2-§3.

| role | `paper_sweep` total | 1 ms | 10 ms |
|---|---|---|---|
| PS-only (`schedule()`, fp64 host) | **~26 ms** | **no — ~26x over** | **no — ~2.6x over** |
| hybrid as shipped (null batch + threshold + the three glue GEMMs on the PL) | PS ~1.9 ms + PL ~1.38 ms | **no — ~1.9x over on the PS side** | **yes — ~3x margin** |
| hybrid before the glue offload | PS ~6 ms + PL ~1.23 ms | no — ~6x over on the PS side | yes — ~1.6x margin |
| hypothetical all-PL (needs a projection kernel and the design step) | ~1.4 ms + projection | borderline | yes — ~7x |

These are A53 cycle-model **projections**, not board measurements, and they moved
for two measured reasons: the three Nt-class glue GEMMs (3.15 of the 3.44 Mflop)
are now `wbb_apply` launches, and the constant-modulus projection is spelled
`x/|x|` instead of `exp(i*arg(x))`, which is **8.83x cheaper** on the same
arithmetic (measured on this host, `ref/frontend.cpp`). PS "glue+projection"
therefore goes 4.6 ms -> **~0.6 ms**: 0.21 ms of small GEMMs that stay on the PS
by the cost rule, 0.12 ms of fp64<->fp32 staging the kernel interface forces on
the PS (measured at 2.2% of the GEMM it replaces), and ~0.24 ms of projection.

The old "PS 10 ms has 1.5x margin" verdict survives **only** for the hybrid role
and for a different reason than it was derived with: after the deficit table the
threshold stopped being the bottleneck, and what replaced it — the null-basis
Jacobi batch (94% of the front end) plus, at Nt=128, the glue GEMMs — is now
itself halved, because the glue went to the fabric. What is left on the PS is
dominated by `design_instant` (~1.3 ms) and by the small GEMMs the cost rule
deliberately keeps there. The threshold-only table the 1.5x came from was also
costed at a geometry that no pool contains and that is RF-BD infeasible (item 4a).

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

**4. The front end IS the bottleneck -- but the shape it was costed at is not a
shape this system can run, and at the real shapes the expensive step is a
different one.** Two corrections, both measured.

**(a) The geometry this section used to call "deployed" is RF-BD infeasible.**
It was `K=8, Nt=16, Nr=4, NRF=32, d=4`. Every row of the interference stack
`A = blkdiag(H_u) (1 x F_RF)` is a combination of `F_RF`'s rows, so
`rank(A) <= min(Nt, NRF)` and `null(A) = null(F_RF)` whenever
`(K-1)*Nr >= min(Nt, NRF)` -- here `28 >= 16`. Then `F_RF N_k == 0` *identically*
and the whitened channel is numerical noise. Measured, at that geometry:
`rank(F_RF) = 16`, `rank(A) = 16` (not 28), `||F_RF N||_F = 2.6e-15` against
`||F_RF||_F = 5.66` (rel 4.5e-16), `eig(C) = 1.4e-33 .. 1.7e-30`,
`sigma = 3.6e-9` -- and **production python degenerates the same way** (its own
null space comes back `(32 x 16)` while `nul = NRF - rows = 4`), so this is a
property of the configuration, not of this port. That shape appears in **no**
config pool. `front_end()` now detects it structurally and reports
`PipeOut::fe_flag = 1` on stderr instead of returning a plausible-looking
schedule, and `capacity_check` fails if any config in
`mix_precode/{classical,ai}/config_pools/*.json` is infeasible or over a ceiling
(**all 21 distinct configs are feasible and in capacity**). `sched_bench` keeps
that geometry as case 10 for the single purpose of proving the flag fires.

**(b) At the real shapes the dominant cost was the ANALOG precoder, not the null
basis -- by 35x -- and it was an algorithm/shape mismatch.** The envelope over
every pool config is `K in {2,3,4,8}`, `Nt in {16..128}`, `Nr in {2..8}`,
`NRF in {8..32}`, `d in {2,3,4,8}`, so the four decompositions per instant are:

| step | shape at `paper_sweep` (K=8, Nt=128, Nr=4, NRF=32, d=4) | what it needs |
|---|---|---|
| analog `svd(H_u)` | **4 x 128**, x8 users | the `take = min(NRF,Nr,Nt) = 4` LEADING right singular vectors |
| BD `svd(A)` | 28 x 32, x8 users | ALL 32 columns -- specifically the trailing 4, which are the null space |
| whitening `svd(C)` | 4 x 4, x8 | everything |
| whitened `svd(Gbar)` | 4 x 4, x8 | everything |

`analog_precoder` was calling the one-sided Jacobi on the 4x128 matrix, which
costs `n(n-1)/2 = 8128` rotations per sweep and a private `V` of `128 x 128` --
to produce four columns. Measured on this host for the eight per-user SVDs:

| | before | after (`svd_wide`, Gram form) |
|---|---|---|
| analog step | **162.5 ms** (87% of the 184 ms front end) | **0.031 ms** -- **5256x** |
| null-basis step | 4.6 ms (2.5%) | 4.6 ms (unchanged: it must stay on Jacobi) |
| front end total | 184.4 ms | **~24 ms** (7.6x) |
| leading-subspace agreement | -- | **5.4e-15** (fp64 roundoff) |

and 28x-5691x across the eight real configs, agreement 1.2e-15..6.7e-16.
`svd_wide()` forms `G = A A^H` (m x m), eigendecomposes *that* (4x4), and gets
`V[:,j] = A^H u_j / sigma_j` for the `take` leading columns. This is also what
production does implicitly: `AnalogPrecoder` calls `svd(H_u, full_matrices=False)`,
whose `vh` is `[min(Nr,Nt) x Nt]` -- an ECONOMY V of which only the first
`min(NRF,Nr,Nt)` columns are used. `ref/frontend.cpp` computing the full
`[Nt x Nt]` V was the port's own waste.

**Why the null basis does NOT get the same treatment**, which is the interesting
part: a Gram matrix squares the condition number, so a singular value at
`sigma_min/sigma_max = r` becomes an eigenvalue at `r^2`. The leading triplets
survive that (fp64 keeps `r >= 1e-8`, fp32 `r >= 1e-3`); the *trailing* ones are
destroyed -- and the trailing ones ARE the BD basis. So the split is by accuracy
requirement, not by shape, and it is expressed in one place: the `SvdProvider`
contract carries a `take` argument, `take < n` is served by `svd_wide` and
`take == n` by the Jacobi. That is also why there is no `svd_wide` *kernel*: the
Gram form costs 4.4 kflop at 4x128 against `min_flops = 4e5`, so the cost rule
keeps it on the PS in **both** roles -- and on the PL it is not merely cheaper
there, it is the only form that fits, since the Jacobi at `n = 128` would need a
private `Vv[128 x 128] float2 = 128 KB` per work-item.

Consequence for the offload story: at `paper_sweep` the front end is **4.19 ms
of null-basis Jacobi (94% measured**, the fp64 host running 9 sweeps, not the 7
the old comment claimed**) + 0.03 ms of the other three SVD families + 0.21 ms of
glue** on this throttled M4. On an A53 the glue is not negligible the way it is
here: at Nt=128 it is ~3.4 Mflop + 16k transcendentals, ~15% of the front end —
which is why the three Nt-class GEMMs in it are now `wbb_apply` launches
(`sched_bench` reports 3 launches / 3.15 Mflop offloaded per instant at Nt=128,
total kernel event time 1 us on this device, against 239 us for the same three
GEMMs as fp64 host loops). The other ~0.29 Mflop of glue — the 4x4 whitened
channel, the `F_BB` assembly, the Gram — stays on the PS by the same cost rule.
`sched_bench` measures the 28x32 C=8 kernel event at **1.20 ms** against 4.19 ms
for the fp64 host loop (**3.5x**), and that 1.20 ms is already the post-`tol_exit`
number -- the kernel runs 8 sweeps when asked for 16, verified by cap-sweep
bit-identity (cap=8 output == cap=16 output exactly, event-time ratio
0.98-1.006 over five alternating rounds; cap=7 is *not* equivalent, `max|dV| =
6.9e-1`, so 8 is the floor and there is no further sweep saving on this batch).

**4b. There is now a device role for the whole instant, and it is gated.**
`runtime/hpocl_schedule.cpp::schedule_device()` runs `ref/pipeline.cpp`'s
`front_end()` and `design_instant()` **unmodified** -- they take an
`SvdProvider`, so the PS role and the device role differ only in where the
decompositions ran, and the threshold solve goes through the `threshold_cf`
kernel. `make sched_bench` is the 22nd gate: 11 cases taken from the real config
pools, each run in **four roles** -- `costed` (the cost rule decides which Jacobi
batches are worth a launch, glue on the PS), `force` (every batch launched),
`glue` (cost rule on both families: the three Nt-class GEMMs become `wbb_apply`
launches where they pay) and `glue!` (everything offloaded, so the sites the cost
rule keeps on the PS at Nt <= 32 are gated too) -- plus a **per-site sweep** that
offloads one GEMM at a time, because "the glue is fp32 now" is three decisions
and not one. Every role compares every field of `PipeOut` against the PS role.

Measured worst over all 11 cases x 4 roles (fp32 device role vs fp64 PS role):

| field | worst | gate | why this field and not a tighter one |
|---|---|---|---|
| `mode`, `cf_flag`, `nul`, `fe_flag` | **exact** | exact | a mode flip is a different schedule |
| `tau_dB` | 5.7e-04 | 5e-2 | the fp32 threshold kernel's own documented budget; measured 90x inside it |
| `sigma` | 1.3e-05 | 1e-4 | |
| `Frf` | **0.0** | 1e-6 | both roles run the Gram analog step on the PS, so it is bit-identical, not merely close |
| `F_RF F_BB` (transmitted) | 5.8e-05 | 1e-3 | |
| `P = F_RF N V_D` | 1.9e-04 | 1e-3 | 3-40x more sensitive than `F_RF F_BB`: it omits the whitener whose eigenvalue floor damps exactly the null-space directions where an fp32 and an fp64 BD basis differ most |
| `B`, `R`, `Q`, `rho`, `pscale` | 1.2e-06 .. 2.3e-05 | 1e-3 / 1e-5 | |
| `diagb` | 5.5e-06 (SVD-only roles) **1.5e-05** (glue roles) | 1e-5 / **1e-4** | the one field the glue offload moves: `diagb = pscale*diag(R)`, and `pscale` itself gets *better* (5.4e-09), so the movement is `diag(R)` -- a function of the whitened `sigma` and of `V_D`, i.e. of the null basis, whose INPUT (`H F_RF`, `F_RF N_k`) is now fp32 as well as the Jacobi that factorises it. 1.5e-05 relative on a d x d gain matrix is 0.0015%. `tol_glue` raises **only** `scale`, to 1e-4 (6.6x margin, the same convention as `tol`'s 20x on a 5.0e-07 measurement); every other field keeps the SVD-only budget, so a glue-specific regression in `sigma` or `F_RF F_BB` still fails at the tight number |

Three things this gate found that are worth more than the pass:

* **Raw `F_BB` is not a gateable quantity, and this bench used to try.**
  `Fbb = N C^{-1/2} V_D`, and `C = (F_RF N)^H (F_RF N)` is rank-deficient
  whenever `nul > Nt`, so `C^{-1/2}` carries `1/sqrt(eigenvalue floor) ~ 1e5` in
  the directions `F_RF` annihilates. Those directions are unobservable -- they
  contribute nothing to the transmitted `F_RF F_BB` -- but they are exactly where
  an fp32 and an fp64 null basis differ most. Measured at K=3/Nt=6/NRF=12:
  `max|Fbb|` was 0.78 (PS) against **1661** (device), a 2.1e3 relative
  difference, while `||F_RF F_BB||_F^2` was **6.000000 on both sides**. The gate
  now compares the transmitted product and reports raw `Fbb` ungated.
* **The whitening eigenvalue floor is a precision assumption.** Production floors
  at `max(1e-12, 1e-10 * lam_max)`, which works in fp64 because a structurally
  zero eigenvalue comes out at `1e-17..3e-17 x lam_max`, i.e. *below* the floor.
  The same eigenvalue comes out of an fp32 Jacobi at **2.5e-09..3.7e-08 x
  lam_max** -- *above* the floor -- so it is not floored and `1/sqrt(noise)` is
  applied to an eigenvector that is arbitrary. `SvdProvider` therefore carries
  the precision it returns, and `inv_sqrt_from_eig` takes a `noise_rel` argument:
  eigenvalues at or below `8*eps` of `lam_max` are snapped to zero *before*
  production's floor is applied. For fp64 this is **provably bit-identical**
  (production's 1e-10 is 13 orders above `8*eps64 = 1.8e-15`, so `max()` already
  returned the floor); for fp32 it puts the floor where the noise is.
* **The fp32 front end has a measured dynamic-range limit.** `sched_bench` prints
  a sweep of the per-user large-scale gain spread (not gated, at
  K=8/Nt=64/Nr=4/NRF=32/d=2 from `generalization_pool_2_4_8`):

  | spread | 0 dB | 6 | 12 | 20 | 30 | 60 | 120 |
  |---|---|---|---|---|---|---|---|
  | `sigma` rel | 4.0e-07 | 5.8e-07 | 2.4e-07 | 4.4e-06 | **8.8e-04** | 1.6e-01 | 2.5e-01 |
  | `F_RF F_BB` rel | 2.2e-06 | 2.9e-06 | 2.3e-06 | 5.4e-04 | **1.1e+00** | 1.5e+00 | 1.6e+00 |
  | mode agrees | yes | yes | yes | yes | yes | yes | **no** |

  i.e. **the fp32 BD front end tracks the fp64 one to ~20 dB of per-user spread
  and breaks between 20 and 30 dB**, because the interference stack's Gram spans
  the same spread and the trailing singular vectors -- the ones `null_basis`
  keeps -- are the first thing fp32 loses. This is a deployment constraint, not a
  bug, and it is the number to check a channel model against before putting the
  front end on the fabric.

**What the device role does and does not run, stated plainly.** The three
Nt-class GEMMs — `H_k F_RF`, `F_RF N_k`, `F_RF F_BB` — are **now offloaded**, one
`wbb_apply` launch each, batched over users by stacking the operands (so K=8
users cost one launch, not eight). They are 3.15 of the glue's 3.44 Mflop at
Nt=128, i.e. the whole of the "~40% of the hybrid PS budget" this section used to
describe as the next real item; the interface is `ref/pipeline.h`'s
`GemmProvider`, so `front_end()`/`design_instant()` are still one body and the
only thing that changed is which side runs the GEMM. `wbb_apply`'s contract
(`Y[sc][m] = sum_n W[m][n] X[sc][n]`, no conjugation, `W` transposed) is why no
new kernel was needed — and that matters, because the `__constant` budget is at
98.4% of the 64 KB spec floor in fp32 and already 1.97x over it in fp64, so
"just add a kernel" is not a free move in this library.

Which sites offload is a **cost decision per shape, not a policy**: the rule is
`flops >= kGemmMinFlops = 1e5`, derived from four named constants
(`kHostFlopRate`, `kDevFlopRate`, `kLaunchS`/`kLaunchWorstS`,
`kConvFlopsPerElem`) and re-derived from them at all 63 sites of all 21 pool
configs by `make glue_cost_check`. Result: **16 sites offloaded, 47 kept on the
PS** — every site at Nt >= 64 goes, and the 0.066 Mflop sites at Nt=32 stay,
because their entire saving (<= 10.7 us) is smaller than the launch overhead's own
10-50 us uncertainty. The 14 such sites are printed by the gate rather than
dropped, and `--launch <s>` re-derives the table once the board's XRT number
exists.

What still stays on the PS in both roles, with the reason in each case: the
4x4 whitened channel and the `F_BB` assembly (0.037 Mflop together — launch-bound,
not flop-bound), the constant-modulus projection (transcendental-bound and the
fabric has no transcendental unit; it is now **8.83x cheaper** on the PS instead,
measured, by spelling `x/|x|` rather than `exp(i*arg(x))` — a 2-ulp difference
that leaves `frontend_bench`'s Frf-vs-production column reading the same
7.2e-16/1.2e-15/1.3e-14 it read before), the gauge pin, the nullity selection,
the transposes that put an operand in `wbb` form, and
`gmd_factors`/`ucd_design` (per-user d x d, microseconds). The receive combiner is still PS-only
(`ucd_k` emits no left basis, so `q_ucd`/`w_ucd`/`q_gmd` have no kernel). And on
**this** host the device role's wall clock is dominated by driver launch latency,
not by kernels -- measured by decomposing one launch into its phases:
`clCreateBuffer`+`clReleaseMemObject` 13-32 us, readback ~1 ms, and
`clEnqueueNDRangeKernel`+`clFinish` **50.2 ms for a 28x32 batch whose kernel
event time is 241 us**. That latency also scales with the kernel's *static
private footprint* rather than with the data: the same 4x4 C=8 launch costs
2.07 ms with `DECOMP_MAXM/MAXD = 16x8` (1.5 KB private per work-item) and
**23.3 ms** with `32x40` (22.8 KB), 11x. Quote the kernel-event column, never the
wall column; on XRT the fixed cost is 10-50 us, which is what `min_flops = 4e5`
is derived from.

> **⚠️ Correction (2026-11-06): on this device the instruction above is inverted.**
> The 50.2 ms `clEnqueueNDRangeKernel`+`clFinish` is **not driver launch latency** -- an
> independent pipelined measurement of that one 28x32 C=8 launch gives **51.3 ms**
> (`docs/hpc_roofline_measured.md` §2, §6), i.e. the wall column was measuring the kernel
> all along, and this repo's own `sched_bench` log (wall ~48-58 ms for an instant
> containing exactly one such launch) says the same. A fourth view confirms it: pipelined
> totals scale with *work* (20 x 1.07 Gflop GEMM = 40.8 ms; 10 x 134 MB fold = 14.6 ms),
> which a fixed launch latency cannot do. Consequently the 11x sensitivity to the
> *static private footprint* (1.5 KB -> 2.07 ms, 22.8 KB -> 23.3 ms) is **kernel execution
> cost from register/private spill, not launch cost** -- which makes it a PL-relevant
> finding (BRAM/register pressure), and the reason `DECOMP_MAXM/MAXD` ceilings are per
> kernel. The XRT fixed cost of 10-50 us and the `min_flops = 4e5` derived from it are
> about the *board* and are unaffected. On this host, quote the **wall** column.

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

Status as of this commit: **rows 1, 3, 5, 6 and 10 have landed** (1 = the
deficit table; 3 = the table-path hoists, `-ffp-contract=off`, and the
below-grid short circuit; 5 = zero allocations on either evaluator path,
164 -> 0 `operator new` per call; 6 = the analytic `psi_du` was **implemented and
measured 2.2x SLOWER** -- two `log()` per node against a 12-flop central
difference -- so it is recorded with its numbers and not shipped; 10 = the device
role and `sched_bench`, section 4b). **Row 2 landed differently than written**:
the ceilings are now per-kernel (`DECOMP_ 32x40` for `jacobi_svd`, `GMD_ 16x8`,
`CBD_ 16x8`) because one shared pair would have cost `gmd_k` 12.5x, `ucd_k` 21.7x
and `cbd_k` 14.4x in private footprint; the sweeps were not cut to 6 but made
*adaptive* (`tol_exit`, measured knees 4/5/5/7 at 4x4/4x16/16x8/28x32, worth
2.25x on the front-end batch); and batching is `C=K` per family. **Rows 4, 8 and 9
are unchanged**: 4 (Newton 20 -> 8) is measured safe (<=3e-4 dB over 760
near-flat/saturated points, 0.0000 dB on all 10 in-scope cases) and deliberately
**not taken**, because `cf_bench`'s 5e-2 dB gate cannot see a systematic bias and
2.6x on a 28 us solve is not worth moving the control law's agreement with its
own derivation; 8 is done in the opposite direction to what the row says
(`threshold_cf.cl` is now **fp32 by default** with `-DHPOCL_FP64` to opt in,
which is what the row wanted); 9 needs the board. Row 7 is now **split**: its
fallback-grid half landed (one `PSI_SCAN_N = 161` definition shared by the kernel,
the fp64 ref and the oracle, PS fallback 402 -> 162 fills / 2.46x, device
bit-identical) and its bisection half is **measured and rejected** -- see the row
and the fallback-scan decision record before re-proposing it. The rest of row 7
still needs the board.

| # | item | gain | numerics? | where |
|---|---|---|---|---|
| 1 | ~~land the deficit table~~ | **done** — measured 6 187x at `M=64`, 28 290x at `M=256`; `make cf_bench` 13 min -> 3.2 s, `make check` ~25 min -> 37 s | yes, <=0.096 dB toward the converged reference (that gap is GH-48 model error being removed, not interpolation error, which is 5.4e-6 dB) | DHV |
| 2 | raise `DECOMP_MAXM` 16->32 / `DECOMP_MAXD` 8->40 and batch the 8 `null_basis` SVDs into one `jacobi_svd` launch (`C = 8`); cut sweeps 16 -> 6 | front end **~19 ms -> ~0.6 ms on the PL** | no (same arithmetic) | BOARD for BRAM/Fmax, DHV for the guards |
| 3 | three hoists in the table path: precompute `log lambda_i` and `g_k`, make the grid index affine in `log u`, branch-test on `t^2 = 2*u*lambda` instead of `sqrt` | threshold **1.21 -> 0.35 ms** on A53; removes ~5 900 `log` + 6 400 `sqrt` + 1 300 `div` per channel | yes, at the 1e-16 level (measured <=2.2e-5 dB well-conditioned, <=9.1e-3 dB on one ill-conditioned channel — the same magnitude as `-ffp-contract=off`, which moves 17 of 24 channels by 2.16e-5 dB) | DHV |
| 4 | Newton 20 -> 8 | another **2.5x** (0.35 -> 0.15 ms) | yes — the ladder must be re-measured on the *table* `Psi`, which is smooth (no `TQ` staircase), so 8 should behave like the well-conditioned regime; re-run the near-flat channel that justified 20 | DHV |
| 5 | kill the PS allocator traffic: `psi_u`/`psi_du` do **164 `operator new` / 26 240 B per channel** under clang, which elides `make_axis`'s and the `d0/d1` `push_back` allocations — **GCC/libstdc++ on aarch64 does not**, where the structural count is ~2.07 M malloc/free pairs per channel. Replace with fixed-size stack arrays, as the `.cl` already does | up to **~90 ms/channel** on the quadrature path with g++; ~23 us on the table path | no, if summation order is preserved | DHV but only *measurable* with the real aarch64 g++ |
| 6 | analytic `psi_du`: add `d ln D / d ln t` as an 11th table row instead of the 3-point central difference | **2.5-2.8x on the whole Newton loop** (`psi_du` = 2.82-2.98x `psi_u`), frees 4 of the 6 private arrays | yes — needs a gate against a central difference, which already exists | DHV for accuracy, BOARD for the +6.25 KB |
| 7 | PL restructure of `threshold_cf.cl`: `-DHPOCL_L/-MA/-K/-D` so every trip count is a literal; delete the quadrature from the PL build (`t < DFT_TLO` -> `log2(M)`, worth <=1e-9 dB); constant-latency bracket expansion (6 unconditional `psi_u` + priority mux) instead of the data-dependent `for (e<6 && psi_u(lo)>0)`; ~~fallback scan -> a fixed 40-step bisection on `ln u`, same grid and band on PS and PL~~ **the grid half LANDED, the bisection half is MEASURED AND REJECTED -- do not re-propose it without reading the fallback-scan decision record.** Shipped: one definition of the fallback grid, `kernels/psi_scan.clh` (`PSI_SCAN_LO/HI`, `PSI_SCAN_N = 161`, the loop itself), compiled into the device and the host and ported verbatim into `tools/cf_ref.py`, gated by `capacity_check`, which kills the live 160-point vs 401-point divergence at the resolution the device already had (so the fp32 kernel is bit-identical) and cuts the PS fallback 402 -> 162 fills / 2.46x. Rejected: the bisection. A fixed-step bisection needs a monotone predicate and Psi's dead-band excursions are not monotone -- Psi -> 0 structurally as u -> 0, so the bottom endpoint sits INSIDE the band on exactly the shallow spectra the fallback exists for (an endpoint bracket test is not scale-covariant: rescaling one channel by 1e-4 flipped 14 of 46 degenerate cases from flag 1 to flag 2, and at one scale reversed the saturation sign); the top endpoint can be inside the band too (Psi = +0.0 exactly once both log-sums collapse onto the same dead-user terms); and on a 7-decade in-band plateau the bisection's u0 puts Newton in a different basin, worth up to 30.6 dB of tau. `PSI_BAND` 1e-3/1e-6 stays precision-keyed, and it is NOT always free: `zero2_M64_d2_UCD` is flag 1 / 40.50 dB in fp64 and flag 2 / +inf in fp32; single exit; `flags \|= 8` when the safeguard rejects too many steps | **~10-30 us/channel** at 200 MHz with 4-8 lanes, ~40-80 DSP48E2, constant latency | R1/R3/R6/R8 no; R2 <=1e-9 dB; R7 measured, not projected: 3.789e-4 dB at the deployed 20 Newton steps and 3.783e-7 dB converged, 0 flag flips over 46 degenerate cases, 0.000e+00 dB and 0 flag flips over `cf_bench`'s 40 channels on both sides (`tools/cf_scan_ab.py`) | the grid half is DHV and landed; the rest is BOARD |
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
| fixed on-chip footprint | all private/local arrays are compile-time sized (`MAXT`/`MAXK`/`MAXD`, `DECOMP_MAXM`/`DECOMP_MAXD` = 32x40, `GMD_MAXM`/`GMD_MAXD` = 16x8, `CBD_MAXM`/`CBD_MAXD` = 16x8, `PMAX`, `KMAX`, `MMAX`, `LLR_CACHE_MAX`, `WBB_LOCAL_MAX`), so Vitis gets a constant register/BRAM budget; `make capacity_check` keeps the host mirrors equal **and** checks every bench geometry plus all 21 configs in `mix_precode/*/config_pools` against them. The decomp ceilings are **per kernel** because a static `__private` array is allocated at its ceiling for every work-item whether or not the launch uses it: sharing `jacobi_svd`'s 32x40 would cost `gmd_k` 12.5x, `ucd_k` 21.7x and `cbd_k` 14.4x in private footprint (2 732 / 1 836 / 3 264 B at their own ceilings vs 34 092 / 39 852 / 47 040 B shared). All are `#ifndef`-guarded so a board build can resize one without editing the source |
| out-of-capacity behaviour | loud, on both sides: the kernel writes `NaN`/`flag = 4`/`act = -1` sentinels and the host wrapper refuses *before* launching and `NaN`-fills, so the PS cannot return a number the fabric could not have produced |
| barriers | only `power_iter` has them (two per iteration, proven minimal for its decomposition — see its BARRIER BUDGET comment). The decomp family, `threshold_cf`, the datapath and the rate layer are barrier-free, one work-item per channel/matrix/stream |
| whole-library build | `make allprog_check` compiles all ten sources into ONE program on a real device and acquires all 29 kernels, which is what the `.xclbin` link does across `.xo` files |
| fp64 kernel path | **now executed, on x86 Linux.** Verified on a Tesla P40 (OpenCL 3.0, driver 550.127.08, `CL_DEVICE_DOUBLE_FP_CONFIG = 0x3`, 64 KB constant limit): `cf_bench` with `-DHPOCL_FP64` passes all 7 cases with `tau=0.00e+00`, **`cl=7.61e-16 dB`** against the fp64 gate of `1e-6` (1.3e9x margin), and 39/40 `tau` are **bit-identical between the fp64 and fp32 kernels**. So the fp64 branch is real code, not a syntax-checked claim. It is still not the shipped configuration: the PL build is fp32. See `docs/x86_opencl_notes.md`. The dev host still cannot do it: the Apple M4 reports `CL_DEVICE_DOUBLE_FP_CONFIG = 0`, so `Runtime::init` sets `dev_fp64_ = false` and Apple's `clang -x cl` rejects `double` outright (it ignores the `cl_khr_fp64` pragma). Reproduce with `docs/x86_opencl_notes.md`; the fp64 run needs an fp64 device, and PoCL was not required because the NVIDIA ICD already had one |
| PL configurations gated | `op_pl_bench` (`-DWBB_LOCAL_MAX=4096`, the tiled `wbb_apply`) and `datapath_pl_bench` (`-DLLR_CACHE_MAX=4`, the LLR recompute fallback) run the *fabric's* code paths, bit-identically, on the dev host |
| xclbin flow | `opencl/build/build_xclbin.sh` compiles all ten sources with one shared flag set and links them; `make registry_check` proves the kernel list, the host registry and the call sites are the same 29 names |
| `__constant` budget | **measured, not argued**: `allprog_check` now sums the tables the build actually selected and compares against `CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE`. fp32 (the PL default) = deficit 64 000 B + Gauss-Hermite 384 B + CQI 128 B = **64 512 B = 98.4% of the 64 KB spec minimum**, i.e. 1 024 B of headroom; fp64 = **128 896 B = 1.97x over**, which is not a slow path but a hard failure -- measured on **two** NVIDIA devices that both report exactly 65 536 B: a Tesla P40 fails at `clCreateKernel` with `CL_INVALID_BUFFER_SIZE` (-61), and an L4 fails one level earlier, in the toolchain, with `ptxas error : File uses too much global constant data (0x1f7a0 bytes, 0x10000 max)` (128 928 B, i.e. the 128 896 B of tables plus 32 B of compiler-generated literals). The second form matters for the board: it means the limit is enforced by the *compiler backend*, so no host-side check can talk a vendor toolchain into accepting it, and an `.xclbin` build would fail the same way at `v++` rather than at load time. The gate prints the diagnosis *before* the missing-kernel bail-out, because that failure mode presents as 29 missing kernels. If XRT reports 64 KB the mitigations are one generator call each: `--N 800` (31.25 KB), `--ms 64,256` (25 KB), `--ms 64` (12.5 KB) |
| device-independent numerics | the whole fp32 suite (`decomp/op/datapath/rate/chain/sched`, 156 gate lines) was re-run on x86 Linux + a different vendor ICD + GCC 9.4 with **zero numeric drift** against the Apple M4 runs; the fp32 Jacobi off-diagonal floor reproduced at **2.9e-9** (Apple 1.4e-8..3.4e-8), i.e. `tol_exit = 1e-7` has ~34x margin on a second architecture. Only fp64-precision columns moved (1e-16 -> 1e-7, as expected) |

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

**This table is not a channel codec**, and what the threshold law, the scheduler
and the paper's Problem 1 need from it is the (m, R) *pair* and the rate it
implies, not the coded bits. A codec does now exist next to it -- CRC +
(133,171)8 K=7 convolutional + puncturing + interleaver + soft Viterbi, host-only,
41 gates -- because the paper's model is MIMO-BICM and an (m, R) pair whose BLER
nobody can measure is an untested claim. See **"The FEC layer"** below for what it
proves, what it does not, and the measured waterfall. NVIDIA Sionna's
`sionna.phy.fec` remains the reference implementation to swap to (the names here
are aligned to it deliberately); neither Sionna nor TensorFlow is installed in
this tree, nothing was installed to write the layer, and **no output of it has
ever been compared against Sionna's**.

## The FEC layer (`ref/fec.h`, `ref/fec.cpp`, `tools/fec_ref.py`, `tools/fec_bler.py`)

The paper this stack implements (`yang_unified_transceiver`, the MART unified
transceiver) is a **MIMO-BICM** model: information bits are FEC-encoded at a code
rate `R`, interleaved, then grouped into `q`-bit labels of an `M = 2^q` QAM
constellation. Until this layer existed, the repo carried only the *second* half of
that sentence — `kernels/mcs_table.clh` + `kernels/rate.cl` turn a measured GMI into
the highest supportable `(m, R)` pair — and the README said so plainly:

> **This is not a channel codec.** There is no LDPC/polar/turbo encoder or decoder
> anywhere in the stack … If you need real FEC, NVIDIA Sionna has it
> (`sionna.phy.fec`) … neither Sionna nor TensorFlow is installed in this tree, so
> nothing here is cross-checked against it.

That paragraph is now only half true. **There is a channel codec**: CRC-24A and
CRC-16 (3GPP TS 38.212 §5.1), the `(133,171)₈` `K=7` rate-1/2 convolutional encoder
with zero-tail termination, punctured to 2/3 and 3/4, an S-random / row-column
interleaver, and a soft-decision Viterbi decoder. Names and semantics are aligned to
`sionna.phy.fec` so a later swap is mechanical. The second half of the paragraph is
still exactly true: **Sionna is still not installed, nothing was `pip install`ed to
write this, and nothing here has ever been run against Sionna.** The ground truth
used instead is (a) arithmetic this stack can prove about itself and (b) three
externally attested CRC check values. And the layer is **host-only**: no `.cl`, no
PL kernel, no `__constant` budget, no OpenCL device — `make fec_bench` runs where
the ICD does not.

| file | role |
|---|---|
| `ref/fec.h` / `ref/fec.cpp` | the fp64 host reference: CRC, trellis, encoder, puncturer, interleaver, Viterbi, free-distance search |
| `tools/fec_ref.py` | the independent numpy oracle; writes `build/fec_vectors.bin` |
| `runtime/fec_main.cpp` | `fec_bench`: 41 gates, host-only, no OpenCL |
| `tools/fec_bler.py` | the waterfall measurement (NOT a gate); prints BER/BLER/coding gain with error bars |
| `Makefile` | `fec_bench` target, added to `check` and `.PHONY` |

### Mapping to `sionna.phy.fec`

Verified against the `NVlabs/sionna@main` source tree (`src/sionna/phy/fec/__init__.py`,
`conv/__init__.py`, `conv/utils.py`, `crc.py`, `interleaving.py`, and
`src/sionna/phy/utils/{metrics,misc}.py`). "Aligned" means same role and same
input/output convention — **not** bit-compatible output, which nobody has measured.

| this repo | Sionna |
|---|---|
| `fec::crc_append` / `crc_compute` | `CRCEncoder` (`fec/crc.py`) |
| `fec::crc_check` | `CRCDecoder` (`fec/crc.py`) |
| `fec::conv_encode` | `ConvEncoder` (`fec/conv/encoding.py`) |
| `fec::viterbi_decode` | `ViterbiDecoder` (`fec/conv/decoding.py`) |
| `fec::build_trellis` / `TrellisTab` | `Trellis` (`fec/conv/utils.py`) — the state indexing is copied from `Trellis._generate_transitions`: window MSB = current input, `s' = (u << (K-2)) | (s >> 1)` |
| `fec::kConvK7R12` | `polynomial_selector(rate=0.5, constraint_length=7)` → `('1011011','1111001')` = `(133,171)₈`, which Sionna attributes to Moon, *Error Correction Coding* |
| `fec::make_interleaver(kRowColumn,…)` | `RowColumnInterleaver` |
| `fec::make_interleaver(kSRandom,…)` | `RandomInterleaver`, plus the classic S-random spread constraint Sionna's random interleaver does not enforce |
| `fec::deinterleave_*` | `Deinterleaver` |
| `tools/fec_bler.py`'s Eb/N0 → `n0` | `sionna.phy.utils.ebnodb2no` (role only; this file uses this repo's `n0` = `E|noise|²` per complex sample, `Es = 1`) |
| `tools/fec_bler.py`'s BER / BLER counting | `sionna.phy.utils.compute_ber` / `compute_bler` (a block error is any differing element) |
| — deliberately absent — | `BCJRDecoder`, `LDPC5G*`, `Polar5G*`, `Turbo*`, `Scrambler`, `llr2mi` |

Two conventions worth stating because they are the ones a Sionna swap would have to
match: **zero-tail termination** (`K-1` zeros appended, `n_coded = 2·(A + K − 1)`,
decoder terminates on state 0), and the **path metric** `M = Σ_b c_b · llr_b`
**maximised**. Since `ln P(c_b|y) = c_b·llr_b − softplus(llr_b)` and the softplus
term depends on the received sample only, maximising `Σ c_b·llr_b` over zero-tailed
trellis paths *is* the maximum-likelihood codeword under the BICM factorisation
`p(y|c) = Π_b p(y_b|c_b)`. "Max-log" here refers to that factorisation (bit
independence inside a symbol — what the interleaver buys), not to an extra
approximation inside the decoder. Punctured positions carry `llr == 0.0` exactly,
the neutral element of that metric.

### How this `R` relates to `mcs_table.clh`'s `(m, R)`

`kernels/mcs_table.clh` is the 3GPP TS 38.214 Table 5.2.2.1-2 CQI table: index →
`(m, R/1024)`, and `cqi_select` picks the largest entry with `m·R/1024 ≤ GMI`. That
`R` is a **promise about an encoder that the table does not name**. This layer
supplies three concrete encoders and measures what they actually cost:

| code delivered here | `R_eff = L/n_punc` | `m` | `m·R_eff` (bits/channel use) | nearest CQI entry | its `SE = m·R/1024` |
|---|---|---|---|---|---|
| `(133,171)` rate 1/2 | 0.4068 | 2 | 0.8136 | CQI 5 = (2, 449/1024) | 0.8770 |
| rate 2/3 punctured | 0.5444 | 2 | 1.0889 | CQI 6 = (2, 602/1024) | 1.1758 |
| rate 3/4 punctured | 0.6076 | 2 | 1.2152 | CQI 6 = (2, 602/1024) | 1.1758 |
| rate 1/2 | 0.4068 | 4 | 1.6271 | CQI 7 = (4, 378/1024) | 1.4766 |
| rate 2/3 | 0.5444 | 4 | 2.1778 | CQI 9 = (4, 616/1024) | 2.4062 |
| rate 3/4 | 0.6125 | 4 | 2.4500 | CQI 9 = (4, 616/1024) | 2.4062 |
| rate 1/2 | 0.4083 | 6 | 2.4500 | CQI 10 = (6, 466/1024) | 2.7305 |
| rate 2/3 | 0.5444 | 6 | 3.2667 | CQI 11 = (6, 567/1024) | 3.3223 |
| rate 3/4 | 0.6111 | 6 | 3.6667 | CQI 12 = (6, 666/1024) | 3.9023 |

`R_eff` is not exactly `1/2`, `2/3` or `3/4` because a frame is finite: `L`
information bits + 16 CRC bits + 6 tail bits, punctured by a pattern whose period
need not divide `2·(L+22)`. Read the table as: **the CQI table's `R` is the target,
`R_eff` is what a 96–99-bit frame actually achieves, and the two agree to 1–3
percent** — close enough that a `(m, R)` chosen by `cqi_select` can be served by one
of these three encoders, which is the link the paper's model assumes and the repo
did not have. The table is printed by `tools/fec_bler.py` itself (parsed out of
`kernels/mcs_table.clh`, not retyped — one definition of a standard table is the
point of that header).

What is still missing for a real link, and is *not* claimed here: rate matching to
an arbitrary transport-block size, HARQ, an interleaver matched to the QAM labels
(the current one is bit-level and does not do the intra-symbol label protection of
3GPP's bit-interleaver), and any MIMO/fading/THP coupling. `tools/fec_bler.py`
measures **one stream over flat AWGN**; it does not run through
`kernels/datapath.cl`, does not see the precoder, and does not see the THP receive
modulo.

### The gates (`make fec_bench`, 41 of them, all PASS)

Every number below is the bench's own output, not a model of it. `fec_bench` links
neither `runtime/hpocl_runtime.cpp` nor OpenCL and needs no device.

**Part 1a — the code proves itself.** `d_free` is *computed*, three independent ways;
the value 10 is not typed in anywhere.

```
[PASS] d_free == 10, Dijkstra vs layered BFS at two budgets   dijkstra = 10, bounded(40 branches) = 10, bounded(80) = 10, both attained at 7 branches
[PASS] the branch budget is provably not binding              bounded(6) = -1 (misses it, depth_out -1), bounded(7) = 10 at 7 branches
[PASS] impulse response weight == d_free (explains the 7-branch optimum)  input 1000000 + 6 tail bits -> output weight 10, d_free = 10
[PASS] no zero-weight cycle but the trivial loop (not catastrophic)  32 zero-weight branches of 128 (16 of them driven by u=1), 0 states self-reachable over the non-trivial ones, d_free = 10 > 0
```

The three ways are: Dijkstra over the 64-node state graph from `to_nodes[0][1]` to
state 0 (exact, no depth bound needed — a finite graph with non-negative integer
edges); layered Bellman-Ford with an explicit branch budget, run at 40 and 80
branches, both attaining the optimum at 7; and, in the numpy oracle, **exhaustive
enumeration of all 2¹⁶ zero-tailed codewords** through the generator matrix. The
`bounded(6) = -1` line is the one that matters: it shows the budget is not what
produced the answer. And the optimum being attained at 7 branches is explained, not
asserted — the single-impulse input already produces a weight-10 output, so the
shortest possible non-zero input is a `d_free` codeword. The zero-weight-cycle gate
is the hypothesis Dijkstra rests on, checked exhaustively (transitive closure over
the zero-weight edges, all 64 states) rather than sampled.

**Part 1b — CRC.** Three externally attested check values for the ASCII string
`"123456789"`, reproduced from the LFSR; they are the only numbers in this layer
that come from outside the repo, and they are used as assertions, never as inputs.

```
[PASS] check value CRC-16/XMODEM (init 0 = the 3GPP configuration)  0x31C3 vs published 0x31C3
[PASS] check value CRC-16/IBM-3740 (init 0xFFFF)                    0x29B1 vs published 0x29B1
[PASS] check value CRC-24/OPENPGP (init 0xB704CE, same poly as CRC-24A)  0x21CF02 vs published 0x21CF02
[PASS] catalogue residue 0 for both nonzero-init variants           residue(OPENPGP) = 0x000000, residue(IBM-3740) = 0x0000
[PASS] 0x864CFB / 0x1021 expand to the TS 38.212 term lists         CRC-24A 0x864CFB (14 taps), CRC-16 0x1021 (4 taps), init 0x0 both
[PASS] table-driven == bitwise LFSR on every length 1..300          3000 comparisons (2 polys x 5 inits x 300 lengths, 263 of them not a multiple of 8), 0 mismatches
[PASS] crc(msg || parity) == 0 for random messages of length 1..257  400 residuals checked, 0 nonzero
[PASS] every single-bit error in the codeword is detected           52000 flipped-bit codewords checked, 0 undetected
```

Sources: polynomials from 3GPP TS 38.212 §5.1
(`g_CRC24A(D) = D²⁴+D²³+D¹⁸+D¹⁷+D¹⁴+D¹¹+D¹⁰+D⁷+D⁶+D⁵+D⁴+D³+D+1` → `0x864CFB`;
`g_CRC16(D) = D¹⁶+D¹²+D⁵+1` → `0x1021`), with the LFSR started at zero as that
section specifies. Check values from Greg Cook's *Catalogue of parametrised CRC
algorithms* (reveng.sourceforge.io/crc-catalogue), all three entries class
"attested" — CRC-16/XMODEM against the ITU-T V.41 lineage, CRC-16/IBM-3740 against
AUTOSAR R22-11 `SWS_CRCLibrary` §7.2.2.1, CRC-24/OPENPGP against IETF RFC 4880 §6.
**CRC-16/XMODEM is the pin that matters**: its parameters (`poly 0x1021, init 0x0000,
refin/refout false, xorout 0, residue 0x0000`) are exactly the configuration
TS 38.212 §5.1 defines for `g_CRC16`, so the deployed path — not merely a
neighbouring variant — is checked against a published number. The hex↔term-list
identity is re-expanded and asserted in both C++ and numpy, so a mistyped constant
cannot survive as a self-consistent wrong CRC. The oracle additionally cross-checks
its matrix form against its division form on every column, at lengths 1, 7, 8, 37,
100, 257 (i.e. mostly **not** byte-aligned, which is where the table-driven
implementation's partial-byte branch lives).

**Part 1c — interleavers.**

```
[PASS] deinterleave(interleave(x)) == x exactly     45 (length, kind) pairs, lengths 1..2048; 0 bit-vector and 0 fp64-LLR round trips failed
[PASS] every permutation is a permutation           45 checked, 0 with a repeated or out-of-range index
[PASS] S-random clamp holds densely: n = 4..600 x 2 seeds   1194 interleavers, 0 spread violations, worst restart count 18 at n=600 (budget 200)
```

The `sqrt(n)/3` clamp on the requested spread is load-bearing, and the numbers that
show it are in `ref/fec.h`: the largest feasible `S` is 5 at `n=64`, 6 at `n=100`,
8 at `n=163`, 10 at `n=244`, 13 at `n=512`, 15 at `n=1000`. A single-pass greedy
walk asked for `S=8` at `n=100` produced **41 violating pairs** (minimum spread 1,
i.e. no interleaving at all) before the restart ladder existed — that is the bug the
clamp and the ladder fix, and it is why the achieved spread is *measured and printed*
rather than assumed.

**Part 1d — loopback, erasures, and the strongest gate.**

```
[PASS] noiseless loopback: decoded == input on every frame   1440 frames (3 CRCs x 3 rates x L in {1,7,96,233} x 40), BLER = 0/1440, bit errors = 0, LLR magnitude 8
[PASS] punctured loopback at rate 2/3 and 3/4: BLER 0        100 frames, 0 block errors
[PASS] depunctured erasures are exactly 0.0 and kept LLRs are untouched   25000 depunctured positions checked, 0 wrong (0.0 compared with ==, kept with |x| == 8)
[PASS] Viterbi metric == exhaustive enumeration over all 2^L codewords    216 trials (110592 codewords enumerated, L in {8,10}, all 3 rates), worst |dv| = 1.066e-14 absolute, 5.433e-16 relative, tol 1e-12 relative
[PASS] Viterbi decoded bits == the enumerated argmax bits    216 trials, 0 disagreements
[PASS] all-zero LLR: metric 0 and the tie-break picks the all-zero path   metric = 0.0, 0 of 8 decoded bits set (degenerate: all 2^8 paths tie at 0)
```

The enumeration runs through `encode_frame`, so a bug in the trellis tables, the
survivor memory or the traceback cannot hide behind a matching bug in the encoder.
The all-zero-LLR case is the tie-break gate: every one of the 2⁸ paths scores
exactly 0.0, so the answer is decided by the tie rule alone, and both sides must
land on the same codeword. Erasures are compared with `== 0.0`, not `< eps`.

**Part 2 — numpy oracle vs C++, bit-exact.** 12 cases, 6 of them also
brute-force cross-checked, covering `L ∈ {1, 8, 10, 96, 200, 512}`, all three CRC
settings, all three rates, both interleavers, LLRs synthesised directly *and* LLRs
out of a real QPSK/16-QAM/64-QAM logsumexp demodulator, one case deep below the
decoding threshold and one fully degenerate.

```
[PASS] numpy vs C++ bit-exact on every case   12 cases, 0 bit-level mismatches in total
[----] vector-file totals   perm 0, coded 0, tx 0, dec 0, crc 0, geom 0 mismatches; worst metric 2.728e-16 rel, worst brute-force metric 0.000e+00 rel
```

The permutation is **regenerated in C++ from `(kind, n, S, seed)` and compared entry
by entry**, not read back from the file — the splitmix64 stream, the rejection
ladder and the restart count all have to agree, or the comparison would be vacuous.
The bits are compared exactly (not "BER ≈ 0"); the only tolerance in the whole file
is on the fp64 path metric, `1e-12` relative, against a measured worst of
**2.728e-16** (one ulp at `metric ≈ 2.5e5`, from summing in a different association
order). Two bugs in this layer were found by these gates and not by reading: the
row-column interleaver's numpy readout was row-major (the identity permutation —
250 of 252 entries wrong in case 8), and the bench's own pre-flight size asked for
one `npunc` too many, so the last case failed to be read at all.

### Measured waterfall (`tools/fec_bler.py`, 19.7 s)

12 curves (3 modulation orders × {uncoded, 1/2, 2/3, 3/4}), **277 360 frames,
26 875 248 information bits, 12 901 728 symbols, 27 773 block errors**, seed
20261026 (each curve derives its stream from `seed + 1000·(rate index + 1)`, and each
*point* from `(seed, Eb/N0)` so a single row is reproducible without re-running the
sweep that found it). Frame payload `L = 96…99` information bits + CRC-16, chosen so
`n_punc` is a whole number of `m`-bit labels — no padding bit is transmitted and
`R_eff = L/n_punc` is exact. Coded and uncoded frames carry the **same information
payload**, so the BLERs are comparable at equal information content, not merely at
equal symbol count.

`Eb/N0` at `BLER = 1e-2`, and the gain of the coded curve over the uncoded one at
that point (uncoded minus coded, so positive = coding helps):

| `m` | uncoded | rate 1/2 | rate 2/3 | rate 3/4 | gain 1/2 | gain 2/3 | gain 3/4 |
|---|---|---|---|---|---|---|---|
| 2 (QPSK) | 8.480 ± 0.356 | 3.682 ± 0.171 | 4.019 ± 0.157 | 4.491 ± 0.155 | **+4.798 ± 0.395** | +4.461 ± 0.390 | +3.988 ± 0.388 |
| 4 (16-QAM) | 12.197 ± 0.197 | 6.410 ± 0.190 | 7.115 ± 0.108 | 7.491 ± 0.179 | **+5.787 ± 0.274** | +5.082 ± 0.225 | +4.706 ± 0.266 |
| 6 (64-QAM) | 16.371 ± 0.161 | 9.322 ± 0.130 | 10.730 ± 0.479 | 11.341 ± 0.189 | **+7.048 ± 0.207** | +5.641 ± 0.505 | +5.029 ± 0.248 |

All in dB, all at `BLER = 1e-2`, obtained by log-linear interpolation between the two
measured points that bracket the target after a coarse-to-fine search (2 dB probe →
bisection to a ≤ 1 dB bracket → a 5-point final grid spanning bracket ± 1 dB). The
`±` is the endpoints' binomial CI propagated through that interpolation; it does
**not** include the curvature error of assuming `log₁₀(BLER)` linear in `Eb/N0`,
which is why the bracket is narrowed first. The `m=6`, rate-2/3 row has the widest
interval (± 0.479 dB) because its two bracketing points landed close together in
BLER (1.17e-2 at 10.50 dB, 8.30e-3 at 11.00 dB), so the local slope estimate — and
hence the lever arm of the CI — is the noisiest of the twelve.

**Statistical hygiene, printed with every point.** Budget ≤ 6000 frames/point
(probe points ≤ 1200), early stop once 15 block errors are in, batch 512. That makes
the resolution floor `1/6000 = 1.7e-4` and the relative 95 % CI at the early-stop
count `1.96/√15 = 51 %`, i.e. about ± 0.22 dB on a waterfall of slope 1 decade/dB —
quoted in the header of every run so no row can be read as more precise than it is.
Points where zero block errors were seen print `95% < 3/N` (rule of three) and
`FLOOR 1/N`, never `BLER = 0`. Every row also prints frames, information bits,
symbols, block errors *and* bit errors, so the reader can see which counts are too
small to mean anything.

**An independent check of the whole normalisation chain.** Every uncoded row also
prints the *exact* per-bit error rate of Gray-labelled square QAM over flat AWGN,
computed from the per-axis L-PAM decision regions (`pam_gray_ber_exact`, no textbook
approximation anywhere in it). Over the 18 uncoded rows whose bit-error count gives
a relative CI below 15 %, measured/analytic lies in **0.913 … 1.021**, worst
deviation 8.7 % and inside its own error bar. That is what rules out an Es/N0-vs-Eb/N0
factor of two, a wrong `n0/2` split between real dimensions, or a mislabelled
constellation — all of which would leave a plausible-looking waterfall and a
completely wrong gain number. (The same check caught a real bug during development:
the first version hard-decoded `llr < 0` as bit 1, i.e. inverted, and reported
`BER = 1.0` against an analytic `1.8e-7`.)

**Undetected errors.** The CRC verdict is recomputed from the decoded bits and
compared with the decoded parity bits, and any frame that is wrong but still passes
would be printed as `N block errors the CRC MISSED`. Over 277 360 frames and 27 773
block errors: **zero**. A 16-bit CRC on a ~100-bit frame gives an undetected-error
fraction of order 2⁻¹⁶ per erroneous frame, so 27 773 erroneous frames predicts
~0.4 — this run is consistent with that, not evidence of a better CRC.

### What this layer does **not** prove

* **No Sionna cross-check.** Sionna / TensorFlow / PyTorch are not installed here and
  nothing was installed to write this. The mapping table above was read off
  `NVlabs/sionna@main`'s source, and the generator polynomials come from its
  `polynomial_selector`, but no output of this layer has ever been compared against
  Sionna's. A swap still has to be validated.
* **No device, no PL.** There is no `.cl` kernel, no `__constant`, no `Runtime`
  link. The precoder chain (`kernels/datapath.cl`) still ends at GMI → MCS and never
  sees a coded bit; nothing in `mix_precode/` or the scheduling path calls this.
* **Flat AWGN, one stream.** No MIMO, no fading, no THP receive modulo, no
  precoder coupling, no channel estimation error. The waterfall says what the code
  does to a BICM link, not what it does to *this* stack's link.
* **Hard-decision Viterbi only.** No BCJR / log-MAP, so no soft output and no
  iterative receiver; no `llr2mi`-style EXIT analysis.
* **One code family.** No LDPC, no polar, no turbo, no 5G rate matching, no HARQ, no
  interleaver matched to the QAM labels.
* **Puncture patterns are not from a spec.** They are written out explicitly in
  `ref/fec.cpp`; the only claim made about them is the rate they achieve, which
  `punct_rate()` derives by counting and the bench asserts equals 2/3 and 3/4. Their
  *performance* is measured in the waterfall and is not comparable to a specific
  standard's pattern without redoing that measurement.
* **The waterfall is a measurement, not a gate.** `make check` runs `fec_bench`, not
  `fec_bler.py`. A BER is a statistic; a gate has to be a statement that can be
  false. Nothing in `check` depends on the numbers in the table above.

### Running it

```
make fec_bench                 # regenerate the oracle (0.4 s) + 41 gates (0.05 s)
python3 tools/fec_bler.py      # full 12-curve waterfall, 19.7 s on this M4
python3 tools/fec_bler.py --quick --m 4 --rates 2/3
```

`fec_bench` needs no device and no `DEV=`; it also passes clean under
`-fsanitize=address,undefined` (0.30 s, same 41 gates), which is worth knowing
because the vector reader in `runtime/fec_main.cpp` does raw pointer arithmetic over
a byte buffer with sizes read from the file.

## Two datapath fixes that were invisible until the gates got wider

Both were found by widening coverage rather than by reading, which is the
argument for the coverage: `datapath_bench` went 4 -> 5 cases and `modem_bench`
9 -> 12 (M=2 and M=256 added), and `tools/modem_common.py` now checks itself
against production instead of against a restatement of itself.

* **`gmi_k`'s running sum was losing bits systematically.** `acc += softplus(t)`
  over `nsym*m` non-negative terms, with `acc` growing to `nsym*m*ln2` = 3.6e5 at
  `nsym = 65536, m = 8`, whose fp32 ulp (0.031) is 4% of one term -- and the loss
  is one-sided because `acc` only grows. Measured against the same LLRs summed in
  fp64, in bits of GMI:

  | nsym | 64 | 1024 | 8192 | 20000 | 65536 |
  |---|---|---|---|---|---|
  | plain `acc +=` | 2.3e-6 | 2.8e-5 | 7.2e-4 | **3.2e-3** | 2.3e-2 |
  | Kahan | 2.5e-7 | 2.4e-7 | 3.1e-7 | **1.0e-7** | 3.6e-7 |

  The plain form's error is 300x the `1e-5` selection slack in
  `kernels/mcs_table.clh` at `nsym = 20000`, so `mcs_select_k` could pick an MCS
  the link cannot carry; at -20 dB it reported `+0.0337` where fp64 says
  `+0.0107`, i.e. a genuinely outage-level GMI came out positive. Kahan rather
  than Neumaier because every summand is non-negative, so `acc` is monotone and
  `|acc| >= |y|` always holds -- the case Neumaier's extra branch exists for
  cannot arise. Cost is 4 flops and one register against the `exp` + `log1p`
  already in the loop. Gated: `datapath_bench`, `datapath_pl_bench`,
  `chain_bench` (all three strategies) ALL PASS.
* **Production spells the THP fold period two ways and they are 1 ulp apart.**
  `thp_precoding.qam_axis_order_and_spacing` (`L*(2/sqrt(2(L^2-1)/3))`, used by
  `thp_receive_equalized` -- the receive fold -- and by `ref/modem.cpp`) versus
  `multiuser_simulation_environment.qam_modulo_period` (`L*sqrt(6/(M-1))`, what
  `tools/modem_common.py` used). Algebraically equal; different at `m = 2` and
  `m = 8`, identical at `m = 4, 6`. `chain_bench` is the one gate that computes
  the period independently on both sides, so this was live exposure the moment a
  case used `m = 2` or `8` -- and `chain_ref.CASES` are M=16 and M=64, where the
  two agree exactly, which is why nothing caught it. The oracle now uses the
  receive-path spelling, so the two sides are bit-identical and
  `check_production`'s period assert can be an exact equality. Measured effect:
  `zf`/`llr_fold` move by <= 9.9e-16 (the fold is continuous in `period` unless
  `floor()` flips, and the closest any probe gets to `+-period/2` is 1e-5, about
  2.2e10 ulps) against gates of 1e-4 / 3e-3.

Still open in this slice, with the measurements in `docs/modem_datapath_audit.md`:
the GMI's own noise floor is `3.1e-3` bits at 128 symbols and `2.5e-4` at 16384
(statistics, not implementation -- the oracle agrees to `2e-16`), which is worth
**1.8 dB of threshold error** at 128 symbols and only converges to `2.5e-4`
(~0.15 dB) at 16384. Any GMI-driven threshold calibration needs >= 1e4 symbols
per block. And `rate_ref.py`'s 15-entry CQI table covers `m = 4` only, so an
`m = 8` chain runs at a code rate whose BLER target it never checked -- adding
256QAM entries needs 3GPP TS 38.214 Table 5.2.2.1-4 and must not be invented.

## Two portability defects that only a second vendor's device could show

Both were found by running the same tree, unchanged, in a configuration this repo
had never been run in: **Linux x86-64, g++ 10.5, NVIDIA L4**, with

```bash
HPOCL_CLOPTS=-DHPOCL_FP32 make check DEV=gpu     # 24 gates, the PL configuration
```

That command is worth keeping in CI precisely because the configuration it forces
is the one the board builds, and because an fp64-capable device is the only place
the two defects below are reachable.

* **The host wrapper chose its buffer widths from the DEVICE, not from the BUILD.**
  `Runtime::threshold_cf` decided whether to marshal `lam`/`tau` as `float` or
  `double` from `dev_fp64_` (`CL_DEVICE_DOUBLE_FP_CONFIG`), while the kernel's
  `real` is decided by the compile options (`threshold_cf.cl` is fp32 by default and
  takes `-DHPOCL_FP64` to opt in). On an fp64-capable device the two can disagree,
  and forcing the PL configuration is exactly how they do. Measured on the L4 with
  `-DHPOCL_FP32`: the kernel wrote 4-byte floats into 8-byte slots, so **every tau
  came back 0.000000 dB** while the `int` flags -- the one buffer whose width does
  not depend on precision -- still looked plausible, and `sched_bench` reported a
  1.8e+01 dB tau error on a channel whose host solve was 17.9730 dB. A minimal
  probe pinned it to the wrapper and not to the program layout: the same kernel
  built from one source, from two and from three concatenated sources all returned
  0.0 on the L4 and 38.265274 dB (host 38.265333) on the Apple M4. The fix is
  `Runtime::program_fp64()`, recorded from the options `build()` was actually given;
  `load_xclbin()` sets it false, because an `.xclbin` for the PL is fp32 by
  construction (`build_xclbin.sh` passes `-DHPOCL_FP32`, the DSP48E2 has no floating
  point, and the kernel defaults to fp32) and a binary's precision is not queryable.
  `device_fp64()` stays as the different question it always was -- "can this device
  compile fp64 at all" -- which is what `cf_main`/`sched_main`/`allprog_main` use to
  decide what to ASK for. Asking for fp64 on a device that cannot do it now says so,
  instead of surfacing as Apple's `newComputePipelineState failed`.
* **`cf_bench` counted a kernel/host flag disagreement without failing on it**, so
  the defect above printed as `[cf_bench] ALL PASS` with `cl=0.00e+00`. That column
  is a maximum over the channels that were *compared*, and every channel had been
  skipped by the flag mismatch -- `fbm=8` of 8 on the L4 against `fbm=0` on the
  Apple device. A maximum over an empty set reads as a perfect score, which is the
  worst possible failure mode for a gate: `nfbm` now fails the bench. The one known
  legitimate PS/PL flag difference (`zero2_M64_d2_UCD`, where the precision-keyed
  `PSI_BAND` puts a +5.4e-5 excursion above the fp64 band and below the fp32 one)
  stays where it belongs -- `tools/cf_scan_ab.py --device`, which reports it.
* **A third finding, and it is a property of the problem, not of the code.** At
  K=2 Nt=20 Nr=4 NRF=10 d=3 (`generalization_pool_v1`) the interference stack is
  4x10, so its six trailing singular values are exactly zero and the BD null basis
  is defined only up to a 6x6 unitary. Two correct fp32 implementations may return
  different bases, and `P = F_RF N V_D` is the one compared quantity the whitener
  `C^{-1/2}` does not damp. Measured, same inputs, same algorithm, two devices:
  `P` = **1.1e-06 on the Apple M4 GPU** against **1.39e-02 on the L4** (|P| ~ 0.18),
  while on both devices `sigma` agrees to <= 9.2e-07, the TRANSMITTED `F_RF F_BB` to
  <= 2.6e-06 and `tau` to <= 2.7e-05 dB -- i.e. the schedule and the transmitted
  precoder are device-independent, the un-whitened intermediate is not. It appears
  only in the `force`/`glue!` coverage roles: at 0.31 Mflop the cost rule keeps all
  four batches of this shape on the PS, where both roles are bit-identical (every
  column 0.0e+00). `sched_main.cpp`'s `tol_force.phys = 2e-2` is that measurement
  plus 1.4x, and the costed roles keep `tol.phys = 1e-3`.
* The `__constant` budget got its second confirmation, in a stronger form: the L4
  fails in the toolchain, `ptxas error : File uses too much global constant data
  (0x1f7a0 bytes, 0x10000 max)`, i.e. the 64 KB limit is enforced by the compiler
  backend and no host-side check can negotiate with it.

**Status of the whole suite on the second architecture, final tree**: `24 gates,
261 PASS/OK, 0 FAIL, 0 compiler warnings` under
`HPOCL_CLOPTS=-DHPOCL_FP32 make check DEV=gpu` on Linux x86-64 / g++ 10.5 / NVIDIA
L4, against `24 gates, 264 PASS/OK, exit 0` for `make check DEV=gpu` on macOS /
clang / Apple M4. The two counts differ by three because the Apple device reports no
`CL_DEVICE_DOUBLE_FP_CONFIG`, so three fp64-only comparisons print on one host and
not the other. What the pair buys: the fp32 PL configuration is now gated on a
vendor toolchain that enforces the 64 KB `__constant` limit for real, on a compiler
whose floating-point contraction and whose `NULL` local-size choice both differ from
clang/Metal's -- which is how the two defects above and the `P` device-dependence
below were found at all.

Compiler status on the second toolchain: g++ 10.5 `-O2 -std=c++17 -Wall -Wextra`
over all 26 host translation units produced **1 warning and 0 errors** on the first
pass, the warning being the deprecated `clCreateCommandQueue`, and **0 warnings** on
the full `make check` after the fix below. The queue is now created through
`clCreateCommandQueueWithProperties` where OpenCL 2.0 headers exist and the 1.2 call
where they do not, with `CL_QUEUE_PROFILING_ENABLE` on both paths -- `last_us()` is
the kernel-event column every timing in this README quotes, so profiling is not
optional. clang on macOS: 0 warnings, unchanged.

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
| `tx_combine_k` | the digital stage `bb[r][sym] = sum_t Fbb[r][t] s[t][sym]` does not depend on the antenna, but one work-item owned one `(antenna, symbol)`, so it was recomputed **Nt times**: `Nt*(NRF*T + NRF)` complex MACs per symbol instead of `NRF*T + Nt*NRF` -- **26.4x redundant arithmetic** at the largest pool shape (Nt=128, NRF=32, T=32: 135 168 vs 5 120 per symbol), on the per-symbol kernel, i.e. paid `nsym` times. A work-group now computes `NRF*LS` of those values once into `__local` (**one** barrier per work-item, not one per iteration) and every antenna in the group reads them from there; `TXC_LOCAL_MAX` bounds the tile (default 8 KB) with the untiled kernel kept verbatim as the fallback, and `Runtime::tx_combine` now passes an explicit **64x4** group with the grid rounded up (the guards make the padding inert) instead of NULL | **bit-identical**: `bb` still accumulates `t` ascending and the RF stage `r` ascending over the same values, so only the operands' address space changed. Measured: `datapath_bench`'s `tx` column reads 2.1e-07 / 2.7e-07 / 1.8e-07 / 9.2e-08 / 1.4e-07 in **all six** combinations of {cpu, gpu} x {default, `-DTXC_LOCAL_MAX=4`, `=4096`}, and a raw-CL probe at Nt=128/nsym=8192 gives the same `|x|` sample (73.988710) with 0.0e+00 drift across 7 group shapes. Kernel EVENT time, warm medians of 11 alternating rounds, Apple M4 GPU: untiled 316-326 us, tiled at NULL (the device picks 32x1) 106-111 us, tiled at 8x32/16x16/32x8/128x1 64.7/43.2/32.6/33.2 us, tiled at **64x4 19.6-24.5 us** -- so the tiling is worth 2.9x at the shape a NULL launch gets and 13-16x at 64x4, and the group shape alone 4.3-5.7x. Cold start understates this by 2.7x (the first configuration in a fresh process reads 66.7 us for 64x4), which is why the number is a warm alternating median |
| `rx_combine_k` / `channel_apply_k` | **unchanged** | neither has `tx_combine_k`'s redundancy: each output element is its own dot product (`Nr` MACs for `rx_combine_k`, `Nt` for `channel_apply_k`), so the only thing a tile would save is *traffic* on the shared `y`/`x` column, not arithmetic. That is the `wbb_apply` case, where the tile measured **slower** on this cached device (338 vs 364 GFLOP/s) and shipped only because the PL has no cache. Doing it here without a board is a guess with a barrier attached; the shapes to measure are in the same probe |
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

> **⚠️ Correction (2026-11-06).** The GFLOP/s column here is **event-derived and ~40x too
> high as an absolute rate** (`wbb_apply` 512x512x512 = 1.0737 Gflop: event 48.5 us ->
> 22.1 Tflop/s, impossible; pipelined wall 2073.6 us -> **517.8 GFLOP/s**). The *relative*
> occupancy ordering in the table is a ratio of event times **of the same kernel at the same
> shape**, so the bias cancels and it survives -- and it was re-measured with wall clock,
> same sign (`docs/hpc_roofline_measured.md` §9). Read this table as "32 KB of static
> `__local` costs occupancy", never as "this kernel does 338 GFLOP/s".

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
make check           # all 24 gates in ~60 s (it was ~25 min before the deficit
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
