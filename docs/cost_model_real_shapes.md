# Cost model at the REAL matrix shapes (Agent F)

Measured 2026-10-25 on the repo at this commit, with the **current** code
(`ref/pipeline.cpp` front_end/design_instant, `ref/threshold_cf.cpp` table path).
Replaces README's former cost table, which was priced at the geometry
(K=8, Nt=16, Nr=4, NRF=32, d=4): that geometry is RF-BD **infeasible**
(rows=(K-1)Nr=28 >= min(Nt,NRF)=16, so F_RF·N_k == 0 identically, `fe_flag=1`,
pipeline.cpp:96-110). None of the 21 configs below is infeasible: rows <
min(Nt,NRF) and nul = NRF-rows >= d for every one (verified by formula, all
runs `fe_flag=0`).

## Method & environment (read before quoting any absolute number)

* Host: fanless Apple M4, **throttles 3-3.7x under load** (load ~3 during these
  runs). All wall times are steady-state medians (300 front_end / 100
  threshold / 200 design / 200 schedule repeats), clock-irrelevant **ratios and
  counts are the defensible quantities**; absolute ms are context.
* Sweep counts: fp64 `svd_cplx` exit sweep count instrumented (a /tmp copy of
  ref/decomp.cpp, one counter per call) — **measured**, not assumed:
  28x32 → 9, 12x16/14x16/8x16/9x16 → 8, 4x8 → 6, 8x8 → 7, 12x18 → 9, 2x2 → 2,
  6x6 → 6. The fp32 kernel burns its 16-sweep cap (fp32 relative off-diagonal
  bottoms out above its tol — README's "measured knees 4/5/5/7" are the fp32
  kernel's, and the fp64 host actually runs *more* sweeps at the real shapes).
* Flop convention: complex MAC = 8 flops. SVD flops = the formulas in
  `runtime/hpocl_schedule.cpp` (`jacobi_batch_flops`, `gram_batch_flops`),
  used at the **measured** sweep count.
* A53 projection model (README's, stated explicitly): 1.4 GHz in-order, 1 fp64
  flop/cycle, x1.5 structural overhead → **0.933 GFLOP/s**; cycles per
  transcendental exp 80 / log 70 / log2 55 / div 29 / sqrt 33. This is a
  PROJECTION, not a measurement.
* PL projection model: fp32 @ 200 MHz, 4-8 lanes, **one measured anchor**:
  the 28x32 C=8 null-basis batch's kernel event = **1.18-1.21 ms** (main
  session, sched_bench) for 40.63 Mflop at cap-16 → 0.0295 ms/Mflop, applied
  linearly to other shapes (estimate). No Vitis synthesis exists; all PL claims
  are structural + this one anchor.
  **⚠️ ANCHOR INVALIDATED 2026-11-06.** That 1.18-1.21 ms is an Apple OpenCL
  `CL_PROFILING` **event** time, and on this device the event column under-reports
  device time by 42-490x (median 63.9x over 130 launches): it implies 3.9 TB/s for a
  134 MB kernel, against this machine's ~120 GB/s. The same launch's pipelined wall
  clock is **51.3 ms**, so the batch executes 20.32 Mflop (8 sweeps) at
  **0.40-0.79 GFLOP/s** -- slower than the 4.19 ms fp64 host path on the same machine.
  Every absolute PL number below that was scaled from `0.0295 ms/Mflop` is therefore
  unsupported; the ratios built from same-kernel event pairs still hold.
  Evidence + re-measured roofs: `docs/hpc_roofline_measured.md` §2-§3; re-baselining the
  cost model on wall clock is that document's §10 item 1.

## 1. The 21 configs (mix_precode/{classical,ai}/config_pools/*.json, distinct)

| config | pool | K | Nt | Nr | NRF | d | T=Kd | nul | null shape (C,rowsxNRF) | fp64 sweeps |
|---|---|---|---|---|---|---|---|---|---|---|
| paper_sweep | paper_sweep | 8 | 128 | 4 | 32 | 4 | 32 | 4 | 8x28x32 | 9 (meas) |
| letter_k8_d3 | letter_k8_d3 | 8 | 128 | 4 | 32 | 3 | 24 | 4 | 8x28x32 | 9 (meas) |
| gp248_a | gen_pool_2_4_8 | 8 | 64 | 4 | 32 | 2 | 16 | 4 | 8x28x32 | 9 (meas) |
| gp248_b | gen_pool_2_4_8 | 4 | 32 | 4 | 16 | 2 | 8 | 4 | 4x12x16 | 8 (meas) |
| gp248_c | gen_pool_2_4_8 | 2 | 16 | 4 | 8 | 4 | 8 | 4 | 2x4x8 | 6 (meas) |
| fair_a | pool_2_4_8_fair | 2 | 32 | 8 | 16 | 8 | 16 | 8 | 2x8x16 | 8 (meas) |
| fair_b | pool_2_4_8_fair | 4 | 32 | 4 | 16 | 4 | 16 | 4 | 4x12x16 | 8 |
| fair_c | pool_2_4_8_fair | 8 | 32 | 2 | 16 | 2 | 16 | 2 | 8x14x16 | 8 (meas) |
| main_a | pool_2_4_8_mainline | 4 | 32 | 3 | 12 | 3 | 12 | 3 | 4x9x12 | 8 (est) |
| main_b | pool_2_4_8_mainline | 8 | 64 | 2 | 16 | 2 | 16 | 2 | 8x14x16 | 8 (meas) |
| v1_a | gen_pool_v1 | 2 | 16 | 4 | 8 | 2 | 4 | 4 | 2x4x8 | 6 |
| v1_b | gen_pool_v1 | 2 | 16 | 4 | 8 | 3 | 6 | 4 | 2x4x8 | 6 |
| v1_c | gen_pool_v1 | 2 | 20 | 4 | 10 | 3 | 6 | 6 | 2x4x10 | 8 (est) |
| v1_d | gen_pool_v1 | 2 | 24 | 6 | 12 | 4 | 8 | 6 | 2x6x12 | 8 (est) |
| v1_e | gen_pool_v1 | 3 | 18 | 2 | 9 | 2 | 6 | 5 | 3x4x9 | 8 (est) |
| v1_f | gen_pool_v1 | 3 | 24 | 4 | 12 | 2 | 6 | 4 | 3x8x12 | 8 (est) |
| v1_g | gen_pool_v1 | 3 | 24 | 4 | 12 | 3 | 9 | 4 | 3x8x12 | 8 (est) |
| v1_h | gen_pool_v1 | 4 | 24 | 2 | 12 | 2 | 8 | 6 | 4x6x12 | 8 (est) |
| v1_i | gen_pool_v1 | 4 | 32 | 3 | 16 | 2 | 8 | 7 | 4x9x16 | 8 (est) |
| v2_a | gen_pool_v2_3x4000 | 3 | 24 | 6 | 18 | 3 | 9 | 6 | 3x12x18 | 9 (meas) |
| paper_k4 | paper_sweep | 4 | 64 | 4 | 16 | 4 | 16 | 4 | 4x12x16 | 8 (meas) |

## 2. Measured host breakdown (steady-state medians, throttled M4, ms)

| config | analog | null | whiten | gbar | front_end | th M64 | design g | schedule |
|---|---|---|---|---|---|---|---|---|
| paper_sweep | 0.031 | 4.19 | 0.008 | 0.008 | 4.45 | 0.022 | 0.106 | 4.67 |
| letter_k8_d3 | 0.031 | 4.38 | 0.009 | 0.008 | 4.68 | 0.018 | 0.079 | 4.77 |
| gp248_a | 0.020 | 4.68 | 0.008 | 0.008 | 4.86 | 0.015 | 0.032 | 4.90 |
| gp248_b | 0.007 | 0.26 | 0.004 | 0.004 | 0.30 | 0.008 | 0.007 | 0.31 |
| gp248_c | 0.003 | 0.014 | 0.002 | 0.002 | 0.025 | 0.006 | 0.003 | 0.034 |
| fair_a | 0.024 | 0.10 | 0.017 | 0.019 | 0.18 | 0.010 | 0.021 | 0.21 |
| fair_b | 0.007 | 0.25 | 0.004 | 0.004 | 0.28 | 0.011 | 0.012 | 0.31 |
| fair_c | 0.003 | 0.56 | 0.001 | 0.001 | 0.58 | 0.014 | 0.011 | 0.61 |
| main_a | 0.004 | 0.11 | 0.002 | 0.002 | 0.13 | 0.010 | 0.007 | 0.14 |
| main_b | 0.004 | 0.54 | 0.001 | 0.001 | 0.59 | 0.015 | 0.016 | 0.63 |
| v1_a | 0.003 | 0.013 | 0.002 | 0.002 | 0.024 | 0.005 | 0.002 | 0.032 |
| v1_b | 0.003 | 0.013 | 0.002 | 0.002 | 0.025 | 0.006 | 0.003 | 0.033 |
| v1_c | 0.003 | 0.025 | 0.007 | 0.008 | 0.052 | 0.006 | 0.004 | 0.061 |
| v1_d | 0.010 | 0.042 | 0.008 | 0.008 | 0.079 | 0.006 | 0.006 | 0.091 |
| v1_e | 0.001 | 0.030 | 0.006 | 0.004 | 0.048 | 0.006 | 0.003 | 0.054 |
| v1_f | 0.005 | 0.072 | 0.003 | 0.003 | 0.093 | 0.006 | 0.004 | 0.100 |
| v1_g | 0.005 | 0.079 | 0.003 | 0.003 | 0.099 | 0.007 | 0.005 | 0.115 |
| v1_h | 0.001 | 0.097 | 0.015 | 0.010 | 0.134 | 0.008 | 0.006 | 0.152 |
| v1_i | 0.004 | 0.24 | 0.025 | 0.022 | 0.32 | 0.008 | 0.010 | 0.34 |
| v2_a | 0.015 | 0.26 | 0.012 | 0.012 | 0.32 | 0.007 | 0.009 | 0.34 |
| paper_k4 | 0.010 | 0.26 | 0.004 | 0.004 | 0.31 | 0.012 | 0.017 | 0.34 |

Takeaway: the null-basis Jacobi batch is **94% of front_end at paper_sweep
(4.19/4.45)**; at the biggest configs the two 4x4 whitening/gbar SVDs and the
analog Gram batch are **< 1.2% each** (at the smallest configs they reach 2-11%
but stay microseconds). They correctly stay on the PS by `min_flops`. The glue
is the part the old model hid: at Nt=128 the host glue is 0.21 ms (4.7% of fe)
and on an A53 it is ~15% of the front end, not 2%.

## 3. Flop & transcendental counts (T=Kd lives in the threshold, not the SVDs)

**Front end, paper_sweep (current code):**

| item | flops | transcendentals | premise |
|---|---|---|---|
| null Jacobi batch (fp64, 9 sweeps) | 22.86 Mflop | 0 | jacobi_batch_flops(8,28,32,9); cap-16 = 40.63 Mflop |
| analog Gram batch (4x128 take4, C=8, 16 sw) | 0.10 Mflop | 0 | gram_batch_flops; ~0 real (per-sweep work ~8k) |
| whiten + gbar SVDs (4x4, C=8 each) | 0.034 Mflop | 0 | 2 x jacobi_batch_flops(8,4,4,5) |
| glue GEMMs (eff, F_RF N, gram(FN), reduced, whiten, Fbb, power_norm) | 3.44 Mflop | 0 | analytic: 8*K*Nr*NRF*Nt + 8*K*Nt*nul*NRF + ... (complex MAC=8) |
| Frf constant-modulus projection | 0 | ~~16 384 (4096 x atan2+cos+sin+div) + 4096 div~~ **STALE -- see the correction below this table**; the shipped code does 4 096 `std::abs`(=hypot) + 4 096 mul | no DFT filler at K*take=NRF=32 |
| gauge pin + nullity + trailing select | ~0 | ~0 | |

> **Correction (2026-11-06).** The `Frf constant-modulus projection` row above prices
> *production's* `exp(i*angle(x))` spelling. The shipped code (`ref/frontend.cpp:149-178`)
> spells it `x * (1/sqrt(Nt)) / |x|` with `1/sqrt(Nt)` hoisted and the per-entry division
> turned into a multiplication: **4 096 `std::abs` (=hypot) + 4 096 mul**, measured on this
> host at **5.86 us vs 51.75 us = 8.83x cheaper**, max |difference| 3.10e-17 (2.2 ulp of an
> entry of modulus 1/sqrt(128)), and `frontend_bench` reads the same before and after. So the
> §4 `glue+proj` A53 projection, which was built on this row, is an **upper bound** for the
> projection half; it has not been re-derived. Evidence: `docs/ps_neon_audit.md` (R4).

Threshold (table path, flag=0, T=32 K=8 M=64 GMD, **measured fill counts
22 psi_u + 20 psi_du = 42 fills**): per fill = (T+K) deficit evals, each 1 log
(index) + ~25 flops (4-pt Lagrange) → per solve ≈ **1680 log + 1930 exp +
~45 kflop**, i.e. the quadrature path's 48.2M exp / 6.1M log are **gone** and
even the tail erfc never fires on these spectra (all t < 32).
Near-flat/saturated (flag=2 via scan): **481 fills (401-point scan)**, ~8x a
flag=0 solve, but then **no Newton** — saturation returns immediately.

## 4. A53 projection (model above; PROJECTION, not measurement)

| config | null ms | glue+proj ms | design ms | **total ms** |
|---|---|---|---|---|
| paper_sweep | 24.5 | 4.6 | ~1.3 | **~30** |
| letter_k8_d3 | 24.5 | 4.4 | ~1.2 | **~30** |
| gp248_a | 24.5 | 2.5 | ~0.5 | **~28** |
| gp248_b | 1.2 | 0.3 | 0.1 | **~1.6** |
| fair_a | 0.5 | 0.5 | 0.2 | **~1.2** |
| fair_c / main_b | 2.6 | 0.5 | 0.2 | **~3.3** |
| v2_a | 1.3 | 0.3 | 0.1 | **~1.7** |
| paper_k4 | 1.2 | 0.7 | 0.2 | **~2.1** |
| (all 21 sum) | 87.2 | — | — | — |

Threshold adds ~0.2 ms (flag=0) / ~2.4 ms (saturated-scan regime) to every
row. The A53 number that matters: **~25-30 ms at the largest config, ~1-3 ms
at the smaller pools.** The old README's "PS, 10 ms: short by 1.5-2.7x" was a
threshold-only statement; the full instant is **2.5-3x over 10 ms** on
paper_sweep-class shapes.

## 5. PL projection (anchor: 28x32 C=8 kernel event 1.2 ms)

| config | null batch (cap-16) | threshold | total-ish |
|---|---|---|---|
| paper_sweep / letter / gp248_a | ~1.20 ms | 10-30 us | **~1.23 ms** |
| gp248_b / fair_b / paper_k4 / v2_a | ~0.065 ms | " | ~0.1 ms |
| fair_c / main_b | ~0.145 ms | " | ~0.17 ms |
| v1_i | ~0.054 ms | " | ~0.08 ms |
| all others | <= 0.03 ms | " | <= 0.05 ms |

Private footprint at the shipped ceilings: jacobi_svd 23,360 B / work-item
(DECOMP 32x40) → **C=8 = 186.9 KB** vs ZU49DR ~7.5 MB on-chip (2.5%, fine).
The threshold kernel is ~106 kflop + ~1900 transcendentals per solve — README's
10-50x headroom holds. The glue (GEMMs, projection, gauge, design = ~5 Mflop at
paper_sweep) stays on the PS in both roles **by design** (hpocl_schedule.cpp).

## 6. Verdict: does one scheduling instant fit the interval? (paper_sweep)

| role | time | 1 ms | 10 ms |
|---|---|---|---|
| PS-only (fp64 host, as `schedule()`) | ~30 ms | **NO, ~30x over** | **NO, ~3x over** |
| hybrid as shipped (null batch + threshold on PL, glue/design on PS) | PS ~6 ms + PL ~1.23 ms | **NO, ~6x over (PS side)** | **YES, ~1.6x margin** |
| hypothetical all-PL (needs GEMM glue kernels; none exist) | ~1.23 ms + glue (not built) | borderline | YES, ~8x |

The old "fits 10 ms with ~1.5x margin" conclusion **survives**, but only for
the hybrid role and for a different reason: the threshold is no longer the
bottleneck (table path); the **null-basis Jacobi batch and the Nt=128 glue**
are. On the PS-only role the answer flips to NO at 10 ms.

## 7. What to optimize next (ranked by (gain x confidence) / effort)

1. **Stop the fp32 null batch burning 16 sweeps.** fp64 host needs 9 at 28x32;
   the kernel runs 16 because its relative off-diagonal bottoms out above
   `tol_exit` (README: Smin/Smax = 5.2e-10 at 6 sweeps). A rank-test-aware
   exit (or a looser fp32 tol) at ~6-9 sweeps cuts the 1.2 ms PL batch to
   ~0.5-0.7 ms and the cap-16 flop count 40.6 → ~20 Mflop. Effort: kernel
   tolerance + full decomp_bench/op_pl_bench/sched_bench re-gate. Confidence:
   high (host measurement of needed sweeps); ~2x on the only PL item that
   matters.
2. **The glue is no longer 0.13 Mflop.** README's "glue = ~0.13 Mflop" was
   measured at Nt=16; at the real Nt=64-128 it is **3.4 Mflop + 16.4k
   transcendentals** (eff channels, F_RF N, Fbb, power-norm are all
   Nt*NRF*T-class) = ~4.6 ms A53, **40% of the hybrid PS budget** and the
   reason "PS 1 ms" fails. Offload as batched `wbb_apply` GEMMs (the kernels
   exist) or NEON-fp32 the loops. Confidence: high (counted); effort: medium.
3. **The saturated-scan regime (flag=2) pays 481 fills for a tau it then throws
   away.** 401-point scan = ~2.4 ms A53 on near-flat channels; a fixed ~40-step
   bisection on ln u (README's item) unifies PS/PL and makes it ~0.2 ms.
   Confidence: high; effort: low-medium; but it only matters when such channels
   are frequent (cf_bench skips them, so the real mix is unmeasured).
4. **NEWTON_ITERS 20 -> 8: keep 20.** Flag-0 solves are 42 fills (0.2 ms A53);
   going to 8 saves ~0.08 ms on a 25-30 ms instant and the near-flat regime
   (which needs the budget) sits exactly where cf_bench cannot see a bias. The
   decision is already right; the numbers now say it is also cheap.

## 8. Broken / stale things found in the host path (fix owner: main session)

* `runtime/hpocl_schedule.cpp` and `ref/decomp.h` say "tol_exit stops at the
  measured knee (7 at 28x32...)" — **the fp64 host runs 9 sweeps at 28x32**
  (and 8 at 12x16/14x16); the knee figures are the fp32 kernel's, the comment
  conflates the two. The flop count `jacobi_batch_flops(C,28,32,16)` overstates
  the PS by 40.6/22.9 = 1.8x vs what the PS actually executes.
* README "fits" table (threshold-only, fake geometry) should be replaced by
  section 6; README "glue ~0.13 Mflop" is 26x low at Nt=128; README "null batch
  = 98% of front end" is 94% measured (4.19/4.45) with the glue at 15% of the
  A53 front end.


---

## Main-session corrections to this document (applied after review)

Two of the numbers above were checked on the device and one recommendation is
already implemented. Recorded here rather than silently edited, so the document
and the code cannot disagree without a trail.

1. **TOP-1 ("fp32 null batch still burns 16 sweeps, 1.2 ms -> 0.5-0.7 ms
   available") is ALREADY DONE, and the further saving does not exist.** The
   `tol_exit` fix landed earlier in the same session: `kernels/decomp.cl` stops
   the sweep loop when `sqrt(off) <= 1e-7 * ||A||_F^2`, which is at the fp32
   Gram floor (measured 1.6e-8 at 28x32). Verified by cap-sweep bit-identity on
   the device, five alternating rounds of cap=8 vs cap=16 at 28x32 C=8:
   `max|dS| = max|dV| = 0.00e+00` and event-time ratio **0.98-1.006**, i.e. the
   kernel already runs 8 sweeps when asked for 16, and the 1.20 ms kernel event
   in the table above IS the post-fix number. `cap=7` is not equivalent
   (`max|dV| = 6.9e-1`), so 8 is the floor: there is no 0.5-0.7 ms to reclaim on
   this batch. (The 2.0x this fix did deliver is against the pre-fix 2.39 ms at
   a full 16 sweeps.) Measured exits, fp32 kernel: 28x32 -> 8, 4x32 -> 6,
   4x4 -> 5, 12x16 -> 7.
2. **The fp64 host knees you measured (28x32 -> 9, 12x16 -> 8, 14x16 -> 8,
   4x8 -> 6, 8x8 -> 7, 12x18 -> 9) are now recorded in
   `runtime/hpocl_schedule.cpp`** next to `jacobi_batch_flops`, together with the
   fp32 kernel's exits, so the "hard cap 16 over-states PS work by ~1.8x" bias is
   documented as a deliberate conservative choice instead of looking like an
   oversight.
3. **TOP-2 and TOP-3 stand.** They are the two remaining real items: the glue
   GEMMs at Nt=128 (3.44 Mflop + 16.4k transcendentals, ~40% of the hybrid PS
   budget, and the reason PS-1ms fails) and the flag=2 401-point scan
   (~2.4 ms A53 to return a value the scan already knew).
3b. **TOP-3 is IMPLEMENTED, and §3/§7.3's fill count for it was wrong.** This
   supersedes the TOP-3 half of item 3 above (item 4 below is the main session's
   TOP-2 correction; the numbering is left alone so neither edit has to touch the
   other). Measured with instrumented counters — one `++` per `psi_u` and per
   `psi_du` call,
   inserted into a *copy* of `ref/threshold_cf.cpp`, never into the repo source;
   reproduce with `python3 tools/cf_scan_ab.py --host` (a fill is one call, the
   same unit §3 uses for "22 psi_u + 20 psi_du = 42 fills"):

   | path | §3/§7.3 claimed | measured before | measured after (`PSI_SCAN_N = 161`) |
   |---|---|---|---|
   | flag 2, near-flat / saturated | **481** fills, ~2.4 ms A53 | **402** (401 grid + the one `psi_u(1.0)` that picks the saturation sign) | **162** (2.48x) |
   | flag 1, scan finds a root (24 cases) | not stated | 223..435, avg **305.8** | 115..200, avg **148.4** (2.06x) |
   | flag 0 control (closed form + 20 Newton) | 42 | **42..43** | **42..43** (1.00x, unchanged) |
   | below-grid short circuit | not stated | **1** | **1** (unchanged) |

   The 481 does not reproduce on any channel: 481 = 401 + 80 is what a
   *three-point central-difference* `psi_du` would cost (4 fills per Newton step),
   but the deployed derivative is the analytic one (1 fill per step), and on a
   flag=2 channel **no Newton step runs at all** — the saturation branch returns
   immediately after its single `psi_u(1.0)`. The worst fallback total measured is
   435 fills, against a 453 ceiling (401 grid + 12 bracket extensions + 20 Newton
   × 2). So §7.3's "~2.4 ms A53" was priced ~1.2x high at 481 fills; the same A53
   model at the measured 402 gives ~2.0 ms, and at the measured **2.46x** host
   speedup on the flag=2 class the shipped code projects to **~0.8 ms**.
   (Projection, as everywhere in §4: the A53 model, not a measurement. The host
   ratios ARE measured — 51 reps × 5 alternating old/new rounds, medians, on a
   host that throttles 3-3.7x under load, which is why only ratios are quoted:
   flag 2 class 0.0847 → 0.0344 ms = 2.46x (a second run of the same command gave
   0.0330 ms = 2.57x), flag 1 class 0.0673 → 0.0328 ms = 2.05x in both runs,
   flag 0 control 0.0116 → 0.0116 ms = 1.00x in both.)

   **§7.3 item 3's "~0.2 ms" target is not reachable, and the reason is worth
   keeping.** It assumed the scan could become "a fixed ~40-step bisection on
   ln u" (~42 fills), which is README's PL-list item 7. That was implemented,
   measured and **rejected**: a fixed-step bisection needs a monotone predicate
   and Ψ's dead-band excursions are not monotone. Ψ → 0 *structurally* as u → 0
   (every deficit → log2 M and T = K·d, so Ψ → ln(T·log2M) − ln(K·d·log2M) = 0),
   which puts the bottom endpoint inside the dead band on exactly the shallow
   spectra the fallback exists for, and an endpoint test against an absolute band
   is not scale-covariant — rescaling ONE channel by 1e-4 flipped 14 of 46
   degenerate cases from flag 1 to flag 2 and at one scale reversed the
   saturation sign, i.e. the decision depended on the gain reference. On a
   7-decade in-band plateau the bisection's u₀ also puts Newton in a different
   basin: up to **30.6 dB** of τ, same grid, only u₀ changed. Full numbers, all
   three counterexamples and their reproduction commands: README's fallback-scan
   decision record. The cheapest resolution that flipped no flag on 46 degenerate
   + 442 pooled channels was 41 grid points (85.9 fills avg, 4.2x cheaper than
   401); 161 ships because it is the grid the fp32 device **already** had, so the
   kernel is bit-identical and only the PS moves. `PSI_SCAN_N` is one `#define`
   and `tools/cf_scan_ab.py --sweep` re-measures the table.

   **§7.3's "cf_bench skips them, so the real mix is unmeasured" is still true of
   the real mix, but the coverage gap itself is closed.** `cf_bench`'s pool is
   38 flag-0 + 2 flag-2 + **0 flag-1** of 40, it skips every channel with
   `flat_ratio < 0.05` or `pyflag != 0` *before* it compares kernel flags, so the
   fallback's root-finding half had no τ comparison and the kernel's fallback path
   had no device-vs-host comparison anywhere in `make check`.
   `tools/cf_scan_ab.py` is that comparison (46 fixed degenerate cases + a
   scale-swept 442-channel pool; `--device` runs the fp32 kernel on the same
   cases). It found one pre-existing PS/PL divergence the gates could not see:
   `zero2_M64_d2_UCD`, where Ψ's only positive excursion peaks at +5.4e-5 — above
   the fp64 dead band of 1e-6, 19x below the fp32 one of 1e-3 — so the host says
   flag 1 / τ = 40.50 dB and the kernel says flag 2 / τ = +inf. Recorded in
   `runtime/cf_main.cpp` at the skip that hides it.

4. **TOP-2 IS NOW DONE, and section 7.2's numbers are the reason it was worth
   doing.** Recorded here with its measurements rather than edited into section 7,
   so the recommendation and its outcome stay separable. See section 9 below.

---

## 9. TOP-2 landed: the glue GEMMs on the fabric, costed per real shape

Section 7.2 said the glue was 3.44 Mflop + 16.4k transcendentals at Nt=128, ~4.6 ms
projected on an A53, ~40% of the hybrid PS budget, and "the reason PS-1ms fails".
Both halves of that are now addressed, and neither needed a new kernel.

**9.1 What was implemented**

* `ref/frontend.h::gemm_wx` is now the ONE complex-GEMM primitive, in
  `kernels/wbb_apply.cl`'s exact form (`Y[sc][m] = sum_n W[m][n] X[sc][n]`, no
  conjugation, `W` = the transposed operand). `effective_channels()`,
  `power_norm_scale()` and the three call sites in `ref/pipeline.cpp` are all
  spelled through it, so there is one formula and two providers.
* `ref/pipeline.h::GemmProvider` is the second role switch (next to
  `SvdProvider`): `front_end()` and `design_instant()` take it, the PS role passes
  `host_gemm()` (fp64 `gemm_wx`), the device role passes `DeviceGemm::provider()`
  (one `wbb_apply` launch per site, fp32 staging, same cost rule shape as the SVD
  one). `schedule_device(..., DeviceGemm*)` with a null pointer is exactly the
  behaviour that shipped before.
* The batch dimension is folded into the operands, so K=8 users cost ONE launch:
  `H_k F_RF` is `W = F_RF^T`, `X = H` read as `[K*Nr x Nt]`; `F_RF N_k` is
  `W = stack_k(N_k^T)` `[K*nul x NRF]`, `X = F_RF`, output `[Nt x K*nul]`;
  `F_RF F_BB` is `W = F_BB^T`, `X = F_RF`.
* `F_RF N_k` is deliberately NOT rearranged as `N_k^H (F_RF^H F_RF) N_k`, which
  would be one `[NRF x NRF]` product instead of a `K*nul`-column one: it squares
  `F_RF`'s condition number before the whitening eigendecomposition reads it, and
  C's structurally-zero eigenvalues already sit at 2.5e-9..3.7e-8 x lam_max
  (section on `inv_sqrt_from_eig`). Same argument as the `svd_wide`/null-basis
  split in section 2.
* The constant-modulus projection is spelled `x * (1/sqrt(Nt)) / |x|` instead of
  production's `exp(i*arg(x))/sqrt(Nt)`.

**9.2 Measured, this host (Apple M4, throttling 3-3.7x under load — ratios only)**

| measurement | value | how |
|---|---|---|
| the three glue GEMMs as fp64 host loops, Nt=128 | 238.6 / 239.5 us (two consecutive runs) | `/tmp/mpx3/stage_probe.cpp`, 200 reps |
| fp64<->fp32 staging for the same three | 5.3 / 5.5 us = **2.2% of the GEMM it replaces** | same probe |
| staging cost expressed as flops/element | **2.5** (the rule assumes 6, i.e. 2.4x conservative) | same probe |
| projection, `exp(i*arg(x))` vs `x/|x|`, 4096 entries | 51.75 us vs 5.86 us = **8.83x cheaper** | `/tmp/mpx3/proj_probe.cpp`, 2000 reps |
| projection, max deviation between the two forms | 3.10e-17 at \|entry\| = 0.0884 (2.2 ulp), 1.39e-16 at \|entry\| = 0.5 (1.25 ulp) | `/tmp/mpx3/proj_check.cpp` |
| `frontend_bench` Frf-vs-production, before and after | 7.2e-16 / 1.2e-15 / 1.3e-14, **unchanged** | `make frontend_bench` |
| glue kernel event time, 3 launches at Nt=128 | **1 us total** | `sched_bench` "glue:" line |
| refactor bit-identity, PS role | every error column of `sched_bench` **byte-identical** to the pre-refactor build; the only differing lines in the two logs are kernel-microsecond columns | `/tmp/mpx3/old/sched_bench` vs `/tmp/mpx3/sched_bench` |

Gates re-run after the change, all green: `frontend_bench`, `pipeline_bench`,
`rx_bench`, `modem_bench`, `lmmse_bench`, `chain_bench`, `sched_bench`
(11 cases x 4 roles + the per-site sweep), `glue_cost_check`.

**9.3 The decision at the real shapes (`make glue_cost_check`, 63 sites)**

`kGemmMinFlops = 1e5`, derived against `kLaunchWorstS = 50 us` (the top of XRT's
documented 10-50 us range) so no decision can be inverted by the one constant in
the model that is a range instead of a measurement. The gate asserts the proxy and
that worst-case model agree on all 63 sites, and prints the 14 sites the mid-range
launch would additionally offload:

| verdict | sites | which |
|---|---|---|
| offload | **16** | all three sites at Nt=128 (1.05 Mflop each), Nt=64 (0.26-0.52), and the 0.115/0.131 Mflop `F_RF N_k` sites at Nt=24-32 |
| keep on the PS | **47** | everything at F <= 0.066 Mflop |
| of those, "conservative by design" | 14 | 0.055-0.066 Mflop sites where the mid-range launch would save **4.0-10.7 us** (10-23% of that site's host cost, ~1% of the instant) |

`--launch <s>` re-derives the whole table; that is the first thing to re-run when
the board's real launch overhead is measured (`docs/QX49DR_SV3_上板依赖清单.md`).

**9.4 Effect on section 6's verdict (A53 projections, same cycle model as before)**

| item | before | after | note |
|---|---|---|---|
| glue GEMMs on the PS at `paper_sweep` | 3.44 Mflop = 2.46 ms | 0.29 Mflop = 0.21 ms + 0.12 ms staging | 3.15 Mflop -> 3 `wbb_apply` launches |
| projection on the PS | ~2.14 ms | **~0.24 ms** | the measured 8.83x, not a model number |
| "glue+proj" total | **4.6 ms** | **~0.6 ms** | |
| PS-only instant | ~30 ms | **~26 ms** | the null-basis Jacobi (24.5 ms) still dominates |
| hybrid PS side | ~6 ms | **~1.9 ms** | design_instant ~1.3 ms is now the largest PS item |
| hybrid PL side | ~1.23 ms | **~1.38 ms** | +3 launches at 30-50 us; device time ~1 us |
| hybrid vs 1 ms / 10 ms | no (~6x over) / yes (~1.6x) | **no (~1.9x over) / yes (~3x)** | |

So section 6's answer survives and improves: "PS 1 ms" goes from 6x over to 1.9x
over, and the item that would close it is now `design_instant` (~1.3 ms: the
per-user `gmd_factors`/`ucd_design`, `stream_snr` and `rx_combiner`), not the
glue.

---

## 10. TOP-3 landed, and section 7.3's number was wrong twice

Section 7.3 priced the near-flat fallback at "**481 fills** (401-point scan), ~8x a
flag=0 solve, ~2.4 ms A53" and proposed "a fixed ~40-step bisection on ln u
(README's item) unifies PS/PL and makes it ~0.2 ms". Both halves were re-measured
with `tools/cf_scan_ab.py` (instrumented fill counters, 46 degenerate / near-flat /
below-grid spectra, medians of 5 alternating rounds x 51 reps):

1. **481 does not reproduce.** 481 = 401 + 80 is what a *central-difference*
   `psi_du` would cost, but the deployed derivative is analytic and a flag-2
   channel runs **no Newton at all**. Measured worst fallback total: **435**
   (ceiling 453 = 401 grid + 12 bracket + 40 Newton), and the flag-2 class is
   **402** (401 grid points + the one `psi_u(1.0)` saturation-sign probe).
2. **The bisection is not merely unverified, it is unsound**, and it was rejected
   rather than shipped. `Psi(u) -> 0` *structurally* as `u -> PSI_SCAN_LO` (the
   deficit difference vanishes with T = K*d streams), so the bottom endpoint of the
   bracket test sits INSIDE the dead band on any shallow spectrum -- measured
   `zeroflat_M16_d2_GMD`: `Psi(1e-6) = -4.27e-07` against a 1e-6 band, with
   `Psi(3.2e-6) = -1.36e-06` one step later. And `Psi` is not monotone at the top:
   `zero2_M64_d2_UCD` peaks at `+5.4e-05` near u = 100 and is **+0.0 exactly** at
   `u = 1e6`, so the top endpoint test fails on a spectrum that does have a root.
   An endpoint bracket therefore reported "no root" on **14 of 46** spectra where
   the scan finds one -- a flag 1 -> flag 2 decision flip, i.e. finite tau ->
   +-inf. A fixed-count bisection needs a monotone predicate and this problem does
   not have one; `tools/cf_scan_ab.py --countexamples` reproduces all three classes
   (it also found a Newton-basin failure: on a 7-decade in-band plateau the
   bisection's u0 landed 30.6 dB from the scan's on 3 of 720 adversarial channels).
3. **What shipped instead: one grid, both sides.** `kernels/psi_scan.clh` now owns
   `PSI_SCAN_LO/HI`, `PSI_SCAN_N = 161` and the first-transition scan itself as a
   macro compiled into the kernel, the fp64 ref and (verbatim) the numpy oracle, so
   the live 161-vs-401 divergence README's PL item 7 complained about is gone.
   Measured: flag-2 fills **402 -> 162 (2.48x)**, flag-1 average **305.8 -> 148.4
   (2.06x)**, flag-0 unchanged at 42-43; host wall **0.0847 -> 0.0344 ms (2.46x)**
   on the flag-2 class, **1.000x** on the flag-0 control; the device arm is
   **bit-identical** (0 flag mismatches, max|dtau| = 0.000e+00 dB over the 46)
   because 161 was already the kernel's resolution and `(real)1e6/(real)1e-6` is
   exactly `(real)1e12`. `cf_bench`'s 12 gate columns are byte-identical before and
   after, and the regenerated `build/cf_vectors.bin` is byte-identical too.
4. **The 0.2 ms target is not reachable this way.** 161 is the resolution the
   device already had, chosen because it is the coarsest setting whose equivalence
   is *provable by construction* rather than sampled: `PSI_SCAN_N = 41` measures
   equally clean on these 46 spectra (0 flag flips, max|dtau| = 2.7e-07 dB through
   the whole policy, 4.2x fewer fills) and 21 is clean too, but **11 flips a
   flag**, and a flag flip is a mode decision. It is one `#define` with a
   `static_assert` whose message is the re-measurement procedure, and
   `tools/cf_scan_ab.py --sweep 401,...,11` reproduces the table; what is missing is
   a proof for spectra outside the 46, so the knob stays where the equivalence is
   structural. A53 projection for the flag-2 class: ~2.0 ms (repriced from the
   measured 402, not the 481) -> **~0.8-1.0 ms**.
