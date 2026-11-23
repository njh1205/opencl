# PS-side (host C++) vectorisation audit — XCZU49DR Cortex-A53 / NEON

Scope: the host C++ control plane in `ref/` (compiled by `Makefile:51`,
`CXXF := -O2 -std=c++17 -Wall -Wextra $(THREAD_FLAGS) $(INC)`). No source file,
Makefile, README entry, `kernels/` file or `build/` artefact was modified while
producing this document; every experiment lives in `/tmp/psneon/` and is listed
in §8.

Target platform: Zynq UltraScale+ XCZU49DR PS = 4x Cortex-A53 (ARMv8.0-A,
128-bit NEON: 4 lanes fp32 / **2 lanes fp64**) + 2x Cortex-R5F. There are **no
DSP slices in the PS** — SIMD on the PS means NEON only. Everything below is
*measured on the Apple M4 dev host*; A53 numbers are projections from the
repo's own A53 cycle model (`docs/cost_model_real_shapes.md` §"Method &
environment": 1.4 GHz in-order, 1 fp64 flop/cycle, x1.5 structural →
0.933 GFLOP/s, transcendental cycles exp 80 / log 70 / log2 55 / div 29 /
sqrt 33) and are marked **[PROJECTION]**. Nothing here needs or claims a board
measurement; §7 says which recommendations do.

---

## 0. Answer first

| question | answer |
|---|---|
| Does `-O2` → `-O3` break the bit-identical gates? | **No.** 0 of 102 291 dumped fp64/fp32 scalars and 0 of 159 output fields differ, across the null-basis SVD batch, the three glue GEMMs, the constant-modulus projection, ten threshold spectra (flags 0/1/2, M = 16/64/256, GMD + UCD, LMMSE), the full paper_sweep scheduling instant and the modem bit-exact fields. Integer fields (`mode`, `cf_flag`, `fe_flag`, `nul`) also identical. |
| Does `-O3` help? | **Barely, and not where the time is.** null-basis Jacobi batch 1.00x, `front_end` 0.98–1.02x, glue GEMM 1.00x, `analog_precoder` 1.00x. Only `threshold_closed_form` moves: **1.18x** (59.3 → 50.2 µs). |
| `-O3 -mcpu=native` (= `-target-cpu apple-m4`)? | Same story: bit-identical, 0.99–1.02x on the Jacobi, 1.19x on the threshold. On the A53 the equivalent is `-mcpu=cortex-a53`, which is worth having for the scheduler model, not for SIMD. |
| `-O3 -ffast-math`? | **1.18–1.20x on `front_end`, 1.74x on the glue GEMM — and it breaks gates.** 84 083 of 102 291 scalars change; τ moves up to **3.97e-6 dB** (cf_bench's `u*0` gate is 1e-9 dB); `modem_bench`'s "constellation/modulo bit-exact" row breaks (up to 4096 ulp); and it makes `INFINITY` at `ref/threshold_cf.cpp:1033` undefined behaviour (clang emits `-Wnan-infinity-disabled`), which is the flag-2 saturation semantics of the mode policy. |
| Is PS vectorisation worth doing at all? | **Yes, but only by hand and only bit-exactly.** A hand-written fp64 NEON `gemm_wx` that vectorises across the *independent* output index is **1.63–1.65x and bit-identical** (0/1024 elements differ). A bit-identical column-major NEON null-basis Jacobi is **1.45–1.49x** on M4. Both capture ~95% of what `-ffast-math` would give, at zero numerical cost. |
| Biggest lever overall? | **Not SIMD.** README's optimisation row 2 (batch the eight `null_basis` SVDs into one `jacobi_svd` PL launch) is quoted at front end ~19 ms → ~0.6 ms on the PL, i.e. ~25–40x. Bit-exact PS NEON on the same work is ~1.5x. Do the PL offload first; do PS NEON only for the residual glue that `glue_cost_check` keeps on the PS (47 of 63 sites today). |

---

## 1. Hotspot map: file:line, and why the vectoriser refuses each loop

Baseline cost, `docs/cost_model_real_shapes.md` §2 (throttled M4, steady-state
medians): `paper_sweep` `front_end` = **4.45 ms**, of which the **null-basis
Jacobi batch = 4.19 ms (94%)**; glue GEMMs at Nt=128 = **0.21 ms (4.7%,
~15% on an A53)**; `schedule` total = 4.67 ms. My own re-measurement of the
live tree (§2, `/tmp/psneon/timing_probe.cpp`) gives `front_end` = 4.686 ms and
the null batch = 4.700 ms at `-O2`, i.e. the doc's split still holds.

Every "vectoriser verdict" below is clang's own `-Rpass-analysis=loop-vectorize`
/`-Rpass-missed=loop-vectorize` output for `-O3` (raw logs:
`/tmp/psneon/remarks/*_vec_O3.txt`, `*_vec_O3ffm.txt`). Two libc++ lines are
responsible for almost all of it:

* `.../c++/v1/complex:785` — `operator*(complex<double>, complex<double>)`
  lowers through `__builtin_complex` into the C99 Annex G form, which carries an
  `isnan` recovery branch. Verdict, verbatim:
  **"loop not vectorized: Control flow cannot be substituted for a select"** and
  **"call instruction cannot be vectorized"**.
* `.../c++/v1/complex:942` — `abs(complex<double>)` is `std::hypot(re, im)`, a
  libm call. Verdict: **"call instruction cannot be vectorized"**.

### 1.1 Null-basis Jacobi — 94% of `front_end`

Call chain: `ref/pipeline.cpp:147` (`svd.run(..., K, rows, NRF, NRF, ...)`
with `take == NRF`, so the provider must use the accuracy-preserving Jacobi —
`ref/pipeline.cpp:16-35`) → `ref/decomp.cpp:33 svd_cplx` → the sweep loop
`ref/decomp.cpp:56-91`.

| loop | file:line | what it does | verdict at `-O3` | verdict at `-O3 -ffast-math` |
|---|---|---|---|---|
| Gram inner | `ref/decomp.cpp:62-66` | `alpha += norm(W[r*n+p])`, `beta += norm(W[r*n+q])`, `gamma += conj(W[r*n+p]) * W[r*n+q]` — three reductions over `r`, m = 28 | not vectorized: *instruction cannot be vectorized* (complex:785 + complex:942) | **vectorized 2x2** |
| rotate W | `ref/decomp.cpp:79-83` | `W[r*n+p] = c*ap - jm*aq; W[r*n+q] = jp*ap + c*aq` — elementwise, no reduction | not vectorized (complex:785) | **vectorized 2x1** |
| rotate Vv | `ref/decomp.cpp:84-88` | same on the n x n basis | not vectorized (complex:785) | **vectorized 2x1** |
| `gfro2` | `ref/decomp.cpp:53-54` | Frobenius norm, once per call | *cost-model: vectorization is not beneficial* | vectorized 2x4 |
| column norms → S | `ref/decomp.cpp:95` | `nrm += norm(W[r*n+j])` | *cost-model: not beneficial* | vectorized 2x4 |
| `off` accumulation | `ref/decomp.cpp:67` | sum of 496 `norm(gamma)` across the whole sweep, gates the `break` at :90 | (inside the pair loop) | — |

Why it does not vectorise today — five independent reasons, each with code
evidence:

1. **libc++ `complex<double>` arithmetic is a branch, not a straight-line
   expression.** `ref/decomp.cpp:65` and `:81-82`/`:86-87` all go through
   `operator*`, whose Annex G NaN recovery clang reports as *"Control flow cannot
   be substituted for a select"*. This is the single dominant blocker: it kills
   `:62`, `:79` and `:84` at once, and it is not a property of the algorithm.
2. **`std::abs`/`std::norm` are libm calls.** `ref/decomp.cpp:68` (`mu =
   std::abs(gamma)` → `hypot`) and the `norm` inside `:63-65`/`:67`/`:54`/`:95`.
   A call in the body ⇒ *"call instruction cannot be vectorized"*.
3. **Strided array-of-structs access, and ARMv8.0-A has no gather.**
   `W[(size_t)r * n + p]` walks the column with a stride of `n = 32` complex =
   256 B. A 2-lane fp64 `vld1q_f64` needs two *consecutive* doubles; the column
   gives two doubles 256 B apart. Any NEON version must either pay two
   `vsetq_lane` scalar loads per vector (which is what ate most of my
   `NEON_EXACT` gain — see §2) or change the layout (which is what recovered it:
   `COLMAJOR_NEON`, 1.46–1.49x).
4. **Reduction order is a gate, not a preference.** `alpha`/`beta`/`gamma`
   accumulate ascending in `r`; `off` (`:67`) accumulates across all 496 pairs
   and decides the sweep-level `break` at `:90`. Reassociating `off` changes the
   *number of sweeps executed*, i.e. the amount of work, not just the rounding —
   measured: my `NEON_PAIR` variant (2-way interleaved accumulators) reaches the
   same 9 sweeps here but changes the trailing near-null singular values by up to
   2.6–4.4x relative (§2).
5. **Every trip count is a runtime `int`.** `m`, `n`, `sweeps` are parameters;
   `nul`, `rows = (K-1)*Nr`, `NRF`, `Nt` come from `PipeCfg`. The vectoriser must
   emit a runtime trip-count check plus a scalar epilogue for a 28-iteration
   loop, which is exactly why the cost model says *"not beneficial"* for `:54`
   and `:95`. The device side fixes this with `-DHPOCL_L/-MA/-K/-D` (README
   optimisation row 7); the host has no equivalent, and adding one would mean
   per-shape specialisation of the oracle.

Not the blocker, contrary to the usual guess: **pointer aliasing.** Nothing in
`ref/` carries `restrict` (verified: `grep -rn '__restrict\|restrict\b' ref/` is
empty), but clang does not cite aliasing for any of these loops — `:62` has no
stores at all, and for `:79`/`:84` the reported reason is the complex multiply.

Batch-level note: the C=8 null-basis SVDs are eight *independent* matrices
(`ref/pipeline.cpp:31-34` loops `c`), so the batch dimension is a legal
bit-exact vectorisation axis — but `ref/decomp.cpp:69` (`continue` on the
rotation-skip test) and `:90` (`break` on the norm-invariant exit) fire per
matrix, so lockstep execution needs a per-lane active mask. That is the one
route to 4-lane fp32-style parallelism without touching any reduction order, and
it is also the most invasive.

### 1.2 Glue GEMM — 4.7% of `front_end` on M4, ~15% on an A53

One definition, `ref/frontend.h:44-54`; inner reduction at
**`ref/frontend.h:51`** (`for (int n = 0; n < N; ++n) acc += w[n] * x[n];`).
Call sites: `ref/pipeline.cpp:97` (`kGemmEff`, H·F_RF), `:188` (`kGemmFn`,
F_RF·N_k), `:327` (`kGemmPower`, F_RF·F_BB), plus `ref/frontend.cpp:19`
(`effective_channels`, which the header comment says is the same call) and the
host GEMM provider `ref/pipeline.cpp:47-53`.

* Verdict at `-O3`: `ref/frontend.h:51` — **not vectorized: instruction cannot be
  vectorized** (complex:785 again). At `-O3 -ffast-math`: **vectorized 2x2**.
* The header pins the order deliberately (`ref/frontend.h:36-38`: *"The summation
  order is n ascending, which is what every loop it replaces ran in, so routing
  the glue through here is BIT-IDENTICAL on the host"*). Reassociating `n` is
  therefore a documented invariant, not an accident.
* Unlike the Jacobi, this loop is **contiguous in `n`** for both `w` and `x`, so
  the AoS striding problem of §1.1 does not apply — but there are two *other*
  independent axes: `sc` (`:46`) and `m` (`:47`), both producing independent
  outputs. Vectorising across `m` keeps each `n`-reduction serial and is bit-exact
  by construction. Measured: **1.63–1.65x, 0/1024 output elements differ** (§3).
* Related reduction loops with the same structure and the same verdict at `-O3`
  (all vectorise under `-ffast-math`): `ref/pipeline.cpp:216` (reduced
  H_k,eff·N_k), `:224` (Gbar = reduced·C^-1/2), `:302-312` (Fbb assembly,
  `:307` inner), `ref/frontend.cpp:51` (`gram_hermitian`), `:90`
  (`inv_sqrt_from_eig`), `:116`, `ref/decomp.cpp:473`/`:481` (`cbd_factors`'
  R recomputation), `:501`/`:509`/`:510` (`stream_snr` MGS).

### 1.3 F_RF constant-modulus projection

**`ref/frontend.cpp:174-178`**:

```cpp
const double rs = 1.0 / std::sqrt((double)Nt);
for (int i = 0; i < Nt * NRF; ++i) {
  const double m = std::abs(src[i]);
  Frf[i] = m > 0.0 ? src[i] * (rs / m) : dcplx(rs, 0.0);
}
```

* Verdict at `-O3`: **loop not vectorized**; reasons complex:942 (`std::abs` →
  `hypot`, a call) and complex:785 (`operator*(complex,double)` plus the
  `m > 0.0` select on a complex store).
* Per entry: 1 `hypot` + 1 divide + 1 complex×real. At Nt=128, NRF=32 that is
  **4096 `hypot` + 4096 divide**, and both are exactly the operations an A53 has
  no hardware for (README's model: div 29 cycles, and `hypot` is a libm call on
  top). On the A53 projection this loop is the reason "glue+proj" is 4.6 ms of
  the ~30 ms instant.
* **Correction to `docs/cost_model_real_shapes.md` §3**, which prices this step at
  *"16 384 transcendentals (4096 x atan2+cos+sin+div) + 4096 div"*. That row is
  stale: `ref/frontend.cpp:143-178` already replaced `exp(i*angle(x))` with
  `x * (rs/|x|)` (the comment at `:149-167` records the measurement: 51.75 µs →
  5.86 µs, 8.83x, 2.2 ulp). The live cost is 4096 `hypot` + 4096 div, roughly
  4x fewer transcendentals than the table claims. The `atan2+cos+sin` form only
  survives in the DFT filler at `ref/frontend.cpp:139-142`
  (`std::exp(dcplx(0, 2*pi*r*(c-col)/Nt))`), and that loop is dead at
  paper_sweep because `K*take == NRF == 32` fills every column
  (`analog_users`, `ref/frontend.cpp:181-190`).
* Vectorising `hypot` is a libm question, not a NEON question: a bit-exact
  vector `hypot` does not exist. The gate-safe win here is the *algebraic* one —
  `|x|` for a constant-modulus projection only needs enough accuracy for the
  phase, and `rsqrt`-style approximations are not bit-exact — so this loop should
  be treated as **not vectorisable without moving its gate**, and instead be
  priced honestly on the A53 as 4096 `hypot` + 4096 divide.

### 1.4 Threshold: `psi`, log-sum-exp, quadrature

`ref/threshold_cf.cpp`. Per solve, `docs/cost_model_real_shapes.md` §3: ~1680 log
+ ~1930 exp + ~45 kflop (table path, flag 0, T=32 K=8 M=64 GMD, 42 fills).

| loop | file:line | verdict at `-O3` | verdict at `-O3 -ffast-math` |
|---|---|---|---|
| `logsumexp_neg_d2` reduction | `ref/threshold_cf.cpp:86` | *cost-model: not beneficial* | **vectorized 2x4** |
| `logsumexp_neg_d2` `min_element` | `:84` (libc++ `min_element.h:37`) | not vectorized (`while (++__i != __last)`) | not vectorized |
| `gh_expect` GH-48 sum | `:92-97` | — | — |
| `axis_deficit` distance split | `:124-132` | not vectorized: the `if (cb == 0) d0[n0++] = d2; else d1[n1++] = d2` at `:130` is a **counted store into one of two arrays** — a data-dependent write index, the classic non-vectorisable scatter | partially (2x1/2x4 at `:127`) |
| `thp_point` UCD product/sum | `:597-600` | not vectorized | **vectorized 2x4** (a `*=` product reduction: reassociation changes `p`) |
| `psi_precompute` log sum | `:705` | *cost-model: not beneficial* | **vectorized 2x4** |
| `psi_u` inner | `:730-735` | not vectorized: *"Control flow cannot be substituted for a select"* + *"value that could not be identified as reduction is used outside the loop"* — `mf = std::max(mf, lf[...])` is a running max that leaves the loop, and `lf[]` is written inside | not vectorized |
| `psi_u` log-sum-exp | `:742`, `:743` | *cost-model: not beneficial* (interleaved only) | **vectorized 2x4** |
| `psi_du_central` | `:778-782` | not vectorized: *"value that could not be identified as reduction is used outside the loop"* | vectorized |
| `psi_du` log-sum-exp / weighted sums | `:844-848` | `:844`/`:845` *not beneficial*; `:847`/`:848` reported *vectorized 2x4* | vectorized |
| bracket expansion | `:1048-1049` | `for (int e = 0; e < 6 && psi_u(lo, ...) > 0.0; ++e) lo /= 4.0;` — data-dependent trip count, and each iteration is a full Psi fill | — |
| near-flat scan | `kernels/psi_scan.clh` via `:927` | 161 grid points, each a full `psi_u` | — |
| `scan_root` | `:953` | *"value that could not be identified as reduction is used outside the loop"* | vectorized |

Two structural facts worth recording, because they are what makes this file
different from §1.1:

* **The transcendental count, not the flop count, sets the A53 time.** README:206
  measures the whole table-path solve at ~106 500 flop + ~13 000 transcendentals;
  at 70–80 cycles per exp/log on a 1.4 GHz A53 the transcendentals are ~1 ms of
  the ~1.2 ms. NEON has **no** vector `exp`/`log` in ARMv8.0-A, so vectorising
  the arithmetic around them cannot help much: the measured `-O3` gain here
  (1.18x on `threshold_closed_form`, §2) comes from inlining and register
  allocation, not SIMD.
* **README:925 already settled `exp`/`log` → `exp2`/`log2` on the PS**: ARMv8.0-A
  has no hardware `exp2`/`log2` and `log2` measured *slower* than `log` on this
  host. So the transcendental side of this file has no PS-side SIMD story at all.

### 1.5 Instruction-level census (hard evidence)

`objdump -d` of the four `ref/` TUs, counting vector fp64 multiply-accumulate
(`fmla.2d`) — i.e. actual SIMD arithmetic, not `movi.2d`/`stp q` data movement:

| TU | `-O2` | `-O3` | `-O3 -ffast-math` |
|---|---|---|---|
| `ref/decomp.cpp` | **0** `fmla.2d`, 187 scalar fmul/fadd/fsub | **0**, 202 scalar | **107**, 62 scalar |
| `ref/frontend.cpp` | **0**, 32 scalar | **0**, 41 scalar | **32**, 7 scalar |
| `ref/threshold_cf.cpp` | 8, 327 scalar | 12, 404 scalar | 39, 279 scalar |
| `ref/pipeline.cpp` | **0**, 75 scalar | **0**, 87 scalar | **47**, 9 scalar |

So: **`-O3` emits essentially no new SIMD arithmetic in the hot TUs** (0 → 0 in
`decomp`, `frontend`, `pipeline`; 8 → 12 in `threshold_cf`), and `-ffast-math`
moves the code wholesale from scalar to vector. That is the machine-level
explanation for §0's timing and bit-exactness results.

---

## 2. Microbenchmark A — the null-basis Jacobi rotation, isolated

`/tmp/psneon/mb_jacobi.cpp` (standalone, ~700 lines). It re-implements
`ref/decomp.cpp:33-133` (`svd_cplx`) at the paper_sweep null shape
**C = 8, m = 28, n = 32**, with the same tolerance (`1e-13`), the same
norm-invariant early exit and the same rotation formula, and runs seven
formulations of the two inner kernels.

**Validation of the replica.** The replica's singular values are **bit-identical
to the real `hpocl::svd_cplx`** compiled from the repo at `-O2` on the same input
(all 32 values of matrix 0, after the ref's descending sort; the ref sorts, the
replica does not):

```
mb S[0] sorted desc == repo nullbatch.S[0:32]: True
  first 6: 13.927894761248352 13.047361313792578 12.53632859904195
           11.873016129980137 11.438024655745096 11.174893423656714
  last  4: 5.307863e-20 8.695127e-27 7.762130e-30 8.716943e-32
```

Flop convention is `runtime/hpocl_schedule.cpp:29-32 jacobi_batch_flops`:
`C * n(n-1)/2 * sweeps * (10m + 6(m+n))` = **22.856 Mflop** at the measured
9 sweeps (40.632 Mflop at the 16-sweep cap) — the same number
`docs/cost_model_real_shapes.md` §3 quotes, so the GFLOP/s below is directly
comparable to its 0.933 GFLOP/s A53 projection.

**Protocol.** 7 flag builds, each run **7 times, interleaved round-robin** (so
every variant sees the same thermal history), each run taking the **median of 7
inner rounds**. `sysctl -n vm.loadavg` recorded around every rep:
**2.95–5.24** during the runs (the doc's environment note says load ~3 under
throttling; this machine is a fanless MacBook Air and `docs/cost_model_real_shapes.md`
measures 3–3.7x throttling). Absolute ms are context; the ratios are the result.

### 2.1 End-to-end on the *real* repo code (not the replica)

`/tmp/psneon/timing_probe.cpp` links `ref/{decomp,frontend,threshold_cf,pipeline,rx,modem}.cpp`
read-only and times the shipped functions. 7 interleaved reps x median of 9
rounds; load 2.56–3.98. Every timed loop leaves an observable checksum, because
without one the `-ffast-math` build **deletes the glue GEMM entirely** (measured:
it reported `0.000000 ms` for a loop that costs 83 µs at `-O2` — dead-store
elimination on the un-read `Y`).

| section | `-O2` | `-O3` | `-O3 -mcpu=native` | `-O3 -ffast-math` | `-O2 -ffp-contract=off` | `-O3 -ffp-contract=off` |
|---|---|---|---|---|---|---|
| null Jacobi batch (8x28x32, 9 sw) | **4.6997 ms** | 4.6994 (1.000x) | 4.7324 (0.993x) | 3.9960 (**1.176x**) | 4.8225 (0.975x) | 4.8284 (0.973x) |
| `front_end` (whole) | **4.6860 ms** | 4.7843 (0.979x) | 4.6749 (1.002x) | 3.9434 (**1.188x**) | 4.7852 (0.979x) | 4.7803 (0.980x) |
| `analog_precoder` (Gram SVD + projection) | 0.0422 ms | 0.0421 (1.004x) | 0.0419 (1.008x) | 0.0326 (**1.297x**) | 0.0423 | 0.0421 |
| one glue `gemm_wx` 32x128x32 | 0.0824 ms | 0.0830 (0.994x) | 0.0828 (0.996x) | 0.0474 (**1.741x**) | 0.0826 | 0.0824 |
| `threshold_closed_form` (1 solve) | 0.0593 ms | 0.0502 (**1.182x**) | 0.0499 (**1.189x**) | 0.0488 (1.216x) | 0.0635 (0.933x) | 0.0518 (1.145x) |
| `schedule` (full instant) | 4.7841 ms | 4.8027 (0.996x) | 4.8167 (0.993x) | 4.0062 (**1.194x**) | 4.9047 | 4.9242 |

Run-to-run spread at `-O2` (min–max over the 7 interleaved reps): null batch
4.663–6.057 ms, `front_end` 4.649–4.988 ms, `schedule` 4.775–4.924 ms; the three
sub-millisecond sections vary by ±2%. So the `front_end` 0.979x at `-O3` is
inside the noise and should be read as 1.00x, while the null batch (1.000x), the
glue GEMM (0.994x) and `analog_precoder` (1.004x) are flat well outside it.

Read this table as: **the only place `-O3` actually pays is the threshold**
(1.18x, bit-identical), because that file is scalar-`double` + `log`/`exp` and
benefits from inlining; the Jacobi-dominated `front_end` does not move at all,
because clang has nothing to vectorise there without `-ffast-math`.

### 2.2 Formulation-by-formulation, on the replica

`sp_in_build` = median of `SCALAR(verbatim)` **in the same build** / median of
this variant, so the flag effect and the formulation effect are separated.
`S_bitex`/`V_hash` compare the 256 singular values and all 8192 `V` entries
bit-for-bit against `SCALAR(verbatim)` of the same build.

Builds: `-O2` / `-O2 -mcpu=native` / `-O3` / `-O3 -mcpu=native` /
`-O3 -ffast-math` / `-O2 -ffp-contract=off` / `-O3 -ffp-contract=off`.
Exact flag strings as passed to `clang++` (Apple clang 21.0.0,
`clang-2100.1.1.101`, target `arm64-apple-darwin25.6.0`); `-mcpu=native`
resolves to `-target-cpu apple-m4 -target-feature +v8.7a +neon +bf16 +i8mm
+dotprod +fullfp16 +sme ...`.

`-O2` build (baseline med_ms **4.6268**, 4.94 GFLOP/s):

| variant | med_ms | GFLOP/s | sp_in_build | S bit-exact | max rel S | V hash |
|---|---|---|---|---|---|---|
| `SCALAR(verbatim)` = `ref/decomp.cpp:62-88` | 4.6268 | 4.94 | 1.000 | REF | 0 | REF |
| `SCALAR(gen)` (same code through the templated driver) | 4.8462 | 4.72 | 0.955 | **yes** | 0.0 | **yes** |
| `SCALAR_PACK` (same r-order, columns packed contiguous first) | 4.4202 | 5.17 | 1.047 | **yes** | 0.0 | **yes** |
| `NEON_EXACT` (fp64x2, alpha\|beta + gamma_re\|gamma_im, ordered adds) | 4.5381 | 5.04 | 1.020 | **yes** | 0.0 | **yes** |
| `NEON_PAIR` (fp64x2, 2-way unrolled over r = reassociated) | 4.5150 | 5.06 | 1.025 | **NO** | 2.72 | NO |
| `VDSP` (Accelerate `vDSP_dotprD`, blocked reduction) | 5.1147 | 4.47 | 0.905 | **NO** | 6.51 | NO |
| `NEON_EXACT+apply` (bit-exact NEON Gram **and** apply loop) | 3.3560 | 6.81 | 1.379 | **yes** | 0.0 | **yes** |
| `COLMAJOR_NEON` (one column-major repack **per sweep**, `vld1q`+`vtrn1/2`, `vfmaq_f64` for `norm`) | **3.1658** | **7.22** | **1.461** | **yes** | **0.0** | **yes** |

`-O3`: same ordering, `COLMAJOR_NEON` **1.485x**, `NEON_EXACT+apply` 1.405x,
`SCALAR_PACK` 1.098x, all bit-exact; `NEON_PAIR` 1.074x **not** bit-exact;
`VDSP` 0.930x.
`-O3 -mcpu=native`: `COLMAJOR_NEON` **1.481x**, bit-exact. (`-mcpu=native` is
worth ≤2% over plain `-O3` here; on Apple Silicon NEON is baseline, so
`-mcpu=native` only adds v8.7a features this code does not use.)
`-O3 -ffast-math`: `COLMAJOR_NEON` 1.561x, `SCALAR_PACK` 1.628x, `SCALAR(gen)`
1.247x — **none bit-exact** (max rel S 1.1e3 … 1.7e1 on the trailing near-null
singular values).

Elementwise rotation-apply loop only (60 rows x one `(p,q)` pair, 496*9 reps):

| build | scalar ms | NEON ms | speedup | bit diffs / 1920 |
|---|---|---|---|---|
| `-O2` | 0.3994 | 0.2332 | **1.713** | **0** |
| `-O2 -mcpu=native` | 0.3994 | 0.2331 | 1.715 | 0 |
| `-O3` | 0.3992 | 0.2330 | 1.715 | 0 |
| `-O3 -mcpu=native` | 0.4018 | 0.2331 | 1.716 | 0 |
| `-O3 -ffast-math` | 0.1785 | 0.1840 | 0.970 | 120 |
| `-O2 -ffp-contract=off` | 0.3996 | 0.2328 | 1.715 | 0 |
| `-O3 -ffp-contract=off` | 0.3994 | 0.2330 | 1.714 | 0 |

Three results worth pulling out:

1. **`VDSP`/Accelerate is a loss.** 0.905–0.940x, i.e. 6–10% *slower* than the
   scalar loop, and not bit-exact. At m = 28 the blocked-reduction setup cost
   dominates. There is no "just call the library" path for this shape.
2. **Reassociation buys nothing on M4.** `NEON_PAIR` (2-way interleaved
   accumulators — what a vectoriser does) is 1.025–1.074x, statistically the same
   as the ordered `NEON_EXACT` (1.020–1.065x), and it is **not** bit-exact. On an
   out-of-order M4 the machine already extracts the ILP. Breaking the reduction
   order is therefore pure cost with no benefit on this host — and on an in-order
   A53 the benefit would have to be re-measured before accepting the cost.
3. **The win comes from the layout, not from the SIMD.** Going from per-`(p,q)`
   packing (`SCALAR_PACK`, 1.047x) to one column-major repack per sweep
   (`COLMAJOR_NEON`, 1.461x) is a 1.4x further gain, and both are bit-exact. The
   repack costs `2*(m*n + n*n)` complex moves per sweep = 3840, against
   `496 pairs x (28 + 32) rows x 2` ≈ 59 500 complex stores per sweep for the
   rotations alone (~6% more store traffic, and `n(n-1)/2 * (10m + 6(m+n))`
   ≈ 317 kflop of arithmetic) — and it converts every strided AoS access into a
   `vld1q_f64`.

---

## 3. Microbenchmark B — the glue GEMM (`gemm_wx`)

`/tmp/psneon/mb_gemm.cpp`. `REF` is a verbatim copy of `ref/frontend.h:44-54`.
`NEON_M` vectorises across the **independent output index `m`** (two rows of `W`
in one `float64x2` pair, `X` packed to SoA once per `sc` and reused by all `M`
rows), keeping the reduction over `n` strictly ascending and performing exactly
libc++'s three roundings per element. `UNROLL_N` is a 2-way unroll over the
reduction index `n` written in scalar form, so the bit effect of reassociation is
isolated from the SIMD effect. 7 rounds, medians, load ~2.6–3.6.

| shape | M | N | SC | `REF` ms | `NEON_M` ms | `UNROLL_N` ms | GFLOP/s (NEON_M) | speedup | bit-exact `NEON_M` | bit-exact `UNROLL_N` |
|---|---|---|---|---|---|---|---|---|---|---|
| `kGemmEff` / `kGemmFn` / `kGemmPower` at Nt=128 | 32 | 128 | 32 | 0.0831 | **0.0505** | 0.0826 | **20.75** | **1.645** | **yes (0/1024)** | NO (1008/1024, max rel 3.1e-14) |
| Nt=64 class | 32 | 64 | 32 | 0.0411 | **0.0251** | 0.0414 | 20.87 | **1.637** | **yes (0/1024)** | NO (996/1024) |
| Nt=32 class | 32 | 32 | 32 | 0.0206 | **0.0126** | 0.0208 | 20.76 | **1.630** | **yes (0/1024)** | NO (959/1024) |

Identical at `-O3` (1.648/1.651/1.654/1.633/1.648x, all bit-exact) and at
`-O3 -mcpu=native` (1.644/1.647/1.648/1.634/1.636x, all bit-exact).

Under `-O3 -ffast-math` the compiler's own vectorised `REF` reaches 0.0480 ms
(**1.73x** vs the `-O2` `REF`) with **986/1024 elements differing** — i.e. the
hand-written bit-exact `NEON_M` captures **95% of `-ffast-math`'s speedup at zero
numerical cost**. That single comparison is the strongest argument in this
document for doing PS SIMD by hand.

`UNROLL_N` is the control that matters: reassociating the `n` reduction is worth
**1.006x** on M4 and costs bit-exactness on 98% of the outputs. There is no
trade to make here.

---

## 4. Bit-exactness of the flag change, measured on the whole control plane

`/tmp/psneon/gate_probe.cpp` links the repo's `ref/*.cpp` read-only and dumps
**every** output as raw IEEE-754 bit patterns: **159 fields, 102 291 scalars**.
Coverage: the 8x28x32 null-basis Jacobi batch (`S`, `V`, `U`, and the trailing
4-column `Nbasis` that `null_basis_trailing` hands downstream); `gemm_wx` at the
three Nt-class shapes; `analog_precoder` (Gram SVD + projection); the projection
loop in isolation including the `x == 0` and `1e-300` branches; ten threshold
spectra x (`tau_cf_dB`, `tau_dB`, `u_star`, `newton_used`, `flag`, `fell_back`,
`psi_u`, `psi_du`, `psi_du_central` at five `u`, `deficit_fin`,
`deficit_fin_quad`, `ln_deficit_mod` at twelve gammas, `threshold_deploy_lmmse`);
the full `front_end` + `threshold_closed_form` + `design_instant` instant at
paper_sweep for both THP variants (every `PipeOut` and `FrontEnd` field); and
`qam_table` / `modulo_fold` / `softdemod_parallel` / `gmi_from_llr` for
M = 16/64/256. Comparators: `/tmp/psneon/tau_report.py`,
`/tmp/psneon/field_report.py`, `/tmp/psneon/oracle_tau.py`.

### 4.1 Bit-exactness vs the `-O2` baseline

| build | fields bit-identical | scalars differing | max abs Δτ (dB) | max rel Δτ | integer fields (`mode`/`cf_flag`/`fe_flag`/`nul`) differing |
|---|---|---|---|---|---|
| `-O3` | **159 / 159** | **0 / 102 291** | **0.0** | 0.0 | **0** |
| `-O3 -mcpu=native` | **159 / 159** | **0 / 102 291** | **0.0** | 0.0 | **0** |
| `-O3 -ffast-math` | 37 / 159 | **84 083** | **3.971e-06** | 9.1e-07 | 0 |
| `-O2 -ffp-contract=off` | 56 / 159 | **75 487** | **1.986e-06** | 4.6e-07 | 0 |
| `-O3 -ffp-contract=off` | 56 / 159 | **75 487** | **1.986e-06** | 4.6e-07 | 0 |

Group-level worst case (`max_rel` over entries with `|x| > 1e-12`):

| group | n | `-O3 -ffast-math` differing / max_rel | `-O2 -ffp-contract=off` differing / max_rel |
|---|---|---|---|
| null-basis batch (`S`,`V`,`U`,`Nbasis`) | 33 024 | 31 059 / **28.4** | 31 053 / 28.4 |
| glue GEMMs | 6 144 | 5 524 / 1.2e-12 | 4 512 / 6.4e-13 |
| `analog_precoder` Frf | 8 192 | 7 337 / 6.3e-11 | 6 919 / 6.8e-11 |
| projection loop | 8 192 | 799 / 4.5e-16 | 593 / 4.4e-16 |
| threshold (all cases) | 610 | 336 / 3.7e-06 | 155 / 2.5e-06 |
| scheduling instant | 33 166 | 30 538 / **149.6** | 29 400 / 152.4 |
| modem | 12 963 | 8 490 / 5.2e-11 | 2 855 / 2.5e-12 |

The O(1) relative deviations are **not** a precision failure — they are the
degenerate null-space basis. A 28x32 stack has 4 structurally-zero singular
values, and any basis of that 4-dimensional null space is a correct answer; a
1-ulp perturbation rotates the basis inside the null space. The absolute numbers
confirm it: `nullbatch.S` max **abs** change is 1.6e-14 while its max **rel**
change is 28 (the tiny trailing values). The repo already knows this — it gates
`N` through the projector and pins the gauge:

| invariant | `-O3 -ffast-math` | `-O2 -ffp-contract=off` | gate |
|---|---|---|---|
| BD projector `N_k N_k^H`, max rel (8 users) | **6.4e-13** (Frobenius rel 1.1e-14) | 3.6e-13 (4.7e-15) | README:104 gates `N(projector)` at 1e-9 → **passes** |
| `o.sigma` (= sigma(Gbar)), max rel | **6.5e-15** | 4.6e-15 | README:104/105 at 1e-9 → **passes** |
| `o.Frf`, max rel | 8.8e-11 | 3.8e-11 | README:104 at 1e-9 → passes, **11x margin** |
| `fe.eff`, max rel | 2.0e-12 | 1.9e-12 | 1e-9 → passes |
| `o.Fbb`, max rel | **5.6e-10** | 6.0e-10 | README:105 `B <= 2.2e-11` measured / 1e-9 bound → passes with only **1.8x margin** |
| `fe.Gbar` itself, max rel | 8.7 (Frobenius 0.39) | 27 (0.40) | **not gated entrywise** — README:104 gates `sigma(Gbar)`. Would break any future entrywise comparison |
| `o.pscale` | 8.9e-16 rel | 1.1e-15 rel | README:105 `P <= 6.9e-12` → passes |

### 4.2 τ in dB against the numpy oracle, vs the gate thresholds

`tools/cf_ref.py` imported **read-only** (`importlib` from the repo path;
`main()` — the only thing that writes `build/cf_vectors.bin` — is never called),
converged Newton at `iters=30`:

| case | flag | τ_oracle (dB) | `-O2` | `-O3` | `-O3 -mcpu=native` | `-O3 -ffast-math` | `-O2 -ffp-contract=off` |
|---|---|---|---|---|---|---|---|
| well_gmd_64 | 1 | 30.2856855481782 | 3.6e-15 | 3.6e-15 | 3.6e-15 | 3.6e-15 | 3.6e-15 |
| well_ucd_64 | 1 | 30.1679916189594 | 7.1e-15 | 7.1e-15 | 7.1e-15 | 1.8e-14 | 0.0 |
| spread_gmd_64 | 1 | 21.6723910551910 | 1.1e-14 | 1.1e-14 | 1.1e-14 | **3.97e-06** | **1.99e-06** |
| dead_gmd_64 | 1 | 84.1006819190230 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| nearflat_64 / deepflat_d2_64 / sat_gmd_64 | 2 | ±inf | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| huge_gmd_64 | 1 | -22.3904796890609 | 1.8e-14 | 1.8e-14 | 1.8e-14 | 0.0 | 1.8e-14 |
| well_gmd_256 | 1 | 34.8480019061011 | 7.1e-15 | 7.1e-15 | 7.1e-15 | 7.1e-15 | 7.1e-15 |
| well_gmd_16 | 1 | 23.0224645442073 | 3.6e-15 | 3.6e-15 | 3.6e-15 | 0.0 | 0.0 |
| **worst over all cases** | | | **1.78e-14** | **1.78e-14** | **1.78e-14** | **3.97e-06** | **1.99e-06** |

`tau_cf_dB` (index 0 of `cf.*.out`) is **unchanged in every case under every
variant** — it is a closed-form expression with no reduction. Only `tau_dB`
moves, and only on the flag-1 (near-flat, `PSI_SCAN_N = 161` fallback) spectra,
where it is a Newton root of a reassociated log-sum-exp. `newton_used` and
`flag` are unchanged everywhere, so no branch of the mode policy flips.

Against the gates in README's validation table:

| gate (README line) | bound | measured today | `-O3` | `-O3 -ffast-math` | `-ffp-contract=off` |
|---|---|---|---|---|---|
| :84 closed form `u*0` C++ vs numpy | **1e-9 dB** | 0.00e+00, 40/40 string-identical | **PASS (identical)** | **FAIL**, 3.97e-6 dB (4000x over) | **FAIL**, 1.99e-6 dB |
| :85 Newton-root τ vs converged numpy oracle (30 steps) | 1e-4 dB | ≤2.49e-13 | PASS | PASS (3.97e-6) | PASS |
| :105 pipeline scheduling instant vs production python | 1e-9 | τ≤2e-10 dB, σ≤1.5e-11, P≤6.9e-12, B≤2.2e-11 | **PASS (identical)** | PASS but B margin down to 1.8x | PASS, same margin issue |
| :104 front end eff/N(projector)/eig/σ(Gbar) | 1e-9 | ≤2e-11 | **PASS (identical)** | PASS (6.4e-13 projector, 6.5e-15 σ) | PASS |
| :98 SVD σ / V_D vs scipy | 1e-9 | 1.3e-15 / 3.0e-14 | **PASS (identical)** | σ PASS (2.0e-15); V_D columns are basis-dependent and already reported `n/a` (:102) | same |
| :106 modem constellation/modulo | **bit-exact** | bit-exact | **PASS (identical)** | **FAIL** — `qam_table` 64/128 entries 1 ulp, `modulo_fold` up to **4096 ulp**, LLR up to 262 144 ulp | FAIL (up to 16 384 ulp) |
| :73 `gh_check`, :79 `deficit_check` | bit-identical / 1 ulp | 16 000 values bit-identical | unaffected — these compare `include/deficit_table.h` against `kernels/deficit_table.clh` and a numpy regeneration; no host arithmetic is involved | unaffected | unaffected |
| :77 `sched_bench` device role vs PS role | τ 5e-2 dB, σ 1e-4, pscale/diagb 1e-5/1e-4 | τ≤5.7e-4 dB | PASS (identical PS role) | PASS on τ; **but** the PS role moves while the fp32 kernel does not, so the two-role agreement degrades toward its bound | same |
| :114 LMMSE deployment | 1e-3 dB | τ≤1.8e-4 dB | PASS | PASS (1.29e-5 dB) | PASS |

Additional `-ffast-math` disqualifier, independent of precision: clang emits

```
ref/threshold_cf.cpp:1033:56: warning: use of infinity via a macro is undefined
behavior due to the currently enabled floating-point options [-Wnan-infinity-disabled]
 1033 |     const double inf = psi_u(1.0, lam, c, &w) > 0.0 ? -INFINITY : INFINITY;
```

`INFINITY` there is the flag-2 saturation value that `decide_mode`
(`ref/pipeline.cpp:340-356`) compares against, and `ref/threshold_cf.h:99-102`
documents it as a contract ("INFINITY, not HUGE_VAL, so `std::isinf()` is
meaningful"). Under `-ffast-math` that contract is UB. The probe did still print
`inf` on this compiler, but shipping a control law whose saturation branch is UB
is not a trade for 1.19x.

### 4.3 The discovery that decides the implementation recipe: FMA contraction

Apple clang's default is **`-ffp-contract=on`**, and the repo does not override
it. That default produces a *mixed* convention inside one expression, which any
hand-written kernel must reproduce exactly:

| expression | what the `-O2`/`-O3` baseline actually computes | measured match rate vs a plain real-arithmetic spelling |
|---|---|---|
| `std::norm(z)` (`ref/decomp.cpp:54,63,64,67,95`) | **`fma(re, re, im*im)`** — libc++'s `norm` *is* contracted | plain `re*re+im*im` matches only **336 146 / 400 000**; `fma(ar,ar,ai*ai)` matches **400 000 / 400 000**; `fma(ai,ai,ar*ar)` only 309 930 (`/tmp/psneon/normprobe.cpp`) |
| `std::conj(a)*b` (`ref/decomp.cpp:65`) | **uncontracted** `(ar*br + ai*bi, ar*bi - ai*br)` — libc++'s `operator*` carries the C99 Annex G `isnan` recovery branch, which blocks contraction | plain matches **200 000 / 200 000** *only inside a `#pragma clang fp contract(off)` region*; with the TU default it matches **112 537 / 200 000** (`/tmp/psneon/pragma_test.cpp`) |

Consequences, all measured:

1. **A naive hand-written NEON kernel is not bit-exact even when its reduction
   order is identical.** Before applying the recipe below, `SCALAR_PACK`,
   `NEON_EXACT` and `COLMAJOR_NEON` all differed from the `-O2` baseline on
   220–233 of 256 singular values (max abs 1.2e-14), and the elementwise apply
   loop differed on 120/1920 entries.
2. **The fix is local and needs no Makefile change.** `#pragma clang fp
   contract(off)` around the rewritten region + `std::fma`/`vfmaq_f64` for the
   `|z|^2` terms + unfused `mul`/`add` for the complex products reproduces the
   baseline **exactly**: `SCALAR_PACK`, `NEON_EXACT`, `NEON_EXACT+apply` and
   `COLMAJOR_NEON` all become bit-identical (S hash *and* V hash match, 0
   differing singular values), and the apply loop goes 120 → **0** bit diffs.
   Verified at `-O2`, `-O2 -mcpu=native`, `-O3` and `-O3 -mcpu=native`.
   For the deployment toolchain (aarch64 GCC), the portable equivalent is to put
   the NEON kernels in their **own translation unit** compiled with
   `-ffp-contract=off` — a one-line addition to a *new* source list, not an edit
   of `CXXF`.
3. **The symmetric trap:** building the *whole* tree with `-ffp-contract=off`
   moves 75 487 of 102 291 scalars and τ by up to 1.99e-6 dB — the same damage
   class as `-ffast-math`. So "make it strictly IEEE" is **not** a safe
   direction either: the recorded gate numbers are properties of the current
   contracted build. This corroborates README:919, which measured
   `-ffp-contract=off` moving 17 of 24 channels by 2.16e-5 dB.

---

## 5. Conclusions and ranked recommendations

Ordered by (benefit) / (risk) / (does it need the board).

**R1. Do not touch the flags for speed. If `-O3` is ever adopted, adopt it for the
threshold file only, and re-run `make check` because it is free.**
Benefit: 1.18x on `threshold_closed_form` (59.3 → 50.2 µs), 0.98–1.02x
everywhere else — i.e. ≈0.9 µs of a 4.79 ms instant at paper_sweep, and ~0.2 ms
of a ~30 ms A53 instant **[PROJECTION]**. Risk: **zero measured** (0 of 102 291
scalars change). Needs board: no. Honest verdict: the gain is too small to be
worth a flag change on its own, but it is the only flag change in this audit that
is numerically free, so if the Makefile is ever touched for another reason,
`-O3` costs nothing and buys the threshold. `-mcpu=cortex-a53` on the deployed
toolchain is the analogous free change (instruction scheduling for an in-order
core), and should be measured on the board, not assumed.

**R2. Hand-write a bit-exact fp64 NEON `gemm_wx` that vectorises across `m`, not
across `n`.** Benefit: **1.63–1.65x measured, bit-identical**, at 20.8 GFLOP/s on
M4 — 95% of what `-ffast-math` extracts (1.73x) with none of its cost. On the A53
this is the *glue* term, which `docs/cost_model_real_shapes.md` puts at ~15% of
the front end after the decompositions go to the fabric, i.e. ~0.7 ms of the
~4.6 ms glue+proj, and it applies to the **47 of 63 sites `glue_cost_check` keeps
on the PS** as well as to `ref/pipeline.cpp:216/224/302-312` and
`ref/frontend.cpp:51/90/116`, which have the same "independent outer index,
reduction inner" shape. Risk: low and *mechanically checkable* — the recipe of
§4.3 (`contract(off)` region + `fma` for `|z|^2` + unfused complex product) plus
a gate that compares the NEON and scalar providers on the same input. Needs
board: no for correctness or M4 speedup; yes for the A53 number.

**R3. Hand-write the bit-exact column-major NEON null-basis Jacobi
(`COLMAJOR_NEON`) — but only after R2, and only if the PL offload of README row 2
is not happening.** Benefit: **1.45–1.49x measured, bit-identical** on 94% of
`front_end`. On the A53 that is 24.5 ms → ~16.5 ms **[PROJECTION]**, i.e. ~30 ms
→ ~22 ms for the whole instant. Risk: medium — it is the most invasive change in
this list (a per-sweep layout transform inside `svd_cplx`, and it must preserve
the `off` accumulation at `ref/decomp.cpp:67` bit-for-bit because that decides
the sweep count). Needs board: no for M4 evidence; yes to confirm the A53 gain,
because A53's single load pipe makes the repack relatively more expensive than on
M4. **Opportunity cost: README row 2 quotes the same batch at ~19 ms → ~0.6 ms on
the PL (~30x).** R3 is what you do for the configs that stay on the PS.

**R4. Leave `ref/frontend.cpp:174-178` alone and fix its cost row instead.**
Benefit: none from SIMD — `std::abs(complex)` is `hypot`, and there is no
bit-exact vector `hypot` in ARMv8.0-A. Risk: high if attempted (it is the one
loop whose output is a *physical* constant-modulus constraint). Needs board: no.
Actionable: `docs/cost_model_real_shapes.md` §3 still prices this step at
"16 384 (4096 x atan2+cos+sin+div) + 4096 div"; the shipped code does
**4096 `hypot` + 4096 divide** (`ref/frontend.cpp:149-178`). Correct the row so
the A53 budget is not chasing a transcendental count that was already removed.

**R5. Do not reassociate anything, and do not use Accelerate/vDSP.**
Benefit: negative. Measured: reassociation (`NEON_PAIR`, `UNROLL_N`) is worth
1.006–1.074x on M4 and costs bit-exactness on 98% of GEMM outputs and on the
trailing singular values; `vDSP_dotprD` is **0.905–0.940x, i.e. slower than
scalar**. Needs board: no. Corollary for the A53 discussion: the usual
"compensated/Kahan summation so we can reassociate safely" idea does not help
here either — Kahan changes the result too (it is a *different*, more accurate
sum), so it still breaks a bit-level gate, and it costs 4x the flops on a
2-lane fp64 machine. If a reduction must be reassociated, the gate has to move
first, not the arithmetic.

**R6. If a gate is ever to be relaxed, relax it to "ulp-bounded on the invariants,
not bit-level on the gauge".** The measurement in §4.1 shows the repo's existing
choice of invariants is already right: under `-ffast-math` the *basis* `N_k`
rotates by O(1) while the *projector* `N_k N_k^H` moves 6.4e-13 and `sigma(Gbar)`
moves 6.5e-15. What actually blocks a flag change is not physics, it is (a) the
`1e-9 dB` closed-form τ row, (b) `modem_bench`'s bit-exact constellation/modulo
row, and (c) README's recorded `0.00e+00` / "string-identical" measured values.
Those three could be restated as `≤1e-5 dB`, `≤4 ulp`, `≤1e-12 rel` respectively
without losing any real assurance — and that would make `-ffast-math`'s 1.19x on
`front_end` and 1.74x on the glue available. Risk: this is a governance decision,
not a technical one, and it is **not recommended**: R2+R3 deliver ~1.5–1.65x
bit-exactly, so buying 1.19–1.74x with the whole bit-identical story is a bad
trade. Needs board: no.

---

## 6. What to actually SIMD-ise on an A53: fp64 2-lane vs fp32 4-lane

**The control plane must be fp64, so the honest ceiling is 2 lanes, not 4.**
`ref/decomp.h:6-12` states the role of these files ("fp64 PS-role reference …
this file stays a bit-comparable oracle model"), and every PS gate in README's
table compares the fp64 host against a numpy/scipy oracle at 1e-9 or bit level.
An fp32 4-lane PS control plane is not a vectorisation decision, it is a
re-definition of what the oracle is — and the fp32 side of that split already
exists, on the PL (`kernels/*.cl`, gated at 3e-4 / 5e-2 / 1e-4).

Concretely, for a 4x Cortex-A53:

| | fp64 (2 lanes) | fp32 (4 lanes) |
|---|---|---|
| legal for `ref/` (the oracle)? | **yes** — it is the required precision | **no** — would move every 1e-9/bit gate |
| lane-width ceiling vs scalar | 2x | 4x |
| measured M4 realisation, bit-exact | **1.63–1.65x** (GEMM), **1.45–1.49x** (Jacobi), **1.71x** (apply loop) | n/a |
| A53 **[PROJECTION]** | the M4 is out-of-order and already extracts ILP, so the *relative* gain on an in-order, single-FP-pipe A53 should be **at least** the M4 number and plausibly closer to the 2x ceiling — but A53 also has one load pipe, so the §1.1 strided-AoS problem and the §2.2 repack cost bite harder. Must be measured. | — |
| transcendentals | **no vector `exp`/`log`/`hypot`/`atan2` in ARMv8.0-A NEON.** This caps the threshold path (README:206: ~13 000 transcendentals ≈ 1 ms of the ~1.2 ms A53 solve) and the projection loop (4096 `hypot`), and it is why README:925's `exp2`/`log2` idea is a PL-only win | same |

Priority list for the A53, best return first:

1. **`gemm_wx` across `m`** (R2): contiguous in `n`, independent in `m`, one
   reduction, bit-exact recipe proven. Highest ratio of measured gain to risk.
2. **The elementwise rotation-apply loops** `ref/decomp.cpp:79-88` (and
   `:268-278`, `:453-463` in `gmd_factors`/`cbd_factors`, which are the same
   shape): no reduction at all, so bit-exactness is trivial. Measured 1.71x
   standalone on M4. In `jacobi_batch_flops`' own split the two apply loops are
   the `6(m+n) = 360` term against the Gram's `10m = 280`, i.e. **56% of the
   per-pair work is elementwise and carries no reduction hazard at all** — this
   is the larger and safer half of R3.
3. **The null-basis Jacobi Gram inner loop** with a per-sweep column-major repack
   (R3). Needs `vfmaq_f64` for `|z|^2` and unfused products for `conj(a)*b`
   (§4.3), and ordered lane extraction to keep the `r`-ascending sum.
4. **`gram_hermitian` / `inv_sqrt_from_eig`** (`ref/frontend.cpp:47-64`,
   `:66-95`) — same "independent outer, reduction inner" shape as `gemm_wx`, and
   they run K times per instant. Small in absolute terms (<1.2% of `front_end`
   each per the cost model) but free once R2's kernel exists.
5. **Not worth it:** the threshold path (transcendental-bound), the projection
   loop (`hypot`-bound), `stream_snr` (m = 4…8, too short for a 2-lane loop with
   a runtime trip count), and anything fp32.

One more A53-specific note: the four A53s are a legitimate *coarse*-grain
parallelism axis that needs no SIMD and no numerical change at all — the C=8
null-basis batch is eight independent matrices (`ref/pipeline.cpp:31-34`), and
`std::thread` is already linked (`Makefile:44`, `THREAD_FLAGS := -pthread` on
Linux). Splitting the batch across 3–4 cores is bit-exact *per matrix* because
each matrix's reductions are untouched, and it is worth up to ~4x — more than any
SIMD path here. It is out of scope for this audit (it is threading, not
vectorisation) but it dominates the list above and should be evaluated first;
the caveat is that a 4-thread PS leaves no core for the OpenCL runtime's own
submission thread or for the R5F-side RF path, and the A53 cluster shares one
L2, so the 8 x 1856-complex working set (237 KB) will contend.

---

## 7. Does any of this need the board?

| claim | status |
|---|---|
| `-O2` → `-O3` / `-O3 -mcpu=native` is bit-identical on the PS control plane | **established on M4 + Apple clang 21.** Should be re-confirmed with aarch64 GCC (the deployed toolchain) before adopting, because GCC's default is `-ffp-contract=fast` and its inlining differs; the probe in §8 does this in one command. |
| `-O3 -ffast-math` / `-ffp-contract=off` move τ by 2–4e-6 dB and break the bit-exact rows | **established on M4.** Magnitudes will differ on GCC; the direction will not. |
| `NEON_M` gemm 1.645x bit-exact; `COLMAJOR_NEON` Jacobi 1.46x bit-exact | **M4 measurements.** Bit-exactness is a *proof-level* property (identical operation sequence per element) and should hold on A53/GCC once the contraction convention is matched; **the speedup is a board question.** |
| A53 ~16.5 ms null batch, ~0.7 ms glue saving | **[PROJECTION]** from the repo's own 0.933 GFLOP/s model. Needs the board. |
| vDSP is slower than scalar at these sizes | M4-only, and Accelerate does not exist on the A53. Irrelevant to deployment; recorded because it rules out the "use the library" shortcut. |

---

## 8. Reproduction

Everything below is in `/tmp/psneon/`; nothing was written into the repo except
this file. `build/` was not touched, and no repo source, Makefile or README line
was modified (`git status` on the repo is unchanged apart from `docs/ps_neon_audit.md`).

| file | what it is |
|---|---|
| `mb_jacobi.cpp` | the null-basis Jacobi microbenchmark, 8 formulations (verbatim replica, templated scalar, per-pair pack, `NEON_EXACT`, `NEON_PAIR`, `VDSP`, `NEON_EXACT+apply`, `COLMAJOR_NEON`), FNV hashes of `S` and `V`, per-round timing |
| `mb_gemm.cpp` | `gemm_wx` microbenchmark: `REF` / `NEON_M` (bit-exact, across `m`) / `UNROLL_N` (reassociated control), 5 shapes |
| `gate_probe.cpp` | read-only driver over `ref/*.cpp`: 159 fields / 102 291 scalars dumped as raw bit patterns |
| `timing_probe.cpp` | steady-state timing of the *real* `svd.run`/`front_end`/`analog_precoder`/`gemm_wx`/`threshold_closed_form`/`schedule` |
| `normprobe.cpp`, `pragma_test.cpp`, `diag.cpp` | the FMA-contraction experiments behind §4.3 |
| `build_probe.sh`, `build_timing.sh`, `run_mb.sh`, `run_timing.sh` | flag matrices and interleaved-repetition runners |
| `tau_report.py`, `field_report.py`, `oracle_tau.py`, `agg_timing.py` | bit/ulp/rel differ, per-field report, `tools/cf_ref.py` oracle comparison, timing aggregator |
| `remarks/*.txt`, `remarks/*.o` | clang `-Rpass*` vectoriser logs and the objects used for the `objdump` census of §1.5 |

```sh
cd /tmp/psneon
./build_probe.sh O2 -O2 && ./bin/probe_O2 O2 > dump_O2.txt          # + O3 / O3NAT / O3FFM / O2NC / O3NC
python3 tau_report.py   O2 O3 O3NAT O3FFM O2NC O3NC                  # §4.1, §4.2
python3 field_report.py O2 O3FFM                                     # §4.1 group table
python3 oracle_tau.py   O2 O3 O3NAT O3FFM O2NC O3NC                  # §4.2 vs tools/cf_ref.py
./run_mb.sh      && cat mb_summary.txt                               # §2.2
REPS=7 ./run_timing.sh && python3 agg_timing.py                      # §2.1
for t in O2 O3 O3NAT O3FFM; do ./gemm_$t 7 $t > gemm_$t.txt; done    # §3
# §1.5 census
clang++ -O3 -ffast-math -std=c++17 -I<repo>/include -I<repo>/ref -I<repo>/runtime \
        -I<repo>/kernels -c <repo>/ref/decomp.cpp -o /tmp/x.o \
        -Rpass=loop-vectorize -Rpass-missed=loop-vectorize -Rpass-analysis=loop-vectorize
```

Environment: MacBook Air, Apple M4 (fanless, throttles under load), 10 CPUs
(4P + 6E), Darwin 25.6.0, Apple clang 21.0.0 (`clang-2100.1.1.101`), target
`arm64-apple-darwin25.6.0`, numpy 2.0.2 / scipy 1.13.1. `sysctl -n vm.loadavg`
during the measured runs: **2.56–5.24**. Date: 2026-11-06.
