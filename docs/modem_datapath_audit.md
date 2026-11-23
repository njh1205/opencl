# Modem / datapath / rate slice: line-by-line audit

Scope: `kernels/datapath.cl`, `kernels/rate.cl`, `kernels/mcs_table.clh`,
`ref/modem.{cpp,h}`, `runtime/{datapath,rate,chain,modem}_main.cpp`,
`tools/{datapath,modem,rate,chain}_ref.py`, `tools/modem_common.py` — i.e. the
chain `bits -> QAM map -> tx combine -> channel -> rx combine -> LLR -> GMI -> MCS`.

Ground truth is the production python (`mix_precode/classical/bicm_metrics.py`,
`thp_precoding.py`), then `tools/*_ref.py` (numpy), then `ref/modem.cpp` (fp64
C++), then the fp32 kernels. **Every number below was measured on this machine's
Apple M4 OpenCL device**, not reasoned about; where a claim is structural it says
so.

---

## 1. What was verified correct (and how)

| Convention | Where it is pinned | Measured |
|---|---|---|
| Square-QAM normalisation `delta = sqrt(6/(M-1))`, `E|s|^2 = 1`; label `= idx ^ (idx>>1)` MSB-first per axis; label -> amplitude needs the full Gray **decode** | `modem_common.check_production()` compares the label -> point map against `get_constellation` for m = 2,4,6,8 and against `modulate_bits_with_labeling` on random bits | `modem_bench` 12 cases: `per/sym/fold = 0.0e+00` (bit-exact), `lab = 0` |
| Exact log-MAP LLR, `max_log=False`, `llr = lnP(b=1|y) - lnP(b=0|y)` (positive favours 1) | `modem_ref.branch_llr` calls production's `bit_llr_from_distances`; the kernel uses the min-stabilised form | `llr_par <= 7.1e-14`, `llr_fold <= 5.7e-14` (fp64 vs production) |
| Noise scale: production scores `-|sqrt(snr)*g*(y-c)|^2`; with unit post-equalisation gain and `snr = 1/n0` that **is** `-|y-c|^2/n0` — no factor of two | `modem_ref.build_case` calls both production estimators with `snr_per_stream = 1/n0` and asserts `\|prod - clamped\| < 1e-9` | holds on all 12 cases |
| THP receive modulo = `centered_modulo_complex`, floor form, per axis | `include/hpocl_modulo.h` + `kernels/cplx.clh` are the two spellings; oracle asserts **exact** equality | `mod_fold == centered_modulo_complex` to 0.0 |
| Replica lattice of the folded branch (real shift outer, imag inner, base innermost, labels copied) | `folded_candidates` vs `enumerate_thp_symbol_replicas` | element-wise for m = 4,6,8; **order-free** for m = 2 (see §2.6) |
| GMI = `m - mean_sym sum_b log2(1+exp(-(2c-1)llr))`, **per stream, unclamped**; production clamps the *aggregate* | `chain_ref` asserts `\|prod - max(0, sum_k gmi_k)\| < 1e-6` | holds on the SVD and GMD cases, and per stream on the UCD case (§2.5) |
| MCS rule `R*m <= GMI`, largest such CQI index, outage otherwise; non-finite or out-of-range GMI is a **refusal**, never a pick | `rate_ref.select` (numpy, table retyped from TS 38.214 and self-checked against the published SE column) vs the host `.clh` vs the kernel | `rate_bench` 10 cases, **0.00e+00** on every column. Device probe: `NaN/+inf/-inf/-0.5/0 -> idx -1 (outage)`, `8.0 -> idx 15`, `8.0001 -> outage` |
| `softdemod_folded_k(radius=0) == softdemod_llr_k` (the `LLR_CACHE_MAX` fallback is the same code, not a second implementation) | `datapath_bench` column `r0` | **0.0e+00** on all 5 cases, in the default *and* the `-DLLR_CACHE_MAX=4` build |
| The LLR does not depend on the work partition `grp` | device probe, `grp = 1..8` and `grp = 64` on the folded kernel | **bit-exact** (0.000e+00) for every `grp` |
| Registry: no kernel registered-but-uncalled or called-but-unregistered | `tools/check_registry.py` | `OK: 29 kernels in 10 sources; registry, xclbin and call sites all agree` |

### The QPSK `log 2` — a model property, not a bug

At m = 2 with radius 2 the folded LLR differs from the parallel one by exactly
`ln 2 = 0.693`. Cause, verified in fp64: the replica period is `2*delta`, so a
shift by `-2*delta` places a replica exactly `delta` from a base point, and the
flipped-bit subset gains a **second co-minimal candidate** — `S_1 = 2.0002`
instead of `1.00005`, hence `log S_1 - log S_0 = log 2`. Both implementations
compute this from the same lattice and agree; it is production's replica model
behaving as written, and it is why the only exact identity claimed anywhere is
`radius == 0`. `datapath_bench` case4 (m = 2, radius 2) and `modem_bench` cases
0-2 exercise it.

---

## 2. Bugs found and fixed

### 2.1 `grp <= 0` silently produced a zero LLR (device-measured)

`Runtime::dp_dims_ok` validates `m` and `nsymtot` but never `grp`, and
`dp_bit_groups` maps any `grp <= 0` to a single bit group. The kernel's bit range
`[g*grp, min(g*grp+grp, m))` is then **empty**, so the loop body never ran and
nothing was written. The output buffer is created `CL_MEM_WRITE_ONLY` from a
`nullptr` host pointer, i.e. uninitialised device memory, and this device returns
zeros for it:

```
softdemod_llr grp=2  -> llr[0..3] = 3.938  4.09839  -2.78819  5.27698
softdemod_llr grp=0  -> llr[0..3] =     0        0         0        0   (before)
softdemod_llr grp=-1 -> llr[0..3] =     0        0         0        0   (before)
```

`LLR = 0` is not an error a caller can see: it reads as "no information", gives
`GMI = 0`, and `mcs_select_k` turns that into an *outage* — a plausible answer
for a launch that never happened. The same hole covered `radius < 0` on the
folded kernel (both replica sweeps empty).

Fix: both demodulators now sentinel-fill on `M > MMAX || grp < 1` (folded: also
`radius < 0`), matching the loud-NaN policy the rest of the runtime uses.
Measured after: `llr = nan -> gmi = nan -> mcs idx = -1, outage = 1`.

Bit-exactness: the guard cannot fire for `grp >= 1`, and it was verified anyway —
`datapath_bench` (5 cases), `datapath_pl_bench` (`-DLLR_CACHE_MAX=4`) and
`chain_bench` (3 cases) are **byte-identical** before and after, on the device.

The host half of this (a `grp >= 1` test in `dp_dims_ok`, so the PS also prints to
stderr) is outside this slice — see §4.

### 2.2 Wrong vector file: 18 s of thrashing and `SIGKILL`, no message

All five benches take their vector file as `argv[1]`, and the case header sizes
every allocation. Feeding `rate_vectors.bin` to `datapath_bench` read `K = 15` as
`Nt` and a float64 mantissa as `U/d/nsym/m`, then asked for gigabytes:

```
before: exit=137 (SIGKILL), wall 18.3 s, no output at all
after : exit=2,  wall 0.04 s
        [datapath] case0: implausible header Nt=15 Nr=0 NRF=0 U=1042022400
        d=1047527424 nsym=1052835840 m=1058668544 ... -- wrong vector file?
        (run: python3 tools/datapath_ref.py)
```

`include/hpocl_vecio.h` documents this exact hazard as fixed, but it only
bounds-checks each **read**, which happens after the allocation. The fix is a
pre-flight in each main: sanity-bound the header, then ask
`VecReader::take(need, ...)` — which validates without consuming — whether the
file still holds this case's bytes.

`need` is computed from a **named table of per-field element counts that also
sizes the vectors**, so the count cannot drift from the allocations. It took two
attempts to get right (chain over-counted by 256 B, datapath by 4096 B = one
`nsymtot` term), which is the argument for the single table: the correct total
equals each case's on-disk size *exactly*, and `expect_done()` at the end of the
run independently confirms zero trailing bytes. A miscount therefore cannot hide.

### 2.3 Three copy-pasted `struct Reader`

`runtime/{datapath,rate,chain}_main.cpp` each carried a private 20-line
little-endian reader — the very struct `include/hpocl_vecio.h` says it was
extracted *from* (`"runtime/datapath_main.cpp already had the correct version;
this is it, shared"`). All three now use `hpocl::VecReader`, which also gives
them the offset in the error message and `expect_done()` contract-drift
detection (datapath and chain previously compared `rd.o != rd.buf.size()` by
hand). ~60 lines and three now-unneeded includes (`<fstream>`, `<cstring>`,
`<string>`) removed. Output of all three benches: **byte-identical**.

`rate_vectors.bin` also carries `float32[]` and `int32[]` columns, which
`VecReader` has no named accessor for; those go through its public `raw()` via
two 3-line local helpers rather than a second reader (§4 asks for the members).

### 2.4 Dead code

* `runtime/datapath_main.cpp`: `const int ng = (m + grp - 1) / grp;` computed,
  never used, then `(void)ng;` to silence the warning. Removed.
* `tools/rate_ref.py`: `j = np.arange(1, 16)` … `_ = j`. Removed.
* `kernels/datapath.cl`: `#define fold_axis(x, period) cmod_axis((x), (period))`
  — a macro whose only purpose was to rename a function that is already spelled
  clearly at its one call site. Removed (identical expansion).
* `kernels/datapath.cl`: the header said "1 KB of private memory per work-item at
  M = 256" twice, in two adjacent paragraphs. Merged.
* `runtime/chain_main.cpp`: the folded replica radius was a bare `2` at the call
  site. Now `const int kRadius = 2` with its provenance
  (`enumerate_thp_symbol_replicas`, must equal the oracle's `RADIUS`).

### 2.5 MMSE-UCD was excluded from the device chain — and from the oracle's production cross-check

Two related gaps, both closed:

* **Kernel side.** `chain_main` skipped the whole fp32 chain for UCD
  (`if (!ucd)`), printing zeros for eight columns, on the stated ground of "no
  scalar-n0 kernel path". That is not needed: the demodulator takes a *pointer*,
  so launching it once per stream with that stream's own floor is the same kernel
  and the same math the fp64 branch already used. UCD also needs
  `rx_combine(..., diag_b = nullptr)`, because `w_ucd` has unit **gain** (which
  is why z is not divided) but not unit **norm** — measured
  `||Q_k col j||^2 = 0.0853..0.0931` on the UCD case, i.e. the noise floor is
  `n0*||Q_j||^2`, an order of magnitude below `n0`. The old comment
  ("w_ucd is pre-normalised") read as unit-norm and was misleading.
  Result, case2, with the **pre-existing** tolerances (nothing was widened):
  `cl sf=8e-08 x=2e-07 y=2e-06 z=6e-07 zf=7e-07 llr=2e-04 gmi=2e-07 ber=0e+00
  thp32=0e+00 PASS`. `thp32` (the fp32 feedback kernel vs the fp64 host
  reference) is now gated on UCD too.
* **Oracle side.** `chain_ref.py` ran its production-GMI cross-check under
  `if variant == 0:`, so the UCD branch's per-stream floor was only ever compared
  against our own two implementations — self-consistent, not independent. It now
  calls `estimate_bicm_gmi_thp_from_received` **per stream** with
  `snr = 1/(n0*qn[k,j])` (production takes one scalar snr per call, which is
  exactly why this branch needs T calls). Negative-tested: with a scalar floor
  the assert fires immediately, `production 5.1791 vs ours 5.9951`.
* `qn` was computed independently in the oracle and in the bench; both sides now
  compute it once and share it between the LLR stage and the cross-check.

### 2.6 QPSK was supported by production and by the kernels, but not by the C++ reference or the benches

`ref/modem.cpp::qam_bits_per_symbol` returned -1 for `M = 4`, so the fp64 side of
`chain_bench` (the thing the kernels are compared against) could not serve the
smallest constellation, and `modem_ref.py` covered only `{16, 64, 256}` — while
production's `get_constellation` accepts `bits_per_symbol = 2`, `dp_dims_ok`
accepts `m = 2`, and `datapath.cl`'s own `LLR_CACHE_MAX` bound is documented as
"4 is QPSK". Added: `M = 4` in the reference, `ORDERS = (4, 16, 64, 256)` in the
oracle (**12 cases**, was 9; QPSK reports `per/sym/fold/lab = 0.0e+00`, GMI error
`0.0e+00`), and a QPSK case in `datapath_ref.py` (**5 cases**, was 4) which runs
the smallest constellation through the whole device chain:
`qam=1.7e-08 tx=1.4e-07 ch=5.1e-07 rx=7.1e-07 fold=4.4e-07 llr=8.9e-06
llrF=4.1e-06 r0=0.0e+00 gmi=1.1e-07 gmiF=3.1e-07 mcs=0 PASS`.

`check_production` also skipped m = 2 in its **replica-lattice** check, and it
had to: production lists its QPSK table in Gray-label order
(`-1-1j, -1+1j, +1+1j, +1-1j`) while this port lists every order in level-index
order, so an element-wise comparison is impossible at m = 2 even though the
label -> point pairing is identical. It now compares the lattice **order-free**
(new `labeled_points()`: label tuple -> sorted points) for every m, keeping the
strict element-wise form where the row orders do agree. Negative-tested twice:
swapping two QPSK labels is caught, and dropping the outermost replica ring is
caught by the new multiplicity assert.

### 2.7 Units in a report line

`chain_ref.py` printed `GMI=5.9888/64` — bits per symbol over the constellation
*size*. Now `/6 bits/symbol`.

---

## 3. Found, **not** applied (numerics-changing) — `docs/gmi_kahan.patch`

Two patches are handed over rather than applied, because both change device
numerics and the slice's rule is that a numerics change gets gated by whoever
owns `make check`. Each is a `patch -p0` diff against `kernels/datapath.cl`, each
was built and measured on this device before being written out, and each
reproduces the measured file byte for byte (`patch --dry-run` checked).

### `gmi_k` loses precision linearly in block length

`gmi_k` runs one work-item per stream and accumulates `nsym*m` softplus terms
into a single fp32 `acc`. `acc` grows to `nsym*m*ln2` — `3.6e5` at
`nsym = 65536, m = 8`, whose ulp (`0.031`) is 4% of one term — and because `acc`
only grows, the loss is systematic, not a random walk. Measured on the M4, same
fp32 LLRs summed by the kernel and in fp64:

| `nsym` | 64 | 1024 | 8192 | 20000 | 65536 |
|---|---|---|---|---|---|
| plain `\|ΔGMI\|` | 2.3e-6 | 2.8e-5 | 7.2e-4 | **3.2e-3** | **2.3e-2** |
| Kahan `\|ΔGMI\|` | 2.5e-7 | 2.4e-7 | 3.1e-7 | 1.0e-7 | 3.6e-7 |

(worst over `-20/0/+10 dB`, `m = 8`.) At `nsym = 20000` — the block length the
README quotes for its low-SNR GMI figures — the plain sum is off by 3.2e-3 bits,
**300x the `1e-5` selection slack of `mcs_table.clh`**, and at `-20 dB` it
reported `+0.0337` where fp64 says `+0.0107`: a 3x error, in the direction that
turns an outage-level GMI into a positive one, i.e. into a scheduled rate.

`docs/gmi_kahan.patch` (`patch -p0 < docs/gmi_kahan.patch`) replaces the
accumulation with Kahan's 4-flop compensated sum. Kahan rather than Neumaier
because every summand is a softplus, hence `>= 0`: `acc` is monotone, so
`|acc| >= |y|` always holds and Neumaier's extra branch (a comparison plus two
selects per bit on the fabric) covers a case that cannot arise. Both forms were
measured **bit-identical** here.

Pre-gated on this device, so it can be applied as-is:
`datapath_bench` 5 cases ALL PASS (`gmi`/`gmiF` unchanged to 1e-7),
`datapath_pl_bench` (`-DLLR_CACHE_MAX=4`) ALL PASS,
`chain_bench` 3 cases ALL PASS (`gmi_cl` 5e-07 / 1e-07 / 2e-07, from
1e-06 / 2e-07 / 2e-07). `patch --dry-run` reproduces the validated file exactly.

The remaining structural point is separate and not patched: one work-item per
stream over 65536 symbols is a long *serial* chain, which is what a PL build
least wants. The proper fix is a two-stage reduction (partial sums per
(stream, chunk), then a reduce), which needs a new kernel plus a Runtime wrapper.

---

## 4. Proposed, deliberately **not** done here

1. **`dp_dims_ok` should reject `grp < 1`** (`runtime/hpocl_runtime.cpp`, outside
   this slice). The kernel now refuses loudly with NaN, but only the host can
   print to stderr, and the host is where every other shape check lives. A
   defaulted parameter keeps the two callers that have no `grp` (`qam_map`, `gmi`)
   untouched:
   ```diff
   -bool dp_dims_ok(const char* who, int m, int nsymtot) {
   -  if (m < 2 || (m & 1) || m > 8 || nsymtot < 1) {
   -    fprintf(stderr, "[hpocl] %s: bad shape m=%d nsymtot=%d (m must be even, "
   -            "2..8; the kernels cap the constellation at MMAX=256)\n",
   -            who, m, nsymtot);
   +// grp is the soft demodulators' bits-per-work-item partition; grp < 1 makes
   +// the kernel's bit range [g*grp, min(g*grp+grp, m)) empty, so it is a shape
   +// fault like any other (the kernel now NaN-fills; see kernels/datapath.cl).
   +bool dp_dims_ok(const char* who, int m, int nsymtot, int grp = 1) {
   +  if (m < 2 || (m & 1) || m > 8 || nsymtot < 1 || grp < 1) {
   +    fprintf(stderr, "[hpocl] %s: bad shape m=%d nsymtot=%d grp=%d (m even and "
   +            "2..8, the kernels cap the constellation at MMAX=256; grp >= 1)\n",
   +            who, m, nsymtot, grp);
         return false;
   ```
   and the two demodulator wrappers pass it through:
   ```diff
   -  if (!dp_dims_ok("softdemod_llr", m, nsymtot)) return;
   +  if (!dp_dims_ok("softdemod_llr", m, nsymtot, grp)) return;
   -  if (!dp_dims_ok("softdemod_folded", m, nsymtot)) return;
   +  if (!dp_dims_ok("softdemod_folded", m, nsymtot, grp)) return;
   ```
   `dp_bit_groups` already special-cases `grp <= 0` (it substitutes `m`), which
   is the smell that hid this: the launch geometry was repaired but the kernel
   argument was not.
2. **`VecReader` should gain `f32()`/`i32()` members** (`include/hpocl_vecio.h`,
   outside this slice) so `rate_main.cpp` can drop its two `raw()` wrappers.
3. **`err_c`/`err_f`/`err_r` are still written three times** (`datapath_main`,
   `chain_main`, `modem_main`). They want a shared `include/hpocl_benchcmp.h`;
   new files outside `docs/` were out of scope for this pass.
4. **`float n0` -> `float[K] n0` in both demodulators** would let UCD run as one
   launch instead of T, and is the shape a deployment wants (a real link has a
   per-stream noise floor). It changes two kernel signatures, two Runtime
   wrappers, the registry's argument marshalling and the `.xclbin` argument list,
   so it belongs with the runtime owner. Today's per-stream launches are correct
   and measured (§2.5); T <= 8, so the cost is not a reason to hurry.
5. **`max_log` mode: implemented, measured, and recommended *against*.**
   Production keeps a `max_log=True` branch ("an explicit approximation retained
   for large candidate sets") that this stack never built. It is attractive for
   the PL — the max-log LLR is literally the first term of the stabilised form,
   so a build that drops the correction needs **no `exp`, no `log`, one sweep
   instead of two, and no candidate array** (which also removes the 1 KB/work-item
   `LLR_CACHE_MAX` problem). A sandbox switch was built and validated: with
   `HPOCL_MAX_LOG=0` the kernel is byte-identical to today's, and with `=1` it
   matches an fp64 max-log reference to `1.4e-06 .. 2.6e-06` for M = 16..256.
   The accuracy cost, however, is not a rounding-level effect — it moves the
   *decision*:

   | M | SNR | GMI exact | GMI max-log | CQI exact | CQI max-log |
   |---|---|---|---|---|---|
   | 16 | 0 dB | 0.57517 | 0.99061 | 3 | **5** |
   | 64 | 12 dB | 4.61343 | 4.47315 | 13 | **12** |
   | 256 | 0 dB | **-0.16687 (outage)** | +0.35827 | **-1** | **2** |
   | 256 | 6 dB | 1.14050 | 1.34230 | 5 | **6** |

   4 of 16 operating points pick a different MCS, and the 256-QAM/0 dB row is the
   worst kind of error: the exact metric says the link carries nothing, max-log
   schedules QPSK at R = 78/1024. Since `gmi_k`'s output *is* the rate adapter's
   input, adopting max-log would need its own GMI-level gate and a back-off
   margin, not just an LLR tolerance. Left out; the measured switch is in
   `docs/maxlog_mode.patch` if the fabric ever forces the trade (`patch -p0`;
   inert unless `-DHPOCL_MAX_LOG=1` is passed, verified by re-running
   `datapath_bench` with the patch applied and no flag: ALL PASS). Both patches
   in this document apply cleanly in sequence and the combined file was re-gated:
   `datapath_bench` 5 cases ALL PASS, `gmi` error back to <=3.6e-7.
6. **No kernel-side odd-`m` guard, on purpose.** `label_bit`'s
   `v >> (mh - 1 - pos)` is only defined for even `m`, so an odd `m` would be UB
   rather than a wrong answer. `dp_dims_ok` already rejects it loudly and
   `tools/check_capacity.py` pins `MMAX` against that host bound; restating "8" in
   the kernel would create a constant no gate watches. The kernel header now
   states the contract instead. If a kernel-side guard is wanted, `MMAX_LOG2`
   must be added to `check_capacity.py` in the same commit.
7. **No per-byte validation of `bits[]` in `qam_map_k`.** Production raises a
   `KeyError` for a label that is not in the table; the kernel folds a non-binary
   byte into the level index and emits an off-constellation point. A complete
   guard costs `m` compares per symbol in the mapper's hot path (a partial
   `iI >= L` check catches some cases only). Both oracles assert binary bits and
   the PL is fed by a hard-decision decoder, so the contract is documented in the
   kernel header rather than paid for.

---

## 5. Why the constellation formula stays written twice

`kernels/datapath.cl` derives amplitudes analytically (`gray_decode`,
`pam_amp`, `qam_delta`) while `ref/modem.cpp` builds an explicit table
(`Qam`, `gray_label`, `pam_from_gray`). This looks like the duplication the
`.clh` pattern (`mcs_table.clh`, `switch_law.clh`) exists to remove, and it was
checked rather than assumed: **unifying it would weaken the validation.**

* `rate_bench`'s own header states the principle — "the host and the device
  compile the SAME header text, so the host column is not an independent
  implementation". For the MCS table that is acceptable because the *numpy*
  column is independent (the table is retyped from TS 38.214).
* For the modem, the independence lives in `ref/modem.cpp`: `chain_bench` calls
  both `ref/modem.cpp` (fp64) and the kernels (fp32) on the same control plane
  and compares them at `2e-3`/`3e-3`. That comparison is the only place where the
  device modem is checked against a second C++ implementation. Sharing one header
  would turn it into a tautology.
* The two are also not the same computation: fp32 analytic vs fp64 table, and the
  spacing is spelled `sqrt(6/(M-1))` on the device and `2/sqrt(2(L^2-1)/3)` in
  the reference (algebraically equal, ~1 ulp apart in fp64, invisible at the
  `1e-12` tolerance `modem_bench` uses). Both are pinned to production
  independently.

So: three implementations of the modem math, each checked against production by a
different route (`modem_ref.py` calls production directly; `modem_common.py`
asserts its own formulas against production on every run; `chain_ref.py` asserts
the GMI against production's estimators). That is the design, and it is now
documented here rather than looking like an oversight.

---

## 6. Gates for this slice

```sh
make modem_bench                                   # host only, 12 cases
make datapath_bench                                # 5 cases, m = 2..8
make datapath_pl_bench                             # same, -DLLR_CACHE_MAX=4
make rate_bench                                    # 10 cases, all columns exact
make chain_bench                                   # 3 cases, all three strategies
patch -p0 < docs/gmi_kahan.patch && make datapath_bench && make datapath_pl_bench \
  && make rate_bench && make chain_bench           # only if §3 is accepted
```

## 7. README rows this work invalidates (README is owned elsewhere)

| line | says | now measures |
|---|---|---|
| 104 | `LLR<=4.3e-14` | `LLR<=7.1e-14`, `GMI<=1.8e-15`, **12 cases** (QPSK added) |
| 105 | `(4 cases, m = 4..8)` … `rx<=5.8e-07` | `(5 cases, m = 2..8)` … `rx<=7.1e-07` (new QPSK case); every other column unchanged |
| 106 | `on all 4 cases` | `on all 5 cases` |
| 109 | `sum SE 10.70 / 27.12 / 3.39 / 7.22 bpcu over the 4 cases` | `… / 7.22 / 6.92 bpcu over the 5 cases` |
| 110 | `<=2e-06 (llr 1e-05), thp32 = 0.0e+00` | `<=2e-04 (llr, the UCD case)`, `thp32 = 0.0e+00`, and **all three strategies now run on the device** — the "UCD is C++/oracle only" caveat is gone |
