# fp64 kernel path: first real execution, and what a second architecture proved

Until this run, README's "fp64 kernel path" row said **"still never executed
anywhere"**, and that was true: the dev host is an Apple M4 whose
`CL_DEVICE_DOUBLE_FP_CONFIG` is `0`, so `Runtime::init` sets `dev_fp64_ = false`
and every gate compiled `-DHPOCL_FP32`. Apple's `clang -x cl` also rejects
`double` outright (it ignores the `cl_khr_fp64` pragma), so the fp64 branch of
`kernels/threshold_cf.cl` had never been through a real OpenCL compiler at all.
The dev host is arm64 macOS 26.6 with no `brew`, no `cmake` and no
`llvm-config`, so PoCL is not buildable there.

## Host and device used

`172.31.12.238` — the personal container `docs/QX49DR_SV3_上板依赖清单.md` §2
already lists as a candidate Linux host. x86_64, Ubuntu 20.04.6, 48 cores,
`/etc/OpenCL/vendors/nvidia.icd` and `/usr/bin/clinfo` already present.

| property | value |
|---|---|
| device | Tesla P40, `CL_DEVICE_TYPE_GPU`, NVIDIA OpenCL 3.0, driver 550.127.08 |
| `CL_DEVICE_DOUBLE_FP_CONFIG` | **`0x3`** — fp64 kernels can be created |
| `CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE` | **65 536 B — exactly the spec minimum** |
| `CL_DEVICE_LOCAL_MEM_SIZE` | 32 768 B (same as the Apple M4) |
| `CL_DEVICE_MAX_MEM_ALLOC_SIZE` | 2 952 790 016 B |
| PoCL | **not needed and not installed** — the existing ICD already had fp64 |

There is no `sudo` on that host, and it was missing `python3-scipy` and
`opencl-headers`. Everything was done in `$HOME/hpocl` with `pip install --user`
plus a shim include dir; nothing system-wide was touched.

## Result 1 — the fp64 branch is real code

`cf_bench` with `HPOCL_CLOPTS=-DHPOCL_FP64`, i.e. the three-way gate numpy vs
fp64 C++ vs **fp64 kernel**:

* all 7 cases PASS, `tau = 0.00e+00` on every one;
* `cl = 7.61e-16 dB` against the fp64 gate of `1e-6 dB` — **1.3e9x margin**;
* **39 of 40 `tau` values are bit-identical between the fp64 and the fp32
  kernels**, the 40th differing in the last bit.

So the fp64 path compiles, runs and agrees with both the fp64 C++ reference and
the fp32 kernel. It is still not the *shipped* configuration — the PL build is
fp32, because the ZU49DR's DSP48E2 has no floating point at all.

## Result 2 — the `__constant` budget is a hard blocker for fp64, and it is now gated

The fp64 run **failed first**, and the failure is the most useful thing here:

```
clCreateKernel -> CL_INVALID_BUFFER_SIZE (-61)
```

The program *builds*; the constant pool cannot be allocated. The arithmetic, now
computed by `allprog_check` from the tables the build actually selected
(`deficit_table.clh` / `gh_table.clh` pick their block on `#ifndef HPOCL_FP32`,
and `threshold_cf.cl` defines `HPOCL_FP32` unless `-DHPOCL_FP64` is passed):

| build | deficit | Gauss-Hermite | CQI | total | vs 65 536 B |
|---|---|---|---|---|---|
| fp32 (PL default) | 64 000 B | 384 B | 128 B | **64 512 B** | fits, **1 024 B (1.6%) headroom** |
| fp64 | 128 000 B | 768 B | 128 B | **128 896 B** | **1.97x over → -61** |

This was invisible on the dev host because Apple reports 1 GB for
`CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE`. `allprog_check` now prints the breakdown,
fails on overflow, warns under 5% headroom, and — importantly — prints the
diagnosis **before** the missing-kernel bail-out, because this failure presents
as 29 missing kernels and hides its own cause. Whether XRT on the ZU49DR reports
64 KB is still unmeasured; if it does, the fp32 build fits with 1.6% to spare and
any new `__constant` table breaks it. Mitigations are one generator call each:
`tools/gen_deficit_table.py --N 800` (31.25 KB, 8.4e-5 dB interpolation error),
`--ms 64,256` (25 KB), `--ms 64` (12.5 KB), then `make deficit_check`.

## Result 3 — the library is not Apple-GPU-specific

The full suite was re-run on x86_64 Linux + a different vendor ICD + GCC 9.4
(vs Apple M4 + Apple ICD + clang):

* **`decomp_bench`: 48/48 PASS, 156 gate lines compared, ZERO numeric drift** —
  every `cl=` / `cli=` value identical to 3 significant figures;
* fp32 Jacobi off-diagonal floor reproduced at **2.9e-9** (Apple measured
  1.4e-8..3.4e-8), so `tol_exit = 1e-7` keeps ~34x margin on a second
  architecture;
* the `tol_exit` cap-invariance re-verified: `cap=8` vs `cap=16` outputs
  bit-identical, 1200.9 vs 1196.4 µs;
* `sched_bench` 11/11 in both costed and forced modes, same worst fields as on
  the M4 (`tau` 5.7e-4 dB, `sigma` 1.3e-5, `F_RF` 0.0, `F_RF F_BB` 5.8e-5);
* `datapath`/`rate`/`chain`/`op` all green.

Only fp64-precision columns moved (1e-16 → 1e-7), which is expected: those
compare against an fp64 oracle through an fp32 kernel.

## One reported bug that did NOT survive checking

The remote run reported that `runtime/sched_main.cpp::physical()` indexes `Nk`
with the wrong stride, making the `P` column garbage and the gate blind. **That
is wrong**, and it is recorded here so the claim is not resurrected:

* the expression is `((size_t)k * c.NRF + t) * o.nul + q`, which expands to
  exactly `k*NRF*nul + t*nul + q` — the correct row-major index into
  `Nk [K][NRF x nul]`;
* it is **byte-identical to `runtime/pipeline_main.cpp:75`**, whose `P` column is
  gated against production python's `Pk = Frf @ N @ vd` at `1e-9` and passes at
  `5.1e-12`. A wrong stride would fail that gate at O(1);
* measured directly: `||F_RF F_BB||_F = 5.656854 = sqrt(T)` exactly (the `pscale`
  identity), `||P||_F = 4.4976`, ratio `0.795` — which is the whitener
  `C^{-1/2}` that `P` legitimately omits, not an indexing error.

A gate can still be *weak* without being wrong, and the fair version of the
concern is that `P` is compared only between the two roles of the same code. That
is what `pipeline_bench`'s production-python gate covers, from the other end.

## Reproduce

```sh
rsync -a --exclude build hpocl/ 172.31.12.238:hpocl/          # + mix_precode/ alongside
ssh 172.31.12.238
pip3 install --user numpy scipy                               # scipy was missing
cd hpocl && make check DEV=gpu                                # host-only benches + device benches
HPOCL_CLOPTS=-DHPOCL_FP64 ./build/cf_bench build/cf_vectors.bin gpu   # the fp64 path
./build/allprog_check gpu                                     # constant-budget pre-flight
```

## What this does NOT prove

No Vitis and no `.xclbin` were involved: this is an OpenCL **source-JIT** run on
an NVIDIA ICD, not XRT on a Zynq. Kernel acquisition, argument marshalling and
the numerics are now validated on two architectures and two vendors; the
`v++ -l` link, the platform, and the board's own constant-buffer limit are still
open, and still blocked on the vendor `.xsa` (see
`docs/QX49DR_SV3_上板依赖清单.md` §8).

Incidental: `gcc -Wall -Wextra` on the host sources is clean except for 28 lines
of unused-parameter / sign-compare warnings in `runtime/cf_main.cpp`,
`chain_main.cpp`, `transformer_predictor.cpp`, `ref/rx.cpp`, `ref/hp_ref.cpp`,
`frontend.cpp` and `tools/`-facing mains. Clang on the dev host does not emit
them. Cosmetic, not gated.
