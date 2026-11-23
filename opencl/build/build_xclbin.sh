#!/usr/bin/env bash
# HPOCL -> Zynq PL: compile the very same kernel sources into an xclbin for
# the QX_49DR_SV3 (XCZU49DR-2FFVF1760) embedded platform.
#
# Prereqs (one-time, on the Vitis host):
#   1. Vivado block design for the board: PS + PL DDR4 banks + RFDC + 100G
#      CMAC as needed; export .xsa.
#   2. Vitis platform:  v++ -t hw --platform -f ... or use the vendor BSP if
#      the board supplier ships one; result: platform.xsa/platform.xpfm.
#
# Usage: PLATFORM=/path/platform.xpfm ./build_xclbin.sh [hw|hw_emu]
set -euo pipefail
TARGET=${1:-hw}
PLATFORM=${PLATFORM:?set PLATFORM=/path/to/platform.xpfm}
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
OUT=$ROOT/build/xclbin_$TARGET
mkdir -p "$OUT"

# fp32 datapath for PL (no fp64 on fabric). -DHPOCL_FP32 is now REDUNDANT for
# threshold_cf.cl, which is fp32 BY DEFAULT and takes -DHPOCL_FP64 to opt into
# double; the flag is kept here because it is explicit, because README.md and
# docs/QX49DR_SV3_上板依赖清单.md both quote this exact COMMON string, and because
# dropping it would silently change the build for anyone who re-adds an fp64
# default later. The inversion is the fix for the footgun this script used to
# have: the one `v++ -c -t hw` run this repo ever did (docs/vitis_flow_notes.md)
# did NOT pass -DHPOCL_FP32, so a forgotten flag used to mean "synthesise fp64 on
# a fabric whose DSP48E2 has no floating point at all". Now a forgotten flag means
# the deployable datapath. Verified: the new no-flag build is IR-BYTE-IDENTICAL to
# the old -DHPOCL_FP32 build (clang -emit-llvm, same md5), and the precision block
# sits ABOVE the two #includes because gh_table.clh and deficit_table.clh select
# their fp64/fp32 constant block on `#ifndef HPOCL_FP32` -- a fp32 `real` with
# that macro undefined compiles fine and silently pulls in 125.0 KB of fp64
# __constant instead of 62.5 KB. threshold_cf.cl now defines it for them.
# NEWTON_ITERS: the *deployed* budget is 20. Do NOT lower it on the strength of
# the number this comment used to carry ("8 steps cost 0.087 dB on the cf_bench
# UCD config"): re-measured over a 390-channel population that deliberately
# includes the 203 near-flat channels with a finite root (flat_ratio < 0.05) that
# cf_bench SKIPS, 8 steps cost 7.46e-1 dB worst and 12 steps 5.06e-2 dB, against
# 1.99e-4 dB at 20. The ladder is not monotone (3 steps happened to land at
# 1.2e-11 dB, 5 steps at 3.98 dB), so it cannot be extrapolated -- see
# newton_iters in ref/threshold_cf.h and docs/exp_threshold_policy.md.
# Keep this in step with ref/threshold_cf.h and kernels/threshold_cf.cl.
# -I$ROOT/kernels is load-bearing, not cosmetic: threshold_cf.cl #includes THREE
# headers -- gh_table.clh, deficit_table.clh (the generated ln D grid, 10 rows x
# 1600 points, 5 constellations x 2 branches) and psi_scan.clh (the near-flat
# fallback's range, grid size and scan loop, shared verbatim with
# ref/threshold_cf.cpp and tools/cf_ref.py). psi_scan.clh adds no __constant data:
# it is two untyped literals, one integer and a macro, so the budget below is
# unchanged by it.
# DEPLOYMENT RISK, check it before anything else on the board: the table lands in
# __constant memory. MEASURED totals (make allprog_check prints this per build):
#   fp32 (what this script builds)  64 000 + 384 (GH) + 128 (CQI) = 64 512 B
#                                   = 98.4% of the 64 KB spec minimum, 1 024 B spare
#   fp64 (-DHPOCL_FP64)            128 000 + 768 + 128 = 128 896 B = 1.97x OVER
# A real device reporting the spec minimum refuses the fp64 build with
# CL_INVALID_BUFFER_SIZE (-61) at clCreateKernel -- measured on a Tesla P40
# (65 536 B), see docs/x86_opencl_notes.md. The fp32 build fits, with 1.6%
# headroom, so ANY new __constant table needs a budget check first. If XRT
# reports 64 KB the fp32 build still fits and the fp64 one cannot be linked. Mitigations, each one generator invocation plus
# `make deficit_check`: `--N 800` (31.25 KB, 8.4e-5 dB interpolation error),
# `--ms 64,256` (25 KB, other M fall back to GH-48 quadrature), `--ms 64`
# (12.5 KB). BRAM cost of the shipped grid: 500 Kibit = 14 BRAM36 by capacity,
# up to 20 with depth granularity, or 2 URAM288 (capacity arithmetic, NOT a
# synthesis report).
# -DHPOCL_DEFICIT_TABLE=0 restores the pre-table GH-48 quadrature path on BOTH
# sides; runtime/cf_main.cpp propagates the host setting to the device build so
# the two cannot silently disagree (they did once, and it cost 9.6e-2 dB of
# host-vs-kernel tau with every host-side column looking perfect).
COMMON="-DHPOCL_FP32 -DNEWTON_ITERS=20 -I$ROOT/kernels"

# control plane: closed form + safeguarded Newton (per-channel work-item)
v++ -c -t "$TARGET" --platform "$PLATFORM" $COMMON \
    -k threshold_cf -o "$OUT/threshold_cf.xo"  "$ROOT/kernels/threshold_cf.cl"
# iterative eigensolver: dominant singular pair + deflation (local size == M <= PMAX=64)
v++ -c -t "$TARGET" --platform "$PLATFORM" $COMMON \
    -k power_iter -k deflate \
    -o "$OUT/powit.xo"        "$ROOT/kernels/power_iter.cl"
# per-user decomposition: one-sided Jacobi SVD, GMD, UCD, control-block diag
v++ -c -t "$TARGET" --platform "$PLATFORM" $COMMON \
    -k jacobi_svd -k gmd_k -k ucd_k -k cbd_k \
    -o "$OUT/decomp.xo"       "$ROOT/kernels/decomp.cl"
# the decision operator (scalar tau_dB + mode from lambda_min / g_min)
v++ -c -t "$TARGET" --platform "$PLATFORM" $COMMON \
    -k switch_threshold -o "$OUT/swthr.xo"     "$ROOT/kernels/switch_threshold.cl"
# baseband precoder GEMV
v++ -c -t "$TARGET" --platform "$PLATFORM" $COMMON \
    -k wbb_apply    -o "$OUT/wbb.xo"         "$ROOT/kernels/wbb_apply.cl"
# THP feedback / modulo datapath
v++ -c -t "$TARGET" --platform "$PLATFORM" $COMMON \
    -k thp_precoder -o "$OUT/thp.xo"         "$ROOT/kernels/thp_precoder.cl"
# covariance accumulation
v++ -c -t "$TARGET" --platform "$PLATFORM" $COMMON \
    -k channel_cov  -o "$OUT/cov.xo"         "$ROOT/kernels/channel_cov.cl"
# per-symbol datapath: QAM map -> tx combine -> channel -> rx combine ->
# modulo fold -> soft demod (parallel + folded LLR) -> GMI
v++ -c -t "$TARGET" --platform "$PLATFORM" $COMMON \
    -k qam_map_k -k tx_combine_k -k channel_apply_k -k rx_combine_k \
    -k modulo_fold_k -k softdemod_llr_k -k softdemod_folded_k -k gmi_k \
    -o "$OUT/datapath.xo"     "$ROOT/kernels/datapath.cl"
# code-rate / MCS layer: per-stream GMI -> highest supportable (m, R) from the
# 3GPP TS 38.214 CQI table in kernels/mcs_table.clh (same text as the host)
v++ -c -t "$TARGET" --platform "$PLATFORM" $COMMON \
    -k mcs_select_k -o "$OUT/rate.xo"        "$ROOT/kernels/rate.cl"
# spectral set transformer (student predictor) inference operators
v++ -c -t "$TARGET" --platform "$PLATFORM" $COMMON \
    -k tf_dense -k tf_gelu -k tf_add_broadcast -k tf_mul_mask \
    -k tf_layernorm -k tf_attn -k tf_pool -k tf_sign_pack -k tf_add_elem \
    -o "$OUT/tf.xo" "$ROOT/kernels/tf_ops.cl"

# link: every .xo above (unique basenames, one .xo per source, and all 29
# kernel names are distinct across sources, so the wildcard cannot collide).
v++ -l -t "$TARGET" --platform "$PLATFORM" \
    "$OUT"/*.xo -o "$OUT/hpocl.xclbin" \
    --config "$ROOT/opencl/build/link.ini" 2>/dev/null || \
v++ -l -t "$TARGET" --platform "$PLATFORM" "$OUT"/*.xo -o "$OUT/hpocl.xclbin"

echo "xclbin: $OUT/hpocl.xclbin"
echo "deploy: scp to PS, then runtime init(\"acc\") picks the PL device via XRT."
