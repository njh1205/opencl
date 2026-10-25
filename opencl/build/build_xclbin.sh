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

# fp32 datapath for PL (no fp64 on fabric). NEWTON_ITERS: the *deployed* budget
# is 20 -- re-measured after kTQ moved 10 -> 5; 8 steps cost 0.087 dB on the
# cf_bench UCD config (see newton_iters in ref/threshold_cf.h for the sweep).
# Keep this in step with ref/threshold_cf.h and kernels/threshold_cf.cl.
# -I$ROOT/kernels is load-bearing, not cosmetic: threshold_cf.cl #includes
# gh_table.clh AND deficit_table.clh, the generated ln D grid (10 rows x 1600
# points, 5 constellations x 2 branches).
# DEPLOYMENT RISK, check it before anything else on the board: the table lands in
# __constant memory, 62.5 KB of fp32 = 97.7% of the 64 KB OpenCL *spec minimum*
# for CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE. If XRT reports 64 KB the fp32
# clBuildProgram fails outright. Mitigations, each one generator invocation plus
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
