#!/usr/bin/env bash
# Xilinx hw_emu (cycle-approximate PL simulation) host run.
# On the Vitis machine:
#   1. PLATFORM=... opencl/build/build_xclbin.sh hw_emu
#   2. scripts/run_emu.sh build/xclbin_hw_emu/hpocl.xclbin
set -euo pipefail
XCLBIN=${1:?xclbin path}
ROOT=$(cd "$(dirname "$0")/.." && pwd)
DIR=$(cd "$(dirname "$XCLBIN")" && pwd)
if [ -n "${PLATFORM:-}" ]; then
  emconfigutil --platform "$PLATFORM" --od "$DIR" >/dev/null 2>&1 || true
fi
export XCL_EMULATION_MODE=hw_emu
export HPOCL_XCLBIN="$XCLBIN"
export HPOCL_FP32=1
cd "$ROOT"
echo "[emu] cf_bench";  ./build/cf_bench  build/cf_vectors.bin acc || true
echo "[emu] op_bench";  ./build/op_bench  acc || true
echo "[emu] tf_bench";  ./build/tf_bench  build/tf_weights.bin acc || true
