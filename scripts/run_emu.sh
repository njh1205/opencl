#!/usr/bin/env bash
# Xilinx hw_emu (cycle-approximate PL simulation) run of the whole device
# surface. Same bench list as run_board.sh, and for the same reason: an emu run
# that only covers three benches tells you nothing about the other 17 kernels.
#
# On the Vitis machine:
#   1. PLATFORM=/path/platform.xpfm opencl/build/build_xclbin.sh hw_emu
#   2. scripts/run_emu.sh build/xclbin_hw_emu/hpocl.xclbin
#
# hw_emu is the first place the xclbin path can be exercised at all without the
# vendor .xsa for the real board, so it is where the kernel-acquisition list and
# the argument marshalling get their first real test.
set -uo pipefail
XCLBIN=${1:?xclbin path}
DEV=${2:-acc}
ROOT=$(cd "$(dirname "$0")/.." && pwd)
DIR=$(cd "$(dirname "$XCLBIN")" && pwd)
if [ -n "${PLATFORM:-}" ]; then
  emconfigutil --platform "$PLATFORM" --od "$DIR" >/dev/null 2>&1 || true
fi
export XCL_EMULATION_MODE=hw_emu
export HPOCL_XCLBIN="$XCLBIN"
cd "$ROOT"
# `|| true` on each stage: hw_emu is slow and partly unsupported for some
# constructs, so a partial report is more useful than stopping at the first
# kernel. The summary at the end is what to read.
fail=0
run() { echo "[emu] $*"; "$@" || { echo "[emu]   ^^ FAILED"; fail=1; }; }
run ./build/allprog_check "$DEV"
run ./build/cf_bench build/cf_vectors.bin "$DEV"
run ./build/op_bench "$DEV"
run ./build/decomp_bench build/decomp_vectors.bin "$DEV"
run ./build/sched_bench "$DEV"
run ./build/datapath_bench build/datapath_vectors.bin "$DEV"
run ./build/rate_bench build/rate_vectors.bin "$DEV"
run ./build/chain_bench build/chain_vectors.bin "$DEV"
run ./build/tf_bench build/tf_weights.bin "$DEV"
echo
[ "$fail" -eq 0 ] && echo "[emu] all stages passed" || echo "[emu] FAILURES above"
exit "$fail"
