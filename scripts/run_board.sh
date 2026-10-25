#!/usr/bin/env bash
# On-board (Zynq PS) run: same host binaries, XRT picks the PL accelerator.
#   scp -r hpocl/ root@<ps-ip>:; ssh root@<ps-ip>
#   cd hpocl && make cf_ref && scripts/run_board.sh /lib/firmware/hpocl.xclbin
set -euo pipefail
XCLBIN=${1:?xclbin path on PS}
ROOT=$(cd "$(dirname "$0")/.." && pwd)
export HPOCL_XCLBIN="$XCLBIN"
export HPOCL_FP32=1
cd "$ROOT"
./build/cf_bench  build/cf_vectors.bin acc
./build/op_bench  acc
./build/tf_bench  build/tf_weights.bin acc
