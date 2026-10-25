# HPOCL - heterogeneous hybrid-precoding operator stack
#
#   make help        : list every target
#   make check       : run all benches (each regenerates its own numpy oracle)
#
# Benches (each is a three-way or two-way cross-check; see README "Validation"):
#   cf_bench         closed-form threshold: numpy vs fp64 C++ vs fp32 OpenCL
#   op_bench         control-plane operators vs fp64 golden reference
#   op_pl_bench      the same, with PL build options (wbb_apply local-memory tile)
#   decomp_bench     svd/gmd/ucd: production python vs C++ vs fp32 kernel
#   frontend_bench   RF-BD front end vs production python
#   pipeline_bench   full per-scheduling pipeline vs production python
#   rx_bench         receive-side combiners vs production python
#   lmmse_bench      LMMSE shrinkage vs production python
#   modem_bench      receive modem (QAM map, LLR, GMI) vs production python
#   datapath_bench   fp32 PL datapath kernels vs numpy reference
#   datapath_pl_bench the same datapath with PL build options (LLR recompute path)
#   rate_bench       code-rate/MCS layer: numpy oracle vs host .clh vs fp32 kernel
#   chain_bench      end-to-end fp32 kernel chain vs numpy reference
#   tf_bench         transformer threshold predictor vs numpy reference
#   gh_check         Gauss-Hermite table: host header vs device header in sync
#   registry_check   every kernel is in the host registry, PL-linked, and called
#   capacity_check   shared constants (PMAX/MAXM/MAXD/KMAX/MMAX/TQ/...) in step
#   deficit_check    generated ln D table: host vs device, Lemma-4, tail handover
#   allprog_check    every kernel source builds as ONE program, all kernels acquired
#   asan_check       host-only benches rebuilt with ASan+UBSan and re-run
#
# Build integrity: -MMD -MP records header dependencies, and every target that
# JIT-loads a kernel lists that .cl as a prerequisite, so `make` cannot report
# a stale binary as up to date. DEV selects the OpenCL device (cpu|gpu|acc).
CXX ?= c++
UNAME := $(shell uname)
ifeq ($(UNAME),Darwin)
CL_LIB := -framework OpenCL
else
CL_LIB := -lOpenCL
endif
# -Ikernels lets the host reference share the device headers (switch_law.clh),
# so one definition of a formula is compiled into both sides.
INC  := -Iinclude -Iref -Iruntime -Ikernels
CXXF := -O2 -std=c++17 -Wall -Wextra $(INC)
# dependency generation is a *compile*-only flag: passing -MMD to the link step
# makes clang try to emit one .d per input object and fail.
DEPFLAGS := -MMD -MP
RT   := runtime/hpocl_runtime.cpp
PY   ?= python3
DEV  ?= cpu

# kernels JIT-loaded at runtime, per bench (keep in sync with the load_program /
# cl.init call in the corresponding runtime/*_main.cpp)
K_OP    := kernels/channel_cov.cl kernels/power_iter.cl kernels/switch_threshold.cl \
           kernels/wbb_apply.cl kernels/thp_precoder.cl
K_CF    := kernels/threshold_cf.cl kernels/gh_table.clh kernels/deficit_table.clh
K_DECOMP:= kernels/decomp.cl
K_PATH  := kernels/datapath.cl kernels/rate.cl kernels/mcs_table.clh
K_CHAIN := kernels/datapath.cl kernels/thp_precoder.cl
K_TF    := kernels/tf_ops.cl
K_RATE  := kernels/rate.cl kernels/mcs_table.clh

# Per-bench translation units. Named here once because two targets need the same
# list: the normal link rule and `asan_check`, which rebuilds them with
# sanitizers. A list typed twice is a list that drifts.
SRC_pipeline := ref/pipeline.cpp ref/frontend.cpp ref/decomp.cpp ref/threshold_cf.cpp ref/rx.cpp runtime/pipeline_main.cpp
SRC_rx       := ref/rx.cpp ref/decomp.cpp runtime/rx_main.cpp
SRC_frontend := ref/frontend.cpp ref/decomp.cpp runtime/frontend_main.cpp
SRC_modem    := ref/modem.cpp runtime/modem_main.cpp
SRC_lmmse    := ref/threshold_cf.cpp runtime/lmmse_main.cpp

# One .o per translation unit, so -MMD emits a *complete* .d per TU. (Compiling
# several .cpp in a single command writes only the first TU's headers to the .d,
# which silently defeats the whole point of dependency tracking.)
DEPS := $(shell find build -name '*.d' 2>/dev/null)
-include $(DEPS)

build:
	mkdir -p build

build/%.o: %.cpp
	@mkdir -p $(dir $@)
	$(CXX) $(CXXF) $(DEPFLAGS) -c $< -o $@

# $(1) = binary, $(2) = .cpp sources, $(3) = kernels JIT-loaded at runtime
# (prerequisites only -- they must never reach the link line), $(4) = link libs.
# $^ is filtered to %.o for the same reason.
define link_rule
build/$(1): $$(patsubst %.cpp,build/%.o,$(2)) $(3) | build
	$$(CXX) $$(CXXF) $$(filter %.o,$$^) -o $$@ $(4)
endef

$(eval $(call link_rule,cf_bench,ref/threshold_cf.cpp $(RT) runtime/cf_main.cpp,$(K_CF),$(CL_LIB)))
$(eval $(call link_rule,op_bench,ref/hp_ref.cpp $(RT) runtime/bench_main.cpp,$(K_OP),$(CL_LIB)))
$(eval $(call link_rule,pipeline_bench,$(SRC_pipeline),,))
$(eval $(call link_rule,rx_bench,$(SRC_rx),,))
$(eval $(call link_rule,lmmse_bench,$(SRC_lmmse),$(K_CF),))
$(eval $(call link_rule,chain_bench,ref/pipeline.cpp ref/frontend.cpp ref/decomp.cpp ref/threshold_cf.cpp ref/rx.cpp ref/modem.cpp ref/hp_ref.cpp $(RT) runtime/chain_main.cpp,$(K_CHAIN),$(CL_LIB)))
$(eval $(call link_rule,frontend_bench,$(SRC_frontend),,))
$(eval $(call link_rule,decomp_bench,ref/decomp.cpp ref/hp_ref.cpp $(RT) runtime/decomp_main.cpp,$(K_DECOMP),$(CL_LIB)))
$(eval $(call link_rule,tf_bench,$(RT) runtime/transformer_predictor.cpp runtime/tf_main.cpp,$(K_TF),$(CL_LIB)))
$(eval $(call link_rule,modem_bench,$(SRC_modem),,))
$(eval $(call link_rule,datapath_bench,$(RT) runtime/datapath_main.cpp,$(K_PATH),$(CL_LIB)))
$(eval $(call link_rule,rate_bench,$(RT) runtime/rate_main.cpp,$(K_RATE),$(CL_LIB)))
$(eval $(call link_rule,allprog_check,$(RT) runtime/allprog_main.cpp,$(K_ALL),$(CL_LIB)))

cf_ref: | build
	$(PY) tools/cf_ref.py

cf_bench: build/cf_bench cf_ref
	./build/cf_bench build/cf_vectors.bin $(DEV)

op_bench: build/op_bench
	./build/op_bench $(DEV)

# Same binary, PL build options: raising WBB_LOCAL_MAX switches wbb_apply from
# the direct path to its local-memory tile. The tile must stay bit-identical, so
# the rel_err this prints has to match the default op_bench run exactly.
op_pl_bench: build/op_bench
	HPOCL_CLOPTS=-DWBB_LOCAL_MAX=4096 ./build/op_bench $(DEV)

pipeline_bench: build/pipeline_bench
	$(PY) tools/pipeline_ref.py
	./build/pipeline_bench build/pipeline_vectors.bin

frontend_bench: build/frontend_bench
	$(PY) tools/frontend_ref.py
	./build/frontend_bench build/frontend_vectors.bin

decomp_bench: build/decomp_bench
	$(PY) tools/decomp_ref.py
	./build/decomp_bench build/decomp_vectors.bin $(DEV)

tf_bench: build/tf_bench
	$(PY) tools/make_random_tf_weights.py
	./build/tf_bench build/tf_weights.bin $(DEV)

modem_bench: build/modem_bench
	$(PY) tools/modem_ref.py
	./build/modem_bench build/modem_vectors.bin

datapath_bench: build/datapath_bench
	$(PY) tools/datapath_ref.py
	./build/datapath_bench build/datapath_vectors.bin $(DEV)

# Same binary, PL build options: -DLLR_CACHE_MAX=4 shrinks softdemod_llr_k's
# private candidate cache to 16 B and therefore routes every m >= 6 case through
# its recompute fallback. The fabric runs that path, so gate it -- a fallback
# nobody tests is where a second implementation goes to drift.
datapath_pl_bench: build/datapath_bench
	HPOCL_CLOPTS=-DLLR_CACHE_MAX=4 ./build/datapath_bench build/datapath_vectors.bin $(DEV)

rate_bench: build/rate_bench
	$(PY) tools/rate_ref.py
	./build/rate_bench build/rate_vectors.bin $(DEV)

rx_bench: build/rx_bench
	$(PY) tools/rx_ref.py
	./build/rx_bench build/rx_vectors.bin

lmmse_bench: build/lmmse_bench
	$(PY) tools/lmmse_ref.py
	./build/lmmse_bench build/lmmse_vectors.bin

chain_bench: build/chain_bench
	$(PY) tools/chain_ref.py
	./build/chain_bench build/chain_vectors.bin $(DEV)

# Whole-library build: every .cl into ONE program, every registered kernel
# acquired from it. Catches duplicate macros / duplicate non-static helpers /
# include-guard and extension-pragma problems that per-bench builds cannot see,
# because Runtime::build concatenates its sources into a single translation unit
# (and the PL link does the same across .xo files).
K_ALL := $(wildcard kernels/*.cl)
allprog_check: build/allprog_check
	./build/allprog_check $(DEV)
	HPOCL_CLOPTS=-DHPOCL_FP32 ./build/allprog_check $(DEV)

# Sanitizer pass over the host-only benches (the ones that need no OpenCL
# device). The fp64 references in ref/ are the oracles every kernel is judged
# against, so undefined behaviour there invalidates the comparison rather than
# merely the run -- and a plain `make check` cannot see it, because the same
# out-of-bounds read usually returns a plausible number.
# UBSan as well as ASan: the vector readers in runtime/*_main.cpp do raw
# pointer arithmetic over a byte buffer with sizes read from the file, which is
# exactly where an out-of-contract offset would hide.
SANF := -fsanitize=address,undefined -fno-omit-frame-pointer -g
SAN_BENCHES := pipeline_bench rx_bench frontend_bench modem_bench lmmse_bench
asan_check: | build
	@mkdir -p build/asan
	@set -e; for pair in \
	    "pipeline_bench:$(SRC_pipeline):pipeline_vectors.bin:$(PY) tools/pipeline_ref.py" \
	    "rx_bench:$(SRC_rx):rx_vectors.bin:$(PY) tools/rx_ref.py" \
	    "frontend_bench:$(SRC_frontend):frontend_vectors.bin:$(PY) tools/frontend_ref.py" \
	    "modem_bench:$(SRC_modem):modem_vectors.bin:$(PY) tools/modem_ref.py" \
	    "lmmse_bench:$(SRC_lmmse):lmmse_vectors.bin:$(PY) tools/lmmse_ref.py"; do \
	  b=$$(echo "$$pair" | cut -d: -f1); src=$$(echo "$$pair" | cut -d: -f2); \
	  vec=$$(echo "$$pair" | cut -d: -f3); gen=$$(echo "$$pair" | cut -d: -f4-); \
	  echo "[asan] building $$b"; \
	  $(CXX) $(CXXF) $(SANF) $$src -o build/asan/$$b; \
	  $$gen > /dev/null; \
	  echo "[asan] running $$b"; \
	  ./build/asan/$$b build/$$vec > /dev/null || { echo "[asan] $$b FAILED"; exit 1; }; \
	done
	@echo "[asan_check] OK: $(words $(SAN_BENCHES)) host benches clean under ASan+UBSan"

# The Gauss-Hermite table exists twice (fp64 host header, fp32/fp64 device
# header) and GH_N is defined in both; this gate fails the build if they drift.
gh_check:
	$(PY) tools/check_gh.py

# Every __kernel in kernels/ must be (a) in the host registry, (b) in the PL
# link, and (c) called by somebody. A kernel missing from (a) is source-JIT-only
# and therefore undeployable -- which is exactly how the eight per-symbol
# datapath kernels were lost (reachable only from a private mini-runtime in
# runtime/datapath_main.cpp, absent from load_xclbin's acquisition list).
registry_check:
	$(PY) tools/check_registry.py

# The deficit table (ln D on a 1600-point ln-t grid, 5 constellations x 2
# branches) is GENERATED, so it gets the same treatment as the Gauss-Hermite
# headers: a gate that regenerates a sample and compares, then checks every
# property the interpolation relies on -- monotonicity, the Lemma-4 ratio
# D_mod/D_fin = L/(L-1), the tail handover step, the fp64/fp32 pair, and the
# 4th-difference smoothness that makes 4-point Lagrange safe. It also reports the
# PL footprint, because 62.5 KB of __constant is the memory item to budget.
deficit_check:
	$(PY) tools/check_deficit_table.py

deficit_gen:
	cd $(CURDIR) && $(PY) tools/gen_deficit_table.py
	$(PY) tools/check_deficit_table.py

# The strong form: regenerate BOTH headers from scratch (a converged-quadrature
# pass, ~25 s) and byte-compare. Not in `check` -- the plain gate re-derives 90 of
# the 16 000 values instead -- but run it after touching the generator.
deficit_full_check:
	$(PY) tools/check_deficit_table.py --full

# Regenerate BOTH Gauss-Hermite headers (host fp64 + device fp64/fp32). Not part
# of `check`: the generators write cwd-relative paths, so the recipe pins the
# directory, and gh_check is what proves the two files still agree afterwards.
gh_gen:
	cd $(CURDIR) && $(PY) tools/gen_gh_table.py && $(PY) tools/gen_gh_cl.py
	$(PY) tools/check_gh.py

# One constant, N definition sites, zero drift: capacity bounds the host must
# honour because the device sizes an array with them (PMAX/MAXM/MAXD/KMAX/MMAX),
# plus the algorithm parameters the fp32 kernel, the fp64 reference and the
# numpy oracle have to agree on (TQ, NEWTON_ITERS, the CQI table). Text only.
capacity_check:
	$(PY) tools/check_capacity.py

# Deployed transformer path: real checkpoint -> flat binary -> OpenCL graph.
# Skips itself when the archived checkpoint is absent. TF_CKPT=/path/model.pt
# overrides the default location.
tf_export_check: build/tf_bench
	$(PY) tools/tf_export_check.py

# cf_ref writes build/cf_vectors.bin which cf_bench reads: never run in parallel.
.NOTPARALLEL:

check: gh_check registry_check capacity_check deficit_check allprog_check cf_bench op_bench op_pl_bench tf_bench tf_export_check decomp_bench frontend_bench \
       pipeline_bench modem_bench datapath_bench datapath_pl_bench rate_bench rx_bench lmmse_bench chain_bench \
       asan_check
	@echo "[hpocl] all checks done"

# clean keeps the generated oracle vectors and build/experiments results;
# distclean removes the whole build tree.
clean:
	rm -f build/*_bench build/*.o build/*.d
distclean:
	rm -rf build

help:
	@echo "targets:"
	@echo "  check         run every bench (regenerates numpy oracles first)"
	@echo "  gh_check      verify host/device Gauss-Hermite tables agree"
	@echo "  registry_check verify every kernel is registered, PL-linked, called"
	@echo "  capacity_check verify shared constants agree across kernel/host/oracle"
	@echo "  allprog_check build all kernels as one program, acquire every kernel"
	@echo "  asan_check    rebuild+run the host-only benches under ASan+UBSan"
	@echo "  gh_gen        regenerate both Gauss-Hermite headers, then gh_check"
	@echo "  deficit_check verify the generated ln D deficit table (no device)"
	@echo "  deficit_gen   regenerate both deficit-table headers, then deficit_check"
	@echo "  deficit_full_check  regenerate from scratch and byte-compare (~25 s)"
	@echo "  tf_export_check  real checkpoint -> binary -> OpenCL graph (skips if absent)"
	@echo "  <name>_bench  build+run one bench (cf op decomp frontend pipeline rate rx"
	@echo "                lmmse modem datapath chain tf)"
	@echo "  clean         remove binaries and .d files (keeps *.bin, experiments/)"
	@echo "  distclean     remove the whole build/ tree"
	@echo "variables: CXX=$(CXX)  PY=$(PY)  DEV=$(DEV) (cpu|gpu|acc)"

.PHONY: cf_ref cf_bench op_bench tf_bench decomp_bench frontend_bench pipeline_bench \
        op_pl_bench modem_bench datapath_bench datapath_pl_bench rate_bench \
        rx_bench \
        lmmse_bench chain_bench \
        gh_check \
        registry_check capacity_check deficit_check allprog_check asan_check \
        gh_gen deficit_gen deficit_full_check \
        tf_export_check check \
        clean distclean help
