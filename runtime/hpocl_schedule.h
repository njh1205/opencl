#pragma once
//-----------------------------------------------------------------------------
// The DEVICE role of the control plane: ref/pipeline.cpp's front_end() and
// design_instant() bodies, unmodified, with the decompositions and the threshold
// solve executed as batched OpenCL launches instead of fp64 host loops.
//
// What this file adds, and it is only three things:
//   1. device_svd()  -- an SvdProvider whose run() is ONE jacobi_svd launch for
//      the whole batch (C matrices of one shape), which is what makes the
//      offload worth its driver overhead: the front end's eight 28x32
//      null-basis factorisations become one launch of 8 work-items instead of
//      eight host SVDs.
//   2. a COST RULE for which batches to offload (DeviceSvd::min_flops). Not
//      every decomposition in the instant is worth a launch: at the deployed
//      geometry the null-basis batch is 20.3 Mflop and the analog batch 1.23
//      Mflop, while the two nul x nul = 4x4 batches are 34 kflop each -- less
//      than the 10-50 us an XRT launch costs. Sending those to the fabric would
//      make the control plane SLOWER, so the provider keeps them on the PS and
//      says so in its counters. `make sched_bench` also runs with force=true,
//      which offloads all four families, so the device path is gated on every
//      shape and not only on the two that pay for themselves.
//   3. schedule_device() -- front_end + device threshold_cf + design_instant,
//      i.e. the same three calls schedule() makes.
//
//   4. device_gemm() -- a GemmProvider whose run() is ONE wbb_apply launch per
//      glue GEMM, with the SAME cost rule, so the three Nt-class GEMMs (H F_RF,
//      F_RF N_k, F_RF F_BB) can come off the PS too. This was added after the
//      cost model was redone at the shapes the pools actually contain: the glue
//      is NOT "~0.13 Mflop for the whole instant" (that was measured at Nt=16),
//      it is 3.08-3.35 Mflop + 4096 atan2/cos/sin at Nt=128, ~4.6 ms projected on
//      an A53, i.e. ~40% of the hybrid PS budget -- see
//      docs/cost_model_real_shapes.md sections 3 and 7.2. Optional (a nullptr
//      DeviceGemm keeps the glue on the PS, which is what shipped before), and
//      gated SITE BY SITE because the three GEMMs feed differently
//      rank-sensitive consumers.
//
// What stays on the PS in both roles, deliberately: the whitened channel
// (Nr x nul = 4 x 4) and the Fbb assembly (nul x nul = 4 x 4) GEMMs, which are
// 0.037 Mflop together at the largest pool shape and would cost more in launch
// overhead than in flops; the constant-modulus projection (4096 atan2+cos+sin at
// Nt=128 -- transcendental-bound, and the fabric has no transcendental unit, so
// a kernel for it would be a CORDIC nobody has written); the gauge pin, the
// rank/nullity selection and the transposes that put an operand in wbb form
// (branchy or O(Nt*NRF) scalar work); and gmd_factors/ucd_design inside
// design_instant (per-user d x d, microseconds).
//-----------------------------------------------------------------------------
#include <cstddef>
#include <string>
#include <vector>
#include "hpocl_executor.h"
#include "hpocl_runtime.h"
#include "pipeline.h"

namespace hpocl {

// One scheduling instant on the device. Returns false if the runtime has no
// program (no device, or the kernels were not built), in which case nothing is
// written and the caller must fall back to schedule().
struct DeviceSvd {
  explicit DeviceSvd(Runtime& r) : rt(&r) {}
  Runtime* rt = nullptr;

  // Estimated fp32 flops a batch must be worth before it is launched. Derived,
  // not tuned -- see the block comment in hpocl_schedule.cpp. 0 offloads
  // everything.
  double min_flops = 4e5;
  bool force = false;        // offload every family regardless of min_flops
  int sweeps = 16;           // hard cap; the kernel's tol_exit stops earlier

  // counters, so a run can REPORT its split instead of asserting one
  int launches = 0;          // batches sent to the device
  int host_batches = 0;      // batches kept on the PS by the cost rule
  double flops_off = 0, flops_host = 0;
  double us_svd = 0, us_threshold = 0;   // device kernel time, from last_us()
  std::vector<const char*> where;   // one entry per batch, in call order
  std::vector<std::string> what;    // its shape, so a report says WHICH batch

  SvdProvider provider();
  std::vector<dcplx> scratch_u;     // jacobi_svd always writes U; callers may
                                    // pass nullptr (host_svd honours that)
};

// The GLUE role: which of the three Nt-class GEMMs run as wbb_apply launches.
// Separate from DeviceSvd because the cost rule is shared but the accuracy risk
// is not -- the SVD batches were already fp32 in the device role, whereas
// kGemmEff feeds the BD interference stack whose TRAILING singular vectors are
// the null basis, and kGemmFn feeds the Gram whose structurally-zero eigenvalues
// are 2.5e-9..3.7e-8 x lam_max. So `sites` is a mask, and sched_bench runs each
// site alone to say which one costs what.
// The four constants BOTH cost rules are derived from, named so the derivation
// is executable rather than prose: tools/glue_cost_table.py reads these four back
// out of this header and re-derives the per-site offload decision for all 21
// configs in mix_precode/*/config_pools, which is how "the rule is right at the
// real shapes" stays a checked statement instead of a remembered one. Sources:
//   kHostFlopRate  Cortex-A53 fp64, README's cycle model (1.4 GHz, in-order,
//                  ~1 fp64 flop/cycle, x1.5 structural overhead).
//   kDevFlopRate   PL fp32 at 200 MHz with 4-8 lanes, README's estimate for this
//                  kernel family (the measured wbb_apply rate on a cached GPU is
//                  not transferable; see the WBB_LOCAL_MAX note in the kernel).
//   kLaunchS       XRT embedded launch, the MIDDLE of the documented 10-50 us
//                  range. This dev host measures ~1 us, which is why the number
//                  comes from the board and not from here.
//   kConvFlopsPerElem  fp64<->fp32 staging the fp32 kernel interface forces on
//                  the PS. 6 is conservative: measured 2.5 flop-equivalent per
//                  element on this host (M4, /tmp/mpx3/stage_probe.cpp: staging is
//                  2.2% of the host GEMM it replaces at Nt=128).
inline constexpr double kHostFlopRate = 1.4e9;
inline constexpr double kDevFlopRate = 1.0e11;
inline constexpr double kLaunchS = 3.0e-5;
// The TOP of the same 10-50 us range. The GLUE rule is derived against this one
// rather than against kLaunchS, so its decisions hold whichever end of the range
// the board turns out to be: a site is offloaded only if offloading still wins
// when a launch costs 50 us. tools/glue_cost_table.py asserts exactly that at all
// 21 pool configs (63 sites) and separately reports what the mid-range launch
// would additionally offload -- 11 sites of 0.066 Mflop, <= 10.7 us each, ~1% of
// those instants -- so the conservatism is a number and not a mood.
inline constexpr double kLaunchWorstS = 5.0e-5;
inline constexpr double kConvFlopsPerElem = 6.0;

// ONE definition of the glue rule's threshold, declared here so the struct
// default and the derivation in hpocl_schedule.cpp cannot drift apart: the
// derivation's table produces this number, and this number is what the struct
// defaults to.
inline constexpr double kGemmMinFlops = 1e5;

struct DeviceGemm {
  explicit DeviceGemm(Runtime& r) : rt(&r) {}
  Runtime* rt = nullptr;

  // Same QUESTION as DeviceSvd::min_flops (is host_time > device_time + launch?)
  // but a different ANSWER, derived in hpocl_schedule.cpp next to kGemmMinFlops:
  // 1e5, not 4e5. Short version -- the SVD rule can afford 10x conservatism
  // because the batches it mis-prices are microseconds of work either way, while
  // 4e5 here would keep the glue on the PS for 15 of the 21 pool configs whose
  // sites are 0.05-0.13 Mflop, i.e. 20-95 us of A53 time each. 1e5 is the
  // crossover that still says "offload" when the launch costs 50 us instead of
  // the model's 30 us, so the decision is robust to the one constant in the model
  // that is a range rather than a measurement. 0 offloads every enabled site.
  double min_flops = kGemmMinFlops;
  bool force = false;        // offload every enabled site regardless of cost
  int sites = (1 << kGemmSites) - 1;   // bitmask of GemmSite

  // counters, so a run REPORTS its split instead of asserting one
  int launches = 0, host_calls = 0;
  double flops_off = 0, flops_host = 0;
  double us_gemm = 0;        // device kernel time, from last_us()
  // fp64<->fp32 staging traffic the offload ADDS to the PS (the kernel interface
  // is fp32, ref/ is fp64). Reported because "the GEMM is free once it is on the
  // fabric" is only true if the conversion is not the new bottleneck: at Nt=128
  // the three sites convert 3*(4096 + 4096 + 1024) ~ 27.6k complex elements
  // against the 3.15 Mflop they replace.
  size_t elems_converted = 0;
  std::vector<const char*> where;
  std::vector<std::string> what;

  GemmProvider provider();

  // fp32 staging buffers, grown on demand and reused across calls
  std::vector<cplx> w32, x32, y32;
};

// dgemm == nullptr keeps every glue GEMM on the PS (the pre-glue-offload
// behaviour, and still the right choice at Nt <= 32 where the largest site is
// 0.13 Mflop).
bool schedule_device(const dcplx* H, const PipeCfg& c, PipeOut& o,
                     DeviceSvd& dsvd, DeviceGemm* dgemm = nullptr);

// Estimated Jacobi work of one batch, counted the way kernels/decomp.cl does
// it. Exposed so sched_bench can print the split it is actually exercising
// rather than trusting the cost rule's prose.
double jacobi_batch_flops(int C, int m, int n, int sweeps);
// Same, for the Gram form the provider uses when a batch asks for take < n.
double gram_batch_flops(int C, int m, int n, int take, int sweeps);
// Same, for one glue GEMM: 8 flops per complex MAC (4 mul + 2 add, the
// convention jacobi_batch_flops uses), M*N*SC of them.
double gemm_flops(int M, int N, int SC);

// One independent Linux/PS control-plane request. `H` must point to a contiguous
// [config.K][config.Nr][config.Nt] complex channel tensor and remain valid until
// schedule_host_batch returns.
struct HostScheduleJob {
  const dcplx* H = nullptr;
  PipeCfg config{};
};

// Parallel PS-only scheduling across independent instants. Work is distributed
// through CpuExecutor (four A53 workers by default on ZU49DR); no OpenCL Runtime
// is used or shared here. This is the safe CPU parallel boundary: device jobs
// remain serialized through schedule_device(), whose Runtime owns mutable kernel
// argument state and a single queue. `outputs` must point to `count` distinct
// PipeOut objects. Invalid null pointers throw std::invalid_argument.
void schedule_host_batch(CpuExecutor& executor, const HostScheduleJob* jobs,
                         size_t count, PipeOut* outputs);

}  // namespace hpocl
