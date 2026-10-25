// allprog_check: build EVERY kernel source into ONE program and acquire every
// registered kernel from it.
//
// Why this gate exists. Runtime::build() hands all of its files to a single
// clCreateProgramWithSource, so they share one translation unit: a duplicate
// macro, a duplicate non-static helper, a missing include guard or an extension
// pragma in the wrong order is a build error or -- worse -- a silent resize of
// somebody else's array. The per-bench builds never see this, because each bench
// loads only the sources it needs. The PL link does see it (v++ links all the
// .xo into one xclbin), and so would any PS "load the whole library" path.
//
// tools/check_registry.py catches the collisions statically; this catches them
// on a real compiler and a real device, and additionally proves that all 29
// kernels of the library are simultaneously creatable -- i.e. that the registry,
// the .xclbin kernel list and the sources really do describe one library.
//
// Pass -DHPOCL_FP32 (or set HPOCL_CLOPTS) to check the deployed PL configuration
// instead of the dev-host fp64 one.
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>
#include "hpocl_runtime.h"

int main(int argc, char** argv) {
  const char* dev = argc > 1 ? argv[1] : "cpu";
  const std::vector<std::string> srcs = {
    "kernels/channel_cov.cl", "kernels/datapath.cl", "kernels/decomp.cl",
    "kernels/power_iter.cl", "kernels/rate.cl", "kernels/switch_threshold.cl",
    "kernels/tf_ops.cl", "kernels/thp_precoder.cl", "kernels/threshold_cf.cl",
    "kernels/wbb_apply.cl"};
  hpocl::Runtime rt;
  rt.set_verbose(true);
  if (!rt.init(dev)) { fprintf(stderr, "[allprog] OpenCL init failed\n"); return 2; }
  // threshold_cf.cl compiles to fp64 unless told otherwise, and a device whose
  // CL_DEVICE_DOUBLE_FP_CONFIG is 0 cannot pipeline it (Apple's Metal-backed
  // OpenCL fails in clCreateKernel with "newComputePipelineState failed" -- the
  // program builds, the kernel object does not). cf_main.cpp makes the same
  // choice; HPOCL_CLOPTS overrides it so the PL configuration can be forced.
  const char* env = getenv("HPOCL_CLOPTS");
  const char* fp32 = rt.device_fp64() ? "" : "-DHPOCL_FP32";
  const char* copts = env ? env : fp32;
  if (!rt.build(srcs, copts)) {
    fprintf(stderr, "[allprog] FAIL: %zu sources do not form one program%s%s\n",
            srcs.size(), copts ? " with opts " : "", copts ? copts : "");
    return 1;
  }
  // Every name the registry asks for must have been acquired from this one
  // program. The list comes from Runtime::kernel_names() rather than being
  // retyped here, so this gate cannot drift into checking a stale manifest:
  // what it proves is that a whole-library build provides the whole library.
  size_t nk = 0;
  const char* const* names = hpocl::Runtime::kernel_names(&nk);
  int missing = 0;
  for (size_t i = 0; i < nk; ++i)
    if (!rt.has_kernel(names[i])) {
      fprintf(stderr, "[allprog] missing kernel %s\n", names[i]);
      ++missing;
    }
  printf("[allprog_check] device: %s (fp64=%d) | %zu sources -> 1 program | "
         "%zu/%zu kernels acquired | opts %s\n", rt.device_name(),
         (int)rt.device_fp64(), srcs.size(), nk - (size_t)missing, nk,
         copts[0] ? copts : "(none, fp64 threshold_cf)");
  if (missing) { printf("[allprog_check] FAIL\n"); return 1; }
  printf("[allprog_check] OK\n");
  return 0;
}
