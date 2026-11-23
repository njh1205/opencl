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
#include <cstring>
#include <string>
#include <vector>
#include "hpocl_runtime.h"
#include "deficit_table.h"   // DFT_ROWS / DFT_N: the __constant footprint to check
#include "gh_table.h"        // GH_N
// mcs_table.clh is plain C arithmetic with the storage class spelled per language
// (`static const` on the host, `__constant` on the device), so including it here
// gives the CQI table's real entry count rather than a retyped 16. -Ikernels is
// already on the compile line; tools/check_capacity.py keeps HPOCL_CQI_N in step
// with tools/rate_ref.py's retyped table.
#include "mcs_table.clh"     // HPOCL_CQI_N

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
  // threshold_cf.cl is fp32 BY DEFAULT and takes -DHPOCL_FP64 to opt into double
  // (the inversion landed with the PL footgun fix: the ZU49DR's DSP48E2 has no
  // floating point at all, so a forgotten flag must mean "the deployable fp32
  // datapath", not "synthesise fp64"). A device whose CL_DEVICE_DOUBLE_FP_CONFIG
  // is 0 cannot create the fp64 kernel at all -- Apple's Metal-backed OpenCL
  // builds the program and then fails clCreateKernel with
  // "newComputePipelineState failed" -- so this gate asks for fp64 only where the
  // device reports it can do fp64, and cf_main.cpp makes the same choice.
  // HPOCL_CLOPTS overrides both, so the PL configuration can be forced anywhere.
  const char* env = getenv("HPOCL_CLOPTS");
  const char* want = rt.device_fp64() ? "-DHPOCL_FP64" : "";
  const char* copts = env ? env : want;
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
         copts[0] ? copts : "(none)");
  // Deliberately NOT returning here: the pre-flight below is the most
  // informative thing this gate prints, and the failure mode it exists for
  // PRODUCES missing kernels. Measured on a Tesla P40 with -DHPOCL_FP64: the
  // program builds, then every clCreateKernel fails with CL_INVALID_BUFFER_SIZE
  // (-61) because the fp64 constant tables need 128 896 B and the device allows
  // 65 536 B. Bailing out at "missing kernel" would print 29 lines of a symptom
  // and hide the one number that explains them, so the diagnosis runs first and
  // the missing count is folded into the final verdict.
  const int missing_kernels = missing;

  //-------------------------------------------------------------------------
  // Board pre-flight: the two device limits this library actually pushes, and
  // the per-kernel resource numbers the PL budget is built from. Both used to
  // be ARGUED from arithmetic in README/build_xclbin.sh; both are queryable, so
  // now they are MEASURED, on every device this gate runs on -- which is the
  // first thing to look at when the tree lands on the Zynq.
  //-------------------------------------------------------------------------
  cl_device_id d = rt.device();
  cl_ulong maxconst = 0, localmem = 0, maxalloc = 0;
  cl_uint dfp = 0;
  size_t maxwi[3] = {0, 0, 0};
  clGetDeviceInfo(d, CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE, sizeof maxconst, &maxconst, nullptr);
  clGetDeviceInfo(d, CL_DEVICE_LOCAL_MEM_SIZE, sizeof localmem, &localmem, nullptr);
  clGetDeviceInfo(d, CL_DEVICE_MAX_MEM_ALLOC_SIZE, sizeof maxalloc, &maxalloc, nullptr);
  clGetDeviceInfo(d, CL_DEVICE_DOUBLE_FP_CONFIG, sizeof dfp, &dfp, nullptr);
  clGetDeviceInfo(d, CL_DEVICE_MAX_WORK_ITEM_SIZES, sizeof maxwi, maxwi, nullptr);

  // The __constant budget, computed for the precision THIS build actually
  // selected. kernels/deficit_table.clh and kernels/gh_table.clh pick their
  // block on `#ifndef HPOCL_FP32`, and threshold_cf.cl defines HPOCL_FP32 by
  // default (with -DHPOCL_FP64 opting into double), so:
  //   default or -DHPOCL_FP32 : deficit 10x1600 fp32 = 64 000 B, gh 48x2 fp32
  //   -DHPOCL_FP64            : deficit 10x1600 fp64 = 128 000 B, gh 48x2 fp64
  // That distinction is not academic. MEASURED on a real non-Apple device
  // (Tesla P40, OpenCL 3.0, driver 550.127.08, x86_64 Linux): it reports
  // CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE = 65 536 B -- exactly the spec minimum --
  // and the fp64 build fails with CL_INVALID_BUFFER_SIZE (-61), the program
  // builds and then clCreateKernel cannot allocate the constant pool. The fp32
  // build fits with 1 024 B (1.6%) to spare. Apple reports 1 GB, which is why
  // this was invisible on the dev host: the #1 xclbin deployment risk in the repo
  // could only be measured off-Apple.
  const bool fp64_tables = (strstr(copts, "HPOCL_FP64") != nullptr);
  const unsigned esz = fp64_tables ? 8u : 4u;
  const unsigned long dft_bytes = (unsigned long)DFT_ROWS * DFT_N * esz;
  const unsigned long gh_bytes  = (unsigned long)GH_N * 2u * esz;
  const unsigned long mcs_bytes = (unsigned long)HPOCL_CQI_N * 2u * 4u;  // 2 int tables
  const unsigned long table_bytes = dft_bytes + gh_bytes + mcs_bytes;
  printf("[allprog_check] device limits: max_constant_buffer=%lu B (%.1f KB), "
         "local_mem=%lu B, max_mem_alloc=%lu B, double_fp_config=0x%x, "
         "max_work_item=%zux%zux%zu\n",
         (unsigned long)maxconst, maxconst / 1024.0, (unsigned long)localmem,
         (unsigned long)maxalloc, dfp, maxwi[0], maxwi[1], maxwi[2]);
  int fails = 0;
  printf("[allprog_check] __constant footprint of this build (%s tables): "
         "deficit %lu B + gauss-hermite %lu B + cqi %lu B = %lu B\n",
         fp64_tables ? "fp64" : "fp32", dft_bytes, gh_bytes, mcs_bytes,
         table_bytes);
  if (maxconst && (cl_ulong)table_bytes > maxconst) {
    printf("[allprog_check] FAIL: the constant tables need %lu B but this "
           "device allows %lu B (%.2fx over) -- clCreateKernel will return "
           "CL_INVALID_BUFFER_SIZE (-61), exactly as measured on a Tesla P40 "
           "with the fp64 tables. Fixes, in order of preference: build fp32 "
           "(the PL default, and the only thing a DSP48E2 fabric can do "
           "anyway); or shrink the table -- python3 "
           "tools/gen_deficit_table.py --N 800 (31.25 KB, 8.4e-5 dB "
           "interpolation error) or --ms 64,256 (25 KB) or --ms 64 (12.5 KB), "
           "then make deficit_check.\n",
           table_bytes, (unsigned long)maxconst,
           (double)table_bytes / (double)maxconst);
    ++fails;
  } else if (!maxconst) {
    printf("[allprog_check] FAIL: the device reports "
           "CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE = 0, i.e. it cannot answer the "
           "one question that decides whether this program is loadable; that is "
           "not a pass.\n");
    ++fails;
  } else {
    const double head = 100.0 * (1.0 - (double)table_bytes / (double)maxconst);
    printf("[allprog_check] constant budget: %lu B used of %lu B allowed "
           "(%.1f%% of the limit, %.1f%% of the 64 KB spec minimum), headroom "
           "%.1f%%%s\n", table_bytes, (unsigned long)maxconst,
           100.0 * table_bytes / maxconst, 100.0 * table_bytes / 65536.0, head,
           head < 5.0 ? " -- TIGHT: any new __constant table, or a device that "
                        "reserves part of the pool, breaks the load" : "");
  }

  // Per-kernel resources, from the compiler rather than from hand arithmetic.
  // CL_KERNEL_PRIVATE_MEM_SIZE is what the static __private arrays actually cost
  // per work-item after the compiler has had its say -- the number the README's
  // capacity table used to estimate (jacobi_svd 23 360 B, gmd_k 2 732 B, ucd_k
  // 1 836 B, cbd_k 3 264 B at their own ceilings). Sorted by size so the one
  // kernel that dominates the PL footprint is the first row.
  struct Row { const char* name; size_t priv, local, wgs; };
  std::vector<Row> rows;
  for (size_t i = 0; i < nk; ++i) {
    cl_kernel k = rt.kernel(names[i]);
    if (!k) continue;
    Row r{names[i], 0, 0, 0};
    clGetKernelWorkGroupInfo(k, d, CL_KERNEL_PRIVATE_MEM_SIZE, sizeof r.priv, &r.priv, nullptr);
    clGetKernelWorkGroupInfo(k, d, CL_KERNEL_LOCAL_MEM_SIZE, sizeof r.local, &r.local, nullptr);
    clGetKernelWorkGroupInfo(k, d, CL_KERNEL_WORK_GROUP_SIZE, sizeof r.wgs, &r.wgs, nullptr);
    rows.push_back(r);
  }
  for (size_t i = 0; i < rows.size(); ++i)
    for (size_t j = i + 1; j < rows.size(); ++j)
      if (rows[j].priv > rows[i].priv) std::swap(rows[i], rows[j]);
  // Does this device actually ANSWER the private-memory query? Apple's ICD
  // returns 0 for every kernel (measured: all 29 zero, while CL_KERNEL_LOCAL_
  // MEM_SIZE comes back correct -- power_iter 1296 B, wbb_apply 1024 B), so a
  // column of zeros here is "unqueryable", NOT "the kernels use no private
  // memory". Printing it as a measurement would be worse than not printing it:
  // jacobi_svd's static __private arrays are 23 360 B/work-item at the shipped
  // DECOMP_ 32x40 ceiling by hand arithmetic, and the README quotes that. On
  // XRT/Vitis and on PoCL the query is implemented, so the board run is what
  // turns the arithmetic into a measurement -- which is exactly why this gate
  // prints the table at all.
  bool priv_known = false;
  for (auto& r : rows) if (r.priv) priv_known = true;
  printf("[allprog_check] per-kernel resources (compiler-reported, this device):\n");
  printf("  %-20s %10s %10s %10s\n", "kernel", "private B", "local B", "max WG");
  for (auto& r : rows)
    printf("  %-20s %10zu %10zu %10zu\n", r.name, r.priv, r.local, r.wgs);
  if (!priv_known) {
    printf("[allprog_check] NOTE: this device reports CL_KERNEL_PRIVATE_MEM_SIZE "
           "= 0 for all %zu kernels, i.e. the query is UNIMPLEMENTED here, not "
           "zero-footprint. The private numbers in README's capacity table are "
           "hand arithmetic (jacobi_svd 23 360 B, gmd_k 2 732 B, ucd_k 1 836 B, "
           "cbd_k 3 264 B at their own ceilings) until this gate runs on XRT or "
           "PoCL.\n", rows.size());
  } else if (!rows.empty()) {
    // The deployed control plane batches C = 8 users through jacobi_svd, so the
    // footprint that matters is 8x the largest private number in the library.
    printf("[allprog_check] largest private = %s at %zu B/work-item; x C=8 = "
           "%.1f KB (%.2f%% of a 7.5 MB ZU49DR on-chip budget)\n",
           rows[0].name, rows[0].priv, 8.0 * rows[0].priv / 1024.0,
           100.0 * 8.0 * rows[0].priv / (7.5 * 1024 * 1024));
  }

  // Local memory headroom, which this device DOES answer and which is a real
  // launch blocker: wbb_apply's PL tile (-DWBB_LOCAL_MAX=4096, what
  // `make op_pl_bench` builds) allocates 32 KB of __local per work-group, and
  // CL_DEVICE_LOCAL_MEM_SIZE here is exactly 32 768 B -- i.e. the tiled path
  // fits with ZERO bytes to spare, and any device reporting less, or any
  // implementation that reserves part of local memory for its own use, fails the
  // launch rather than running slowly. That is why the tile is opt-in and the
  // shipped default is 1 KB.
  size_t maxloc = 0; const char* locwho = "";
  for (auto& r : rows) if (r.local > maxloc) { maxloc = r.local; locwho = r.name; }
  if (localmem && maxloc) {
    const double frac = 100.0 * (double)maxloc / (double)localmem;
    printf("[allprog_check] local memory: largest is %s at %zu B = %.1f%% of "
           "CL_DEVICE_LOCAL_MEM_SIZE (%lu B)%s\n", locwho, maxloc, frac,
           (unsigned long)localmem, frac >= 90.0 ? " -- NO HEADROOM: a device "
           "that reserves any local memory, or reports less, will fail this "
           "launch; lower WBB_LOCAL_MAX (default 1 KB) before blaming the kernel"
           : "");
    if (frac > 100.0) ++fails;
  }
  fails += missing_kernels;
  printf(fails ? "[allprog_check] FAIL (%d)\n" : "[allprog_check] OK\n", fails);
  return fails ? 1 : 0;
}
