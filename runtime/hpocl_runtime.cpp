#include "hpocl_runtime.h"
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include "hpocl_clsrc.h"   // read_cl_source(): the one .cl #include inliner

namespace hpocl {
namespace {

// A kernel that is not in the loaded program is a deployment error (wrong .cl
// set, or an .xclbin built without that operator), not a data condition. Every
// wrapper is void and cannot report failure through its return value, so report
// it here instead of returning silently: a silent no-op leaves the caller's
// output buffer holding its previous contents, which reads exactly like a
// plausible wrong answer.
void no_kernel(const char* name) {
  fprintf(stderr, "[hpocl] kernel '%s' is not in the loaded program -- output "
          "buffers were NOT written\n", name);
}


// Every kernel name the operator library can expose, in exactly one place.
// Both the source-JIT path (build) and the pre-synthesised .xclbin path
// (load_xclbin) acquire from this list, so a kernel cannot be reachable on one
// and missing from the other -- which is precisely how the eight per-symbol
// datapath kernels ended up source-JIT-only and therefore undeployable.
// Acquisition is tolerant: a program that exposes a subset (cf_bench builds
// only threshold_cf.cl) simply leaves the rest absent.
const char* const kKernelNames[] = {
  // control plane ------------------------------------------ op_bench
  "channel_cov", "power_iter", "deflate", "switch_threshold", "wbb_apply",
  "thp_precoder", "threshold_cf",
  // per-user decompositions -------------------------------- decomp_bench
  "jacobi_svd", "gmd_k", "ucd_k", "cbd_k",
  // per-symbol datapath ------------------------ datapath_bench / chain_bench
  "qam_map_k", "tx_combine_k", "channel_apply_k", "rx_combine_k",
  "modulo_fold_k", "softdemod_llr_k", "softdemod_folded_k", "gmi_k",
  // code-rate / MCS layer ------------------------------------- rate_bench
  "mcs_select_k",
  // transformer threshold predictor ------------------------ tf_bench
  "tf_dense", "tf_gelu", "tf_add_broadcast", "tf_mul_mask", "tf_layernorm",
  "tf_attn", "tf_pool", "tf_sign_pack", "tf_add_elem",
};
constexpr size_t kNumKernelNames = sizeof kKernelNames / sizeof *kKernelNames;

}  // namespace

cl_kernel Runtime::kernel(const char* name) const {
  auto it = kernels_.find(name);
  return it == kernels_.end() ? nullptr : it->second;
}
bool Runtime::has_kernel(const char* name) const { return kernel(name) != nullptr; }

const char* const* Runtime::kernel_names(size_t* count) {
  if (count) *count = kNumKernelNames;
  return kKernelNames;
}

bool Runtime::acquire_kernels() {
  for (size_t i = 0; i < kNumKernelNames; ++i) {
    cl_int err = CL_SUCCESS;
    cl_kernel k = clCreateKernel(prog_, kKernelNames[i], &err);
    if (err == CL_SUCCESS) kernels_[kKernelNames[i]] = k;
  }
  if (kernels_.empty()) {
    fprintf(stderr, "[hpocl] no operator kernels found in program\n");
    return false;
  }
  return true;
}

void Runtime::release_program() {
  // TransformerPredictor borrows these handles without owning them, so release
  // kernels before the program on every reload, not only at destruction.
  for (auto& kv : kernels_) clReleaseKernel(kv.second);
  kernels_.clear();
  if (prog_) clReleaseProgram(prog_);
  prog_ = nullptr;
  prog_fp64_ = false;
}

void Runtime::reset() {
  if (ev_) clReleaseEvent(ev_);
  ev_ = nullptr;
  release_program();
  if (q_) clReleaseCommandQueue(q_);
  q_ = nullptr;
  if (ctx_) clReleaseContext(ctx_);
  ctx_ = nullptr;
  dev_ = nullptr;
  dev_name_.clear();
  dev_fp64_ = false;
}

Runtime::~Runtime() { reset(); }

bool Runtime::init(const char* pref) {
  // Reinitialisation is supported for device-policy retries. Drop the complete
  // old ownership graph first so no stale kernels can outlive their program.
  reset();
  const char* preference = pref ? pref : "cpu";
  cl_uint nplat = 0;
  cl_int err = clGetPlatformIDs(0, nullptr, &nplat);
  if (err != CL_SUCCESS || !nplat) {
    fprintf(stderr, "[hpocl] clGetPlatformIDs failed (%d)\n", err);
    return false;
  }
  std::vector<cl_platform_id> plats(nplat);
  err = clGetPlatformIDs(nplat, plats.data(), nullptr);
  if (err != CL_SUCCESS) {
    fprintf(stderr, "[hpocl] clGetPlatformIDs enumeration failed (%d)\n", err);
    return false;
  }

  cl_device_type want = CL_DEVICE_TYPE_CPU;
  if (!strcmp(preference, "gpu")) want = CL_DEVICE_TYPE_GPU;
  if (!strcmp(preference, "acc")) want = CL_DEVICE_TYPE_ACCELERATOR;

  bool found = false;
  for (cl_uint p = 0; p < nplat && !found; ++p) {
    if (clGetDeviceIDs(plats[p], want, 1, &dev_, nullptr) == CL_SUCCESS)
      found = true;
  }
  if (!found) {  // fallback policy: any device keeps the stack functional
    for (cl_uint p = 0; p < nplat && !found; ++p) {
      if (clGetDeviceIDs(plats[p], CL_DEVICE_TYPE_ALL, 1, &dev_, nullptr) == CL_SUCCESS)
        found = true;
    }
  }
  if (!found) {
    fprintf(stderr, "[hpocl] no OpenCL device available for preference '%s'\n",
            preference);
    return false;
  }

  ctx_ = clCreateContext(nullptr, 1, &dev_, nullptr, nullptr, &err);
  if (err != CL_SUCCESS) {
    fprintf(stderr, "[hpocl] clCreateContext failed (%d)\n", err);
    reset();
    return false;
  }
  // clCreateCommandQueue is OpenCL 1.2 and DEPRECATED from 2.0 on, where the
  // replacement takes a property list. Which one exists depends on the headers the
  // target ships, not on us: XRT embedded has historically been 1.2, newer Vitis
  // and every desktop ICD are 2.0+, and a bare call here is the one g++ warning
  // this file produces (g++ 10.5, Ubuntu 20.04, -Wall -Wextra: 26 translation
  // units, 1 warning, 0 errors). Profiling must stay on either way -- last_us() is
  // the kernel-event column every timing in the README quotes, and the wall column
  // on this dev host is 200x larger than the event it wraps.
#ifdef CL_VERSION_2_0
  const cl_queue_properties props[] = {CL_QUEUE_PROPERTIES,
                                       CL_QUEUE_PROFILING_ENABLE, 0};
  q_ = clCreateCommandQueueWithProperties(ctx_, dev_, props, &err);
#else
  q_ = clCreateCommandQueue(ctx_, dev_, CL_QUEUE_PROFILING_ENABLE, &err);
#endif
  if (err != CL_SUCCESS) {
    fprintf(stderr, "[hpocl] command-queue creation failed (%d)\n", err);
    reset();
    return false;
  }

  char name[256] = {0};
  clGetDeviceInfo(dev_, CL_DEVICE_NAME, sizeof(name), name, nullptr);
  dev_name_ = name;
  cl_device_fp_config fp64 = 0;
  clGetDeviceInfo(dev_, CL_DEVICE_DOUBLE_FP_CONFIG, sizeof(fp64), &fp64, nullptr);
  // HPOCL_FP32=1 forces the single-precision host marshalling even if the
  // device claims fp64 (xclbins are compiled fp32 for PL).
  dev_fp64_ = (fp64 != 0) && getenv("HPOCL_FP32") == nullptr;
  if (verbose_) printf("[hpocl] device: %s (pref=%s)\n", dev_name_.c_str(), preference);
  return true;
}

bool Runtime::build(const std::vector<std::string>& files,
                    const char* extra_opts) {
  if (!ctx_ || !dev_) {
    fprintf(stderr, "[hpocl] build requested before successful init()\n");
    return false;
  }
  if (files.empty()) {
    fprintf(stderr, "[hpocl] build requested with no kernel sources\n");
    return false;
  }
  // A new build replaces the active program; clear stale kernel handles first.
  release_program();
  std::vector<std::string> srcs;
  std::vector<const char*> strs;
  std::vector<size_t> lens;
  for (const auto& f : files) {
    std::string src = read_cl_source(f);   // inlines #include "x.clh"
    if (src.empty()) {
      fprintf(stderr, "[hpocl] missing kernel source %s\n", f.c_str());
      return false;
    }
    srcs.push_back(std::move(src));
  }
  for (auto& s : srcs) { strs.push_back(s.c_str()); lens.push_back(s.size()); }

  cl_int err = CL_SUCCESS;
  prog_ = clCreateProgramWithSource(ctx_, (cl_uint)strs.size(), strs.data(),
                                    lens.data(), &err);
  if (err != CL_SUCCESS) {
    fprintf(stderr, "[hpocl] clCreateProgramWithSource failed (%d)\n", err);
    prog_ = nullptr;
    return false;
  }
  std::string opts = "-cl-std=CL1.2 -cl-mad-enable";
  if (extra_opts) { opts += " "; opts += extra_opts; }
  // The precision this program will have, recorded from the options actually
  // passed -- NOT from what the device supports. kernels/threshold_cf.cl is fp32
  // by default and takes -DHPOCL_FP64 to opt in, so "the device can do fp64" says
  // nothing about what got compiled; the wrappers that marshal `real`-typed
  // buffers (threshold_cf) must agree with the build or they read and write the
  // wrong width. HPOCL_FP32 in the environment still wins, matching dev_fp64_'s
  // existing override, so one knob forces both halves.
  prog_fp64_ = opts.find("HPOCL_FP64") != std::string::npos &&
               getenv("HPOCL_FP32") == nullptr;
  if (prog_fp64_ && !dev_fp64_)
    fprintf(stderr, "[hpocl] warning: -DHPOCL_FP64 was requested but %s reports "
            "no CL_DEVICE_DOUBLE_FP_CONFIG -- the build or clCreateKernel will "
            "fail (on Apple's Metal-backed OpenCL it fails as "
            "'newComputePipelineState failed', which does not say why)\n",
            dev_name_.c_str());
  err = clBuildProgram(prog_, 1, &dev_, opts.c_str(), nullptr, nullptr);
  if (err != CL_SUCCESS) {
    char log[8192] = {0};
    clGetProgramBuildInfo(prog_, dev_, CL_PROGRAM_BUILD_LOG, sizeof(log), log, nullptr);
    fprintf(stderr, "[hpocl] source build failed (%d):\n%s\n", err, log);
    release_program();
    return false;
  }
  if (acquire_kernels()) return true;
  release_program();
  return false;
}

cl_mem Runtime::mkbuf(const void* host, size_t bytes, cl_mem_flags extra) {
  cl_int err = CL_SUCCESS;
  cl_mem b = clCreateBuffer(ctx_, extra | (host ? CL_MEM_COPY_HOST_PTR : 0),
                            bytes, const_cast<void*>(host), &err);
  if (err != CL_SUCCESS)
    fprintf(stderr, "[hpocl] clCreateBuffer(%zu bytes) failed (%d)\n", bytes, err);
  return err == CL_SUCCESS ? b : nullptr;
}
void Runtime::rdbuf(cl_mem b, void* host, size_t bytes) {
  if (!b || !host) {
    fprintf(stderr, "[hpocl] refusing read from a null OpenCL buffer or into a null host pointer\n");
    return;
  }
  const cl_int err = clEnqueueReadBuffer(q_, b, CL_TRUE, 0, bytes, host, 0, nullptr, nullptr);
  if (err != CL_SUCCESS)
    fprintf(stderr, "[hpocl] clEnqueueReadBuffer(%zu bytes) failed (%d)\n", bytes, err);
}
bool Runtime::ndrange(cl_kernel k, cl_uint dims, const size_t* g, const size_t* l) {
  if (ev_) { clReleaseEvent(ev_); ev_ = nullptr; }
  cl_int err = clEnqueueNDRangeKernel(q_, k, dims, nullptr, g, l, 0, nullptr, &ev_);
  // CL_INVALID_WORK_GROUP_SIZE / CL_OUT_OF_RESOURCES mean nothing ran at all.
  // Say so, with the kernel name and the shape: a failed launch otherwise looks
  // exactly like a successful one that produced zeros. clFinish() is checked as
  // well: asynchronous device-side failures are reported there by several ICDs.
  if (err == CL_SUCCESS) err = clFinish(q_);
  if (err != CL_SUCCESS) {
    const char* name = "?";
    for (const auto& kv : kernels_)
      if (kv.second == k) { name = kv.first.c_str(); break; }
    fprintf(stderr, "[hpocl] %s: OpenCL launch/finish -> %d (global", name, err);
    for (cl_uint i = 0; i < dims; ++i) fprintf(stderr, " %zu", g[i]);
    fprintf(stderr, ") -- the kernel did NOT complete; outputs are not valid\n");
    return false;
  }
  return true;
}
double Runtime::last_us() const {
  if (!ev_) return 0.0;
  cl_ulong t0 = 0, t1 = 0;
  if (clGetEventProfilingInfo(ev_, CL_PROFILING_COMMAND_START, sizeof(t0), &t0, nullptr) != CL_SUCCESS ||
      clGetEventProfilingInfo(ev_, CL_PROFILING_COMMAND_END, sizeof(t1), &t1, nullptr) != CL_SUCCESS ||
      t1 < t0)
    return 0.0;
  return (double)(t1 - t0) / 1e3;
}

bool Runtime::load_xclbin(const std::string& path) {
  if (!ctx_ || !dev_) {
    fprintf(stderr, "[hpocl] load_xclbin requested before successful init()\n");
    return false;
  }
  // A precompiled binary carries no queryable precision, so this asserts the only
  // one the PL can have: fp32. The ZU49DR's DSP48E2 has no floating point at all,
  // build_xclbin.sh passes -DHPOCL_FP32, and kernels/threshold_cf.cl defaults to
  // fp32 -- three independent reasons. If an fp64 xclbin ever exists this has to
  // become a parameter, and the wrappers that marshal `real` (threshold_cf) are
  // what would silently break: see prog_fp64_'s comment for the measured failure.
  release_program();
  std::ifstream f(path, std::ios::binary);
  if (!f) { fprintf(stderr, "[hpocl] missing xclbin %s\n", path.c_str()); return false; }
  std::vector<char> blob((std::istreambuf_iterator<char>(f)), {});
  if (blob.empty()) { fprintf(stderr, "[hpocl] xclbin %s is empty\n", path.c_str()); return false; }
  const unsigned char* bp = reinterpret_cast<const unsigned char*>(blob.data());
  size_t bl = blob.size();
  cl_int bin_status = CL_SUCCESS, err = CL_SUCCESS;
  prog_ = clCreateProgramWithBinary(ctx_, 1, &dev_, &bl, &bp,
                                    &bin_status, &err);
  if (err != CL_SUCCESS || bin_status != CL_SUCCESS) {
    fprintf(stderr, "[hpocl] clCreateProgramWithBinary failed (%d/%d)\n", err, bin_status);
    release_program();
    return false;
  }
  // OpenCL binaries are not executable until built. Some XRT stacks accept a
  // kernel request without this call, while conforming ICDs return
  // CL_INVALID_PROGRAM_EXECUTABLE; building here makes the deployed xclbin path
  // portable and surfaces the device's build log on an ABI/device mismatch.
  err = clBuildProgram(prog_, 1, &dev_, nullptr, nullptr, nullptr);
  if (err != CL_SUCCESS) {
    char log[8192] = {0};
    clGetProgramBuildInfo(prog_, dev_, CL_PROGRAM_BUILD_LOG, sizeof(log), log, nullptr);
    fprintf(stderr, "[hpocl] xclbin build failed (%d):\n%s\n", err, log);
    release_program();
    return false;
  }
  if (acquire_kernels()) return true;
  release_program();
  return false;
}

bool Runtime::load_program(const std::vector<std::string>& files,
                           const char* extra_opts) {
  const char* xb = getenv("HPOCL_XCLBIN");
  if (xb && xb[0]) return load_xclbin(xb);
  return build(files, extra_opts);
}

//---------------------------------------------------------------------------
// operators
//---------------------------------------------------------------------------
void Runtime::channel_cov(const cplx* H, int M, int N, int S, cplx* R) {
  cl_kernel k = kernel("channel_cov");
  if (!k) return no_kernel("channel_cov");   // absent from this program
  cl_mem bH = mkbuf(H, sizeof(cplx) * S * M * N, CL_MEM_READ_ONLY);
  cl_mem bR = mkbuf(nullptr, sizeof(cplx) * M * M, CL_MEM_WRITE_ONLY);
  clSetKernelArg(k, 0, sizeof(cl_mem), &bH);
  clSetKernelArg(k, 1, sizeof(int), &M);
  clSetKernelArg(k, 2, sizeof(int), &N);
  clSetKernelArg(k, 3, sizeof(int), &S);
  clSetKernelArg(k, 4, sizeof(cl_mem), &bR);
  size_t g[2] = {(size_t)M, (size_t)M};
  ndrange(k, 2, g, nullptr);
  rdbuf(bR, R, sizeof(cplx) * M * M);
  clReleaseMemObject(bH); clReleaseMemObject(bR);
}

void Runtime::power_iter(const cplx* R, int M, int iters, int nmat,
                         cplx* v, float* lambda) {
  cl_kernel k = kernel("power_iter");
  if (!k) return no_kernel("power_iter");   // absent from this program
  cl_mem bR = mkbuf(R, sizeof(cplx) * nmat * M * M, CL_MEM_READ_ONLY);
  cl_mem bV = mkbuf(nullptr, sizeof(cplx) * nmat * M, CL_MEM_WRITE_ONLY);
  cl_mem bL = mkbuf(nullptr, sizeof(float) * nmat, CL_MEM_WRITE_ONLY);
  clSetKernelArg(k, 0, sizeof(cl_mem), &bR);
  clSetKernelArg(k, 1, sizeof(int), &M);
  clSetKernelArg(k, 2, sizeof(int), &iters);
  clSetKernelArg(k, 3, sizeof(cl_mem), &bV);
  clSetKernelArg(k, 4, sizeof(cl_mem), &bL);
  size_t g[1] = {(size_t)nmat * M};
  // kernels/power_iter.cl sizes its __local scratch lv/lw by PMAX and this
  // launch uses local size == M, so M > PMAX would overrun local memory and
  // corrupt neighbouring work-groups. The kernel now writes NaN sentinels
  // instead of corrupting memory, but refuse here so the caller hears about it.
  static const int kPowerIterMaxM = 64;   // must match PMAX in power_iter.cl
  if (M > kPowerIterMaxM) {
    fprintf(stderr, "[hpocl] power_iter: M=%d exceeds PMAX=%d -- raise PMAX in "
            "kernels/power_iter.cl (2*PMAX*8 bytes of local memory per "
            "work-group) or batch the matrices\n", M, kPowerIterMaxM);
    const float nan = std::nanf("");
    for (size_t i = 0; i < (size_t)nmat * M; ++i) v[i] = cplx(nan, nan);
    for (int i = 0; i < nmat; ++i) lambda[i] = nan;
    return;
  }
  size_t l[1] = {(size_t)M};
  ndrange(k, 1, g, l);
  rdbuf(bV, v, sizeof(cplx) * nmat * M);
  rdbuf(bL, lambda, sizeof(float) * nmat);
  clReleaseMemObject(bR); clReleaseMemObject(bV); clReleaseMemObject(bL);
}

void Runtime::deflate(cplx* R, int M, int nmat, const cplx* v, const float* lambda) {
  cl_kernel k = kernel("deflate");
  if (!k) return no_kernel("deflate");   // absent from this program
  // The kernel takes lambda as a SCALAR arg, so a batch cannot share one launch:
  // passing `lambda` with sizeof(float) hands every matrix lambda[0], which is
  // what this wrapper used to do while its comment advertised "batch via 3rd
  // dimension". Deflate one matrix per launch instead. (Making the kernel read
  // __global const float* lambda[gid.z] would allow a single launch; that is a
  // kernels/power_iter.cl change, deliberately not done here.)
  for (int i = 0; i < nmat; ++i) {
    cplx* Ri = R + (size_t)i * M * M;
    const cplx* vi = v + (size_t)i * M;
    cl_mem bR = mkbuf(Ri, sizeof(cplx) * M * M, CL_MEM_READ_WRITE);
    cl_mem bV = mkbuf(vi, sizeof(cplx) * M, CL_MEM_READ_ONLY);
    clSetKernelArg(k, 0, sizeof(cl_mem), &bR);
    clSetKernelArg(k, 1, sizeof(int), &M);
    clSetKernelArg(k, 2, sizeof(cl_mem), &bV);
    clSetKernelArg(k, 3, sizeof(float), &lambda[i]);
    size_t g[3] = {(size_t)M, (size_t)M, 1};
    ndrange(k, 3, g, nullptr);
    rdbuf(bR, Ri, sizeof(cplx) * M * M);
    clReleaseMemObject(bR); clReleaseMemObject(bV);
  }
}

void Runtime::switch_threshold(const float* lambda_min, const float* g_min,
                               const float* metric_dB, int count, float nt,
                               float nrf, float eta, float* tau_dB, uint8_t* mode) {
  cl_kernel k = kernel("switch_threshold");
  if (!k) return no_kernel("switch_threshold");   // absent from this program
  cl_mem b1 = mkbuf(lambda_min, sizeof(float) * count, CL_MEM_READ_ONLY);
  cl_mem b2 = mkbuf(g_min, sizeof(float) * count, CL_MEM_READ_ONLY);
  cl_mem b3 = mkbuf(metric_dB, sizeof(float) * count, CL_MEM_READ_ONLY);
  cl_mem b4 = mkbuf(nullptr, sizeof(float) * count, CL_MEM_WRITE_ONLY);
  cl_mem b5 = mkbuf(nullptr, count, CL_MEM_WRITE_ONLY);
  clSetKernelArg(k, 0, sizeof(cl_mem), &b1);
  clSetKernelArg(k, 1, sizeof(cl_mem), &b2);
  clSetKernelArg(k, 2, sizeof(cl_mem), &b3);
  clSetKernelArg(k, 3, sizeof(int), &count);
  clSetKernelArg(k, 4, sizeof(float), &nt);
  clSetKernelArg(k, 5, sizeof(float), &nrf);
  clSetKernelArg(k, 6, sizeof(float), &eta);
  clSetKernelArg(k, 7, sizeof(cl_mem), &b4);
  clSetKernelArg(k, 8, sizeof(cl_mem), &b5);
  size_t g[1] = {(size_t)count};
  ndrange(k, 1, g, nullptr);
  rdbuf(b4, tau_dB, sizeof(float) * count);
  rdbuf(b5, mode, count);
  clReleaseMemObject(b1); clReleaseMemObject(b2); clReleaseMemObject(b3);
  clReleaseMemObject(b4); clReleaseMemObject(b5);
}

void Runtime::wbb_apply(const cplx* W, int M, int N, const cplx* X, int SC, cplx* Y) {
  cl_kernel k = kernel("wbb_apply");
  if (!k) return no_kernel("wbb_apply");   // absent from this program
  cl_mem bW = mkbuf(W, sizeof(cplx) * M * N, CL_MEM_READ_ONLY);
  cl_mem bX = mkbuf(X, sizeof(cplx) * SC * N, CL_MEM_READ_ONLY);
  cl_mem bY = mkbuf(nullptr, sizeof(cplx) * SC * M, CL_MEM_WRITE_ONLY);
  clSetKernelArg(k, 0, sizeof(cl_mem), &bW);
  clSetKernelArg(k, 1, sizeof(int), &M);
  clSetKernelArg(k, 2, sizeof(int), &N);
  clSetKernelArg(k, 3, sizeof(cl_mem), &bX);
  clSetKernelArg(k, 4, sizeof(int), &SC);
  clSetKernelArg(k, 5, sizeof(cl_mem), &bY);
  size_t g[2] = {(size_t)M, (size_t)SC};
  ndrange(k, 2, g, nullptr);
  rdbuf(bY, Y, sizeof(cplx) * SC * M);
  clReleaseMemObject(bW); clReleaseMemObject(bX); clReleaseMemObject(bY);
}

void Runtime::thp_precoder(const cplx* B, const cplx* F, const cplx* X, int SC,
                           int M, int K, float tau, cplx* Y) {
  cl_kernel k = kernel("thp_precoder");
  if (!k) return no_kernel("thp_precoder");
  // thp_precoder.cl keeps the feedback history in __private float2 b[KMAX] and
  // rejects K > KMAX by returning, which would leave Y holding whatever the
  // caller put there. Refuse here instead and NaN-fill, as power_iter does.
  // tools/check_capacity.py keeps kThpMaxK in step with KMAX.
  static const int kThpMaxK = 16;   // must match KMAX in kernels/thp_precoder.cl
  if (K > kThpMaxK || K < 1) {
    fprintf(stderr, "[hpocl] thp_precoder: K=%d outside [1, KMAX=%d] -- raise "
            "KMAX in kernels/thp_precoder.cl (KMAX*8 bytes of private memory "
            "per work-item) or split the streams\n", K, kThpMaxK);
    const float nan = std::nanf("");
    for (size_t i = 0; i < (size_t)SC * M; ++i) Y[i] = cplx(nan, nan);
    return;
  }   // absent from this program
  cl_mem bB = mkbuf(B, sizeof(cplx) * K * K, CL_MEM_READ_ONLY);
  cl_mem bF = mkbuf(F, sizeof(cplx) * M * K, CL_MEM_READ_ONLY);
  cl_mem bX = mkbuf(X, sizeof(cplx) * SC * K, CL_MEM_READ_ONLY);
  cl_mem bY = mkbuf(nullptr, sizeof(cplx) * SC * M, CL_MEM_WRITE_ONLY);
  clSetKernelArg(k, 0, sizeof(cl_mem), &bB);
  clSetKernelArg(k, 1, sizeof(cl_mem), &bF);
  clSetKernelArg(k, 2, sizeof(cl_mem), &bX);
  clSetKernelArg(k, 3, sizeof(int), &SC);
  clSetKernelArg(k, 4, sizeof(int), &M);
  clSetKernelArg(k, 5, sizeof(int), &K);
  clSetKernelArg(k, 6, sizeof(float), &tau);
  clSetKernelArg(k, 7, sizeof(cl_mem), &bY);
  size_t g[1] = {(size_t)SC};
  ndrange(k, 1, g, nullptr);
  rdbuf(bY, Y, sizeof(cplx) * SC * M);
  clReleaseMemObject(bB); clReleaseMemObject(bF); clReleaseMemObject(bX);
  clReleaseMemObject(bY);
}

void Runtime::threshold_cf(const double* lam, int C, int K, int d, int M,
                           int variant, double* tau_cf_dB, double* tau_dB,
                           int* flags) {
  cl_kernel k_cf = kernel("threshold_cf");
  if (!k_cf) return no_kernel("threshold_cf");
  // Host mirror of the kernel's entry guard, so the PS path refuses the same
  // configurations the PL does instead of launching a kernel that will reject
  // every channel and hand back NaN. The constants must match MAXT/MAXK/MAXD in
  // kernels/threshold_cf.cl; tools/check_capacity.py enforces MAXT and MAXD.
  //   T = K*d streams            <= MAXT
  //   K users                    <= MAXK (== MAXT, since d >= 1)
  //   5*ceil(L/2) distance slots <= MAXD  (the folded deficit is the binding one)
  //   M must be a perfect square: the kernel derives L = round(sqrt(M)) and a
  //   non-square M silently computes a deficit for the wrong constellation
  //   instead of failing, so reject it here (it is not a legal QAM order).
  static const int kCfMaxT = 64;    // == MAXT in kernels/threshold_cf.cl
  static const int kCfMaxD = 128;   // == MAXD in kernels/threshold_cf.cl
  const int T = K * d;
  const int L = (int)(std::sqrt((double)M) + 0.5);
  if (C < 1 || d < 1 || T < 1 || T > kCfMaxT || K > kCfMaxT || M < 4 ||
      L * L != M || 5 * ((L + 1) / 2) > kCfMaxD) {
    fprintf(stderr, "[hpocl] threshold_cf: C=%d K=%d d=%d M=%d outside the "
            "compiled capacity (MAXT=%d, 5*ceil(L/2) <= MAXD=%d, M a perfect "
            "square >= 4) -- see the capacity block in kernels/threshold_cf.cl. "
            "Reporting flag=4 for all %d channels without launching.\n",
            C, K, d, M, kCfMaxT, kCfMaxD, C > 0 ? C : 0);
    for (int c = 0; c < C; ++c) {
      tau_cf_dB[c] = std::nan(""); tau_dB[c] = std::nan(""); flags[c] = 4;
    }
    return;
  }
  // The element width is the BUILD's, not the device's: `real` in
  // kernels/threshold_cf.cl is float unless the program was built with
  // -DHPOCL_FP64, and a wrapper that guessed from CL_DEVICE_DOUBLE_FP_CONFIG would
  // hand an fp32 kernel double buffers -- measured on an NVIDIA L4 (fp64-capable)
  // with the PL configuration forced: every tau came back 0.0 while the int flags
  // still looked sane, so the bench passed. The host interface stays double.
  std::vector<float> lamf, cff, tauf;
  const void* src = (const void*)lam;
  size_t es = sizeof(double);
  if (!prog_fp64_) {
    lamf.assign(lam, lam + (size_t)C * K * d);
    cff.resize(C); tauf.resize(C);
    src = lamf.data(); es = sizeof(float);
  }
  cl_mem bL = mkbuf(src, es * C * K * d, CL_MEM_READ_ONLY);
  cl_mem b1 = mkbuf(nullptr, es * C, CL_MEM_WRITE_ONLY);
  cl_mem b2 = mkbuf(nullptr, es * C, CL_MEM_WRITE_ONLY);
  cl_mem b3 = mkbuf(nullptr, sizeof(int) * C, CL_MEM_WRITE_ONLY);
  clSetKernelArg(k_cf, 0, sizeof(cl_mem), &bL);
  clSetKernelArg(k_cf, 1, sizeof(int), &C);
  clSetKernelArg(k_cf, 2, sizeof(int), &K);
  clSetKernelArg(k_cf, 3, sizeof(int), &d);
  clSetKernelArg(k_cf, 4, sizeof(int), &M);
  clSetKernelArg(k_cf, 5, sizeof(int), &variant);
  clSetKernelArg(k_cf, 6, sizeof(cl_mem), &b1);
  clSetKernelArg(k_cf, 7, sizeof(cl_mem), &b2);
  clSetKernelArg(k_cf, 8, sizeof(cl_mem), &b3);
  size_t g[1] = {(size_t)C};
  ndrange(k_cf, 1, g, nullptr);
  if (prog_fp64_) {           // ... and the readback matches the write width
    rdbuf(b1, tau_cf_dB, sizeof(double) * C);
    rdbuf(b2, tau_dB, sizeof(double) * C);
  } else {
    rdbuf(b1, cff.data(), sizeof(float) * C);
    rdbuf(b2, tauf.data(), sizeof(float) * C);
    for (int i = 0; i < C; ++i) { tau_cf_dB[i] = cff[i]; tau_dB[i] = tauf[i]; }
  }
  rdbuf(b3, flags, sizeof(int) * C);
  clReleaseMemObject(bL); clReleaseMemObject(b1);
  clReleaseMemObject(b2); clReleaseMemObject(b3);
}

//---------------------------------------------------------------------------
// decomposition operators: fp32 kernels, double host interface
//---------------------------------------------------------------------------
namespace {
using dcmp = std::complex<double>;

// kernels/decomp.cl sizes every __private array by a PER-KERNEL ceiling and
// rejects out-of-contract dimensions by returning, which leaves the output
// buffers at the zeros they were created with. Refuse on the host instead, the
// way power_iter already does for PMAX, so the PS can never hand back a number
// the fabric could not have produced. tools/check_capacity.py cross-checks all
// three pairs against the #defines in the kernel source, so they cannot drift.
//
// jacobi_svd gets the large pair because the BD front end feeds it the
// interference stack: null_basis factorises A [(K-1)*Nr x NRF] = 28 x 32 at
// K=8, Nr=4, NRF=32 and analog_precoder factorises H_u [Nr x Nt] = 4 x 16
// (ref/frontend.cpp). gmd/ucd/cbd only ever see rows = nul = NRF-(K-1)*Nr = 4
// and d <= 8 streams, and a static __private array is allocated at its ceiling
// for every work-item whether or not the launch uses it -- so sharing one large
// pair would cost gmd_k 12.5x, ucd_k 21.7x and cbd_k 14.4x in private footprint
// for nothing (the arithmetic is in the decomp.cl header comment).
constexpr int kJacobiMaxM = 32;   // == DECOMP_MAXM in kernels/decomp.cl
constexpr int kJacobiMaxD = 40;   // == DECOMP_MAXD in kernels/decomp.cl
constexpr int kGmdMaxM    = 16;   // == GMD_MAXM    (gmd_k and ucd_k)
constexpr int kGmdMaxD    = 8;    // == GMD_MAXD
constexpr int kCbdMaxM    = 16;   // == CBD_MAXM    (cbd_k urows and vrows)
constexpr int kCbdMaxD    = 8;    // == CBD_MAXD

// A refused or failed launch must not leave the caller's buffer holding its
// previous contents: NaN for floats, -1 for counts (active_count is >= 1 by
// contract, so -1 cannot be mistaken for a real answer).
void nan_fill(dcmp* p, size_t n) {
  if (!p) return;
  const dcmp v(std::nan(""), std::nan(""));
  for (size_t i = 0; i < n; ++i) p[i] = v;
}
void nan_fill(double* p, size_t n) {
  if (!p) return;
  for (size_t i = 0; i < n; ++i) p[i] = std::nan("");
}
void nan_fill(int* p, size_t n) {
  if (!p) return;
  for (size_t i = 0; i < n; ++i) p[i] = -1;
}
// maxm/maxd are passed in rather than read from a global because each decomp
// kernel has its own ceiling; `macro` names it in the message so the fix is
// unambiguous (the kernel #define and this mirror move together, and
// tools/check_capacity.py fails the build if they do not).
bool dims_ok(const char* who, int rows, int d, int maxm, int maxd,
             const char* macro) {
  if (rows >= 1 && rows <= maxm && d >= 1 && d <= maxd) return true;
  fprintf(stderr, "[hpocl] %s: rows=%d d=%d outside the kernel's capacity "
          "(%s: MAXM=%d MAXD=%d in kernels/decomp.cl) -- raise them there and "
          "in the matching k*Max* mirror, outputs NaN-filled\n", who, rows, d,
          macro, maxm, maxd);
  return false;
}
std::vector<float> pack2(const dcmp* v, size_t n) {
  std::vector<float> f(2 * n);
  for (size_t i = 0; i < n; ++i) { f[2 * i] = (float)v[i].real(); f[2 * i + 1] = (float)v[i].imag(); }
  return f;
}
void unpack2(const std::vector<float>& f, dcmp* v, size_t n) {
  for (size_t i = 0; i < n; ++i) v[i] = dcmp(f[2 * i], f[2 * i + 1]);
}
cl_mem buf(cl_context cx, std::vector<float>& h) {
  cl_int e = CL_SUCCESS;
  return clCreateBuffer(cx, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                        4 * h.size(), h.data(), &e);
}
cl_mem buf(cl_context cx, std::vector<int>& h) {   // same 4-byte element size
  cl_int e = CL_SUCCESS;
  return clCreateBuffer(cx, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                        4 * h.size(), h.data(), &e);
}
}  // namespace

void Runtime::jacobi_svd(const dcmp* A, int C, int m, int d, int sweeps,
                         dcmp* U, double* S, dcmp* V) {
  cl_kernel k = kernel("jacobi_svd");
  if (!k) return no_kernel("jacobi_svd");
  if (!dims_ok("jacobi_svd", m, d, kJacobiMaxM, kJacobiMaxD, "DECOMP_MAXM/DECOMP_MAXD")) {
    nan_fill(U, (size_t)C * m * d); nan_fill(S, (size_t)C * d);
    nan_fill(V, (size_t)C * d * d);
    return;
  }
  std::vector<float> hA = pack2(A, (size_t)C * m * d), hU(2 * (size_t)C * m * d),
      hS(C * d), hV(2 * (size_t)C * d * d);
  cl_mem bA = buf(ctx_, hA), bU = buf(ctx_, hU), bS = buf(ctx_, hS), bV = buf(ctx_, hV);
  clSetKernelArg(k, 0, sizeof(cl_mem), &bA);
  clSetKernelArg(k, 1, sizeof(int), &C);
  clSetKernelArg(k, 2, sizeof(int), &m);
  clSetKernelArg(k, 3, sizeof(int), &d);
  clSetKernelArg(k, 4, sizeof(int), &sweeps);
  clSetKernelArg(k, 5, sizeof(cl_mem), &bU);
  clSetKernelArg(k, 6, sizeof(cl_mem), &bS);
  clSetKernelArg(k, 7, sizeof(cl_mem), &bV);
  size_t g[1] = {(size_t)C};
  if (!ndrange(k, 1, g, nullptr)) {
    nan_fill(U, (size_t)C * m * d); nan_fill(S, (size_t)C * d);
    nan_fill(V, (size_t)C * d * d);
    clReleaseMemObject(bA); clReleaseMemObject(bU);
    clReleaseMemObject(bS); clReleaseMemObject(bV);
    return;
  }
  std::vector<float> rU(hU.size()), rS(hS.size()), rV(hV.size());
  clEnqueueReadBuffer(q_, bU, CL_TRUE, 0, 4 * rU.size(), rU.data(), 0, nullptr, nullptr);
  clEnqueueReadBuffer(q_, bS, CL_TRUE, 0, 4 * rS.size(), rS.data(), 0, nullptr, nullptr);
  clEnqueueReadBuffer(q_, bV, CL_TRUE, 0, 4 * rV.size(), rV.data(), 0, nullptr, nullptr);
  if (U) unpack2(rU, U, (size_t)C * m * d);
  if (S) for (int i = 0; i < C * d; ++i) S[i] = rS[i];
  if (V) unpack2(rV, V, (size_t)C * d * d);
  clReleaseMemObject(bA); clReleaseMemObject(bU);
  clReleaseMemObject(bS); clReleaseMemObject(bV);
}

void Runtime::gmd_op(const dcmp* P0, const double* sigma, int C, int rows,
                     int d, dcmp* R, dcmp* P) {
  cl_kernel k = kernel("gmd_k");
  if (!k) return no_kernel("gmd_k");
  if (!dims_ok("gmd_op", rows, d, kGmdMaxM, kGmdMaxD, "GMD_MAXM/GMD_MAXD")) {
    nan_fill(R, (size_t)C * d * d); nan_fill(P, (size_t)C * rows * d);
    return;
  }
  std::vector<float> hP0 = pack2(P0, (size_t)C * rows * d), hS(C * d),
      hR(2 * (size_t)C * d * d), hP(2 * (size_t)C * rows * d);
  for (int i = 0; i < C * d; ++i) hS[i] = (float)sigma[i];
  cl_mem bP0 = buf(ctx_, hP0), bS = buf(ctx_, hS), bR = buf(ctx_, hR), bP = buf(ctx_, hP);
  clSetKernelArg(k, 0, sizeof(cl_mem), &bP0);
  clSetKernelArg(k, 1, sizeof(cl_mem), &bS);
  clSetKernelArg(k, 2, sizeof(int), &C);
  clSetKernelArg(k, 3, sizeof(int), &rows);
  clSetKernelArg(k, 4, sizeof(int), &d);
  clSetKernelArg(k, 5, sizeof(cl_mem), &bR);
  clSetKernelArg(k, 6, sizeof(cl_mem), &bP);
  size_t g[1] = {(size_t)C};
  if (!ndrange(k, 1, g, nullptr)) {
    nan_fill(R, (size_t)C * d * d); nan_fill(P, (size_t)C * rows * d);
    clReleaseMemObject(bP0); clReleaseMemObject(bS);
    clReleaseMemObject(bR);  clReleaseMemObject(bP);
    return;
  }
  std::vector<float> rR(hR.size()), rP(hP.size());
  clEnqueueReadBuffer(q_, bR, CL_TRUE, 0, 4 * rR.size(), rR.data(), 0, nullptr, nullptr);
  clEnqueueReadBuffer(q_, bP, CL_TRUE, 0, 4 * rP.size(), rP.data(), 0, nullptr, nullptr);
  unpack2(rR, R, (size_t)C * d * d);
  unpack2(rP, P, (size_t)C * rows * d);
  clReleaseMemObject(bP0); clReleaseMemObject(bS);
  clReleaseMemObject(bR); clReleaseMemObject(bP);
}

void Runtime::ucd_op(const dcmp* Veff, const double* sigma, int C, int rows,
                     int d, double snr, bool waterfill, double min_pl, dcmp* Vd,
                     double* rho_eq, dcmp* upper, double* pl) {
  cl_kernel k = kernel("ucd_k");
  if (!k) return no_kernel("ucd_k");
  if (!dims_ok("ucd_op", rows, d, kGmdMaxM, kGmdMaxD, "GMD_MAXM/GMD_MAXD")) {
    nan_fill(Vd, (size_t)C * rows * d); nan_fill(rho_eq, C);
    nan_fill(upper, (size_t)C * d * d); nan_fill(pl, (size_t)C * d);
    return;
  }
  std::vector<float> hV = pack2(Veff, (size_t)C * rows * d), hS(C * d),
      hVd(2 * (size_t)C * rows * d), hRho(C), hUp(2 * (size_t)C * d * d), hPl(C * d);
  for (int i = 0; i < C * d; ++i) hS[i] = (float)sigma[i];
  cl_mem bV = buf(ctx_, hV), bS = buf(ctx_, hS), bVd = buf(ctx_, hVd),
         bRho = buf(ctx_, hRho), bUp = buf(ctx_, hUp), bPl = buf(ctx_, hPl);
  float fs = (float)snr; int wf = waterfill ? 1 : 0; float mpl = (float)min_pl;
  clSetKernelArg(k, 0, sizeof(cl_mem), &bV);
  clSetKernelArg(k, 1, sizeof(cl_mem), &bS);
  clSetKernelArg(k, 2, sizeof(int), &C);
  clSetKernelArg(k, 3, sizeof(int), &rows);
  clSetKernelArg(k, 4, sizeof(int), &d);
  clSetKernelArg(k, 5, sizeof(float), &fs);
  clSetKernelArg(k, 6, sizeof(int), &wf);
  clSetKernelArg(k, 7, sizeof(float), &mpl);
  clSetKernelArg(k, 8, sizeof(cl_mem), &bVd);
  clSetKernelArg(k, 9, sizeof(cl_mem), &bRho);
  clSetKernelArg(k, 10, sizeof(cl_mem), &bUp);
  clSetKernelArg(k, 11, sizeof(cl_mem), &bPl);
  size_t g[1] = {(size_t)C};
  if (!ndrange(k, 1, g, nullptr)) {
    nan_fill(Vd, (size_t)C * rows * d); nan_fill(rho_eq, C);
    nan_fill(upper, (size_t)C * d * d); nan_fill(pl, (size_t)C * d);
    clReleaseMemObject(bV); clReleaseMemObject(bS); clReleaseMemObject(bVd);
    clReleaseMemObject(bRho); clReleaseMemObject(bUp); clReleaseMemObject(bPl);
    return;
  }
  std::vector<float> rVd(hVd.size()), rRho(C), rUp(hUp.size()), rPl(hPl.size());
  clEnqueueReadBuffer(q_, bVd, CL_TRUE, 0, 4 * rVd.size(), rVd.data(), 0, nullptr, nullptr);
  clEnqueueReadBuffer(q_, bRho, CL_TRUE, 0, 4 * rRho.size(), rRho.data(), 0, nullptr, nullptr);
  clEnqueueReadBuffer(q_, bUp, CL_TRUE, 0, 4 * rUp.size(), rUp.data(), 0, nullptr, nullptr);
  clEnqueueReadBuffer(q_, bPl, CL_TRUE, 0, 4 * rPl.size(), rPl.data(), 0, nullptr, nullptr);
  unpack2(rVd, Vd, (size_t)C * rows * d);
  for (int i = 0; i < C; ++i) rho_eq[i] = rRho[i];
  unpack2(rUp, upper, (size_t)C * d * d);
  for (int i = 0; i < C * d; ++i) pl[i] = rPl[i];
  clReleaseMemObject(bV); clReleaseMemObject(bS); clReleaseMemObject(bVd);
  clReleaseMemObject(bRho); clReleaseMemObject(bUp); clReleaseMemObject(bPl);
}

void Runtime::cbd_op(const dcmp* U, const dcmp* V, const double* sigma, int C,
                     int urows, int vrows, int d, double snr,
                     double pair_threshold, dcmp* Q, dcmp* R, dcmp* P,
                     int* active_count) {
  cl_kernel k = kernel("cbd_k");
  if (!k) return no_kernel("cbd_k");
  // cbd_k bounds its private q/p arrays by CBD_MAXM and A/B by CBD_MAXD, so
  // urows and vrows are both checked against CBD_MAXM.
  if (!dims_ok("cbd_op", urows, d, kCbdMaxM, kCbdMaxD, "CBD_MAXM/CBD_MAXD") ||
      !dims_ok("cbd_op", vrows, d, kCbdMaxM, kCbdMaxD, "CBD_MAXM/CBD_MAXD")) {
    nan_fill(Q, (size_t)C * urows * d); nan_fill(R, (size_t)C * d * d);
    nan_fill(P, (size_t)C * vrows * d); nan_fill(active_count, C);
    return;
  }
  std::vector<float> hU = pack2(U, (size_t)C * urows * d),
      hV = pack2(V, (size_t)C * vrows * d), hS(C * d),
      hQ(2 * (size_t)C * urows * d), hR(2 * (size_t)C * d * d),
      hP(2 * (size_t)C * vrows * d);
  std::vector<int> hAct(C, 0);
  for (int i = 0; i < C * d; ++i) hS[i] = (float)sigma[i];
  cl_mem bU = buf(ctx_, hU), bV = buf(ctx_, hV), bS = buf(ctx_, hS),
         bQ = buf(ctx_, hQ), bR = buf(ctx_, hR), bP = buf(ctx_, hP),
         bA = buf(ctx_, hAct);
  float fs = (float)snr, fthr = (float)pair_threshold;
  clSetKernelArg(k, 0, sizeof(cl_mem), &bU);
  clSetKernelArg(k, 1, sizeof(cl_mem), &bV);
  clSetKernelArg(k, 2, sizeof(cl_mem), &bS);
  clSetKernelArg(k, 3, sizeof(int), &C);
  clSetKernelArg(k, 4, sizeof(int), &urows);
  clSetKernelArg(k, 5, sizeof(int), &vrows);
  clSetKernelArg(k, 6, sizeof(int), &d);
  clSetKernelArg(k, 7, sizeof(float), &fs);
  clSetKernelArg(k, 8, sizeof(float), &fthr);
  clSetKernelArg(k, 9, sizeof(cl_mem), &bQ);
  clSetKernelArg(k, 10, sizeof(cl_mem), &bR);
  clSetKernelArg(k, 11, sizeof(cl_mem), &bP);
  clSetKernelArg(k, 12, sizeof(cl_mem), &bA);
  size_t g[1] = {(size_t)C};
  if (!ndrange(k, 1, g, nullptr)) {
    nan_fill(Q, (size_t)C * urows * d); nan_fill(R, (size_t)C * d * d);
    nan_fill(P, (size_t)C * vrows * d); nan_fill(active_count, C);
    clReleaseMemObject(bU); clReleaseMemObject(bV); clReleaseMemObject(bS);
    clReleaseMemObject(bQ); clReleaseMemObject(bR); clReleaseMemObject(bP);
    clReleaseMemObject(bA);
    return;
  }
  std::vector<float> rQ(hQ.size()), rR(hR.size()), rP(hP.size());
  std::vector<int> rA(C, 0);
  clEnqueueReadBuffer(q_, bQ, CL_TRUE, 0, 4 * rQ.size(), rQ.data(), 0, nullptr, nullptr);
  clEnqueueReadBuffer(q_, bR, CL_TRUE, 0, 4 * rR.size(), rR.data(), 0, nullptr, nullptr);
  clEnqueueReadBuffer(q_, bP, CL_TRUE, 0, 4 * rP.size(), rP.data(), 0, nullptr, nullptr);
  clEnqueueReadBuffer(q_, bA, CL_TRUE, 0, 4 * C, rA.data(), 0, nullptr, nullptr);
  if (Q) unpack2(rQ, Q, (size_t)C * urows * d);
  if (R) unpack2(rR, R, (size_t)C * d * d);
  if (P) unpack2(rP, P, (size_t)C * vrows * d);
  if (active_count) for (int i = 0; i < C; ++i) active_count[i] = rA[i];
  clReleaseMemObject(bU); clReleaseMemObject(bV); clReleaseMemObject(bS);
  clReleaseMemObject(bQ); clReleaseMemObject(bR); clReleaseMemObject(bP);
  clReleaseMemObject(bA);
}

//---------------------------------------------------------------------------
// per-symbol datapath operators (kernels/datapath.cl)
//
// These are the fp32 kernels, so the host buffers are fp32 too (cplx ==
// complex<float>, bit-identical to the device float2): unlike the
// decompositions there is no double interface to convert to and from.
// Each wrapper is one synchronous launch, because ndrange() calls clFinish().
// A PS control path and the benches want exactly that. A PL deployment that
// needs the stages to stay on device should enqueue them itself through
// kernel()/queue() and read back only the GMI.
//
// MMAX guard: softdemod_llr_k caches all M = 2^m squared distances in private
// memory, so the kernels cap the constellation at 256 (kernels/datapath.cl).
// Rejecting it here turns a silent all-outputs-untouched launch into a message.
namespace {
bool dp_dims_ok(const char* who, int m, int nsymtot) {
  if (m < 2 || (m & 1) || m > 8 || nsymtot < 1) {
    fprintf(stderr, "[hpocl] %s: bad shape m=%d nsymtot=%d (m must be even, "
            "2..8; the kernels cap the constellation at MMAX=256)\n",
            who, m, nsymtot);
    return false;
  }
  return true;
}

// `grp` is bits per demod work-item: the launch is gid(1) = ceil(m/grp) and group
// g owns bits [g*grp, min(g*grp+grp, m)). grp < 1 makes EVERY group's range empty
// and grp > m leaves the bits above the first group unwritten, so both would hand
// the caller a buffer the kernel never filled. kernels/datapath.cl already refuses
// them by writing a NaN sentinel; this refuses one level earlier, where the
// caller's own argument is still visible, and NaN-fills the SAME sentinel, so the
// observable result is unchanged (thp_precoder does this for K > KMAX). The fill
// is the point: an early return without it would leave the caller reading whatever
// its uninitialised buffer held -- a plausible number instead of an error, which
// is the failure mode dp_dims_ok exists to prevent.
bool dp_grp_ok(const char* who, int m, int grp, float* llr, int nsymtot) {
  if (grp >= 1 && grp <= m) return true;
  fprintf(stderr, "[hpocl] %s: bad grp=%d for m=%d (bits per work-item, must be "
          "in 1..m); the LLR output is NaN-filled\n", who, grp, m);
  for (size_t i = 0; i < (size_t)nsymtot * m; ++i) llr[i] = NAN;
  return false;
}

// tx_combine_k's work-group shape, chosen from a MEASUREMENT rather than left to
// the implementation. The kernel shares its digital stage through __local, so the
// group shape decides how many antennas reuse one bb[r][sym]; and a NULL local
// size -- what this wrapper used to pass -- lets the implementation pick, which on
// the Apple M4 GPU means 32x1. Kernel EVENT time at the largest shape in any
// config pool (Nt=128, NRF=32, T=32, nsym=8192), Apple M4 GPU, medians of 11
// rounds with the group shapes run in ALTERNATING order inside one process, and
// the output bit-identical in EVERY configuration (same |x| sample 73.988710,
// drift 0.0e+00 between rounds):
//     untiled (-DTXC_LOCAL_MAX=4), any group shape   316-326 us
//     tiled, NULL local size (the device picks 32x1) 106-111 us
//     tiled, 8x32 / 16x16 / 32x8 / 128x1              64.7 / 43.2 / 32.6 / 33.2
//     tiled, 64x4                                     19.6-24.5 us
// i.e. the tiling is worth 2.9x at the shape a NULL launch gets and 13-16x at the
// best explicit one, and the group shape alone is worth 4.3-5.7x on top of the
// tiling. COLD-START CAVEAT, because it produced one wrong conclusion before the
// alternating measurement: the FIRST configuration run in a fresh process reads
// ~66.7 us for 64x4 and settles to 19.6-24.5 us after a few rounds of the same
// kernel, while NULL reads 106-111 us cold or warm. Quote warm medians from an
// alternating sequence; a single-configuration fresh process understates the win
// by 2.7x and can make a 4.5x change look like a wash.
// The rule: the largest power of two <= 64 along the antenna that the kernel's
// CL_KERNEL_WORK_GROUP_SIZE and the device's CL_DEVICE_MAX_WORK_ITEM_SIZES both
// accept, times 4 along the symbol, with the grid ROUNDED UP to a multiple of it.
// Rounding up is safe because tx_combine_k guards a >= Nt and sym >= nsym on every
// path, including the barrier side of the tiled one, so the extra work-items are
// inert; it is also the only way to pass an explicit local size at all, since
// OpenCL 1.2 requires global % local == 0 while Nt/nsym are the caller's.
// A launch the device still rejects falls back to NULL at the call site, so the
// worst case is the old behaviour and not a failed operator.
size_t txc_group(cl_kernel k, cl_device_id dev, int want0, size_t* l1_out) {
  size_t kg = 0, wis[3] = {0, 0, 0};
  clGetKernelWorkGroupInfo(k, dev, CL_KERNEL_WORK_GROUP_SIZE, sizeof kg, &kg,
                           nullptr);
  clGetDeviceInfo(dev, CL_DEVICE_MAX_WORK_ITEM_SIZES, sizeof wis, wis, nullptr);
  size_t l0 = (size_t)want0, l1 = 4;
  if (wis[1] && l1 > wis[1]) l1 = wis[1];
  if (wis[0] && l0 > wis[0]) l0 = wis[0];
  while (l0 > 1 && kg && l0 * l1 > kg) l0 >>= 1;     // shrink the antenna side
  while (l1 > 1 && kg && l0 * l1 > kg) l1 >>= 1;     // first, then the symbol one
  *l1_out = l1 ? l1 : 1;
  return l0 ? l0 : 1;
}
}  // namespace

void Runtime::qam_map(const uint8_t* bits, int m, int nsymtot, cplx* s) {
  cl_kernel k = kernel("qam_map_k");
  if (!k) return no_kernel("qam_map_k");
  if (!dp_dims_ok("qam_map", m, nsymtot)) return;
  cl_mem bB = mkbuf(bits, (size_t)nsymtot * m, CL_MEM_READ_ONLY);
  cl_mem bS = mkbuf(nullptr, sizeof(cplx) * nsymtot, CL_MEM_WRITE_ONLY);
  clSetKernelArg(k, 0, sizeof(cl_mem), &bB);
  clSetKernelArg(k, 1, sizeof(int), &m);
  clSetKernelArg(k, 2, sizeof(int), &nsymtot);
  clSetKernelArg(k, 3, sizeof(cl_mem), &bS);
  size_t g[1] = {(size_t)nsymtot};                 // one work-item per symbol
  ndrange(k, 1, g, nullptr);
  rdbuf(bS, s, sizeof(cplx) * nsymtot);
  clReleaseMemObject(bB); clReleaseMemObject(bS);
}

void Runtime::tx_combine(const cplx* Frf, int Nt, int NRF, const cplx* Fbb,
                         int T, const cplx* s, int nsym, cplx* x) {
  cl_kernel k = kernel("tx_combine_k");
  if (!k) return no_kernel("tx_combine_k");
  cl_mem bRf = mkbuf(Frf, sizeof(cplx) * (size_t)Nt * NRF, CL_MEM_READ_ONLY);
  cl_mem bBb = mkbuf(Fbb, sizeof(cplx) * (size_t)NRF * T, CL_MEM_READ_ONLY);
  cl_mem bS  = mkbuf(s,   sizeof(cplx) * (size_t)T * nsym, CL_MEM_READ_ONLY);
  cl_mem bX  = mkbuf(nullptr, sizeof(cplx) * (size_t)Nt * nsym, CL_MEM_WRITE_ONLY);
  clSetKernelArg(k, 0, sizeof(cl_mem), &bRf);
  clSetKernelArg(k, 1, sizeof(int), &Nt);
  clSetKernelArg(k, 2, sizeof(int), &NRF);
  clSetKernelArg(k, 3, sizeof(cl_mem), &bBb);
  clSetKernelArg(k, 4, sizeof(int), &T);
  clSetKernelArg(k, 5, sizeof(cl_mem), &bS);
  clSetKernelArg(k, 6, sizeof(int), &nsym);
  clSetKernelArg(k, 7, sizeof(cl_mem), &bX);
  // (antenna, symbol) grid, ROUNDED UP to a multiple of the measured-best group
  // shape; see txc_group() for the numbers and for why the padding is inert.
  size_t l1 = 4;
  const size_t l0 = txc_group(k, dev_, 64, &l1);
  size_t g[2] = {((size_t)Nt + l0 - 1) / l0 * l0,
                 ((size_t)nsym + l1 - 1) / l1 * l1};
  size_t loc[2] = {l0, l1};
  if (!ndrange(k, 2, g, loc)) {                    // rejected: the old behaviour
    size_t gn[2] = {(size_t)Nt, (size_t)nsym};
    if (!ndrange(k, 2, gn, nullptr))
      fprintf(stderr, "[hpocl] tx_combine: launch failed with AND without an "
              "explicit local size (%zux%zu) -- x NOT written\n", l0, l1);
  }
  rdbuf(bX, x, sizeof(cplx) * (size_t)Nt * nsym);
  clReleaseMemObject(bRf); clReleaseMemObject(bBb);
  clReleaseMemObject(bS);  clReleaseMemObject(bX);
}

void Runtime::channel_apply(const cplx* H, int Nr, int Nt, const cplx* x,
                            int nsym, const cplx* n, cplx* y) {
  cl_kernel k = kernel("channel_apply_k");
  if (!k) return no_kernel("channel_apply_k");
  cl_mem bH = mkbuf(H, sizeof(cplx) * (size_t)Nr * Nt, CL_MEM_READ_ONLY);
  cl_mem bX = mkbuf(x, sizeof(cplx) * (size_t)Nt * nsym, CL_MEM_READ_ONLY);
  cl_mem bN = mkbuf(n, sizeof(cplx) * (size_t)Nr * nsym, CL_MEM_READ_ONLY);
  cl_mem bY = mkbuf(nullptr, sizeof(cplx) * (size_t)Nr * nsym, CL_MEM_WRITE_ONLY);
  clSetKernelArg(k, 0, sizeof(cl_mem), &bH);
  clSetKernelArg(k, 1, sizeof(int), &Nr);
  clSetKernelArg(k, 2, sizeof(int), &Nt);
  clSetKernelArg(k, 3, sizeof(cl_mem), &bX);
  clSetKernelArg(k, 4, sizeof(int), &nsym);
  clSetKernelArg(k, 5, sizeof(cl_mem), &bN);
  clSetKernelArg(k, 6, sizeof(cl_mem), &bY);
  size_t g[2] = {(size_t)Nr, (size_t)nsym};        // (rx antenna, symbol)
  ndrange(k, 2, g, nullptr);
  rdbuf(bY, y, sizeof(cplx) * (size_t)Nr * nsym);
  clReleaseMemObject(bH); clReleaseMemObject(bX);
  clReleaseMemObject(bN); clReleaseMemObject(bY);
}

void Runtime::rx_combine(const cplx* W, int U, int Nr, int d, const cplx* y,
                         int nsym, const cplx* diag_b, cplx* z) {
  cl_kernel k = kernel("rx_combine_k");
  if (!k) return no_kernel("rx_combine_k");
  const int T = U * d;
  cl_mem bW = mkbuf(W, sizeof(cplx) * (size_t)U * Nr * d, CL_MEM_READ_ONLY);
  cl_mem bY = mkbuf(y, sizeof(cplx) * (size_t)Nr * nsym, CL_MEM_READ_ONLY);
  // a NULL cl_mem is a legal kernel argument and is how the kernel distinguishes
  // "divide by diag_b" from "return the raw combiner output"
  cl_mem bD = diag_b ? mkbuf(diag_b, sizeof(cplx) * T, CL_MEM_READ_ONLY) : nullptr;
  cl_mem bZ = mkbuf(nullptr, sizeof(cplx) * (size_t)T * nsym, CL_MEM_WRITE_ONLY);
  clSetKernelArg(k, 0, sizeof(cl_mem), &bW);
  clSetKernelArg(k, 1, sizeof(int), &U);
  clSetKernelArg(k, 2, sizeof(int), &Nr);
  clSetKernelArg(k, 3, sizeof(int), &d);
  clSetKernelArg(k, 4, sizeof(cl_mem), &bY);
  clSetKernelArg(k, 5, sizeof(int), &nsym);
  clSetKernelArg(k, 6, sizeof(cl_mem), &bD);
  clSetKernelArg(k, 7, sizeof(cl_mem), &bZ);
  size_t g[2] = {(size_t)T, (size_t)nsym};         // (stream, symbol)
  ndrange(k, 2, g, nullptr);
  rdbuf(bZ, z, sizeof(cplx) * (size_t)T * nsym);
  clReleaseMemObject(bW); clReleaseMemObject(bY); clReleaseMemObject(bZ);
  if (bD) clReleaseMemObject(bD);
}

void Runtime::modulo_fold(const cplx* in, int nsymtot, float period, cplx* out) {
  cl_kernel k = kernel("modulo_fold_k");
  if (!k) return no_kernel("modulo_fold_k");
  cl_mem bI = mkbuf(in, sizeof(cplx) * nsymtot, CL_MEM_READ_ONLY);
  cl_mem bO = mkbuf(nullptr, sizeof(cplx) * nsymtot, CL_MEM_WRITE_ONLY);
  clSetKernelArg(k, 0, sizeof(cl_mem), &bI);
  clSetKernelArg(k, 1, sizeof(int), &nsymtot);
  clSetKernelArg(k, 2, sizeof(float), &period);
  clSetKernelArg(k, 3, sizeof(cl_mem), &bO);
  size_t g[1] = {(size_t)nsymtot};
  ndrange(k, 1, g, nullptr);
  rdbuf(bO, out, sizeof(cplx) * nsymtot);
  clReleaseMemObject(bI); clReleaseMemObject(bO);
}

// one launch covers (symbol, bit-group); the group owns bits
// [g*grp, min(g*grp+grp, m)), so gid(1) = ceil(m/grp). grp >= 1 is guaranteed by
// dp_grp_ok, which both callers run first -- the `grp > 0 ? grp : m` fallback this
// used to carry was a second, silent definition of the same guard, and it made
// grp = 0 mean "one group of m bits" on the host while the kernel read it as an
// empty range and NaN-filled the row.
static size_t dp_bit_groups(int m, int grp) {
  return (size_t)((m + grp - 1) / grp);
}

void Runtime::softdemod_llr(const cplx* z, int nsymtot, int m, int grp,
                            float n0, float* llr) {
  cl_kernel k = kernel("softdemod_llr_k");
  if (!k) return no_kernel("softdemod_llr_k");
  if (!dp_dims_ok("softdemod_llr", m, nsymtot)) return;
  if (!dp_grp_ok("softdemod_llr", m, grp, llr, nsymtot)) return;
  cl_mem bZ = mkbuf(z, sizeof(cplx) * nsymtot, CL_MEM_READ_ONLY);
  cl_mem bL = mkbuf(nullptr, sizeof(float) * (size_t)nsymtot * m, CL_MEM_WRITE_ONLY);
  clSetKernelArg(k, 0, sizeof(cl_mem), &bZ);
  clSetKernelArg(k, 1, sizeof(int), &nsymtot);
  clSetKernelArg(k, 2, sizeof(int), &m);
  clSetKernelArg(k, 3, sizeof(int), &grp);
  clSetKernelArg(k, 4, sizeof(float), &n0);
  clSetKernelArg(k, 5, sizeof(cl_mem), &bL);
  size_t g[2] = {(size_t)nsymtot, dp_bit_groups(m, grp)};
  ndrange(k, 2, g, nullptr);
  rdbuf(bL, llr, sizeof(float) * (size_t)nsymtot * m);
  clReleaseMemObject(bZ); clReleaseMemObject(bL);
}

void Runtime::softdemod_folded(const cplx* z, int nsymtot, int m, int grp,
                               int radius, float period, float n0, float* llr) {
  cl_kernel k = kernel("softdemod_folded_k");
  if (!k) return no_kernel("softdemod_folded_k");
  if (!dp_dims_ok("softdemod_folded", m, nsymtot)) return;
  if (!dp_grp_ok("softdemod_folded", m, grp, llr, nsymtot)) return;
  cl_mem bZ = mkbuf(z, sizeof(cplx) * nsymtot, CL_MEM_READ_ONLY);
  cl_mem bL = mkbuf(nullptr, sizeof(float) * (size_t)nsymtot * m, CL_MEM_WRITE_ONLY);
  clSetKernelArg(k, 0, sizeof(cl_mem), &bZ);
  clSetKernelArg(k, 1, sizeof(int), &nsymtot);
  clSetKernelArg(k, 2, sizeof(int), &m);
  clSetKernelArg(k, 3, sizeof(int), &grp);
  clSetKernelArg(k, 4, sizeof(int), &radius);
  clSetKernelArg(k, 5, sizeof(float), &period);
  clSetKernelArg(k, 6, sizeof(float), &n0);
  clSetKernelArg(k, 7, sizeof(cl_mem), &bL);
  size_t g[2] = {(size_t)nsymtot, dp_bit_groups(m, grp)};
  ndrange(k, 2, g, nullptr);
  rdbuf(bL, llr, sizeof(float) * (size_t)nsymtot * m);
  clReleaseMemObject(bZ); clReleaseMemObject(bL);
}

void Runtime::gmi(const float* llr, const uint8_t* bits, int m, int nsym,
                  int K, float* gmi_out) {
  cl_kernel k = kernel("gmi_k");
  if (!k) return no_kernel("gmi_k");
  if (!dp_dims_ok("gmi", m, K * nsym)) return;
  const size_t n = (size_t)K * nsym * m;
  cl_mem bL = mkbuf(llr, sizeof(float) * n, CL_MEM_READ_ONLY);
  cl_mem bB = mkbuf(bits, n, CL_MEM_READ_ONLY);
  cl_mem bG = mkbuf(nullptr, sizeof(float) * K, CL_MEM_WRITE_ONLY);
  clSetKernelArg(k, 0, sizeof(cl_mem), &bL);
  clSetKernelArg(k, 1, sizeof(cl_mem), &bB);
  clSetKernelArg(k, 2, sizeof(int), &m);
  clSetKernelArg(k, 3, sizeof(int), &nsym);
  clSetKernelArg(k, 4, sizeof(int), &K);
  clSetKernelArg(k, 5, sizeof(cl_mem), &bG);
  size_t g[1] = {(size_t)K};                       // one work-item per stream
  ndrange(k, 1, g, nullptr);
  rdbuf(bG, gmi_out, sizeof(float) * K);
  clReleaseMemObject(bL); clReleaseMemObject(bB); clReleaseMemObject(bG);
}

void Runtime::mcs_select(const float* gmi, int K, float eps, McsSel* out) {
  cl_kernel k = kernel("mcs_select_k");
  if (!k) return no_kernel("mcs_select_k");
  if (K < 1) {
    fprintf(stderr, "[hpocl] mcs_select: K=%d, nothing to select\n", K);
    return;
  }
  cl_mem bG = mkbuf(gmi, sizeof(float) * K, CL_MEM_READ_ONLY);
  cl_mem bI = mkbuf(nullptr, sizeof(int) * K, CL_MEM_WRITE_ONLY);
  cl_mem bM = mkbuf(nullptr, sizeof(int) * K, CL_MEM_WRITE_ONLY);
  cl_mem bR = mkbuf(nullptr, sizeof(int) * K, CL_MEM_WRITE_ONLY);
  cl_mem bS = mkbuf(nullptr, sizeof(float) * K, CL_MEM_WRITE_ONLY);
  cl_mem bA = mkbuf(nullptr, sizeof(float) * K, CL_MEM_WRITE_ONLY);
  cl_mem bF = mkbuf(nullptr, sizeof(float) * K, CL_MEM_WRITE_ONLY);
  cl_mem bO = mkbuf(nullptr, sizeof(uint8_t) * K, CL_MEM_WRITE_ONLY);
  clSetKernelArg(k, 0, sizeof(cl_mem), &bG);
  clSetKernelArg(k, 1, sizeof(int), &K);
  clSetKernelArg(k, 2, sizeof(float), &eps);
  clSetKernelArg(k, 3, sizeof(cl_mem), &bI);
  clSetKernelArg(k, 4, sizeof(cl_mem), &bM);
  clSetKernelArg(k, 5, sizeof(cl_mem), &bR);
  clSetKernelArg(k, 6, sizeof(cl_mem), &bS);
  clSetKernelArg(k, 7, sizeof(cl_mem), &bA);
  clSetKernelArg(k, 8, sizeof(cl_mem), &bF);
  clSetKernelArg(k, 9, sizeof(cl_mem), &bO);
  size_t g[1] = {(size_t)K};                       // one work-item per stream
  std::vector<int> hi(K), hm(K), hr(K);
  std::vector<float> hs(K), ha(K), hf(K);
  std::vector<uint8_t> ho(K);
  if (!ndrange(k, 1, g, nullptr)) {
    for (int i = 0; i < K; ++i) {                  // defined sentinels, not stale
      out[i] = McsSel();
      out[i].margin = gmi[i];
    }
  } else {
    rdbuf(bI, hi.data(), sizeof(int) * K);
    rdbuf(bM, hm.data(), sizeof(int) * K);
    rdbuf(bR, hr.data(), sizeof(int) * K);
    rdbuf(bS, hs.data(), sizeof(float) * K);
    rdbuf(bA, ha.data(), sizeof(float) * K);
    rdbuf(bF, hf.data(), sizeof(float) * K);
    rdbuf(bO, ho.data(), K);
    for (int i = 0; i < K; ++i) {
      out[i].index = hi[i]; out[i].m = hm[i]; out[i].rate = hr[i];
      out[i].se = hs[i]; out[i].margin = ha[i]; out[i].info = hf[i];
      out[i].outage = ho[i] != 0;
    }
  }
  clReleaseMemObject(bG); clReleaseMemObject(bI); clReleaseMemObject(bM);
  clReleaseMemObject(bR); clReleaseMemObject(bS); clReleaseMemObject(bA);
  clReleaseMemObject(bF); clReleaseMemObject(bO);
}

}  // namespace hpocl
