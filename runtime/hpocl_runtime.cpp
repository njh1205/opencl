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

Runtime::~Runtime() {
  if (ev_) clReleaseEvent(ev_);
  // TransformerPredictor borrows these handles without owning them, so the map
  // must be emptied here to make a use-after-free impossible rather than
  // merely unlikely.
  for (auto& kv : kernels_) clReleaseKernel(kv.second);
  kernels_.clear();
  if (prog_) clReleaseProgram(prog_);
  if (q_)    clReleaseCommandQueue(q_);
  if (ctx_)  clReleaseContext(ctx_);
}

bool Runtime::init(const char* pref) {
  cl_uint nplat = 0;
  if (clGetPlatformIDs(0, nullptr, &nplat) != CL_SUCCESS || !nplat) return false;
  std::vector<cl_platform_id> plats(nplat);
  clGetPlatformIDs(nplat, plats.data(), nullptr);

  cl_device_type want = CL_DEVICE_TYPE_CPU;
  if (!strcmp(pref, "gpu")) want = CL_DEVICE_TYPE_GPU;
  if (!strcmp(pref, "acc")) want = CL_DEVICE_TYPE_ACCELERATOR;

  cl_int err = CL_SUCCESS;
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
  if (!found) return false;

  ctx_ = clCreateContext(nullptr, 1, &dev_, nullptr, nullptr, &err);
  if (err != CL_SUCCESS) return false;
  q_ = clCreateCommandQueue(ctx_, dev_, CL_QUEUE_PROFILING_ENABLE, &err);
  if (err != CL_SUCCESS) return false;

  char name[256] = {0};
  clGetDeviceInfo(dev_, CL_DEVICE_NAME, sizeof(name), name, nullptr);
  dev_name_ = name;
  cl_device_fp_config fp64 = 0;
  clGetDeviceInfo(dev_, CL_DEVICE_DOUBLE_FP_CONFIG, sizeof(fp64), &fp64, nullptr);
  // HPOCL_FP32=1 forces the single-precision host marshalling even if the
  // device claims fp64 (xclbins are compiled fp32 for PL).
  dev_fp64_ = (fp64 != 0) && getenv("HPOCL_FP32") == nullptr;
  if (verbose_) printf("[hpocl] device: %s (pref=%s)\n", dev_name_.c_str(), pref);
  return true;
}

bool Runtime::build(const std::vector<std::string>& files,
                    const char* extra_opts) {
  std::vector<std::string> srcs;
  std::vector<const char*> strs;
  std::vector<size_t> lens;
  for (auto& f : files) {
    std::string src = read_cl_source(f);   // inlines #include "x.clh"
    if (src.empty()) { fprintf(stderr, "[hpocl] missing kernel source %s\n", f.c_str()); return false; }
    srcs.push_back(src);
  }
  for (auto& s : srcs) { strs.push_back(s.c_str()); lens.push_back(s.size()); }

  cl_int err = CL_SUCCESS;
  prog_ = clCreateProgramWithSource(ctx_, (cl_uint)strs.size(), strs.data(),
                                    lens.data(), &err);
  if (err != CL_SUCCESS) return false;
  std::string opts = "-cl-std=CL1.2 -cl-mad-enable";
  if (extra_opts) { opts += " "; opts += extra_opts; }
  err = clBuildProgram(prog_, 1, &dev_, opts.c_str(), nullptr, nullptr);
  if (err != CL_SUCCESS) {
    char log[8192];
    clGetProgramBuildInfo(prog_, dev_, CL_PROGRAM_BUILD_LOG, sizeof(log), log, nullptr);
    fprintf(stderr, "[hpocl] build failed:\n%s\n", log);
    return false;
  }
  return acquire_kernels();
}

cl_mem Runtime::mkbuf(const void* host, size_t bytes, cl_mem_flags extra) {
  cl_int err = CL_SUCCESS;
  cl_mem b = clCreateBuffer(ctx_, extra | (host ? CL_MEM_COPY_HOST_PTR : 0),
                            bytes, const_cast<void*>(host), &err);
  return err == CL_SUCCESS ? b : nullptr;
}
void Runtime::rdbuf(cl_mem b, void* host, size_t bytes) {
  clEnqueueReadBuffer(q_, b, CL_TRUE, 0, bytes, host, 0, nullptr, nullptr);
}
bool Runtime::ndrange(cl_kernel k, cl_uint dims, const size_t* g, const size_t* l) {
  if (ev_) { clReleaseEvent(ev_); ev_ = nullptr; }
  cl_int err = clEnqueueNDRangeKernel(q_, k, dims, nullptr, g, l, 0, nullptr, &ev_);
  clFinish(q_);
  // CL_INVALID_WORK_GROUP_SIZE / CL_OUT_OF_RESOURCES mean nothing ran at all.
  // Say so, with the kernel name and the shape: a failed launch otherwise looks
  // exactly like a successful one that produced zeros.
  if (err != CL_SUCCESS) {
    const char* name = "?";
    for (const auto& kv : kernels_)
      if (kv.second == k) { name = kv.first.c_str(); break; }
    fprintf(stderr, "[hpocl] %s: clEnqueueNDRangeKernel -> %d (global", name, err);
    for (cl_uint i = 0; i < dims; ++i) fprintf(stderr, " %zu", g[i]);
    fprintf(stderr, ") -- the kernel did NOT run, outputs NOT written\n");
    return false;
  }
  return true;
}
double Runtime::last_us() const {
  if (!ev_) return 0.0;
  cl_ulong t0 = 0, t1 = 0;
  clGetEventProfilingInfo(ev_, CL_PROFILING_COMMAND_START, sizeof(t0), &t0, nullptr);
  clGetEventProfilingInfo(ev_, CL_PROFILING_COMMAND_END, sizeof(t1), &t1, nullptr);
  return (double)(t1 - t0) / 1e3;
}

bool Runtime::load_xclbin(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) { fprintf(stderr, "[hpocl] missing xclbin %s\n", path.c_str()); return false; }
  std::vector<char> blob((std::istreambuf_iterator<char>(f)), {});
  const unsigned char* bp = (const unsigned char*)blob.data();
  size_t bl = blob.size();
  cl_int bin_status = CL_SUCCESS, err = CL_SUCCESS;
  cl_program prog = clCreateProgramWithBinary(ctx_, 1, &dev_, &bl, &bp,
                                              &bin_status, &err);
  if (err != CL_SUCCESS || bin_status != CL_SUCCESS) {
    fprintf(stderr, "[hpocl] clCreateProgramWithBinary failed (%d/%d)\n", err, bin_status);
    return false;
  }
  prog_ = prog;
  return acquire_kernels();
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
  // fp32-capable-only devices (Apple Metal GPU, area-constrained PL builds)
  // take the single-precision datapath; host interface stays double.
  std::vector<float> lamf, cff, tauf;
  const void* src = (const void*)lam;
  size_t es = sizeof(double);
  if (!dev_fp64_) {
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
  if (dev_fp64_) {
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

// kernels/decomp.cl sizes every __private array by MAXM/MAXD and rejects
// out-of-contract dimensions by returning, which leaves the output buffers at
// the zeros they were created with. Refuse on the host instead, the way
// power_iter already does for PMAX. tools/check_registry.py cross-checks these
// three pairs against the #defines in the kernel sources, so they cannot drift.
constexpr int kDecompMaxM = 16;   // == MAXM in kernels/decomp.cl
constexpr int kDecompMaxD = 8;    // == MAXD in kernels/decomp.cl

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
bool dims_ok(const char* who, int rows, int d) {
  if (rows >= 1 && rows <= kDecompMaxM && d >= 1 && d <= kDecompMaxD) return true;
  fprintf(stderr, "[hpocl] %s: rows=%d d=%d outside the kernel's capacity "
          "(MAXM=%d MAXD=%d in kernels/decomp.cl) -- raise them there and in "
          "kDecompMaxM/kDecompMaxD, outputs NaN-filled\n", who, rows, d,
          kDecompMaxM, kDecompMaxD);
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
  if (!dims_ok("jacobi_svd", m, d)) {
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
  if (!dims_ok("gmd_op", rows, d)) {
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
  if (!dims_ok("ucd_op", rows, d)) {
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
  // cbd_k bounds its private q/p arrays by MAXM and A/B by MAXD, so urows and
  // vrows are both checked against MAXM.
  if (!dims_ok("cbd_op", urows, d) || !dims_ok("cbd_op", vrows, d)) {
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
  size_t g[2] = {(size_t)Nt, (size_t)nsym};        // (antenna, symbol)
  ndrange(k, 2, g, nullptr);
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
// [g*grp, min(g*grp+grp, m)), so gid(1) = ceil(m/grp)
static size_t dp_bit_groups(int m, int grp) {
  return (size_t)((m + (grp > 0 ? grp : m) - 1) / (grp > 0 ? grp : m));
}

void Runtime::softdemod_llr(const cplx* z, int nsymtot, int m, int grp,
                            float n0, float* llr) {
  cl_kernel k = kernel("softdemod_llr_k");
  if (!k) return no_kernel("softdemod_llr_k");
  if (!dp_dims_ok("softdemod_llr", m, nsymtot)) return;
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
