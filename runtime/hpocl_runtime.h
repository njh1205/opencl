#pragma once
//-----------------------------------------------------------------------------
// HPOCL heterogeneous runtime.
// Plain OpenCL 1.2 C API on purpose: the very same host binary compiles
//   * on the x86 dev/cluster CPU  (PoCL / Intel / Apple CPU runtime)  <- today
//   * on Zynq PS Cortex-A53       (XRT's libOpenCL, embedded platform)
//   * against the PL accelerator  (same API, device = ACCELERATOR, .xclbin)
// Device policy + CPU fallback live here, mirroring how AI inference
// runtimes expose execution-provider selection.
//-----------------------------------------------------------------------------
#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif
#include <map>
#include <string>
#include <vector>
#include "hpocl_types.h"

namespace hpocl {

class Runtime {
 public:
  ~Runtime();
  bool init(const char* device_pref);            // "cpu" | "gpu" | "acc"
  bool build(const std::vector<std::string>& cl_files,
             const char* extra_opts = nullptr);
  // XRT/PL path: load a pre-synthesised xclbin as an OpenCL binary.
  // Preferred on Zynq; source JIT is dev-host only.
  bool load_xclbin(const std::string& path);
  // env-driven: HPOCL_XCLBIN set -> load_xclbin, else source build
  bool load_program(const std::vector<std::string>& cl_files,
                    const char* extra_opts = nullptr);
  const char* device_name() const { return dev_name_.c_str(); }

  // ---- operators (synchronous wrappers; events kept for profiling) -------
  void channel_cov(const cplx* H, int M, int N, int S, cplx* R);
  void power_iter(const cplx* R, int M, int iters, int nmat,
                  cplx* v, float* lambda);
  void deflate(cplx* R, int M, int nmat, const cplx* v, const float* lambda);
  void switch_threshold(const float* lambda_min, const float* g_min,
                        const float* metric_dB, int count, float nt,
                        float nrf, float eta, float* tau_dB, uint8_t* mode);
  void wbb_apply(const cplx* W, int M, int N, const cplx* X, int SC, cplx* Y);
  void thp_precoder(const cplx* B, const cplx* F, const cplx* X, int SC,
                    int M, int K, float tau, cplx* Y);

  // ---- per-symbol datapath (fp32 on device AND in the host buffers: these
  // run per symbol, so a double host interface would only add a conversion
  // per stage). Stream order is transmit order, k = u*d + j. See
  // kernels/datapath.cl for the per-kernel index conventions.
  // bits[nsymtot][m] (MSB first) -> symbols
  void qam_map(const uint8_t* bits, int m, int nsymtot, cplx* s);
  // x = Frf[Nt x NRF] * Fbb[NRF x T] * s[T x nsym] -> x[Nt x nsym]
  void tx_combine(const cplx* Frf, int Nt, int NRF, const cplx* Fbb, int T,
                  const cplx* s, int nsym, cplx* x);
  // y = H[Nr x Nt] * x + n -> y[Nr x nsym]
  void channel_apply(const cplx* H, int Nr, int Nt, const cplx* x, int nsym,
                     const cplx* n, cplx* y);
  // z = (W_u^H y) / diag_b -> z[U*d x nsym]; diag_b == nullptr skips the
  // normalization (the un-equalized receive path)
  void rx_combine(const cplx* W, int U, int Nr, int d, const cplx* y, int nsym,
                  const cplx* diag_b, cplx* z);
  // centered THP receive modulo, per axis, period = L*delta
  void modulo_fold(const cplx* in, int nsymtot, float period, cplx* out);
  // exact log-MAP LLRs over the M base candidates; grp = bits per work-item
  void softdemod_llr(const cplx* z, int nsymtot, int m, int grp, float n0,
                     float* llr);
  // same over the (2*radius+1)^2*M THP replica set (radius 0 == softdemod_llr)
  void softdemod_folded(const cplx* z, int nsymtot, int m, int grp, int radius,
                        float period, float n0, float* llr);
  // GMI_k = m - mean_symbols sum_b log2(1+exp(-(2*bits-1)*llr)) -> gmi[K]
  void gmi(const float* llr, const uint8_t* bits, int m, int nsym, int K,
           float* gmi_out);

  // ---- code-rate / MCS layer (kernels/rate.cl, table in kernels/mcs_table.clh)
  // Selects, per stream, the highest 3GPP TS 38.214 CQI entry whose spectral
  // efficiency m*R/1024 the measured GMI supports; eps is the selection slack
  // in bits (see cqi_select). Fills all K entries on every path, including
  // outage, so a caller never reads a stale slot.
  void mcs_select(const float* gmi, int K, float eps, McsSel* out);
  // closed-form SVD/THP switching threshold, batched over channels
  void threshold_cf(const double* lam, int C, int K, int d, int M, int variant,
                    double* tau_cf_dB, double* tau_dB, int* flags);
  // decomposition operators (fp32 PL datapath, double host interface)
  void jacobi_svd(const std::complex<double>* A, int C, int m, int d,
                  int sweeps, std::complex<double>* U, double* S,
                  std::complex<double>* V);
  // P0/Veff/Vd/R/P are [C][rows x d] (R, upper: [C][d x d]); rows >= d
  void gmd_op(const std::complex<double>* P0, const double* sigma, int C,
              int rows, int d, std::complex<double>* R, std::complex<double>* P);
  void ucd_op(const std::complex<double>* Veff, const double* sigma, int C,
              int rows, int d, double snr, bool waterfill, double min_pl,
              std::complex<double>* Vd, double* rho_eq,
              std::complex<double>* upper, double* pl);
  // GP-CBD paired block decomposition. U [C][urows x d] / V [C][vrows x d] are
  // the SVD bases, sigma [C][d] DESCENDING; pair_threshold is production's
  // cbd_pair_threshold (1.7). Q [C][urows x d], R [C][d x d] (2x2 blocks at
  // (2k,2k+1)), P [C][vrows x d], active_count [C]. See kernels/decomp.cl::cbd_k.
  void cbd_op(const std::complex<double>* U, const std::complex<double>* V,
              const double* sigma, int C, int urows, int vrows, int d,
              double snr, double pair_threshold, std::complex<double>* Q,
              std::complex<double>* R, std::complex<double>* P,
              int* active_count);

  double last_us() const;                        // profiling of last operator
  void set_verbose(bool v) { verbose_ = v; }
  bool device_fp64() const { return dev_fp64_; }
  cl_context context() const { return ctx_; }
  cl_command_queue queue() const { return q_; }
  // lookup a kernel object created during build() by name
  cl_kernel kernel(const char* name) const;
  bool has_kernel(const char* name) const;
  // The names build()/load_xclbin() try to acquire, and how many. Exposed so
  // allprog_check can assert a whole-library program really provides every one
  // of them instead of keeping a second copy of the list -- a second copy is a
  // list that drifts, and a drifted list makes the gate check the wrong thing.
  static const char* const* kernel_names(size_t* count);

 private:
  bool acquire_kernels();   // shared by build() and load_xclbin()
  cl_mem mkbuf(const void* host, size_t bytes, cl_mem_flags extra);
  void rdbuf(cl_mem b, void* host, size_t bytes);
  bool ndrange(cl_kernel k, cl_uint dims, const size_t* g, const size_t* l);

  cl_context       ctx_  = nullptr;
  cl_command_queue q_    = nullptr;
  cl_device_id     dev_  = nullptr;
  cl_program       prog_ = nullptr;
  cl_event         ev_   = nullptr;
  std::string      dev_name_;
  bool             verbose_ = false;
  bool             dev_fp64_ = false;
  // One registry, keyed by kernel name: wrappers call kernel("gmd_k") instead
  // of caching handles in members, so adding a kernel means adding one string
  // to kKernelNames[] and one wrapper -- not editing two acquisition tables.
  std::map<std::string, cl_kernel> kernels_;
};

}  // namespace hpocl
