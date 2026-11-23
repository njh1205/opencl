#include "hpocl_schedule.h"
#include <cstdio>
#include <cmath>

namespace hpocl {

//---------------------------------------------------------------------------
// The cost model, in one place, because it is the only thing here that is a
// judgement rather than a port.
//
// Work in one Jacobi sweep, counted the way kernels/decomp.cl performs it:
//   pairs    = n(n-1)/2                          column pairs orthogonalised
//   per pair = 10*m  (the Gram: two |a|^2 and one conj-mul-accumulate per row)
//            + 6*(m+n)  (the two rotations: W's m rows, Vv's n rows)
//   flops    = C * pairs * sweeps * (10m + 6(m+n))
// `sweeps` is the hard cap (16), not the number either side actually runs, so
// this OVER-estimates -- deliberately, because an over-estimate offloads more
// and the shapes it over-estimates are the ones the launch decision is least
// sensitive to. The measured exits, if you want the real numbers:
//   fp32 kernel (tol_exit = 1e-7, kernels/decomp.cl): 28x32 -> 8, 4x32 -> 6,
//     4x4 -> 5, 12x16 -> 7 sweeps; established by cap-sweep bit-identity, not by
//     timing (cap=8 output == cap=16 output exactly; cap=7 differs by 6.9e-1).
//   fp64 host (tol = 1e-13, ref/decomp.cpp, instrumented sweep counter):
//     28x32 -> 9, 12x16 -> 8, 14x16 -> 8, 4x8 -> 6, 8x8 -> 7, 12x18 -> 9.
// So the cap over-states the PS work by ~1.8x at the largest shape and up to 4x
// on the small ones. That is a conservative bias in the OFFLOAD decision and it
// is not worth removing: the batches it mis-prices are the 0.03-0.07 Mflop ones
// that stay on the PS either way.
double jacobi_batch_flops(int C, int m, int n, int sweeps) {
  const double pairs = 0.5 * n * (n - 1);
  return (double)C * pairs * sweeps * (10.0 * m + 6.0 * (m + n));
}

// The Gram form's cost, for the batches that ask for take < n: the m x m Gram
// (m*m*n), its eigendecomposition (a Jacobi on m columns: m(m-1)/2 pairs x
// (10m + 12m) x sweeps), and the take leading right vectors (m*n*take). At
// m=4, n=128, take=4, sweeps=8 that is 4.4 kflop against the 6.8 Mflop PER
// SWEEP the full one-sided Jacobi needs -- the measured 5256x, and the reason
// the analog batch falls below min_flops and stays on the PS in BOTH roles.
double gram_batch_flops(int C, int m, int n, int take, int sweeps) {
  const double pairs = 0.5 * m * (m - 1);
  return (double)C * ((double)m * m * n + pairs * sweeps * 22.0 * m +
                      (double)m * n * take);
}

double gemm_flops(int M, int N, int SC) {
  return 8.0 * (double)M * (double)N * (double)SC;
}

// Why min_flops = 4e5. The question is host_time > device_time + launch:
//   host   A53 fp64, ~1.4 GFLOP/s (README's cycle model: 1.4 GHz, in-order,
//          1 fp64 flop/cycle, x1.5 structural overhead)
//   device PL fp32, ~100 GFLOP/s at 200 MHz with 4-8 lanes (README's estimate
//          for this kernel family; C work-items run concurrently, so the batch
//          is one device-time, not C)
//   launch 10-50 us on XRT embedded (this dev host measures ~1 us, which is why
//          the number is derived from the board and not from the M4)
// With the same arithmetic on both sides, F/1.4e9 > F/1e11 + 3e-5 gives
// F > 4.3e4. 4e5 is ~10x above that, so it is conservative: it keeps the
// 20.3 Mflop null-basis batch and the 1.23 Mflop analog batch on the fabric and
// the two 34 kflop 4x4 batches on the PS, and if the board's real launch
// overhead is lower, lowering min_flops only moves more work to the PL.
namespace {
constexpr double kDefaultMinFlops = 4e5;

// The GLUE rule, derived rather than inherited. Same three constants as above
// (A53 fp64 1.4 GFLOP/s, PL fp32 100 GFLOP/s at 200 MHz with 4-8 lanes, XRT
// launch 10-50 us) plus the one cost the SVD batches do not have: the kernel
// interface is fp32 and ref/ is fp64, so an offloaded GEMM stages E = M*N +
// SC*N + SC*M complex elements through the PS at ~6 flop/element.
//   offload  iff  F/1.4e9  >  F/1e11 + LAUNCH + 6E/1.4e9
// Ignoring staging the crossover is F > 1.4e9*LAUNCH/(1-1.4e9/1e11) = 4.3e4 for
// LAUNCH = 3e-5. Evaluated with staging at the shapes the 21 pool configs
// actually have (table in docs/cost_model_real_shapes.md section 9):
//   F = 1.05 Mflop (Nt=128)  host 749 us vs device  80 us -> offload
//   F = 0.52 Mflop (Nt=64)   host 375 us vs device  57 us -> offload
//   F = 0.26 Mflop           host 187 us vs device  46 us -> offload
//   F = 0.13 Mflop           host  94 us vs device  41 us -> offload
//   F = 0.066 Mflop          host  47 us vs device  36 us -> WASH: inside the
//                                   model's own launch uncertainty
//   F = 0.037 Mflop          host  26 us vs device  34 us -> keep on the PS
// So kGemmMinFlops = 1e5 (declared in hpocl_schedule.h, where the struct default
// reads it, so this derivation and the value cannot drift) is the smallest round
// number that agrees with the time model evaluated at kLaunchWorstS -- the 50 us
// end of the launch range -- at EVERY site of EVERY pool config: it offloads the
// 0.13 Mflop sites and up, and keeps the 0.066 Mflop wash on the PS. Using the
// SVD rule's 4e5 instead would leave 15 of the 21 configs' glue on the PS for no
// accuracy reason; using the naive 4.3e4 crossover would offload 11 sites whose
// entire saving (<= 10.7 us each, ~1% of those instants) is smaller than the
// launch estimate's own uncertainty.
// `python3 tools/glue_cost_table.py` re-derives all 63 site decisions from the
// constants in the header and FAILS if the proxy and the model disagree, so this
// paragraph cannot outlive the rule it describes. Run it after touching the rule,
// and with `--launch <s>` once the board's real XRT overhead is measured.

void device_svd_run(void* ctx_, const dcplx* A, int C, int m, int n, int take,
                    dcplx* U, double* S, dcplx* V) {
  DeviceSvd& d = *static_cast<DeviceSvd*>(ctx_);
  // A take < n batch is served by the Gram form, which is so much cheaper that
  // it is never worth a launch: at the largest deployed shape (K=8, Nr=4,
  // Nt=128) all eight analog factorisations together cost 0.031 ms on this host
  // against 4.6 ms for the null-basis batch, i.e. 0.7% of the front end. There
  // is therefore no svd_wide KERNEL, and this is the reason: adding one would
  // save 0.7% and cost a 30th kernel, its own ceilings and its own gate. What
  // the fabric cannot do is the full Jacobi at n=128 -- that needs a private
  // Vv[128x128] float2 = 128 KB per work-item -- so the Gram form is not merely
  // faster here, it is the only form that fits. It stays on the PS.
  const bool wide = (take < n && m < n);
  const double flops = wide ? gram_batch_flops(C, m, n, take, d.sweeps)
                            : jacobi_batch_flops(C, m, n, d.sweeps);
  const bool offload = d.force || flops >= d.min_flops;
  d.where.push_back(offload ? "device" : "host  ");
  { char b[64]; snprintf(b, sizeof b, "C=%d %dx%d take=%d %s, %.3f Mflop", C, m, n,
                         take, wide ? "gram" : "jacobi", flops / 1e6);
    d.what.push_back(b); }
  if (wide || !offload) {               // one definition of the host role too
    SvdProvider h = host_svd();
    h.run(h.ctx, A, C, m, n, take, U, S, V);
    ++d.host_batches; d.flops_host += flops;
    return;
  }
  if (!U) {                             // jacobi_svd always writes a left basis
    d.scratch_u.assign((size_t)C * m * n, dcplx(0, 0));
    U = d.scratch_u.data();
  }
  // A wide batch never reaches here (see above), so take == n and jacobi_svd's
  // [C][n x n] V layout is the one the caller allocated.
  if (take != n) {
    fprintf(stderr, "[sched] internal: wide batch reached the device path\n");
    SvdProvider h = host_svd();
    h.run(h.ctx, A, C, m, n, take, U, S, V);
    ++d.host_batches; d.flops_host += flops;
    return;
  }
  d.rt->jacobi_svd(A, C, m, n, d.sweeps, U, S, V);
  d.us_svd += d.rt->last_us();
  ++d.launches; d.flops_off += flops;
}
}  // namespace

//---------------------------------------------------------------------------
// The glue. Same cost rule as the SVD batches, because the question is the same
// one (is F/1.4e9 s of A53 time worth F/1e11 s of fabric time plus a 10-50 us
// launch), and at the real pool shapes the answer flips relative to Nt=16: the
// three Nt-class GEMMs are 1.05 Mflop each at Nt=128 (offload, 2.6x above
// min_flops) and 0.03-0.13 Mflop at Nt<=32 (stay on the PS). One launch per
// site, not per user: the batch dimension is folded into SC/M by stacking the
// operands (ref/pipeline.cpp), so K=8 users cost one launch.
namespace {
const char* gemm_site_name(int site) {
  return site == kGemmEff ? "H*F_RF" : site == kGemmFn ? "F_RF*N_k"
                                                       : "F_RF*F_BB";
}

void device_gemm_run(void* ctx_, int site, const dcplx* W, int M, int N,
                     const dcplx* X, int SC, dcplx* Y) {
  DeviceGemm& g = *static_cast<DeviceGemm*>(ctx_);
  const double flops = gemm_flops(M, N, SC);
  const bool enabled = ((g.sites >> site) & 1) != 0;
  const bool offload =
      enabled && (g.force || flops >= g.min_flops) && g.rt &&
      g.rt->has_kernel("wbb_apply");
  g.where.push_back(offload ? "device" : "host  ");
  { char b[96];
    snprintf(b, sizeof b, "%-9s M=%d N=%d SC=%d, %.3f Mflop", gemm_site_name(site),
             M, N, SC, flops / 1e6);
    g.what.push_back(b); }
  if (!offload) {
    gemm_wx(W, M, N, X, SC, Y);          // one definition of the host role too
    ++g.host_calls; g.flops_host += flops;
    return;
  }
  // fp32 staging. The kernel's interface is float2 and ref/ is dcplx, so the
  // offload ADDS a conversion to the PS budget; elems_converted is reported by
  // sched_bench so that cost is visible instead of being folded into "free".
  const size_t nw = (size_t)M * N, nx = (size_t)SC * N, ny = (size_t)SC * M;
  g.w32.assign(nw, cplx()); g.x32.assign(nx, cplx()); g.y32.assign(ny, cplx());
  for (size_t i = 0; i < nw; ++i) g.w32[i] = cplx((float)W[i].real(), (float)W[i].imag());
  for (size_t i = 0; i < nx; ++i) g.x32[i] = cplx((float)X[i].real(), (float)X[i].imag());
  g.rt->wbb_apply(g.w32.data(), M, N, g.x32.data(), SC, g.y32.data());
  g.us_gemm += g.rt->last_us();
  for (size_t i = 0; i < ny; ++i) Y[i] = dcplx(g.y32[i].real(), g.y32[i].imag());
  g.elems_converted += nw + nx + ny;
  ++g.launches; g.flops_off += flops;
}
}  // namespace

GemmProvider DeviceGemm::provider() {
  if (min_flops <= 0) force = true;     // 0 means "offload every enabled site"
  return GemmProvider{device_gemm_run, this};
}

SvdProvider DeviceSvd::provider() {
  if (min_flops <= 0) force = true;     // 0 means "offload everything"
  SvdProvider p{device_svd_run, this};
  // The kernels are fp32-only (kernels/decomp.cl disables cl_khr_fp64), and
  // batches this provider routes back to the host are still fp32-representable
  // shapes, so the declared precision is fp32's. front_end() derives the
  // whitening eigenvalue noise floor from it -- 8*eps32 = 9.5e-7, which is
  // 26-380x above the 2.5e-9..3.7e-8 x lam_max band the structurally-zero
  // eigenvalues of C actually land in at fp32, and 5 orders above production's
  // 1e-10 relative floor (see inv_sqrt_from_eig for the measurement).
  p.eps = 1.1920929e-7f;                // FLT_EPSILON
  return p;
}

bool schedule_device(const dcplx* H, const PipeCfg& c, PipeOut& o,
                     DeviceSvd& dsvd, DeviceGemm* dgemm) {
  if (!dsvd.rt || !dsvd.rt->has_kernel("jacobi_svd") ||
      !dsvd.rt->has_kernel("threshold_cf")) {
    fprintf(stderr, "[sched] device role unavailable: the program has no "
            "jacobi_svd/threshold_cf -- call schedule() instead\n");
    return false;
  }
  if (dsvd.min_flops <= 0) dsvd.min_flops = kDefaultMinFlops;

  // One GemmProvider for the whole instant: dgemm == nullptr means the PS role's
  // fp64 gemm_wx at all three sites, which is what shipped before the glue was
  // costed at the real pool shapes.
  const GemmProvider gemm = dgemm ? dgemm->provider() : host_gemm();

  FrontEnd fe;
  front_end(H, c, dsvd.provider(), gemm, fe, o);

  // The threshold on the device. Unlike the SVD batches this one is ALWAYS
  // offloaded: with the deficit table the host solve is ~1.2 ms on an A53
  // (README, measured 51.3 us on this host scaled by the cycle model) against
  // ~106 kflop, i.e. it is transcendental- and branch-bound rather than
  // flop-bound and the fabric's 10-30 us is 40-100x better. Once the front end
  // is offloaded this is the largest remaining PS cost in the instant.
  const int T = c.K * c.d;
  std::vector<double> lam((size_t)T);
  for (int i = 0; i < T; ++i) lam[i] = o.sigma[i] * o.sigma[i];
  double tau_cf = 0, tau = 0;
  int flag = 0;
  dsvd.rt->threshold_cf(lam.data(), 1, c.K, c.d, c.M, c.variant, &tau_cf, &tau,
                        &flag);
  dsvd.us_threshold = dsvd.rt->last_us();
  o.tau_dB = tau;
  o.mode = decide_mode(c, tau, flag, &o.cf_flag);

  design_instant(c, o.mode, gemm, fe, o);
  return true;
}

}  // namespace hpocl
