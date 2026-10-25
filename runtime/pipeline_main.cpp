// pipeline_bench: one scheduling instant of the PS control plane vs the
// production oracle. Only gauge-invariant quantities are compared: LAPACK and
// the deployed Jacobi SVD choose different singular-vector phases, and the BD
// basis / analog precoder carry their own unitary gauges, so the physical
// precoder P_k = F_RF N_k V_D,k (up to column phase) replaces N_k and V_D,k.
//   tau    threshold policy (may saturate to +-inf)   mode  SVD/THP branch
//   sigma  whitened per-stream spectrum               B     THP feedback
//   P      physical precoder                          rr/rt realised/target SNR
#include <cstdio>
#include <cmath>
#include <vector>
#include <fstream>
#include "hpocl_vecio.h"
#include <cstring>
#include "pipeline.h"

using namespace hpocl;

namespace {
double err_c(const dcplx* a, const dcplx* b, size_t n) {
  double m = 0; for (size_t i = 0; i < n; ++i) m = fmax(m, std::abs(a[i] - b[i])); return m;
}
double err_r(const double* a, const double* b, size_t n) {
  double m = 0; for (size_t i = 0; i < n; ++i) m = fmax(m, fabs(a[i] - b[i])); return m;
}
double err_tau(double a, double b) {   // +-inf when the branch order is uniform
  if (std::isinf(a) || std::isinf(b)) return a == b ? 0.0 : HUGE_VAL;
  return fabs(a - b);
}
void align_phase(std::vector<dcplx>& mine, const std::vector<dcplx>& ref,
                 int rows, int cols) {  // columns are free up to a phase
  for (int j = 0; j < cols; ++j) {
    dcplx dot(0, 0);
    for (int r = 0; r < rows; ++r)
      dot += std::conj(ref[(size_t)r * cols + j]) * mine[(size_t)r * cols + j];
    if (std::abs(dot) > 1e-12) {
      dcplx ph = std::conj(dot / std::abs(dot));
      for (int r = 0; r < rows; ++r) mine[(size_t)r * cols + j] *= ph;
    }
  }
}
}  // namespace

int main(int argc, char** argv) {
  hpocl::VecReader b(argc > 1 ? argv[1] : "build/pipeline_vectors.bin");
  if (!b.ok()) { fprintf(stderr, "[pipeline_bench] %s\n", b.error().c_str()); return 2; }
  int ncase = b.i(), fails = 0;
  for (int ci = 0; ci < ncase; ++ci) {
    PipeCfg c;
    c.K = b.i(); c.Nt = b.i(); c.Nr = b.i(); c.NRF = b.i(); c.d = b.i();
    c.M = b.i(); c.variant = b.i(); int nul = b.i();
    c.gamma_tot_dB = b.f(); double e_tau = b.f();
    const int K = c.K, Nt = c.Nt, d = c.d, T = K * d;
    std::vector<dcplx> H((size_t)K * c.Nr * Nt), eP((size_t)K * Nt * d),
        eB((size_t)K * d * d);
    std::vector<double> eSig(T), eRr(T), eRt(T);
    b.cx(H); b.cx(eP); b.re(eSig); b.cx(eB); b.re(eRr); b.re(eRt);
    int eMode = b.i();

    PipeOut o;
    schedule(H.data(), c, o);

    e_tau = err_tau(o.tau_dB, e_tau);
    bool mode_ok = o.mode == eMode && o.nul == nul;
    double e_sig = err_r(o.sigma.data(), eSig.data(), T);
    double e_P = 0, e_B = 0;
    for (int k = 0; k < K; ++k) {
      std::vector<dcplx> P((size_t)Nt * d, dcplx(0, 0));
      for (int i = 0; i < Nt; ++i)
        for (int j = 0; j < d; ++j) {
          dcplx a(0, 0);
          for (int t = 0; t < c.NRF; ++t) {
            dcplx nv(0, 0);                       // F_BB column = N_k V_D,k
            for (int q = 0; q < nul; ++q)
              nv += o.Nk[((size_t)k * c.NRF + t) * nul + q] *
                    o.Vd[((size_t)k * nul + q) * d + j];
            a += o.Frf[(size_t)i * c.NRF + t] * nv;
          }
          P[(size_t)i * d + j] = a;
        }
      std::vector<dcplx> Pr(eP.begin() + (size_t)k * Nt * d,
                            eP.begin() + (size_t)(k + 1) * Nt * d);
      align_phase(P, Pr, Nt, d);
      e_P = fmax(e_P, err_c(P.data(), Pr.data(), (size_t)Nt * d));
      e_B = fmax(e_B, err_c(o.B.data() + (size_t)k * d * d,
                            eB.data() + (size_t)k * d * d, (size_t)d * d));
    }
    double e_rr = err_r(o.rho_real.data(), eRr.data(), T);
    double e_rt = err_r(o.rho_tgt.data(), eRt.data(), T);

    bool ok = e_tau < 1e-5 && mode_ok && e_sig < 1e-9 && e_P < 1e-9 &&
              e_B < 1e-9 && e_rr < 1e-8 && e_rt < 1e-8;
    fails += !ok;
    printf("case%d K=%d NRF=%d d=%d M=%3d var=%d s=%.1fdB mode=%d: tau=%.1e sig=%.1e "
           "P=%.1e B=%.1e rr=%.1e rt=%.1e %s\n",
           ci, K, c.NRF, d, c.M, c.variant, c.gamma_tot_dB, o.mode,
           e_tau, e_sig, e_P, e_B, e_rr, e_rt, ok ? "PASS" : "FAIL");
  }
  if (!b.expect_done("pipeline_bench")) ++fails;   // drift vs tools/pipeline_ref.py
  printf(fails ? "[pipeline_bench] FAIL\n" : "[pipeline_bench] ALL PASS\n");
  return fails ? 1 : 0;
}
