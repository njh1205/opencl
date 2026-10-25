// frontend_bench: multi-user RF-BD front end vs production-python oracle.
// Null-space bases carry rotational freedom, so the basis is compared via
// its projector and the whitened channel via sigma(Gbar).
#include <cstdio>
#include <cmath>
#include <vector>
#include <fstream>
#include "hpocl_vecio.h"
#include <cstring>
#include <algorithm>
#include "frontend.h"
#include "decomp.h"

using namespace hpocl;

namespace {
double err_c(const dcplx* a, const dcplx* b, size_t n) {
  double m = 0; for (size_t i = 0; i < n; ++i) m = fmax(m, std::abs(a[i] - b[i])); return m;
}
// P = A A^H for A [r x c]
void projector(const dcplx* A, int r, int c, std::vector<dcplx>& P) {
  P.assign((size_t)r * r, dcplx(0, 0));
  for (int i = 0; i < r; ++i)
    for (int j = 0; j < r; ++j) {
      dcplx acc(0, 0);
      for (int k = 0; k < c; ++k) acc += A[(size_t)i * c + k] * std::conj(A[(size_t)j * c + k]);
      P[(size_t)i * r + j] = acc;
    }
}
std::vector<double> sigmas(const dcplx* A, int m, int n) {
  std::vector<double> S(n);
  std::vector<dcplx> U(m * n), V(n * n);
  svd_cplx(A, m, n, U.data(), S.data(), V.data());   // sweeps: decomp.h default
  return S;
}
}  // namespace

int main(int argc, char** argv) {
  hpocl::VecReader b(argc > 1 ? argv[1] : "build/frontend_vectors.bin");
  if (!b.ok()) { fprintf(stderr, "[frontend_bench] %s\n", b.error().c_str()); return 2; }
  int ncase = b.i(), fails = 0;
  for (int ci = 0; ci < ncase; ++ci) {
    int K = b.i(), Nt = b.i(), Nr = b.i(), NRF = b.i(), d = b.i(), nul = b.i();
    std::vector<dcplx> H((size_t)K * Nr * Nt), Frf((size_t)Nt * NRF),
        eEff((size_t)K * Nr * NRF), eN((size_t)NRF * nul), eG((size_t)Nr * nul);
    std::vector<double> eEig(nul);
    b.cx(H); b.cx(Frf); b.cx(eEff); b.cx(eN); b.cx(eG); b.re(eEig);
    double eFloor = b.f();
    std::vector<dcplx> eFrf((size_t)Nt * NRF);
    b.cx(eFrf);

    // 1) effective channels: plain GEMM, elementwise comparison
    std::vector<dcplx> eff((size_t)K * Nr * NRF);
    effective_channels(H.data(), K, Nr, Nt, Frf.data(), NRF, eff.data());
    double e_eff = err_c(eff.data(), eEff.data(), eff.size());

    // 2) BD null basis of the stacked interfering users (user 0)
    int rows = (K - 1) * Nr;
    std::vector<dcplx> A((size_t)rows * NRF);
    for (int u = 1; u < K; ++u)
      memcpy(A.data() + (size_t)(u - 1) * Nr * NRF,
             eff.data() + (size_t)u * Nr * NRF, sizeof(dcplx) * Nr * NRF);
    std::vector<dcplx> N((size_t)NRF * NRF);
    int nul_m = null_basis(A.data(), rows, NRF, N.data());
    std::vector<dcplx> Pm, Pe;
    projector(N.data(), NRF, nul_m, Pm);
    projector(eN.data(), NRF, nul, Pe);
    double e_proj = (nul_m == nul) ? err_c(Pm.data(), Pe.data(), Pm.size()) : 1.0;

    // 3) power whitening on F_RF N
    std::vector<dcplx> FN((size_t)Nt * nul_m), inv((size_t)nul_m * nul_m);
    for (int i = 0; i < Nt; ++i)
      for (int j = 0; j < nul_m; ++j) {
        dcplx acc(0, 0);
        for (int k = 0; k < NRF; ++k) acc += Frf[(size_t)i * NRF + k] * N[(size_t)k * nul_m + j];
        FN[(size_t)i * nul_m + j] = acc;
      }
    // production defaults for the C^{-1/2} eigenvalue floor
    // (multiuser_simulation_environment.power_whitening); ref/pipeline.cpp
    // passes the same pair.
    const double kRelFloor = 1e-10, kAbsFloor = 1e-12;
    std::vector<double> eig(nul_m);
    power_whitening(FN.data(), Nt, nul_m, kRelFloor, kAbsFloor, inv.data(), eig.data());
    std::vector<double> eig_s = eig, eeig_s = eEig;
    std::sort(eig_s.begin(), eig_s.end(), std::greater<double>());
    std::sort(eeig_s.begin(), eeig_s.end(), std::greater<double>());
    double e_eig = 0;
    for (int i = 0; i < nul_m; ++i)
      e_eig = fmax(e_eig, fabs(eig_s[i] - eeig_s[i]) / fmax(1.0, fabs(eeig_s[i])));

    // 4) whitened channel: sigma(Gbar) is basis-choice invariant
    std::vector<dcplx> red((size_t)Nr * nul_m), G((size_t)Nr * nul_m);
    for (int i = 0; i < Nr; ++i)
      for (int j = 0; j < nul_m; ++j) {
        dcplx acc(0, 0);
        for (int k = 0; k < NRF; ++k)
          acc += eff[(size_t)0 * Nr * NRF + i * NRF + k] * N[(size_t)k * nul_m + j];
        red[(size_t)i * nul_m + j] = acc;
      }
    whiten_channel(red.data(), Nr, nul_m, inv.data(), G.data());
    auto sM = sigmas(G.data(), Nr, nul_m), sE = sigmas(eG.data(), Nr, nul);
    double e_g = 0;
    for (size_t i = 0; i < sE.size(); ++i)
      e_g = fmax(e_g, fabs(sM[i] - sE[i]) / fmax(1.0, fabs(sE[i])));
    // 4b) the floor itself. power_whitening() does not return it, but it is
    //     max(abs_floor, rel_floor * lambda_max) and lambda_max survives the
    //     flooring (floor <= lambda_max whenever rel_floor < 1; if every
    //     eigenvalue is below abs_floor then max(eig) == abs_floor and the
    //     formula still reproduces the floor), so it can be recomputed and
    //     compared instead of read and thrown away.
    double eig_max = 0;
    for (int i = 0; i < nul_m; ++i) eig_max = fmax(eig_max, eig[i]);
    const double floor_m = fmax(kAbsFloor, kRelFloor * eig_max);
    const double e_floor = fabs(floor_m - eFloor) / fmax(1.0, fabs(eFloor));

    // 5) analog precoder: constant-modulus, column phases are gauge --
    //    align per column, then elementwise
    std::vector<dcplx> Frf_m((size_t)Nt * NRF);
    analog_precoder(H.data(), K, Nr, Nt, NRF, Frf_m.data());
    for (int j = 0; j < NRF; ++j) {
      dcplx dot(0, 0);
      for (int r = 0; r < Nt; ++r)
        dot += std::conj(eFrf[(size_t)r * NRF + j]) * Frf_m[(size_t)r * NRF + j];
      if (std::abs(dot) > 1e-12) {
        dcplx ph = std::conj(dot / std::abs(dot));
        for (int r = 0; r < Nt; ++r) Frf_m[(size_t)r * NRF + j] *= ph;
      }
    }
    double e_frf = err_c(Frf_m.data(), eFrf.data(), eFrf.size());

    bool ok = e_eff < 1e-12 && e_proj < 1e-9 && e_eig < 1e-9 && e_g < 1e-9 &&
              e_frf < 1e-12 && e_floor < 1e-12;
    fails += !ok;
    printf("case%d K=%d Nt=%d Nr=%d NRF=%d d=%d nul=%d: eff=%.1e proj=%.1e eig=%.1e "
           "sigmaG=%.1e floor=%.1e frf=%.1e %s\n",
           ci, K, Nt, Nr, NRF, d, nul, e_eff, e_proj, e_eig, e_g, e_floor, e_frf,
           ok ? "PASS" : "FAIL");
  }
  if (!b.expect_done("frontend_bench")) ++fails;   // drift vs tools/frontend_ref.py
  printf(fails ? "[frontend_bench] FAIL\n" : "[frontend_bench] ALL PASS\n");
  return fails ? 1 : 0;
}
