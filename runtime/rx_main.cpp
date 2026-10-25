// rx_bench: receive-side control plane (ref/rx.cpp) against the
// production-python oracle in tools/rx_ref.py, which calls
// build_structured_digital_chain's per-strategy factors, the digital power
// normalization and compute_effective_channel. Per case: combiner Q, r_chain R,
// the normalization scale, the normalized r_chain and the THP receive diagonal.
// Field order mirrors tools/rx_ref.py::field_list.
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include "hpocl_vecio.h"
#include <vector>
#include "rx.h"

using namespace hpocl;

namespace {

double err_c(const dcplx* a, const dcplx* b, size_t n) {
  double m = 0; for (size_t i = 0; i < n; ++i) m = fmax(m, std::abs(a[i] - b[i])); return m;
}
}  // namespace

int main(int argc, char** argv) {
  const char* path = argc > 1 ? argv[1] : "build/rx_vectors.bin";
  hpocl::VecReader b(path);
  if (!b.ok()) {
    fprintf(stderr, "missing %s (run: python3 tools/rx_ref.py)\n", path);
    return 2;
  }
  const int ncase = b.i();
  int fails = 0;
  for (int ci = 0; ci < ncase; ++ci) {
    const int Nr = b.i(), rows = b.i(), d = b.i(), Nt = b.i(), T = b.i();
    const char strategy = (char)b.i();
    const double snr = b.f(), p_constraint = b.f();
    if (Nr <= 0 || rows < d || d <= 0 || Nt <= 0 || T <= 0) {
      fprintf(stderr, "bad case header: Nr=%d rows=%d d=%d Nt=%d T=%d\n",
              Nr, rows, d, Nt, T);
      return 2;
    }
    std::vector<dcplx> eU((size_t)Nr * d), eV((size_t)rows * d), eQ((size_t)Nr * d),
        eR((size_t)d * d), eFrf((size_t)Nt * rows), eFbb((size_t)rows * T),
        eRn((size_t)d * d), eDiag(d);
    std::vector<double> eSigma(d), eScale(1);
    b.cx(eU); b.cx(eV); b.re(eSigma); b.cx(eQ); b.cx(eR);
    b.cx(eFrf); b.cx(eFbb); b.re(eScale); b.cx(eRn); b.cx(eDiag);

    // 1) per-strategy combiner and r_chain (rows is the RF-chain count of the
    //    reduced channel, i.e. the width of Frf)
    std::vector<dcplx> Q((size_t)Nr * d), R((size_t)d * d);
    rx_combiner(eU.data(), Nr, eV.data(), rows, eSigma.data(), d, snr, strategy,
                Q.data(), R.data());
    const double e_q = err_c(Q.data(), eQ.data(), Q.size());
    const double e_r = err_c(R.data(), eR.data(), R.size());
    // 2) digital power normalization, applied to the r_chain as in production
    const double scale = power_norm_scale(eFrf.data(), Nt, rows, eFbb.data(), T,
                                          p_constraint);
    const double e_scale = fabs(scale - eScale[0]);
    std::vector<dcplx> Rn(R.size());
    for (size_t i = 0; i < Rn.size(); ++i) Rn[i] = scale * R[i];
    const double e_rn = err_c(Rn.data(), eRn.data(), Rn.size());
    // 3) THP receive diagonal of the normalized r_chain
    std::vector<dcplx> diag((size_t)d);
    rx_diag(Rn.data(), d, diag.data());
    const double e_diag = err_c(diag.data(), eDiag.data(), diag.size());

    const bool ok = e_q < 1e-9 && e_r < 1e-9 && e_scale < 1e-9 && e_rn < 1e-9 &&
                    e_diag < 1e-9;
    fails += !ok;
    printf("case%2d Nr=%d rows=%d d=%d %c snr=%6.1f: Q=%.1e R=%.1e "
           "scale=%.9f(%.1e) Rnorm=%.1e diag=%.1e %s\n",
           ci, Nr, rows, d, strategy, snr, e_q, e_r, scale, e_scale, e_rn,
           e_diag, ok ? "PASS" : "FAIL");
  }
  if (!b.expect_done("rx_bench")) {      // contract drift vs tools/rx_ref.py
    ++fails;
  }
  printf(fails ? "[rx_bench] FAIL\n" : "[rx_bench] ALL PASS\n");
  return fails ? 1 : 0;
}
