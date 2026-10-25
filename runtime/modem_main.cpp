// modem_bench: receive-side modem reference (ref/modem.cpp) against the
// production-python oracle in tools/modem_ref.py (bicm_metrics + thp_precoding
// are the ground truth there). Covers gray_standard 16/64/256-QAM:
//   bits -> symbols, THP receive modulo, logsumexp LLRs of the parallel and
//   the modulo-folded branch, and the GMI built from those LLRs.
// Field order mirrors tools/modem_ref.py::field_list.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include "hpocl_vecio.h"
#include <vector>
#include "modem.h"

using namespace hpocl;

namespace {

double err_c(const dcplx* a, const dcplx* b, size_t n) {
  double m = 0; for (size_t i = 0; i < n; ++i) m = fmax(m, std::abs(a[i] - b[i])); return m;
}
double err_r(const double* a, const double* b, size_t n) {
  double m = 0; for (size_t i = 0; i < n; ++i) m = fmax(m, fabs(a[i] - b[i])); return m;
}

// Labeling consistency: hard-slicing a mapped symbol on the table must return
// the bit label it was mapped from (qam_map vs qam_table).
int labeling_errors(int M, const dcplx* sym, const uint8_t* bits, int n) {
  const int m = qam_bits_per_symbol(M);
  std::vector<dcplx> tab(M);
  std::vector<uint8_t> tb((size_t)M * m);
  qam_table(M, tab.data(), tb.data());
  int bad = 0;
  for (int i = 0; i < n; ++i) {
    int best = 0;
    double bd = 1e300;
    for (int s = 0; s < M; ++s) {
      const double d = std::norm(sym[i] - tab[s]);
      if (d < bd) { bd = d; best = s; }
    }
    bad += !std::equal(bits + (size_t)i * m, bits + (size_t)(i + 1) * m,
                       tb.begin() + (size_t)best * m);
  }
  return bad;
}
}  // namespace

int main(int argc, char** argv) {
  const char* path = argc > 1 ? argv[1] : "build/modem_vectors.bin";
  hpocl::VecReader b(path);
  if (!b.ok()) {
    fprintf(stderr, "missing %s (run: python3 tools/modem_ref.py)\n", path);
    return 2;
  }
  const int ncase = b.i();
  int fails = 0;
  for (int ci = 0; ci < ncase; ++ci) {
    const int M = b.i(), n = b.i();
    const double n0 = b.f(), period = b.f();
    const int m = qam_bits_per_symbol(M);
    if (m < 0 || n <= 0) {
      fprintf(stderr, "bad case header: M=%d n=%d\n", M, n);
      return 2;
    }
    const size_t nb = (size_t)n * m;
    std::vector<uint8_t> bits(nb);
    std::vector<dcplx> esym(n), y_par(n), y_thp(n), ez(n);
    std::vector<double> ellr_p(nb), ellr_f(nb), egmi(2);
    b.by(bits); b.cx(esym); b.cx(y_par); b.cx(y_thp); b.cx(ez);
    b.re(ellr_p); b.re(ellr_f); b.re(egmi);

    // 1) geometry: THP modulo period of qam_axis_order_and_spacing
    const double e_per = fabs(qam_modulo_period(M) - period);
    // 2) bits -> symbols, plus the labeling round trip through the table
    std::vector<dcplx> sym(n);
    qam_map(bits.data(), M, sym.data(), n);
    const double e_sym = err_c(sym.data(), esym.data(), n);
    const int lab = labeling_errors(M, sym.data(), bits.data(), n);
    // 3) THP receive modulo fold
    std::vector<dcplx> z(n);
    modulo_fold(y_thp.data(), n, period, z.data());
    const double e_fold = err_c(z.data(), ez.data(), n);
    // 4) logsumexp LLRs and GMI on both branches
    std::vector<double> llr_p(nb), llr_f(nb);
    softdemod_parallel(y_par.data(), n0, M, llr_p.data(), n);
    softdemod_folded(z.data(), n0, M, period, llr_f.data(), n);
    const double e_lp = err_r(llr_p.data(), ellr_p.data(), nb);
    const double e_lf = err_r(llr_f.data(), ellr_f.data(), nb);
    const double gmi_p = gmi_from_llr(llr_p.data(), bits.data(), m, n);
    const double gmi_f = gmi_from_llr(llr_f.data(), bits.data(), m, n);
    const double e_gp = fabs(gmi_p - egmi[0]), e_gf = fabs(gmi_f - egmi[1]);

    const bool ok = e_per < 1e-15 && e_sym < 1e-12 && lab == 0 &&
                    e_fold < 1e-12 && e_lp < 1e-9 && e_lf < 1e-9 &&
                    e_gp < 1e-9 && e_gf < 1e-9;
    fails += !ok;
    printf("case%d M=%3d n0=%5.3f n=%d: per=%.1e sym=%.1e fold=%.1e lab=%d "
           "llr_par=%.1e llr_fold=%.1e gmi_par=%.6f(%.1e) gmi_fold=%.6f(%.1e) %s\n",
           ci, M, n0, n, e_per, e_sym, e_fold, lab, e_lp, e_lf,
           gmi_p, e_gp, gmi_f, e_gf, ok ? "PASS" : "FAIL");
  }
  if (!b.expect_done("modem_bench")) {   // contract drift vs tools/modem_ref.py
    ++fails;
  }
  printf(fails ? "[modem_bench] FAIL\n" : "[modem_bench] ALL PASS\n");
  return fails ? 1 : 0;
}
