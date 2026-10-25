//-----------------------------------------------------------------------------
// ref/modem.cpp -- receive-side modem reference. modem.h maps every entry
// point to the production function it ports and fixes the LLR/GMI conventions.
//-----------------------------------------------------------------------------
#include "modem.h"
#include "hpocl_modulo.h"

#include <algorithm>
#include <cmath>
#include <vector>

namespace hpocl {
namespace {

constexpr double kLn2 = 0.6931471805599453;  // np.log(2.0)
constexpr int kReplicaRadius = 2;            // enumerate_thp_symbol_replicas

// bicm_metrics.stable_binary_logloss_from_signed_llr = logaddexp(0, -t)/ln2,
// split so that neither branch overflows for |t| > 700.
double logloss_bits(double t) {
  return t >= 0.0 ? std::log1p(std::exp(-t)) / kLn2
                  : (-t + std::log1p(std::exp(t))) / kLn2;
}

// bicm_metrics.stable_logsumexp: max-shifted sum in the natural log domain.
double logsumexp(const double* v, int n) {
  const double mx = *std::max_element(v, v + n);
  double sum = 0.0;
  for (int i = 0; i < n; ++i) sum += std::exp(v[i] - mx);
  return mx + std::log(sum);
}

// Unit-average-power scale of get_constellation: sqrt(10), sqrt(42), sqrt(170)
// for L = 4, 8, 16 PAM points per axis.
double pam_norm(int L) { return std::sqrt(2.0 * (L * L - 1) / 3.0); }

// PAM index -> Gray label (idx ^ (idx >> 1)) written MSB first into out[ma];
// the labeling of all three get_constellation branches.
void gray_label(int idx, int ma, uint8_t* out) {
  const int g = idx ^ (idx >> 1);
  for (int k = 0; k < ma; ++k) out[k] = static_cast<uint8_t>((g >> (ma - 1 - k)) & 1);
}

// Inverse of gray_label: ma-bit label -> amplitude -(L-1) + 2*idx, L = 2^ma.
double pam_from_gray(const uint8_t* b, int ma) {
  int g = 0;
  for (int k = 0; k < ma; ++k) g = (g << 1) | (b[k] & 1);
  int idx = g;                              // Gray -> binary
  for (int s = g >> 1; s != 0; s >>= 1) idx ^= s;
  return -static_cast<double>((1 << ma) - 1) + 2.0 * idx;
}

// One constellation table plus its geometry; empty (ok() false) if the order is
// outside {16, 64, 256}, mirroring the ValueError of get_constellation.
struct Qam {
  int m = -1;
  std::vector<dcplx> sym;      // [M]
  std::vector<uint8_t> bits;   // [M*m]

  explicit Qam(int M) : m(qam_bits_per_symbol(M)) {
    if (m < 0) return;
    const int ma = m / 2, L = 1 << ma;
    const double norm = pam_norm(L);
    sym.resize(M);
    bits.assign(static_cast<size_t>(M) * m, 0);
    for (int ir = 0; ir < L; ++ir) {                 // get_constellation loop
      for (int ii = 0; ii < L; ++ii) {               // order: I axis outer
        const int s = ir * L + ii;
        const double re = -static_cast<double>(L - 1) + 2.0 * ir;
        const double im = -static_cast<double>(L - 1) + 2.0 * ii;
        sym[s] = dcplx(re / norm, im / norm);
        gray_label(ir, ma, &bits[static_cast<size_t>(s) * m]);
        gray_label(ii, ma, &bits[static_cast<size_t>(s) * m + ma]);
      }
    }
  }
  bool ok() const { return m >= 0; }
};

// Shared body of both soft demodulators: LLRs of n samples against a fixed
// candidate set, metric -|y - c|^2/n0 and one logsumexp per label bit
// (bicm_metrics.bit_llr_from_distances with max_log=False). Every label bit is
// one for exactly half of the candidates, so both subsets are non-empty.
void softdemod_set(const dcplx* y, int n, double n0, const dcplx* cand,
                   const uint8_t* cbits, int ncand, int m, double* llr) {
  std::vector<double> scratch(3 * static_cast<size_t>(ncand));
  double* metric = scratch.data();
  double* one = metric + ncand;
  double* zero = one + ncand;
  for (int i = 0; i < n; ++i) {
    for (int c = 0; c < ncand; ++c)
      metric[c] = -std::norm(y[i] - cand[c]) / n0;
    for (int b = 0; b < m; ++b) {
      int n_one = 0, n_zero = 0;
      for (int c = 0; c < ncand; ++c) {
        if (cbits[static_cast<size_t>(c) * m + b]) one[n_one++] = metric[c];
        else                                       zero[n_zero++] = metric[c];
      }
      llr[static_cast<size_t>(i) * m + b] =
          logsumexp(one, n_one) - logsumexp(zero, n_zero);
    }
  }
}

// thp_precoding.centered_modulo -- single definition in include/hpocl_modulo.h,
// shared with hpref::thp_precoder and with kernels/cplx.clh.
double fold_axis(double x, double period) { return hpocl::cmod_axis(x, period); }
}  // namespace

int qam_bits_per_symbol(int M) {
  switch (M) {
    case 16:  return 4;
    case 64:  return 6;
    case 256: return 8;
    default:  return -1;
  }
}

int qam_axis_order(int M) {
  const int m = qam_bits_per_symbol(M);
  return m < 0 ? 0 : 1 << (m / 2);
}

double qam_axis_spacing(int M) {
  const int L = qam_axis_order(M);
  return L == 0 ? 0.0 : 2.0 / pam_norm(L);
}

double qam_modulo_period(int M) {
  return qam_axis_order(M) * qam_axis_spacing(M);
}

void qam_table(int M, dcplx* sym, uint8_t* bits) {
  const Qam q(M);
  if (!q.ok()) return;
  std::copy_n(q.sym.begin(), M, sym);
  std::copy_n(q.bits.begin(), static_cast<size_t>(M) * q.m, bits);
}

void qam_map(const uint8_t* bits, int M, dcplx* sym, int n) {
  const int m = qam_bits_per_symbol(M);
  if (m < 0) return;
  const int ma = m / 2, L = 1 << ma;
  const double norm = pam_norm(L);
  for (int i = 0; i < n; ++i) {
    const uint8_t* b = bits + static_cast<size_t>(i) * m;
    sym[i] = dcplx(pam_from_gray(b, ma) / norm, pam_from_gray(b + ma, ma) / norm);
  }
}

void modulo_fold(const dcplx* z, int n, double period, dcplx* out) {
  for (int i = 0; i < n; ++i)
    out[i] = dcplx(fold_axis(z[i].real(), period), fold_axis(z[i].imag(), period));
}

void softdemod_parallel(const dcplx* y, double n0, int M, double* llr, int n) {
  const Qam q(M);
  if (!q.ok()) return;
  softdemod_set(y, n, n0, q.sym.data(), q.bits.data(), M, q.m, llr);
}

void softdemod_folded(const dcplx* y, double n0, int M, double period,
                      double* llr, int n) {
  const Qam q(M);
  if (!q.ok()) return;
  const int shifts = 2 * kReplicaRadius + 1;
  const int ncand = shifts * shifts * M;
  std::vector<dcplx> cand(ncand);
  std::vector<uint8_t> cbits(static_cast<size_t>(ncand) * q.m);
  // replica order of enumerate_thp_symbol_replicas: real shift outer,
  // imaginary shift inner, base constellation innermost (same labels).
  int c = 0;
  for (int lr = -kReplicaRadius; lr <= kReplicaRadius; ++lr) {
    for (int li = -kReplicaRadius; li <= kReplicaRadius; ++li) {
      for (int s = 0; s < M; ++s, ++c) {
        cand[c] = dcplx(q.sym[s].real() + period * lr,
                        q.sym[s].imag() + period * li);
        std::copy_n(&q.bits[static_cast<size_t>(s) * q.m], q.m,
                    &cbits[static_cast<size_t>(c) * q.m]);
      }
    }
  }
  softdemod_set(y, n, n0, cand.data(), cbits.data(), ncand, q.m, llr);
}

// Unclamped per stream, matching kernels/datapath.cl::gmi_k: production clamps
// the AGGREGATE (max(0.0, num_streams*m - loss)), not each stream, so the clamp
// belongs to whoever sums the streams. See the CLAMP CONTRACT in datapath.cl.
double gmi_from_llr(const double* llr, const uint8_t* bits, int m, int n) {
  double loss = 0.0;
  for (int i = 0; i < n; ++i)
    for (int b = 0; b < m; ++b) {
      const size_t k = static_cast<size_t>(i) * m + b;
      loss += logloss_bits((2.0 * bits[k] - 1.0) * llr[k]);
    }
  return m - loss / (n > 0 ? n : 1);  // production divides by max(n, 1)
}
}  // namespace hpocl
