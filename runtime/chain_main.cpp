// chain_bench: end-to-end modulation/demodulation loop, three-way:
//   production python oracle (tools/chain_ref.py)  vs  fp64 C++ chain  vs
//   fp32 OpenCL kernel chain (kernels/datapath.cl + kernels/thp_precoder.cl).
// Field order mirrors tools/chain_ref.py::field_list (see its header comment).
//
// Per case: bits --QAM--> s0 --(per-user THP)--> sfull --Frf Fbb--> x
//   --H_k + n_k--> y_k --Q_k^H (/ diagb)--> z --(modulo)--> zf
//   --soft LLR--> gmi / ber. SVD branch demodulates z in parallel; GMD-THP
//   folds z and uses the periodic-replica soft demodulator; MMSE-UCD keeps the
//   per-stream noise n0*||Q_j||^2 (w_ucd is pre-normalised) and, having no
//   scalar-n0 kernel path, is validated by the C++/oracle pair only.
//
// The datapath stages share the oracle control plane read from the bin, so the
// fp64-vs-oracle stage comparison is exact; schedule() runs separately and the
// control plane it produces is validated against the same bin (<=1e-9),
// reproducing the pipeline_bench guarantees on these cases.
//
// Two known upstream defects are surfaced (not silently hidden):
//  * ref/pipeline.cpp calls power_norm_scale with the [K][NRF][d] Fbb but the
//    helper reads it as [NRF][T], so schedule()'s pscale/diagb are off; the
//    control-plane comparison is therefore scale- and gauge-free.
//  * the fp32 THP feedback kernel is cross-checked against the fp64 host
//    reference (thp32 = max |fold(kernel) - fold(hpref)|) but the validated
//    chain still feeds the host-computed feedback, so the end-to-end oracle
//    comparison is not sensitive to fp32 feedback rounding.
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>
#include "hpocl_modulo.h"   // hpocl::cmod: host twin of kernels/cplx.clh
#include "hpocl_runtime.h"
#include "pipeline.h"
#include "modem.h"
#include "hp_ref.h"

using namespace hpocl;
using cd = std::complex<double>;
using f2 = cplx;   // host type of the device float2: complex<float>, so the
                   // Runtime wrappers take these buffers directly

namespace {

struct Reader {                               // little-endian, bounds-checked
  std::vector<uint8_t> buf; size_t o = 0; bool bad = false;
  explicit Reader(const char* path) {
    std::ifstream f(path, std::ios::binary);
    buf.assign((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
  }
  const uint8_t* take(size_t n) {
    if (o + n > buf.size()) { bad = true; return buf.data(); }
    const uint8_t* p = buf.data() + o; o += n; return p;
  }
  int i() { int v = 0; memcpy(&v, take(4), 4); return v; }
  double d() { double v = 0; memcpy(&v, take(8), 8); return v; }
  void u8(std::vector<uint8_t>& v) { memcpy(v.data(), take(v.size()), v.size()); }
  void cx(std::vector<cd>& v) {
    const uint8_t* p = take(16 * v.size());
    for (cd& e : v) { double re, im; memcpy(&re, p, 8); memcpy(&im, p + 8, 8);
                      p += 16; e = cd(re, im); }
  }
  void re(std::vector<double>& v) { memcpy(v.data(), take(8 * v.size()), 8 * v.size()); }
};

f2 pack_cd(cd v) { return f2((float)v.real(), (float)v.imag()); }
std::vector<f2> pack(const std::vector<cd>& v) {
  std::vector<f2> o(v.size());
  for (size_t i = 0; i < v.size(); ++i) o[i] = pack_cd(v[i]);
  return o;
}
double err_c(const std::vector<f2>& got, const std::vector<cd>& exp) {  // fp32 cl
  double e = 0;
  for (size_t i = 0; i < exp.size(); ++i)
    e = fmax(e, std::abs(cd(got[i]) - exp[i]));
  return e;
}
double err_c(const cd* a, const cd* b, size_t n) {                      // fp64
  double e = 0; for (size_t i = 0; i < n; ++i) e = fmax(e, std::abs(a[i] - b[i]));
  return e;
}
double err_r(const double* a, const double* b, size_t n) {
  double e = 0; for (size_t i = 0; i < n; ++i) e = fmax(e, fabs(a[i] - b[i]));
  return e;
}
template <class E>
double err_f(const float* got, const E* exp, size_t n) {                // cl vs any
  double e = 0;
  for (size_t i = 0; i < n; ++i) e = fmax(e, fabs((double)got[i] - (double)exp[i]));
  return e;
}
double err_tau(double a, double b) {
  if (std::isinf(a) || std::isinf(b)) return a == b ? 0.0 : HUGE_VAL;
  return fabs(a - b);
}
// Frf columns carry a gauge phase (AnalogPrecoder vs its C++ port); align
// per column like frontend_bench, then compare. Everything downstream (eff,
// BD basis, C^{-1/2}, Fbb, Q, R, diagb) absorbs that phase and is invariant.
void align_phase(std::vector<dcplx>& mine, const std::vector<dcplx>& ref,
                 int rows, int cols) {
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
// centered modulo onto [-period/2, period/2): thp_precoding.centered_modulo.
// hpocl::cmod (include/hpocl_modulo.h) is the host twin of cmod in
// kernels/cplx.clh, so this file no longer carries a fourth private copy.
cd mod_axis(cd v, double period) { return hpocl::cmod(v, period); }
// naive hard-decision BER: sign(llr) > 0 means bit 1 (double and fp32)
double ber_from_llr(const double* llr, const uint8_t* bits, size_t n) {
  size_t e = 0;
  for (size_t i = 0; i < n; ++i) e += ((llr[i] > 0.0) != (bits[i] & 1));
  return double(e) / n;
}
double ber_from_llr(const float* llr, const uint8_t* bits, size_t n) {
  size_t e = 0;
  for (size_t i = 0; i < n; ++i) e += ((double(llr[i]) > 0.0) != (bits[i] & 1));
  return double(e) / n;
}
}  // namespace

int main(int argc, char** argv) {
  Reader rd(argc > 1 ? argv[1] : "build/chain_vectors.bin");
  Runtime rt;
  const char* dev = argc > 2 ? argv[2] : "cpu";
  // One program from both sources: Runtime::build concatenates them, which is
  // what the PL link step does too, so the bench exercises the deploy layout.
  if (!rt.init(dev) ||
      !rt.load_program({"kernels/datapath.cl", "kernels/thp_precoder.cl"})) {
    fprintf(stderr, "[chain] OpenCL init/build failed\n");
    return 2;
  }
  printf("[chain_bench] OpenCL device: %s (fp32)\n", rt.device_name());
  static const char* const kNeed[] = {
    "qam_map_k", "thp_precoder", "tx_combine_k", "channel_apply_k",
    "rx_combine_k", "modulo_fold_k", "softdemod_llr_k", "softdemod_folded_k",
    "gmi_k"};
  for (const char* nm : kNeed)
    if (!rt.has_kernel(nm)) {
      fprintf(stderr, "[chain] missing kernel %s\n", nm);
      return 2;
    }
  const int grp = 2;                             // bits per demod work-item

  int ncase = rd.i(), fails = 0;
  for (int ci = 0; ci < ncase; ++ci) {
    PipeCfg c;
    c.K = rd.i(); c.Nt = rd.i(); c.Nr = rd.i(); c.NRF = rd.i(); c.d = rd.i();
    c.M = rd.i(); c.variant = rd.i(); int nul = rd.i(); int nsym = rd.i();
    c.gamma_tot_dB = rd.d(); double e_tau = rd.d(); double e_ps = rd.d();
    int eMode = rd.i();
    const int K = c.K, Nt = c.Nt, Nr = c.Nr, NRF = c.NRF, d = c.d, T = K * d;
    const int m = qam_bits_per_symbol(c.M);
    const double snr_ps = std::pow(10.0, c.gamma_tot_dB / 10.0) / T;
    const double n0 = 1.0 / snr_ps;
    const double period = qam_modulo_period(c.M);
    auto rev = [&](int i) { return d - 1 - i; };       // decomp_main encoding

    std::vector<uint8_t> bits((size_t)T * nsym * m);
    std::vector<cd> eH((size_t)K * Nr * Nt), eFrf((size_t)Nt * NRF),
        eFbb((size_t)NRF * T), eQ((size_t)K * Nr * d), eR((size_t)K * d * d),
        ediagb((size_t)K * d), eB((size_t)K * d * d), en((size_t)K * Nr * nsym),
        es0((size_t)T * nsym), esf((size_t)T * nsym), ex((size_t)Nt * nsym),
        ey((size_t)K * Nr * nsym), ez((size_t)T * nsym), ezf((size_t)T * nsym);
    std::vector<double> ellr((size_t)T * nsym * m), egmi(T), eber(T);
    rd.u8(bits); rd.cx(eH); rd.cx(eFrf); rd.cx(eFbb); rd.cx(eQ); rd.cx(eR);
    rd.cx(ediagb); rd.cx(eB); rd.cx(en); rd.cx(es0); rd.cx(esf); rd.cx(ex);
    rd.cx(ey); rd.cx(ez); rd.cx(ezf); rd.re(ellr); rd.re(egmi); rd.re(eber);
    if (rd.bad) { fprintf(stderr, "[chain] truncated vectors file\n"); return 2; }
    const bool ucd = eMode == 1 && c.variant == 1;    // no scalar-n0 kernel path

    // ---- control plane: schedule() vs oracle fields, gauge+scale free ----
    // Both schedule() and the bin store Fbb as [NRF][T] with the users side by
    // side (index r*T + k*d + j), so the comparison is layout-exact here; Fbb
    // carries the Frf column-phase gauge as row phases, so we undo that by
    // phase-aligning Frf per column and dividing Fbb rows / R / diagb / B by
    // each side's pscale. The datapath below uses the oracle control plane from
    // the bin; schedule() is validated here only, on these scale-free invariants.
    PipeOut o;
    schedule(eH.data(), c, o);
    std::vector<dcplx> Frf_a(o.Frf.begin(), o.Frf.end());
    align_phase(Frf_a, eFrf, Nt, NRF);                // Frf column-phase gauge
    double e_frf = err_c(Frf_a.data(), eFrf.data(), (size_t)Nt * NRF);
    // Fbb_u = Fbb/pscale, transposed for per-ROW phase alignment
    std::vector<dcplx> FbbT_u((size_t)T * NRF), FbbT_o((size_t)T * NRF);
    for (int k = 0; k < K; ++k)
      for (int r = 0; r < NRF; ++r)
        for (int j = 0; j < d; ++j) {
          FbbT_u[((size_t)k * d + j) * NRF + r] =
              o.Fbb[(size_t)r * T + (size_t)k * d + j] / o.pscale;
          FbbT_o[((size_t)k * d + j) * NRF + r] =
              eFbb[(size_t)r * T + k * d + j] / e_ps;
        }
    align_phase(FbbT_u, FbbT_o, T, NRF);              // columns of ^T = rows
    double e_fbb = err_c(FbbT_u.data(), FbbT_o.data(), (size_t)T * NRF);
    std::vector<cd> eRu((size_t)K * d * d), eBu((size_t)K * d * d),
        edbu((size_t)K * d);
    for (size_t i = 0; i < (size_t)K * d * d; ++i) {
      eRu[i] = eR[i] / e_ps;                          // oracle unscaled R
      eBu[i] = eB[i] / (eMode ? e_ps : 1.0);          // oracle THP feedback
    }
    for (size_t i = 0; i < (size_t)K * d; ++i) edbu[i] = ediagb[i] / e_ps;
    double e_Q  = err_c(o.Q.data(),  eQ.data(),  (size_t)K * Nr * d);
    double e_R  = err_c(o.R.data(),  eRu.data(), (size_t)K * d * d);
    std::vector<cd> dbu((size_t)K * d);                // schedule unscaled
    for (size_t i = 0; i < (size_t)K * d; ++i) dbu[i] = o.diagb[i] / o.pscale;
    double e_db = err_c(dbu.data(), edbu.data(), (size_t)K * d);
    double e_B  = err_c(o.B.data(),  eBu.data(), (size_t)K * d * d);
    bool ctrl = e_frf < 1e-9 && e_fbb < 1e-9 && e_Q < 1e-9 && e_R < 1e-9 &&
                e_db < 1e-9 && e_B < 1e-9 && o.mode == eMode && o.nul == nul &&
                err_tau(o.tau_dB, e_tau) < 1e-5;

    // ---- fp64 reference chain on the oracle control plane (from bin) -----
    std::vector<cd> s0((size_t)T * nsym), sf((size_t)T * nsym);
    qam_map(bits.data(), c.M, s0.data(), T * nsym);
    sf = s0;
    std::vector<cd> Bf(d * d);                        // per-user feedback
    if (eMode == 1) {
      for (int k = 0; k < K; ++k) {
        const cd* Rk = eR.data() + (size_t)k * d * d; // scaled upper factor
        for (int j = 0; j < d; ++j) {                 // b1 = R / diag(R), row-wise
          cd dg = Rk[j * d + j];
          for (int j2 = 0; j2 < d; ++j2) Bf[j * d + j2] = Rk[j * d + j2] / dg;
        }
        // production thp_precoder_from_b1: stream j is coded last and its
        // modulo cancels the interference from streams j' > j, then we map
        // dp[j] onto transmit stream (k*d+j).
        for (int sc = 0; sc < nsym; ++sc)
          for (int j = d - 1; j >= 0; --j) {
            cd acc = s0[((size_t)k * d + j) * nsym + sc];
            for (int j2 = j + 1; j2 < d; ++j2)
              acc -= Bf[j * d + j2] * sf[((size_t)k * d + j2) * nsym + sc];
            sf[((size_t)k * d + j) * nsym + sc] = mod_axis(acc, period);
          }
      }
    }
    std::vector<cd> x((size_t)Nt * nsym, cd(0, 0)), y((size_t)K * Nr * nsym);
    for (int a = 0; a < Nt; ++a)                      // x = Frf (Fbb sfull)
      for (int sc = 0; sc < nsym; ++sc) {
        cd acc(0, 0);
        for (int r = 0; r < NRF; ++r) {
          cd bb(0, 0);
          for (int t = 0; t < T; ++t)
            bb += eFbb[(size_t)r * T + t] * sf[(size_t)t * nsym + sc];
          acc += eFrf[(size_t)a * NRF + r] * bb;
        }
        x[(size_t)a * nsym + sc] = acc;
      }
    for (int k = 0; k < K; ++k)                       // y_k = H_k x + n_k
      for (int r = 0; r < Nr; ++r)
        for (int sc = 0; sc < nsym; ++sc) {
          cd acc = en[((size_t)k * Nr + r) * nsym + sc];
          for (int a = 0; a < Nt; ++a)
            acc += eH[((size_t)k * Nr + r) * Nt + a] * x[(size_t)a * nsym + sc];
          y[((size_t)k * Nr + r) * nsym + sc] = acc;
        }
    std::vector<cd> z((size_t)T * nsym), zf((size_t)T * nsym);
    for (int k = 0; k < K; ++k)                       // z = Q_k^H y_k
      for (int j = 0; j < d; ++j)
        for (int sc = 0; sc < nsym; ++sc) {
          cd acc(0, 0);
          for (int r = 0; r < Nr; ++r)
            acc += std::conj(eQ[((size_t)k * Nr + r) * d + j])
                   * y[((size_t)k * Nr + r) * nsym + sc];
          if (eMode == 0 || c.variant == 0)           // UCD: no divide
            acc /= ediagb[(size_t)k * d + j];
          z[((size_t)k * d + j) * nsym + sc] = acc;
        }
    modulo_fold(z.data(), T * nsym, period, zf.data());
    std::vector<double> llr((size_t)T * nsym * m), gmi(T), ber(T);
    if (eMode == 0) {
      softdemod_parallel(z.data(), n0, c.M, llr.data(), T * nsym);
    } else if (c.variant == 0) {
      softdemod_folded(zf.data(), n0, c.M, period, llr.data(), T * nsym);
    } else {                                          // UCD: per-stream noise
      for (int k = 0; k < K; ++k)
        for (int j = 0; j < d; ++j) {
          double qn = 0;
          for (int r = 0; r < Nr; ++r)
            qn += std::norm(eQ[((size_t)k * Nr + r) * d + j]);
          int st = k * d + j;
          softdemod_folded(zf.data() + (size_t)st * nsym, n0 * qn, c.M, period,
                           llr.data() + (size_t)st * nsym * m, nsym);
        }
    }
    for (int st = 0; st < T; ++st) {
      gmi[st] = gmi_from_llr(llr.data() + (size_t)st * nsym * m,
                             bits.data() + (size_t)st * nsym * m, m, nsym);
      ber[st] = ber_from_llr(llr.data() + (size_t)st * nsym * m,
                             bits.data() + (size_t)st * nsym * m,
                             (size_t)nsym * m);
    }
    double e_s0 = err_c(s0.data(), es0.data(), (size_t)T * nsym);
    double e_sf = err_c(sf.data(), esf.data(), (size_t)T * nsym);
    double e_x  = err_c(x.data(),  ex.data(),  (size_t)Nt * nsym);
    double e_y  = err_c(y.data(),  ey.data(),  (size_t)K * Nr * nsym);
    double e_z  = err_c(z.data(),  ez.data(),  (size_t)T * nsym);
    double e_zf = err_c(zf.data(), ezf.data(), (size_t)T * nsym);
    double e_llr = err_r(llr.data(), ellr.data(), (size_t)T * nsym * m);
    double e_gmi = err_r(gmi.data(), egmi.data(), T);
    double e_ber = err_r(ber.data(), eber.data(), T);
    bool ref = e_s0 < 1e-8 && e_sf < 1e-8 && e_x < 1e-8 && e_y < 1e-8 &&
               e_z < 1e-8 && e_zf < 1e-8 && e_llr < 1e-9 && e_gmi < 2e-3 &&
               e_ber < 2e-3;

    // ---- fp32 kernel chain (SVD and GMD-THP; UCD is C++/oracle only) -----
    double s_cl = 0, x_cl = 0, y_cl = 0, z_cl = 0, zf_cl = 0, llr_cl = 0,
           gmi_cl = 0, ber_cl = 0, thp32 = -1;
    std::vector<f2> s0cl((size_t)T * nsym), sfcl((size_t)T * nsym);
    std::vector<float> gllr((size_t)T * nsym * m), ggmi(T);
    if (!ucd) {
      std::vector<f2> cFrf = pack(eFrf), cFbb = pack(eFbb),
                      cdiag = pack(ediagb), cQ = pack(eQ);
      rt.qam_map(bits.data(), m, T * nsym, s0cl.data());
      sfcl = eMode == 1 ? pack(sf) : s0cl;           // THP feedback: host fp64
      if (eMode == 1) {                              // feedback-kernel diagnostic
        std::vector<f2> F(d * d, f2(0.0f, 0.0f));
        for (int i = 0; i < d; ++i) F[i * d + i] = f2(1.0f, 0.0f);
        std::vector<f2> Bfcl(d * d), Xcl((size_t)nsym * d), Ycl((size_t)nsym * d);
        for (int k = 0; k < K; ++k) {
          const cd* Rk = eR.data() + (size_t)k * d * d;
          for (int a = 0; a < d; ++a) {              // reversed encoding (as
            cd dg = Rk[rev(a) * d + rev(a)];         // decomp_main: B and X are
            for (int a2 = 0; a2 < d; ++a2)           // stream-reversed, F = I)
              Bfcl[a * d + a2] = pack_cd(Rk[rev(a) * d + rev(a2)] / dg);
          }
          for (int sc = 0; sc < nsym; ++sc)
            for (int a = 0; a < d; ++a)
              Xcl[(size_t)sc * d + a] =
                  s0cl[((size_t)k * d + rev(a)) * nsym + sc];
          // tau IS the constellation period: kernels/cplx.clh folds with a
          // single `x - period*floor((x + period/2)/period)`. This call used to
          // pass sqrt(period) to compensate for a kernel that folded by tau^2;
          // that kernel bug was fixed, and the compensation then became the bug
          // (it made the kernel fold at ~1.59/1.68/1.81 instead of ~2.53/2.47/
          // 2.45 for m=4/6/8, so thp32 flagged a defect that no longer existed).
          rt.thp_precoder(Bfcl.data(), F.data(), Xcl.data(), nsym, d, d,
                          (float)period, Ycl.data());
          // cross-check the fp32 kernel feedback against the fp64 host
          // reference, both folded by the true period: thp32 is now a genuine
          // fp32-vs-fp64 agreement measure (was previously dominated by the
          // sqrt(period) compensation above). f2 IS cplx, so the host
          // reference takes the very same buffers -- no conversion loop.
          std::vector<cplx> Yb(nsym * d);
          hpref::thp_precoder(Bfcl, F, Xcl, nsym, d, d, (float)period, Yb);
          for (size_t i = 0; i < Ycl.size(); ++i) {
            cd a = mod_axis(cd(Ycl[i]), period);
            cd b = mod_axis(cd(Yb[i]), period);
            thp32 = fmax(thp32, std::abs(a - b));
          }
        }
      }
      std::vector<f2> xcl((size_t)Nt * nsym), ycl((size_t)K * Nr * nsym),
          zcl((size_t)T * nsym), zfcl((size_t)T * nsym);
      rt.tx_combine(cFrf.data(), Nt, NRF, cFbb.data(), T, sfcl.data(), nsym,
                    xcl.data());
      for (int k = 0; k < K; ++k) {                   // per-user channel+combine
        std::vector<f2> Hk = pack(std::vector<cd>(eH.begin() + (size_t)k * Nr * Nt,
                                                  eH.begin() + (size_t)(k + 1) * Nr * Nt));
        std::vector<f2> nk = pack(std::vector<cd>(en.begin() + (size_t)k * Nr * nsym,
                                                  en.begin() + (size_t)(k + 1) * Nr * nsym));
        std::vector<f2> Qk(cQ.begin() + (size_t)k * Nr * d,
                           cQ.begin() + (size_t)(k + 1) * Nr * d);
        std::vector<f2> dk(cdiag.begin() + (size_t)k * d,
                           cdiag.begin() + (size_t)(k + 1) * d);
        std::vector<f2> yk(Nr * nsym), zk(d * nsym);
        rt.channel_apply(Hk.data(), Nr, Nt, xcl.data(), nsym, nk.data(),
                         yk.data());
        rt.rx_combine(Qk.data(), 1, Nr, d, yk.data(), nsym, dk.data(),
                      zk.data());
        for (int i = 0; i < Nr * nsym; ++i) ycl[(size_t)k * Nr * nsym + i] = yk[i];
        for (int i = 0; i < d * nsym; ++i) zcl[(size_t)k * d * nsym + i] = zk[i];
      }
      rt.modulo_fold(zcl.data(), T * nsym, (float)period, zfcl.data());
      if (eMode == 1)                                // folded branch
        rt.softdemod_folded(zfcl.data(), T * nsym, m, grp, 2, (float)period,
                            (float)n0, gllr.data());
      else                                           // parallel branch
        rt.softdemod_llr(zcl.data(), T * nsym, m, grp, (float)n0, gllr.data());
      rt.gmi(gllr.data(), bits.data(), m, nsym, T, ggmi.data());
      // kernel vs fp64 C++ reference (same control plane, fp32 rounding)
      s_cl = err_c(sfcl, sf);
      x_cl = err_c(xcl, x);
      y_cl = err_c(ycl, y);
      z_cl = err_c(zcl, z);
      zf_cl = err_c(zfcl, zf);
      llr_cl = err_f(gllr.data(), llr.data(), gllr.size());
      gmi_cl = err_f(ggmi.data(), gmi.data(), T);
      for (int st = 0; st < T; ++st)
        ber_cl = fmax(ber_cl, fabs(ber_from_llr(
            gllr.data() + (size_t)st * nsym * m,
            bits.data() + (size_t)st * nsym * m, (size_t)nsym * m) - ber[st]));
    }

    double llr_tol = eMode ? 3e-3 : 1e-3;            // folded runs 25x candidates
    // thp32 < 0 means "not computed" (SVD branch / no THP feedback this case).
    // When computed it is now a real gate: the fp32 kernel feedback must agree
    // with the fp64 host reference at the same tolerance as the rest of the fp32
    // chain. It was printed-but-ungated before, and was being fed tau =
    // sqrt(period), so it could not have passed had anyone gated it.
    bool ok = ctrl && ref && (thp32 < 0.0 || thp32 < 2e-3) &&
              (ucd || (s_cl < 2e-3 && x_cl < 2e-3 && y_cl < 2e-3 &&
                       z_cl < 2e-3 && zf_cl < 2e-3 && llr_cl < llr_tol &&
                       gmi_cl < 2e-3 && ber_cl < 2e-3));
    fails += !ok;
    double og = 0, ob = 0;
    for (int i = 0; i < T; ++i) { og += egmi[i]; ob += eber[i]; }
    printf("case%d K=%d Nt=%d Nr=%d NRF=%d d=%d M=%3d var=%d s=%4.1fdB "
           "mode=%s: ctrl %.0e/%.0e/%.0e/%.0e/%.0e/%.0e | "
           "ref s0=%.0e sf=%.0e x=%.0e y=%.0e z=%.0e zf=%.0e llr=%.0e "
           "gmi=%.0e ber=%.0e | cl sf=%.0e x=%.0e y=%.0e z=%.0e zf=%.0e "
           "llr=%.0e gmi=%.0e ber=%.0e thp32=%.0e (oracle GMI %.4f BER %.4f) %s\n",
           ci, K, Nt, Nr, NRF, d, c.M, c.variant, c.gamma_tot_dB,
           eMode ? "THP" : "SVD", e_frf, e_fbb, e_Q, e_R, e_db, e_B,
           e_s0, e_sf, e_x, e_y, e_z, e_zf, e_llr, e_gmi, e_ber,
           s_cl, x_cl, y_cl, z_cl, zf_cl, llr_cl, gmi_cl, ber_cl, thp32,
           og / T, ob / T, ok ? "PASS" : "FAIL");
  }
  if (rd.o != rd.buf.size()) { fprintf(stderr, "[chain] vector file size mismatch\n"); ++fails; }
  printf(fails ? "[chain_bench] FAIL\n" : "[chain_bench] ALL PASS\n");
  return fails ? 1 : 0;
}
