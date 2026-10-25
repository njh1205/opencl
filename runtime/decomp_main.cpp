// decomp_bench: SVD / GMD / UCD / GP-CBD operators, three-way:
//   production python oracle  vs  fp64 C++ reference  vs  fp32 OpenCL kernel.
// Field order mirrors tools/decomp_ref.py::field_list.
//
// Binary contract (little endian), kept in sync with tools/decomp_ref.py:
//   int32 ncase
//   per case: int32 strat, m, d, nsym, bits, float64 snr, float64 aux,
//             then the extension block float64 waterfill, float64 min_pl,
//             int32 active_count, int32 degenerate, int32 thp_ok,
//             then the 14 payload fields in field_list order:
//             h, S, v, vd, rr, rt, P, R, pl, b1, cod, xtx, Q, u.
// The extension block and the Q/u fields were appended when GP-CBD (strat 3)
// and the UCD waterfill/min_pl knobs landed; they must be consumed here or
// every case after the first misaligns.
#include <cstdio>
#include <string>
#include <cmath>
#include <vector>
#include <fstream>
#include <cstring>
#include "decomp.h"
#include "hp_ref.h"
#include "hpocl_runtime.h"

using namespace hpocl;
using dcplx = std::complex<double>;

namespace {

struct Reader {                               // bounds-checked, no UB on truncation
  std::vector<uint8_t> buf; size_t o = 0; bool bad = false;
  explicit Reader(const char* p) {
    std::ifstream f(p, std::ios::binary);
    buf.assign((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
  }
  const uint8_t* take(size_t n) {             // null + bad flag when exhausted
    if (o + n > buf.size()) { bad = true; return nullptr; }
    const uint8_t* p = buf.data() + o; o += n; return p;
  }
  int i() { int v = 0; if (const uint8_t* p = take(4)) memcpy(&v, p, 4); return v; }
  double f() { double v = 0; if (const uint8_t* p = take(8)) memcpy(&v, p, 8); return v; }
  void cx(std::vector<dcplx>& v) {
    if (const uint8_t* p = take(16 * v.size()))
      memcpy(v.data(), p, 16 * v.size());
  }
  void re(std::vector<double>& v) {
    if (const uint8_t* p = take(8 * v.size()))
      memcpy(v.data(), p, 8 * v.size());
  }
};
// abort the bench with an explicit message instead of corrupting allocations
int truncated(const char* what) {
  fprintf(stderr, "[decomp] vector file truncated while reading %s\n", what);
  return 2;
}

double err_c(const dcplx* a, const dcplx* b, size_t n) {
  double m = 0; for (size_t i = 0; i < n; ++i) m = fmax(m, std::abs(a[i] - b[i])); return m;
}
double err_r(const double* a, const double* b, size_t n) {
  double m = 0; for (size_t i = 0; i < n; ++i) m = fmax(m, fabs(a[i] - b[i])); return m;
}
double rel(double a, double b) { return fabs(a - b) / fmax(1.0, fabs(b)); }
// singular vectors are unique up to per-column phase
void align_phase(std::vector<dcplx>& mine, const std::vector<dcplx>& ref, int n) {
  for (int j = 0; j < n; ++j) {
    dcplx dot(0, 0);
    for (size_t r = 0; r < mine.size() / n; ++r)
      dot += std::conj(ref[r * n + j]) * mine[r * n + j];
    if (std::abs(dot) > 1e-12) {
      dcplx ph = std::conj(dot / std::abs(dot));
      for (size_t r = 0; r < mine.size() / n; ++r) mine[r * n + j] *= ph;
    }
  }
}
}  // namespace

int main(int argc, char** argv) {
  Reader b(argc > 1 ? argv[1] : "build/decomp_vectors.bin");
  Runtime rt;
  bool have_cl = rt.init(argc > 2 ? argv[2] : "cpu") &&
                 rt.load_program({"kernels/decomp.cl"});
  if (have_cl) {
    printf("[decomp_bench] OpenCL device: %s\n", rt.device_name());
  } else {
    // The `(!have_cl || e_cl < 3e-4)` gate below passes VACUOUSLY without a
    // device: e_cl stays 0 and every fp32 kernel in kernels/decomp.cl goes
    // unexecuted, while the per-case lines still print PASS. Say so once,
    // loudly, so a green run on a device-less host cannot be read as kernel
    // coverage. (arm64 macOS has no OpenCL device at all; see ref/decomp.h.)
    fprintf(stderr, "[decomp_bench] WARNING: no OpenCL device -- the fp32 kernel "
                    "path is SKIPPED and the e_cl gate is vacuous. Only the "
                    "production-oracle vs fp64-C++ comparison below is real.\n");
  }

  int ncase = b.i();
  if (b.bad) return truncated("case count");
  static const char* const kStrat[4] = {"svd", "gmd", "ucd", "cbd"};
  int fails = 0;
  for (int ci = 0; ci < ncase; ++ci) {
    int strat = b.i(), m = b.i(), d = b.i(), nsym = b.i(), bits = b.i();
    double snr = b.f(); b.f();                // legacy 5i2d header
    double wf = b.f(), min_pl = b.f();        // <2d3i extension block
    int act_ref = b.i(), deg = b.i(), thp_ok = b.i();
    // deg=1: tied / rank-deficient / ill-conditioned spectrum. The singular
    // VECTORS are then unique only up to an arbitrary rotation of the
    // degenerate subspace (LAPACK, the Jacobi SVD and the fp32 kernel each
    // pick different bases), so entrywise vd/P/Q/R and even the same-input
    // kernel-vs-ref comparison have no mathematical meaning. Only the singular
    // VALUES and the CBD active count are rotation-invariant and compared.
    const bool degenerate = deg != 0;
    if (b.bad) return truncated("case header/extension");
    if (strat < 0 || strat > 3) {
      fprintf(stderr, "[decomp] unknown strategy %d at case %d\n", strat, ci);
      return 2;
    }
    std::vector<dcplx> h(m * d), eV(d * d), eVd(d * d), eP(d * d), eR(d * d),
        b1(d * d), cod(d * nsym), xtx(d * nsym), eQ(m * d), eU(m * d);
    std::vector<double> eS(d), rr(d), rt_(d), pl(d);
    b.cx(h); b.re(eS); b.cx(eV); b.cx(eVd); b.re(rr); b.re(rt_);
    b.cx(eP); b.cx(eR); b.re(pl); b.cx(b1); b.cx(cod); b.cx(xtx);
    b.cx(eQ); b.cx(eU);                       // appended receive chain + left basis
    if (b.bad) return truncated("payload fields");

    // ---- fp64 C++ reference, phase-aligned to the oracle SVD basis
    std::vector<dcplx> U(m * d), V(d * d), Vd(d * d), P(d * d), R(d * d),
        Q(m * d);
    std::vector<double> S(d), rho(d), pl_m(d);
    svd_cplx(h.data(), m, d, U.data(), S.data(), V.data(), 16);
    align_phase(V, eV, d);
    if (strat == 3) align_phase(U, eU, d);    // CBD's left factor is phase-bound
    double e_sv = err_r(S.data(), eS.data(), d);

    double rho_eq = 0;
    int act_m = 0;
    if (strat == 0) Vd = V;
    else if (strat == 1) { gmd_factors(V.data(), d, S.data(), d, R.data(), P.data()); Vd = P; }
    else if (strat == 2) {
      ucd_design(V.data(), d, S.data(), d, snr, wf, min_pl, Vd.data(), &rho_eq,
                 R.data(), pl_m.data());
      P = Vd;                       // UCD tx factor is V_D itself
    } else {
      cbd_factors(U.data(), m, V.data(), d, S.data(), d, snr, Q.data(),
                  R.data(), P.data(), &act_m);
      Vd = P;
    }
    stream_snr(h.data(), m, d, Vd.data(), d, snr, rho.data());

    std::vector<double> rt_m(d);
    if (strat == 0) for (int i = 0; i < d; ++i) rt_m[i] = S[i] * S[i] * snr;
    else if (strat == 1) {
      double g = 0;
      for (int i = 0; i < d; ++i) g += log(fmax(S[i] * S[i] * snr, 1e-12));
      g = exp(g / d);
      rt_m.assign(d, g);
    } else if (strat == 2) rt_m.assign(d, rho_eq);
    else for (int i = 0; i < d; ++i) rt_m[i] = std::norm(R[(size_t)i * d + i]) * snr;

    double e_vd = err_c(Vd.data(), eVd.data(), d * d);
    double e_rho = err_r(rho.data(), rr.data(), d);
    double e_rt = err_r(rt_m.data(), rt_.data(), d);
    double e_P = strat ? err_c(P.data(), eP.data(), d * d) : 0.0;
    double e_R = strat ? err_c(R.data(), eR.data(), d * d) : 0.0;
    double e_pl = strat == 2 ? err_r(pl_m.data(), pl.data(), d) : 0.0;
    double e_Q = strat == 3 ? err_c(Q.data(), eQ.data(), m * d) : 0.0;
    double e_act = strat == 3 ? fabs((double)act_m - act_ref) : 0.0;

    // ---- CBD block detectability -------------------------------------------
    // Production _contiguous_cbd_blocks() declares a 2x2 pair when
    // |R[i,i+1]| > 1e-9 * max(||R||_F, 1). A genuinely paired block carries the
    // analytic coupling delta2 - delta1 = O(sigma), so fp32 rounding (~1e-7
    // absolute here) cannot push it under the threshold -- but that is exactly
    // the kind of claim worth measuring rather than assuming, and it is the
    // invariant the PL integration depends on (it must consume cbd_k's act
    // output rather than re-detecting blocks from an fp32 R, where the
    // analytically-zero off-block entries sit ~100x ABOVE the threshold).
    // cbd_margin = min over paired blocks of |R[2k,2k+1]| / threshold; 0 = n/a.
    double cbd_margin = 0;
    if (strat == 3 && act_m >= 2) {
      double fro = 0;
      for (int i = 0; i < d * d; ++i) fro += std::norm(R[i]);
      const double thr = 1e-9 * fmax(sqrt(fro), 1.0);
      cbd_margin = HUGE_VAL;
      for (int k = 0; k + 1 < act_m; k += 2)
        cbd_margin = fmin(cbd_margin, std::abs(R[(size_t)k * d + k + 1]) / thr);
      if (cbd_margin == HUGE_VAL) cbd_margin = 0;
    }

    // ---- THP end-to-end against the oracle (fp32 datapath, fp64 oracle);
    //      only when the oracle actually ran the feedback (thp_ok)
    double e_tx = 0;
    if (strat == 1 && thp_ok) {
      auto rev = [&](int i) { return d - 1 - i; };
      std::vector<dcplx> Bf(d * d), F(d * d), X(nsym * d), Y(nsym * d);
      for (int k = 0; k < d; ++k)
        for (int j = 0; j < d; ++j) Bf[k * d + j] = b1[rev(k) * d + rev(j)];
      for (int a = 0; a < d; ++a)
        for (int k = 0; k < d; ++k) F[a * d + k] = eP[a * d + rev(k)];
      for (int sc = 0; sc < nsym; ++sc)
        for (int k = 0; k < d; ++k) X[sc * d + k] = cod[rev(k) * nsym + sc];
      double Mq = pow(2.0, bits);
      double period = sqrt(Mq) * sqrt(6.0 / (Mq - 1.0));
      std::vector<cplx> Bff(Bf.begin(), Bf.end()), Ff(F.begin(), F.end()),
          Xf(X.begin(), X.end()), Yf(nsym * d);
      hpref::thp_precoder(Bff, Ff, Xf, nsym, d, d, (float)period, Yf);
      for (int sc = 0; sc < nsym; ++sc)
        for (int a = 0; a < d; ++a)
          e_tx = fmax(e_tx, std::abs(dcplx(Yf[sc * d + a]) - xtx[a * nsym + sc]));
    }

    // ---- fp32 OpenCL kernels, fed their own SVD output so the Jacobi
    //      column-sign freedom cancels from the comparison
    double e_cl = 0;
    // Kernel outputs that stay MEANINGFUL under a degenerate spectrum, and are
    // therefore gated even on deg=1 rows (see the gate below): the singular
    // values, the CBD active count, and diag(R)/diag(upper) -- the equalised
    // target diagonal, which is what the rate calculation consumes and which is
    // invariant to the pivot permutation within a block. Everything else the
    // kernel emits (V, P, Q, the strictly-upper part of R) is basis-dependent
    // and is reported as n/a on deg rows. Before this split the deg=1 rows
    // skipped e_cl entirely, which is how a tied-spectrum pivot flip worth
    // max|dR| = 7.9e-4 went ungated.
    double e_cl_sv = 0, e_cl_diag = 0, e_cl_act = 0, e_cl_spur = 0, e_cl_pl = 0;
    if (have_cl) {
      std::vector<dcplx> kU(m * d), kV(d * d), kVd(d * d), kR(d * d), kP(d * d);
      std::vector<double> kS(d), kRho(1), kPl(d);
      rt.jacobi_svd(h.data(), 1, m, d, 16, kU.data(), kS.data(), kV.data());
      std::vector<dcplx> kVa = kV;
      align_phase(kVa, V, d);
      for (int i = 0; i < d; ++i) e_cl_sv = fmax(e_cl_sv, rel(kS[i], S[i]));
      e_cl = fmax(e_cl, e_cl_sv);
      e_cl = fmax(e_cl, err_c(kVa.data(), V.data(), d * d));
      auto diag_err = [d](const dcplx* a, const dcplx* b) {
        double e = 0;
        for (int i = 0; i < d; ++i)
          e = fmax(e, std::abs(a[(size_t)i * d + i] - b[(size_t)i * d + i]));
        return e;
      };
      if (strat == 1) {
        rt.gmd_op(kV.data(), kS.data(), 1, d, d, kR.data(), kP.data());
        std::vector<dcplx> R2(d * d), P2(d * d);
        gmd_factors(kV.data(), d, kS.data(), d, R2.data(), P2.data());
        e_cl = fmax(e_cl, err_c(kR.data(), R2.data(), d * d));
        e_cl = fmax(e_cl, err_c(kP.data(), P2.data(), d * d));
        e_cl_diag = fmax(e_cl_diag, diag_err(kR.data(), R2.data()));
        if (m > d) {
          // TALL right basis, rows = m > d: the shape ref/pipeline.cpp:136 and
          // ref/rx.cpp:33 actually feed gmd_factors (the BD null basis is
          // [nul x d]). kU is m x d with orthonormal columns, so it is a legal
          // tall P0 and it drives every `rows` loop in gmd_dev -- previously
          // unexercised, since the call above always passed rows == d.
          std::vector<dcplx> kRt(d * d), kPt((size_t)m * d), Rt2(d * d),
              Pt2((size_t)m * d);
          rt.gmd_op(kU.data(), kS.data(), 1, m, d, kRt.data(), kPt.data());
          gmd_factors(kU.data(), m, kS.data(), d, Rt2.data(), Pt2.data());
          e_cl = fmax(e_cl, err_c(kRt.data(), Rt2.data(), d * d));
          e_cl = fmax(e_cl, err_c(kPt.data(), Pt2.data(), (size_t)m * d));
        }
      } else if (strat == 2) {
        std::vector<dcplx> Up(d * d);
        rt.ucd_op(kV.data(), kS.data(), 1, d, d, snr, wf, min_pl, kVd.data(),
                  kRho.data(), Up.data(), kPl.data());
        std::vector<dcplx> Vd2(d * d), Up2(d * d);
        double rho2 = 0; std::vector<double> pl2(d);
        ucd_design(kV.data(), d, kS.data(), d, snr, wf, min_pl, Vd2.data(), &rho2,
                   Up2.data(), pl2.data());
        e_cl = fmax(e_cl, err_c(kVd.data(), Vd2.data(), d * d));
        e_cl = fmax(e_cl, err_c(Up.data(), Up2.data(), d * d));
        e_cl = fmax(e_cl, rel(kRho[0], rho2));
        // power_loading is a per-stream SCALAR derived from the spectrum alone:
        // no basis enters it, so it is rotation-invariant and belongs with the
        // quantities that stay gated on degenerate rows. It is also the one
        // output that catches the fp32 water-level cancellation (see the
        // ucd_k waterfill comment): with active mis-selected, pl comes out
        // all-zero and rel(pl, pl_ref) = 1.0 even though diag(upper) still
        // agrees to 1e-7.
        for (int i = 0; i < d; ++i) {
          e_cl_pl = fmax(e_cl_pl, rel(kPl[i], pl2[i]));
          e_cl = fmax(e_cl, rel(kPl[i], pl2[i]));
        }
        e_cl_diag = fmax(e_cl_diag, diag_err(Up.data(), Up2.data()));
        if (m > d) {
          // tall Veff, rows = m > d (ref/rx.cpp:36 uses rows = nul > d)
          std::vector<dcplx> kVdT((size_t)m * d), Vd2T((size_t)m * d), Upt(d * d),
              Up2T(d * d);
          std::vector<double> kPlT(d), pl2T(d), kRhoT(1);
          double rho2T = 0;
          rt.ucd_op(kU.data(), kS.data(), 1, m, d, snr, wf, min_pl, kVdT.data(),
                    kRhoT.data(), Upt.data(), kPlT.data());
          ucd_design(kU.data(), m, kS.data(), d, snr, wf, min_pl, Vd2T.data(),
                     &rho2T, Up2T.data(), pl2T.data());
          e_cl = fmax(e_cl, err_c(kVdT.data(), Vd2T.data(), (size_t)m * d));
          e_cl = fmax(e_cl, err_c(Upt.data(), Up2T.data(), d * d));
          e_cl = fmax(e_cl, rel(kRhoT[0], rho2T));
          for (int i = 0; i < d; ++i) {
            e_cl_pl = fmax(e_cl_pl, rel(kPlT[i], pl2T[i]));
            e_cl = fmax(e_cl, rel(kPlT[i], pl2T[i]));
          }
        }
      } else if (strat == 3) {
        std::vector<dcplx> kQ(m * d), kQ2(m * d), kR2(d * d), kP2(d * d);
        int kAct = 0, kAct2 = 0;
        rt.cbd_op(kU.data(), kV.data(), kS.data(), 1, m, d, d, snr, 1.7,
                  kQ.data(), kR.data(), kP.data(), &kAct);
        cbd_factors(kU.data(), m, kV.data(), d, kS.data(), d, snr, kQ2.data(),
                    kR2.data(), kP2.data(), &kAct2);
        e_cl = fmax(e_cl, err_c(kQ.data(), kQ2.data(), m * d));
        e_cl = fmax(e_cl, err_c(kR.data(), kR2.data(), d * d));
        e_cl = fmax(e_cl, err_c(kP.data(), kP2.data(), d * d));
        e_cl_act = kAct == kAct2 ? 0.0 : 1.0;
        e_cl = fmax(e_cl, e_cl_act);
        e_cl_diag = fmax(e_cl_diag, diag_err(kR.data(), kR2.data()));
        // Off-block spur of the KERNEL's own fp32 R, against production's
        // _contiguous_cbd_blocks re-detection gate 1e-9*max(||R||_F,1). The
        // cbd_margin above is the paired-block margin of the fp64 ref's R and is
        // gated; this is the complementary measurement decomp.cl quotes: the
        // analytically-zero off-block entries of an fp32 R sit ABOVE the gate
        // (measured 12-32x), so the PL must take the block structure from `act`
        // rather than re-detect it. Reported, deliberately NOT gated -- a factor
        // > 1 is the expected property, not a failure.
        if (kAct >= 2) {
          double fro = 0;
          for (int i = 0; i < d * d; ++i) fro += std::norm(kR[i]);
          const double g32 = 1e-9 * fmax(sqrt(fro), 1.0);
          std::vector<char> blk((size_t)d * d, 0);
          for (int k = 0; k + 1 < kAct && k + 1 < d; k += 2) {
            blk[(size_t)k * d + k] = 1; blk[(size_t)k * d + k + 1] = 1;
            blk[(size_t)(k + 1) * d + k] = 1; blk[(size_t)(k + 1) * d + k + 1] = 1;
          }
          for (int i = 0; i < d; ++i)
            for (int j = 0; j < d; ++j)
              if (i != j && !blk[(size_t)i * d + j])
                e_cl_spur = fmax(e_cl_spur, std::abs(kR[(size_t)i * d + j]) / g32);
        }
      }
    }

    double rt_mag = 1.0;                            // relative rt bound
    for (int i = 0; i < d; ++i) rt_mag = fmax(rt_mag, fabs(rt_[i]));
    bool ok;
    if (degenerate) {
      // Rotation-invariant quantities only -- but that includes the KERNEL's
      // rotation-invariant outputs, which used to be skipped along with the
      // basis-dependent ones. Gates: singular values at the same 3e-4 the
      // non-degenerate path uses; diag(R)/diag(upper) at 1e-5, measured
      // <= 8.9e-8 on every degenerate vector (tied [3,3,1,1], rank-deficient
      // [3,2,0,0], [3,2,1e-8,1e-9], d=8 near-tie) so 1e-5 is ~110x margin while
      // still catching a pc/shape change, which would move the diagonal by
      // O(sbar); active count exact; and the UCD power loading at 1e-5 -- pl is
      // a per-stream scalar derived from the spectrum alone, so it is invariant
      // too, and it is the only one of the four that catches the fp32
      // water-level cancellation (measured worst diag error on the two
      // waterfill-band vectors: 5.0e-7 at snr=1e-4 and 5.0e-8 at snr=1e-2,
      // while rel(pl, pl_ref) is 1.0 pre-fix and 0.0 post-fix).
      ok = e_sv < 1e-9 && e_act < 0.5 &&
           (!have_cl || (e_cl_sv < 3e-4 && e_cl_diag < 1e-5 && e_cl_act < 0.5 &&
                         e_cl_pl < 1e-5));
    } else {
      ok = e_sv < 1e-9 && e_vd < 1e-9 && e_rho < 1e-8 && e_rt < 1e-8 * rt_mag &&
           e_P < 1e-9 && e_R < 1e-9 && e_pl < 1e-9 && e_Q < 1e-9 &&
           e_act < 0.5 && e_tx < 1e-5 && (cbd_margin == 0 || cbd_margin > 1e3) &&           // fp32 THP vs fp64 oracle
           // fp32 kernel vs fp64 ref. The figure quoted here used to read
           // "worst 4.0e-05 over all 43 vectors (incl. ... rank-deficient)",
           // which was not true of the degenerate vectors: those take the branch
           // above and never reached this gate, and measuring them anyway gave
           // max|dR| = 7.9e-4 on the tied [3,3,1,1] gmd vector. The 4.0e-05
           // figure applies to the NON-degenerate vectors only. 3e-4 is ~7.5x
           // headroom over that for a different device/compiler and 3x tighter
           // than the old 1e-3 gate; do not tighten below ~1e-4 without
           // re-measuring on the target fabric. Random-spectrum sampling (120
           // healthy spectra, d 2..8) puts the pre-fix worst case at 2.75e-3,
           // i.e. this gate was reachable-by-luck rather than comfortably clear;
           // see the gmd_dev sbar comment in kernels/decomp.cl.
           (!have_cl || e_cl < 3e-4);
    }
    fails += !ok;
    // A degenerate spectrum makes every basis-dependent column meaningless, so
    // print "n/a" instead of a big number next to PASS: in that case the gate
    // above compares only the rotation-invariant svd/act columns (see the deg=1
    // comment at the top of the loop). Reading "vd=1.3e+00 ... PASS" without
    // that is indistinguishable from a broken tolerance.
    auto col = [degenerate](double v) {
      char b[16];
      if (degenerate) snprintf(b, sizeof b, "n/a");
      else            snprintf(b, sizeof b, "%.1e", v);
      return std::string(b);
    };
    // cli = the kernel's rotation-INVARIANT error (S, diag(R), active count).
    // Unlike cl= it is gated and printed on degenerate rows too, so a deg row
    // that prints "cl=n/a cli=3.1e-08 PASS" has really been checked.
    // spur32 = the fp32 CBD off-block spur as a multiple of production's
    // re-detection gate; 0x means "not a CBD case / no pair selected".
    const double e_cl_inv = fmax(fmax(e_cl_sv, e_cl_diag), fmax(e_cl_act, e_cl_pl));
    char spur[24];
    snprintf(spur, sizeof spur, "%.0fx", e_cl_spur);
    printf("case%d %-3s m=%d d=%d%s: svd=%.1e vd=%s rho=%s rt=%s P=%s "
           "R=%s pl=%s Q=%s act=%d/%d tx=%s cl=%s cli=%.1e spur32=%s %s\n",
           ci, kStrat[strat], m, d, degenerate ? " [deg]" : "", e_sv,
           col(e_vd).c_str(), col(e_rho).c_str(), col(e_rt).c_str(),
           col(e_P).c_str(), col(e_R).c_str(), col(e_pl).c_str(),
           col(e_Q).c_str(), act_m, act_ref, col(e_tx).c_str(),
           col(e_cl).c_str(), e_cl_inv, spur, ok ? "PASS" : "FAIL");
  }
  if (b.o != b.buf.size()) {
    fprintf(stderr, "[decomp] %zu trailing bytes after case %d (contract drift?)\n",
            b.buf.size() - b.o, ncase);
    ++fails;
  }
  printf(fails ? "[decomp_bench] FAIL\n" : "[decomp_bench] ALL PASS\n");
  return fails ? 1 : 0;
}
