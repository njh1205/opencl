#include "decomp.h"
#include <cmath>
#include <algorithm>
#include <numeric>

//-----------------------------------------------------------------------------
// fp64 PS-role reference for the decomposition operators (see decomp.h for the
// per-operator contracts). Every threshold below is the production python value
// kept verbatim, so this file stays a bit-comparable oracle model. The fp32
// kernels in kernels/decomp.cl deliberately use *looser* constants at the same
// decision points; the rationale and the resulting divergence bands are
// documented next to each constant there and summarised in decomp.h.
//-----------------------------------------------------------------------------
namespace hpocl {

// private-array bounds of the fp32 mirror, identical to kernels/decomp.cl
enum { MAXD_ = 8, MAXM_ = 16 };

//---------------------------------------------------------------------------
// One-sided Jacobi SVD. Each sweep orthogonalises every column pair (p, q)
// with a 2x2 unitary J chosen from the pair Gram matrix
//   G = [[alpha, gamma], [conj(gamma), beta]],  gamma = a_p^H a_q .
// With gamma = mu e^{i th}, J = [[c, s e^{i th}], [-s e^{-i th}, c]] and
// t = s/c the small root of  mu t^2 - (alpha-beta) t - mu = 0.
//---------------------------------------------------------------------------
void svd_cplx(const dcplx* A, int m, int n, dcplx* U, double* S, dcplx* V,
              int sweeps) {
  std::vector<dcplx> W(A, A + (size_t)m * n);          // working columns
  std::vector<dcplx> Vv(n * n, dcplx(0, 0));
  for (int i = 0; i < n; ++i) Vv[(size_t)i * n + i] = 1;
  // Rotation-skip threshold, relative to sqrt(alpha*beta) and therefore
  // scale-invariant. The fp32 kernel uses 1e-10f; measured, the value is inert
  // at fp32 (1e-10f and 1e-13f give bit-identical S/U/V there), so the two
  // constants do not create a divergence band. See kernels/decomp.cl.
  const double tol = 1e-13;

  // Norm-invariant early exit, mirroring kernels/decomp.cl. `off` is a sum of
  // SQUARED Gram off-diagonals and scales as ||A||_F^4, so the absolute test
  // `off <= tol*tol` (=1e-26 here) was a test on the norm of A: measured, this
  // fp64 reference stops after 1-2 sweeps for ||A||_F <~ 1e-6 (3 sweeps at
  // 1e-5, 2 at 1e-6, 1 at 1e-7). sqrt(off) <= tol*||A||_F^2 is dimensionless.
  // ||W||_F == ||A||_F is invariant under the column rotations, so gfro2 is
  // accumulated once. Keeping the two sides on the same test is what makes
  // e_cl in decomp_bench a comparison of precision rather than of sweep count.
  double gfro2 = 0.0;
  for (int j = 0; j < n; ++j)
    for (int r = 0; r < m; ++r) gfro2 += std::norm(W[(size_t)r * n + j]);

  for (int sw = 0; sw < sweeps; ++sw) {
    double off = 0.0;
    for (int p = 0; p + 1 < n; ++p)
      for (int q = p + 1; q < n; ++q) {
        double alpha = 0, beta = 0;
        dcplx gamma(0, 0);
        for (int r = 0; r < m; ++r) {
          alpha += std::norm(W[(size_t)r * n + p]);
          beta  += std::norm(W[(size_t)r * n + q]);
          gamma += std::conj(W[(size_t)r * n + p]) * W[(size_t)r * n + q];
        }
        off += std::norm(gamma);
        double mu = std::abs(gamma);
        if (mu <= tol * std::sqrt(alpha * beta + 1e-300)) continue;

        double tau = (alpha - beta) * 0.5;
        double t = std::fabs(tau) <= 1e-300
                       ? 1.0
                       : -std::copysign(1.0, tau) * mu /
                             (std::fabs(tau) + std::sqrt(tau * tau + mu * mu));
        double c = 1.0 / std::sqrt(1.0 + t * t);
        dcplx jm = (t * c) * std::conj(gamma / mu);    // mixes a_q into a_p
        dcplx jp = (t * c) * (gamma / mu);             // mixes a_p into a_q
        for (int r = 0; r < m; ++r) {
          dcplx ap = W[(size_t)r * n + p], aq = W[(size_t)r * n + q];
          W[(size_t)r * n + p] = c * ap - jm * aq;
          W[(size_t)r * n + q] = jp * ap + c * aq;
        }
        for (int r = 0; r < n; ++r) {
          dcplx vp = Vv[(size_t)r * n + p], vq = Vv[(size_t)r * n + q];
          Vv[(size_t)r * n + p] = c * vp - jm * vq;
          Vv[(size_t)r * n + q] = jp * vp + c * vq;
        }
      }
    if (std::sqrt(off) <= tol * gfro2) break;   // scale-relative; see above
  }

  for (int j = 0; j < n; ++j) {
    double nrm = 0;
    for (int r = 0; r < m; ++r) nrm += std::norm(W[(size_t)r * n + j]);
    S[j] = std::sqrt(nrm);
  }
  std::vector<int> ord(n);
  std::iota(ord.begin(), ord.end(), 0);
  std::sort(ord.begin(), ord.end(), [&](int a, int b) { return S[a] > S[b]; });
  std::vector<double> Ss(n);                       // permuted copy (in-place
  for (int j = 0; j < n; ++j) Ss[j] = S[ord[j]];   // would clobber ord sources)
  for (int j = 0; j < n; ++j) {
    double sj = Ss[j];
    S[j] = sj;
    // Zero-column test on the singular value: 1e-12, the SAME cut the fp32
    // kernel uses (kernels/decomp.cl), so U is defined identically on both sides
    // and e_cl in decomp_bench measures precision rather than a threshold
    // difference. Dividing a noise-level column (|a| ~ 1e-7 * sigma_max) by its
    // own noise-level norm emits a UNIT-NORM column of pure garbage into U, and
    // U is cbd_k's left basis, where every column is subsequently scaled by
    // 1/sbar -- so the garbage is amplified, not damped. Cutting at 1e-12 still
    // keeps every stream above -240 dB.
    //   The kernel's behaviour was the defensible one and this file moved to it,
    // not the other way round. Effect on the fp64 oracle model: none for the
    // current vectors -- the fp64 Jacobi residual of a rank-deficient input is
    // ~1e-16*sigma_max (measured: sigma_max=1, rank 2 of 6x4 -> tail 1.4e-8 at
    // fp32 and ~1e-16 at fp64), and the smallest non-zero singular value any
    // decomp_ref.py vector specifies is 1e-9, so no vector has sigma in
    // (1e-14, 1e-12] and both cuts zero exactly the same columns. A spectrum
    // that DID sit in that band would now be zeroed here too, which is a
    // deliberate 2-decade widening of the fp64 cut, not an oversight.
    //   NOTE stream_snr()'s own `nrm > 1e-14` below is a different quantity (an
    // MGS diagonal floor, porting qr_with_positive_diagonal) and has no kernel
    // counterpart, so it is left at production's value.
    if (U)
      for (int r = 0; r < m; ++r)
        U[(size_t)r * n + j] = sj > 1e-12 ? W[(size_t)r * n + ord[j]] / sj
                                          : dcplx(0, 0);
    if (V)
      for (int r = 0; r < n; ++r) V[(size_t)r * n + j] = Vv[(size_t)r * n + ord[j]];
  }
}

//---------------------------------------------------------------------------
// Hager/Jiang gmd.m, exact port of _apply_original_gmd_factors with
// left basis = I (the left factor is never used downstream).
//---------------------------------------------------------------------------
void gmd_factors(const dcplx* P0, int rows, const double* sigma, int d,
                 dcplx* R, dcplx* P, const dcplx* Q0, int qrows, dcplx* Q) {
  std::vector<dcplx> p(P0, P0 + (size_t)rows * d);
  std::vector<dcplx> q(Q0 ? Q0 : P0, (Q0 ? Q0 : P0) + (size_t)(Q0 ? qrows : 0) * d);
  std::vector<double> Rr(d * d, 0.0);
  std::vector<double> df(d + 1, 0.0);                  // 1-based singulars
  for (int i = 0; i < d; ++i) df[i + 1] = sigma[i];
  // production "positive count" floor: sigma >= 1e-12 counts as a live stream.
  // The fp32 kernel uses 1e-9f (its Jacobi residual sits at ~1e-7*sigma_max,
  // so 1e-12 would promote numerical noise to a live stream and change pc,
  // i.e. the whole pivot structure). Divergence band: a spectrum with
  // sigma in (1e-12, 1e-9) is decided differently by ref and kernel ON PURPOSE;
  // such a stream carries <= -180 dB of the power and is dead for the PL.
  const double tol = 1e-12;

  int pc = 0;                                          // positive count
  for (int i = d; i >= 1; --i)
    if (df[i] >= tol) { pc = i; break; }

  if (pc == 1) {
    Rr[0] = df[1];
  } else if (pc >= 2) {
    std::vector<double> z(pc, 0.0);
    std::vector<int> perm(pc + 1), invperm(pc + 1);
    std::iota(perm.begin(), perm.end(), 0);
    std::iota(invperm.begin(), invperm.end(), 0);
    int large = 2, small = pc;
    double sbar = 1.0;                                 // equal target diagonal
    for (int i = 1; i <= pc; ++i) sbar *= df[i];
    sbar = std::pow(sbar, 1.0 / pc);

    for (int k = 1; k < pc; ++k) {
      // pair the running element with one from the opposite side of sbar
      bool flag = false;
      int pivot;
      if (df[k] >= sbar) {
        pivot = perm[small--];
        flag = df[pivot] >= sbar;
      } else {
        pivot = perm[large++];
        flag = df[pivot] <= sbar;
      }
      int nx = k + 1;
      if (pivot != nx) {
        std::swap(df[nx], df[pivot]);
        int slot = invperm[nx];
        perm[slot] = pivot;
        invperm[pivot] = slot;
        for (int r = 0; r < rows; ++r)
          std::swap(p[(size_t)r * d + nx - 1], p[(size_t)r * d + pivot - 1]);
        for (int r = 0; r < qrows; ++r)
          std::swap(q[(size_t)r * d + nx - 1], q[(size_t)r * d + pivot - 1]);
      }
      double d1 = df[k], d2 = df[nx], total = d1 + d2;
      double c, s;
      if (flag) {
        c = 1.0; s = 0.0;
      } else {
        double den = d1 - d2;
        if (std::fabs(den) <= tol) den = std::copysign(tol, den ? den : 1.0);
        double frac = (d1 - sbar) / den;
        s = std::sqrt(std::max(frac * (d1 + sbar) / total, 0.0));
        c = std::sqrt(std::max((1.0 - frac) * (d2 + sbar) / total, 0.0));
      }
      df[nx] = d1 * d2 / sbar;
      z[k] = s * c * (d2 - d1) * total / sbar;
      Rr[(size_t)(k - 1) * d + (k - 1)] = sbar;
      for (int i = 1; i < k; ++i) {
        Rr[(size_t)(i - 1) * d + (k - 1)] = z[i] * c;
        z[i] *= -s;
      }
      // right Givens g1 on the column pair (k, nx)
      for (int r = 0; r < rows; ++r) {
        dcplx pa = p[(size_t)r * d + k - 1], pb = p[(size_t)r * d + nx - 1];
        p[(size_t)r * d + k - 1] = c * pa + s * pb;
        p[(size_t)r * d + nx - 1] = -s * pa + c * pb;
      }
      // left factor: q[:, pair] @ g2 with g2 = [[c d1, -s d2], [s d2, c d1]]/sbar
      for (int r = 0; r < qrows; ++r) {
        dcplx qa = q[(size_t)r * d + k - 1], qb = q[(size_t)r * d + nx - 1];
        q[(size_t)r * d + k - 1] = (c * d1 * qa + s * d2 * qb) / sbar;
        q[(size_t)r * d + nx - 1] = (c * d1 * qb - s * d2 * qa) / sbar;
      }
    }
    Rr[(size_t)(pc - 1) * d + (pc - 1)] = sbar;
    for (int i = 1; i < pc; ++i) Rr[(size_t)(i - 1) * d + (pc - 1)] = z[i];
  }

  if (P) for (int i = 0; i < rows * d; ++i) P[i] = p[i];
  if (Q) for (int i = 0; i < qrows * d; ++i) Q[i] = q[i];
  if (R) for (int i = 0; i < d * d; ++i) R[i] = dcplx(Rr[i], 0.0);
}

//---------------------------------------------------------------------------
// Strict UCD: waterfilling (optional) -> MMSE augmentation -> GMD equalise.
//---------------------------------------------------------------------------
void ucd_design(const dcplx* Veff, int rows, const double* sigma, int d,
                double snr, bool waterfill, double min_pl, dcplx* Vd,
                double* rho_equal, dcplx* upper, double* power_loading,
                const dcplx* U, int urows, dcplx* Q) {
  const double alpha = 1.0 / std::max(snr, 1e-12);

  // production _build_ucd_power_loading, clamp order preserved exactly:
  //   no waterfill -> pl = max(1, min_pl)
  //   waterfill    -> pl = max( max(level - alpha/sp, 0), min_pl ) for the
  //                   active set and max(0, min_pl) for the rest.
  // The inner >= 0 clamp comes FIRST. Doing min_pl first (as an earlier version
  // of this port did) lets a negative min_pl push pl below zero, and pl feeds a
  // sqrt two lines down -> NaN. Production is NaN-free for min_pl < 0 for
  // exactly that reason.
  std::vector<double> pl(d, std::max(1.0, min_pl));
  if (waterfill) {
    std::vector<double> sp(d);
    for (int i = 0; i < d; ++i) sp[i] = std::max(sigma[i] * sigma[i], 1e-12);
    double level = (double)d;
    int active = d;
    for (int cnt = d; cnt >= 1; --cnt) {
      double inv = 0;
      for (int i = 0; i < cnt; ++i) inv += alpha / sp[i];
      level = (d + inv) / cnt;
      if (level > alpha / sp[cnt - 1]) { active = cnt; break; }
    }
    for (int i = 0; i < d; ++i)
      pl[i] = i < active
                  ? std::max(std::max(level - alpha / sp[i], 0.0), min_pl)
                  : std::max(0.0, min_pl);
  }
  // belt and braces: sqrt(pl) below must never see a negative loading, whatever
  // min_pl the caller passes. (Not reachable through the branches above -- the
  // water-level loop guarantees level > alpha/sp[active-1] -- but the sqrt is
  // the one place a sign error turns into a silent NaN.)
  for (int i = 0; i < d; ++i) pl[i] = std::max(pl[i], 0.0);

  std::vector<double> saug(d);
  for (int i = 0; i < d; ++i) {
    double loaded = pl[i] * sigma[i] * sigma[i];
    saug[i] = std::sqrt(std::max(loaded + alpha, 1e-12));
  }
  double t = 0;
  for (int i = 0; i < d; ++i) t += std::log(std::max(saug[i], 1e-12));
  t = std::exp(t / d);

  // scaled bases of the production construction: right = Veff diag(sqrt(pl)),
  // left = U diag(sigma_loaded / saug); one GMD call yields p_ucd, q_ucd and
  // the equal-diagonal augmented upper factor r_aug_ucd.
  std::vector<dcplx> rb((size_t)rows * d), up(d * d), lb(U ? (size_t)urows * d : 0);
  for (int r = 0; r < rows; ++r)
    for (int c = 0; c < d; ++c)
      rb[(size_t)r * d + c] = Veff[(size_t)r * d + c] * std::sqrt(pl[c]);
  if (U)
    for (int r = 0; r < urows; ++r)
      for (int c = 0; c < d; ++c)
        lb[(size_t)r * d + c] =
            U[(size_t)r * d + c] * (std::sqrt(pl[c]) * sigma[c] / saug[c]);
  gmd_factors(rb.data(), rows, saug.data(), d, up.data(), Vd,
              U ? lb.data() : nullptr, urows, Q);
  *rho_equal = std::max(t * t / alpha - 1.0, 0.0);
  if (upper) for (int i = 0; i < d * d; ++i) upper[i] = up[i];
  if (power_loading) for (int i = 0; i < d; ++i) power_loading[i] = pl[i];
}

//---------------------------------------------------------------------------
// GP-CBD -- exact port of full_digital_mu/digital_precoder.py
//   _select_cbd_active_count / _apply_gp_cbd_factors /
//   design_cbd_transceiver_from_svd
//---------------------------------------------------------------------------
int cbd_active_count(const double* sigma, int d, double snr,
                     double pair_threshold) {
  if (d < 2) return d;                       // python: size < 2 -> size
  // max(sigma^2, 1e-12) is applied BEFORE the snr scaling, and snr is clamped
  // at 0 -- a rank-deficient stream therefore contributes 1e-12*snr, which can
  // still clear pair_threshold at very high snr (production behaviour).
  std::vector<double> ss(d);
  const double snrp = std::max(snr, 0.0);
  for (int i = 0; i < d; ++i)
    ss[i] = std::max(sigma[i] * sigma[i], 1e-12) * snrp;
  for (int count = d; count >= 1; --count)
    if (ss[0] * ss[count - 1] >= pair_threshold) return count;
  return 1;
}

namespace {
// strongest/weakest interleave used by _apply_gp_cbd_factors. Pairs
// (i, count-1-i) into block i; an odd count leaves the middle stream
// (count-1)/2 unpaired as a 1x1 block. Indices >= count keep their order and
// are appended (the "extra" slot of the odd branch is python's
// `row2.append(count)`, which only happens when count < d).
std::vector<int> cbd_permutation(int count, int d) {
  std::vector<int> row1, row2, perm;
  if (count % 2 == 0) {
    for (int i = 0; i < count / 2; ++i) row1.push_back(i);
    for (int i = count - 1; i >= count / 2; --i) row2.push_back(i);
    for (size_t k = 0; k < row1.size() && k < row2.size(); ++k) {
      perm.push_back(row1[k]);
      perm.push_back(row2[k]);
    }
    for (int i = count; i < d; ++i) perm.push_back(i);
  } else {
    const int hmid = (count + 1) / 2;
    for (int i = 0; i < hmid; ++i) row1.push_back(i);
    for (int i = count - 1; i >= hmid; --i) row2.push_back(i);
    const int extra = count;
    bool used = extra >= d;
    for (size_t k = 0; k < row1.size() && !used; ++k) used = row1[k] == extra;
    for (size_t k = 0; k < row2.size() && !used; ++k) used = row2[k] == extra;
    if (!used) row2.push_back(extra);
    for (size_t k = 0; k < row1.size() && k < row2.size(); ++k) {
      perm.push_back(row1[k]);
      perm.push_back(row2[k]);
    }
    for (int i = 0; i < d; ++i) {
      bool in = false;
      for (size_t k = 0; k < perm.size(); ++k) in = in || perm[k] == i;
      if (!in) perm.push_back(i);
    }
  }
  perm.resize(d);                            // python: permutation[:d]
  return perm;
}
}  // namespace

void cbd_factors(const dcplx* U, int urows, const dcplx* V, int vrows,
                 const double* sigma, int d, double snr, dcplx* Q, dcplx* R,
                 dcplx* P, int* active_count, double pair_threshold) {
  const int selected = cbd_active_count(sigma, d, snr, pair_threshold);
  if (active_count) *active_count = selected;
  const int count = std::min(std::max(selected, 1), d);   // np.clip(., 1, d)
  const std::vector<int> perm = cbd_permutation(count, d);

  std::vector<dcplx> q((size_t)urows * d), p((size_t)vrows * d);
  for (int r = 0; r < urows; ++r)
    for (int c = 0; c < d; ++c) q[(size_t)r * d + c] = U[(size_t)r * d + perm[c]];
  for (int r = 0; r < vrows; ++r)
    for (int c = 0; c < d; ++c) p[(size_t)r * d + c] = V[(size_t)r * d + perm[c]];
  std::vector<double> dd(d);
  for (int c = 0; c < d; ++c) dd[c] = sigma[perm[c]];

  const double tol = 1e-12;                  // production default, verbatim
  for (int idx = 0; idx < std::max(count - 1, 0); idx += 2) {
    const double d1 = dd[idx], d2 = dd[idx + 1];
    const double sbar = std::sqrt(std::max(d1 * d2, tol));   // pair geo-mean
    const double total = d1 + d2;
    double c, s;
    if (std::fabs(d1 - d2) <= tol || total <= tol) {
      c = 1.0; s = 0.0;                      // degenerate pair -> identity
    } else {
      // frac = (d1 - sbar)/(d1 - d2); equivalently s^2 = d1/total and
      // c^2 = d2/total (sbar^2 = d1*d2 cancels the (d1-d2) factor). The
      // production form is kept so this file stays a line-by-line oracle port.
      const double frac = (d1 - sbar) / (d1 - d2);
      s = std::sqrt(std::max(frac * (d1 + sbar) / total, 0.0));
      c = std::sqrt(std::max((1.0 - frac) * (d2 + sbar) / total, 0.0));
    }
    dd[idx + 1] = d1 * d2 / std::max(sbar, tol);   // running update; dead
    // state -- R is recomputed from the factors below, exactly like production.
    const double sb = std::max(sbar, tol);
    // right rotation: p[:, pair] @ g1, g1 = [[c, -s], [s, c]]
    for (int r = 0; r < vrows; ++r) {
      dcplx pa = p[(size_t)r * d + idx], pb = p[(size_t)r * d + idx + 1];
      p[(size_t)r * d + idx]     = c * pa + s * pb;
      p[(size_t)r * d + idx + 1] = -s * pa + c * pb;
    }
    // left rotation: q[:, pair] @ g2, g2 = [[c d1, -s d2], [s d2, c d1]]/sbar
    for (int r = 0; r < urows; ++r) {
      dcplx qa = q[(size_t)r * d + idx], qb = q[(size_t)r * d + idx + 1];
      q[(size_t)r * d + idx]     = (c * d1 * qa + s * d2 * qb) / sb;
      q[(size_t)r * d + idx + 1] = (c * d1 * qb - s * d2 * qa) / sb;
    }
  }

  // R = Q^H (U diag(sigma) V^H) P, recomputed from the ORIGINAL bases and the
  // ORIGINAL (unpermuted) spectrum -- production does not trust the
  // incremental dd[] update, and neither do we.
  std::vector<dcplx> red((size_t)urows * vrows, dcplx(0, 0));
  for (int r = 0; r < urows; ++r)
    for (int cc = 0; cc < vrows; ++cc) {
      dcplx acc(0, 0);
      for (int k = 0; k < d; ++k)
        acc += U[(size_t)r * d + k] * sigma[k] * std::conj(V[(size_t)cc * d + k]);
      red[(size_t)r * vrows + cc] = acc;
    }
  for (int i = 0; i < d; ++i)
    for (int j = 0; j < d; ++j) {
      dcplx acc(0, 0);
      for (int r = 0; r < urows; ++r)
        for (int cc = 0; cc < vrows; ++cc)
          acc += std::conj(q[(size_t)r * d + i]) * red[(size_t)r * vrows + cc] *
                 p[(size_t)cc * d + j];
      R[(size_t)i * d + j] = acc;
    }

  if (Q) for (size_t i = 0; i < (size_t)urows * d; ++i) Q[i] = q[i];
  if (P) for (size_t i = 0; i < (size_t)vrows * d; ++i) P[i] = p[i];
}

//---------------------------------------------------------------------------
// |diag(R)| of the QR of H V_D is phase-invariant, so plain MGS reproduces
// qr_with_positive_diagonal for the SNR magnitudes.
//---------------------------------------------------------------------------
void stream_snr(const dcplx* H, int m, int n, const dcplx* Vd, int d,
                double snr, double* rho) {
  std::vector<dcplx> M(m * d), Qq(m * d, dcplx(0, 0));
  for (int r = 0; r < m; ++r)
    for (int c = 0; c < d; ++c) {
      dcplx acc(0, 0);
      for (int k = 0; k < n; ++k) acc += H[(size_t)r * n + k] * Vd[(size_t)k * d + c];
      M[(size_t)r * d + c] = acc;
    }
  for (int j = 0; j < d; ++j) {
    std::vector<dcplx> a(m);
    for (int r = 0; r < m; ++r) a[r] = M[(size_t)r * d + j];
    for (int i = 0; i < j; ++i) {
      dcplx dot(0, 0);
      for (int r = 0; r < m; ++r) dot += std::conj(Qq[(size_t)r * d + i]) * a[r];
      for (int r = 0; r < m; ++r) a[r] -= dot * Qq[(size_t)r * d + i];
    }
    double nrm = 0;
    for (int r = 0; r < m; ++r) nrm += std::norm(a[r]);
    rho[j] = nrm * snr;
    nrm = std::sqrt(nrm);
    for (int r = 0; r < m; ++r) Qq[(size_t)r * d + j] = nrm > 1e-14 ? a[r] / nrm : dcplx(0, 0);
  }
}

// NOTE: an `hpocl::f32::cbd_factors` device-free mirror of kernels/decomp.cl
// was announced here but never existed in-tree (no `namespace f32` anywhere,
// and build/ref/decomp.o exports only cbd_factors / cbd_active_count). The
// announcement is removed rather than left promising a cross-check that no test
// performs. kernels/decomp.cl::cbd_k is validated directly against cbd_factors
// below by decomp_bench's e_cl term whenever an OpenCL device is present.
// One property of that kernel is worth recording here because the ref does it
// differently: cbd_k forms R as (Q^H U) diag(sigma) (V^H P) instead of
// Q^H (U diag(sigma) V^H) P. Algebraically the same product -- measured
// max|R_prod - R_reassoc| = 4.8e-16 on ||R||_F = 3.6..4.5 -- but it needs only
// MAXD*MAXD scratch instead of a urows x vrows private array on the PL.
}  // namespace hpocl
