//-----------------------------------------------------------------------------
// operator group: per-symbol PL datapath (fp32 only -- the OpenCL devices in
// this stack report CL_DEVICE_DOUBLE_FP_CONFIG == 0). Complex = float2(re,im).
//
//   bits --qam_map_k--> s --tx_combine_k--> x --channel_apply_k--> y
//     --rx_combine_k--> z (=/ diag_b) --+--softdemod_llr_k--> llr --gmi_k--> GMI
//                                       `--modulo_fold_k--> zf
//                                            `--softdemod_folded_k--> llr (THP)
//
// Modem conventions are the production ones (mix_precode/classical:
// bicm_metrics.get_constellation / enumerate_thp_symbol_replicas,
// thp_precoding.qam_axis_order_and_spacing / centered_modulo_complex /
// thp_receive_equalized); tools/datapath_ref.py asserts this oracle against
// them for M in {4, 16, 64, 256}. Square QAM, M = 2^m with m even,
// L = 2^(m/2) levels per axis, spacing delta = sqrt(6/(M-1)) so E|s|^2 = 1.
// Per axis the m/2 label bits are packed MSB first and
//   label = idx ^ (idx >> 1),  amplitude = (2*idx - (L-1)) * delta/2,
// i.e. label -> amplitude needs the full Gray *decode* (XOR cascade, see
// gray_decode); a single XOR would be the encoder and is only its own inverse
// for L <= 4, which silently breaks 64/256-QAM labeling.
// LLR_b = ln P(b=1|y) - ln P(b=0|y) (bicm_metrics.bit_llr_from_distances with
// max_log=False), so a positive LLR favours bit 1 -- the sign gmi_k expects.
//-----------------------------------------------------------------------------
// Largest constellation (m <= 8). softdemod_llr_k caches all M squared
// distances, so this is 1 KB of private memory per work-item at M = 256; on a
// PL build that becomes 1 KB of on-chip local memory (BRAM/URAM) per
// work-item, which is why softdemod_folded_k stores no candidate array at all
// (its (2r+1)^2*M set would need 25 KB per work-item at M=256, r=2) and
// re-derives the distances instead.
#define MMAX 256

// softdemod_llr_k caches all M squared distances in private memory: 4*MMAX =
// 1 KB per work-item at M=256, which on a PL build is 1 KB of on-chip storage
// per work-item and quickly exhausts BRAM/URAM when several work-items are in
// flight. LLR_CACHE_MAX bounds that array; a constellation larger than the
// bound falls back to the two-sweep recompute that softdemod_folded_k already
// uses (radius 0 == no replica shift), which is O(1) memory at the cost of
// evaluating the M distances twice per bit instead of once.
//
// The fallback is not a second implementation that can drift: it calls the same
// replica_sweep() with radius 0, and datapath_bench gates the identity
// "softdemod_folded_k(radius=0) == softdemod_llr_k" at 1e-5 on every case, so a
// PL build with -DLLR_CACHE_MAX=4 exercises exactly that path for m >= 6.
// Default = MMAX keeps the cached path for every supported constellation, i.e.
// the default build is bit-identical to the pre-LLR_CACHE_MAX kernel.
#ifndef LLR_CACHE_MAX
#define LLR_CACHE_MAX MMAX
#endif
#if LLR_CACHE_MAX < 4 || LLR_CACHE_MAX > MMAX
#error "LLR_CACHE_MAX must be in [4, MMAX]: 4 is QPSK, MMAX is the largest supported constellation"
#endif

#include "cplx.clh"   // cmul / cmulc / cadd / cdiv / cmod_axis: one definition

static int gray_decode(int lab) {                    // Gray label -> level idx
  int v = lab;
  for (int s = lab >> 1; s != 0; s >>= 1) v ^= s;
  return v;
}
static float qam_delta(int M) { return sqrt(6.0f / (float)(M - 1)); }
static float pam_amp(int lab, int L, float delta) {  // per-axis amplitude
  return (float)(2 * gray_decode(lab) - (L - 1)) * (0.5f * delta);
}
// squared distance from r to candidate (iI,iQ) of the base constellation,
// shifted by the replica offset off (zero for the parallel branch)
static float cand_d2(float2 r, int iI, int iQ, int L, float delta, float2 off) {
  float dr = r.x - off.x - pam_amp(iI, L, delta);
  float di = r.y - off.y - pam_amp(iQ, L, delta);
  return dr * dr + di * di;
}
static int label_bit(int iI, int iQ, int b, int mh) {  // b-th MSB-first bit
  int v = (b < mh) ? iI : iQ;
  int pos = (b < mh) ? b : b - mh;
  return (v >> (mh - 1 - pos)) & 1;
}
// fold_axis is cmod_axis (kernels/cplx.clh); the alias keeps the call sites in
// this file readable next to the QAM helpers.
#define fold_axis(x, period) cmod_axis((x), (period))

//-----------------------------------------------------------------------------
// qam_map_k : bits -> constellation symbols
//   in  bits    : uchar[T*nsym][m], m bytes per symbol, MSB first; the first
//                 m/2 bytes label the in-phase axis, the last m/2 the
//                 quadrature axis
//   out s       : float2[T*nsym], same row-major order as bits
//   work-item   : one symbol (1-D), gid(0) = q < T*nsym
//-----------------------------------------------------------------------------
__kernel void qam_map_k(__global const uchar* restrict bits, int m, int nsymtot,
                        __global float2* restrict s) {
  int q = get_global_id(0);
  if (q >= nsymtot) return;
  // Capacity reject: the host wrapper refuses m > 8 before launching
  // (Runtime::qam_map, dp_dims_ok), so this is defence in depth for any other
  // caller. It writes a sentinel rather than returning silently, because
  // leaving s[] untouched reads exactly like a legitimate result.
  if ((1 << m) > MMAX) { s[q] = (float2)(NAN, NAN); return; }

  int mh = m >> 1, L = 1 << mh, iI = 0, iQ = 0;
  float delta = qam_delta(1 << m);
  __global const uchar* b = bits + (size_t)q * m;
  for (int j = 0; j < mh; ++j) iI = (iI << 1) | b[j];
  for (int j = 0; j < mh; ++j) iQ = (iQ << 1) | b[mh + j];
  s[q] = (float2)(pam_amp(iI, L, delta), pam_amp(iQ, L, delta));
}

//-----------------------------------------------------------------------------
// tx_combine_k : x = Frf[Nt x NRF] * Fbb[NRF x T] * s[T x nsym]
//   in  Frf : float2[Nt][NRF]  RF (phase-shifter) precoder
//       Fbb : float2[NRF][T]   baseband precoder
//       s   : float2[T][nsym]  mapped symbols (stream-major)
//   out x   : float2[Nt][nsym]
//   work-item : (antenna, symbol) (2-D), gid(0) = a < Nt, gid(1) = sym < nsym
//-----------------------------------------------------------------------------
__kernel void tx_combine_k(__global const float2* restrict Frf, int Nt, int NRF,
                           __global const float2* restrict Fbb, int T,
                           __global const float2* restrict s, int nsym,
                           __global float2* restrict x) {
  int a = get_global_id(0), sym = get_global_id(1);
  if (a >= Nt || sym >= nsym) return;

  float2 acc = (float2)(0.0f, 0.0f);
  for (int r = 0; r < NRF; ++r) {                    // digital stage per chain
    float2 bb = (float2)(0.0f, 0.0f);
    __global const float2* fbb = Fbb + (size_t)r * T;
    for (int t = 0; t < T; ++t)
      bb = cadd(bb, cmul(fbb[t], s[(size_t)t * nsym + sym]));
    acc = cadd(acc, cmul(Frf[(size_t)a * NRF + r], bb));   // RF stage
  }
  x[(size_t)a * nsym + sym] = acc;
}

//-----------------------------------------------------------------------------
// channel_apply_k : y = H[Nr x Nt] x + n  ->  y[Nr x nsym]
//   in  H : float2[Nr][Nt]  flat-fading channel
//       x : float2[Nt][nsym]  transmit block   n : float2[Nr][nsym] noise
//   out y : float2[Nr][nsym]
//   work-item : (receive antenna, symbol), gid(0) = r < Nr, gid(1) = sym
//-----------------------------------------------------------------------------
__kernel void channel_apply_k(__global const float2* restrict H, int Nr, int Nt,
                              __global const float2* restrict x, int nsym,
                              __global const float2* restrict n,
                              __global float2* restrict y) {
  int r = get_global_id(0), sym = get_global_id(1);
  if (r >= Nr || sym >= nsym) return;

  float2 acc = n[(size_t)r * nsym + sym];
  __global const float2* h = H + (size_t)r * Nt;
  for (int a = 0; a < Nt; ++a)
    acc = cadd(acc, cmul(h[a], x[(size_t)a * nsym + sym]));
  y[(size_t)r * nsym + sym] = acc;
}

//-----------------------------------------------------------------------------
// rx_combine_k : z = (W_u^H[Nr x d] y) / diag_b  ->  z[U*d x nsym]
//   in  W      : float2[U][Nr][d]  combiners stacked per user, user u starts
//                                  at offset u*Nr*d (one burst feeds all users)
//       y      : float2[Nr][nsym]  received block, shared by the U users
//       diag_b : float2[U*d] effective-channel diagonal, or NULL to skip the
//                normalization (thp_precoding.thp_receive_equalized divides by
//                diag_b[:, None] after w.conj().T @ y and before the modulo)
//   out z      : float2[U*d][nsym] stream k = u*d + j, i.e. transmit order
//   work-item  : (stream, symbol), gid(0) = k < U*d, gid(1) = sym < nsym
//-----------------------------------------------------------------------------
__kernel void rx_combine_k(__global const float2* restrict W, int U, int Nr,
                           int d, __global const float2* restrict y, int nsym,
                           __global const float2* restrict diag_b,
                           __global float2* restrict z) {
  int k = get_global_id(0), sym = get_global_id(1);
  if (k >= U * d || sym >= nsym) return;

  int u = k / d, j = k - u * d;
  __global const float2* w = W + (size_t)u * Nr * d;  // per-user block
  float2 acc = (float2)(0.0f, 0.0f);
  for (int r = 0; r < Nr; ++r)
    acc = cadd(acc, cmulc(w[(size_t)r * d + j], y[(size_t)r * nsym + sym]));
  z[(size_t)k * nsym + sym] = diag_b ? cdiv(acc, diag_b[k]) : acc;
}

//-----------------------------------------------------------------------------
// modulo_fold_k : THP receive modulo, centered and per axis
//   out[q] = in[q] - period * floor((in[q] + period/2) / period)
//   the floor form of thp_precoding.centered_modulo (and of mod_tau in
//   kernels/thp_precoder.cl), mapping into [-period/2, period/2). period is
//   L*delta (thp_precoding.qam_axis_order_and_spacing).
//   in/out : float2[nsymtot]   work-item : one symbol (1-D), gid(0) = q
//   fp32 caveat: the map is discontinuous at +-period/2, so an input within
//   ~1 ulp of an edge can land on the opposite edge -- a full period jump.
//-----------------------------------------------------------------------------
__kernel void modulo_fold_k(__global const float2* restrict in, int nsymtot,
                            float period, __global float2* restrict out) {
  int q = get_global_id(0);
  if (q >= nsymtot) return;

  float2 v = in[q];
  out[q] = (float2)(fold_axis(v.x, period), fold_axis(v.y, period));
}

// One sweep of the (2*radius+1)^2 * M replica set for bit b: mode 0 collects
// the two subset minima, mode 1 accumulates the stabilised sums (it needs the
// minima of mode 0, hence two sweeps). No candidate array is materialised.
static void replica_sweep(float2 r, int L, float delta, int radius, float period,
                          int b, int mh, int mode, float inv_n0, float* dmin,
                          float* S) {
  for (int lr = -radius; lr <= radius; ++lr) {        // real shift outer, as in
    for (int li = -radius; li <= radius; ++li) {      // enumerate_thp_symbol_..
      float2 off = (float2)(period * (float)lr, period * (float)li);
      for (int iI = 0; iI < L; ++iI)
        for (int iQ = 0; iQ < L; ++iQ) {
          int bit = label_bit(iI, iQ, b, mh);         // labels copied per shift
          float dd = cand_d2(r, iI, iQ, L, delta, off);
          if (mode) S[bit] += exp((dmin[bit] - dd) * inv_n0);
          else dmin[bit] = fmin(dmin[bit], dd);
        }
    }
  }
}

//-----------------------------------------------------------------------------
// softdemod_llr_k : exact log-MAP soft demodulation (parallel branch, M
//   candidates), fp32-stabilised
//   in  z   : float2[K*nsym]    equalized symbols, row-major [K][nsym]
//       n0  : post-equalization noise variance per complex symbol (scalar)
//       m   : bits per symbol   grp : bits handled by one work-item
//   out llr : float[K*nsym][m]
//   work-item : (symbol, bit group), gid(0) = q < K*nsym,
//               gid(1) = g < ceil(m/grp); the group owns bits
//               [g*grp, min(g*grp+grp, m))
//   The metric -|z-c|^2/n0 is kept as the squared distance d2 and divided by
//   n0 only after the per-subset minimum has been subtracted:
//     llr_b = (d2min_0 - d2min_1)/n0 + log S1 - log S0,
//     S_i   = sum over {c : bit_b(c)=i} of exp(-(d2(c)-d2min_i)/n0) >= 1
//   so every exp() argument is <= 0 (no overflow), S_i >= 1 (no log(0)) and
//   the dominant term is a difference of O(1) distances instead of a
//   difference of two huge -d2/n0 values (fp32 cancellation).
//-----------------------------------------------------------------------------
__kernel void softdemod_llr_k(__global const float2* restrict z, int nsymtot,
                              int m, int grp, float n0,
                              __global float* restrict llr) {
  int q = get_global_id(0), g = get_global_id(1);
  int M = 1 << m, mh = m >> 1, L = 1 << mh;
  if (q >= nsymtot || g * grp >= m) return;
  if (M > MMAX) {                       // capacity reject: sentinel, see qam_map_k
    for (int b = g * grp; b < min(g * grp + grp, m); ++b)
      llr[(size_t)q * m + b] = NAN;
    return;
  }

  float2 r = z[q];
  float delta = qam_delta(M), inv_n0 = 1.0f / n0;

  for (int b = g * grp; b < min(g * grp + grp, m); ++b) {
    float dmin[2] = {INFINITY, INFINITY}, S[2] = {0.0f, 0.0f};
    if (M <= LLR_CACHE_MAX) {
      // cached: M distance evaluations per bit-group, 4*LLR_CACHE_MAX bytes of
      // private memory per work-item
      float d2[LLR_CACHE_MAX];
      for (int iI = 0; iI < L; ++iI)
        for (int iQ = 0; iQ < L; ++iQ)
          d2[iI * L + iQ] = cand_d2(r, iI, iQ, L, delta,
                                    (float2)(0.0f, 0.0f));   // no replica shift
      for (int iI = 0; iI < L; ++iI)
        for (int iQ = 0; iQ < L; ++iQ) {
          int bit = label_bit(iI, iQ, b, mh);
          dmin[bit] = fmin(dmin[bit], d2[iI * L + iQ]);
        }
      for (int iI = 0; iI < L; ++iI)
        for (int iQ = 0; iQ < L; ++iQ) {
          int bit = label_bit(iI, iQ, b, mh);
          S[bit] += exp((dmin[bit] - d2[iI * L + iQ]) * inv_n0);
        }
    } else {
      // recompute: two sweeps over the same M candidates, no private array.
      // radius 0 and period 0 => replica_sweep degenerates to the base
      // constellation, i.e. exactly the cached path's candidate set.
      replica_sweep(r, L, delta, 0, 0.0f, b, mh, 0, inv_n0, dmin, S);
      replica_sweep(r, L, delta, 0, 0.0f, b, mh, 1, inv_n0, dmin, S);
    }
    llr[(size_t)q * m + b] =
        (dmin[0] - dmin[1]) * inv_n0 + log(S[1]) - log(S[0]);
  }
}

//-----------------------------------------------------------------------------
// softdemod_folded_k : THP receive-side soft demodulation over the periodic
//   replica set of a modulo receiver (bicm_metrics.enumerate_thp_symbol_
//   replicas): candidates c = point(iI,iQ) + period*(lr + i*li) with
//   lr,li in [-radius, radius] and the base labels copied onto every replica,
//   because the receive modulo destroyed the one-to-one map between y and the
//   transmitted point. Same stabilized LLR form as softdemod_llr_k.
//   in  z   : float2[K*nsym]  modulo-folded symbols (modulo_fold_k output)
//       n0, period, m, grp as in softdemod_llr_k; radius >= 0 selects the
//       replica ring count (radius 0 reproduces softdemod_llr_k exactly)
//   out llr : float[K*nsym][m]
//   work-item : (symbol, bit group), gid(0) = q, gid(1) = g
//   Cost: 2*(2*radius+1)^2*M distance evaluations per bit per symbol, traded
//   against O(1) private memory (see the MMAX comment).
//-----------------------------------------------------------------------------
__kernel void softdemod_folded_k(__global const float2* restrict z, int nsymtot,
                                 int m, int grp, int radius, float period,
                                 float n0, __global float* restrict llr) {
  int q = get_global_id(0), g = get_global_id(1);
  int M = 1 << m, mh = m >> 1, L = 1 << mh;
  if (q >= nsymtot || g * grp >= m) return;
  if (M > MMAX || radius < 0) {         // capacity/argument reject: sentinel
    for (int b = g * grp; b < min(g * grp + grp, m); ++b)
      llr[(size_t)q * m + b] = NAN;
    return;
  }

  float2 r = z[q];
  float delta = qam_delta(M), inv_n0 = 1.0f / n0;
  for (int b = g * grp; b < min(g * grp + grp, m); ++b) {
    float dmin[2] = {INFINITY, INFINITY}, S[2] = {0.0f, 0.0f};
    replica_sweep(r, L, delta, radius, period, b, mh, 0, inv_n0, dmin, S);
    replica_sweep(r, L, delta, radius, period, b, mh, 1, inv_n0, dmin, S);
    llr[(size_t)q * m + b] =
        (dmin[0] - dmin[1]) * inv_n0 + log(S[1]) - log(S[0]);
  }
}

//-----------------------------------------------------------------------------
// gmi_k : per-stream generalized mutual information
//   GMI_k = m - mean_over_symbols sum_b log2(1 + exp(-(2 c_b - 1) llr_b))
//   in  llr  : float[K][nsym][m]  from softdemod_llr_k or softdemod_folded_k
//       bits : uchar[K][nsym][m]  transmitted bits (qam_map_k input buffer)
//   out gmi  : float[K]  NOT clamped: see the clamp contract below
//
// CLAMP CONTRACT. production's bicm_metrics.estimate_bicm_gmi_*_from_received
// returns max(0.0, ideal_rate - loss) where ideal_rate is the AGGREGATE
// num_streams*bits_per_symbol, i.e. it clamps the SUM over streams at zero.
// This kernel deliberately does not clamp: it emits one GMI per stream, and a
// negative per-stream GMI is exactly the information a rate adapter needs (it is
// why kernels/mcs_table.clh::cqi_select treats gmi <= 0 as an outage instead of
// as a small positive rate). Clamping per stream would NOT reproduce production
// either, since production clamps after summing. Whoever forms the aggregate is
// responsible for the clamp -- tools/chain_ref.py does max(0.0, gmi.sum()) when
// it cross-checks against the production estimator.
// Measured, so this is not a hypothetical: with a matched demodulator the
// per-stream GMI stays positive down to -20 dB (m=8, 20000 symbols: +0.0098 at
// -20 dB, +0.1014 at -10 dB, +0.2443 at -6 dB), as E[GMI] = I >= 0 requires.
// The clamp therefore only binds on a mismatched demodulator or a short block.
//   work-item : one stream (1-D), gid(0) = k < K, loops over nsym symbols.
//   softplus is evaluated as log(1+e^t) = max(t,0) + log1p(e^{-|t|}) so the
//   exp() argument stays <= 0 for both very confident and very wrong bits.
//-----------------------------------------------------------------------------
__kernel void gmi_k(__global const float* restrict llr,
                    __global const uchar* restrict bits, int m, int nsym,
                    int K, __global float* restrict gmi) {
  int k = get_global_id(0);
  if (k >= K) return;

  float acc = 0.0f;
  for (int s = 0; s < nsym; ++s) {
    size_t o = ((size_t)k * nsym + s) * m;
    for (int b = 0; b < m; ++b) {
      float t = (1.0f - 2.0f * (float)bits[o + b]) * llr[o + b];
      acc += fmax(t, 0.0f) + log1p(exp(-fabs(t)));
    }
  }
  gmi[k] = (float)m - acc * (1.4426950408889634f / (float)nsym);   // 1/ln 2
}
