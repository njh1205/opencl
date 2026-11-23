// datapath.cl: per-symbol TX/RX datapath (fp32; complex = float2(re,im)).
//   bits ->qam_map_k-> s ->tx_combine_k-> x ->channel_apply_k-> y ->rx_combine_k-> z
//   ->softdemod_llr_k (or modulo_fold_k + softdemod_folded_k)-> llr ->gmi_k-> GMI
#define MMAX 256

#ifndef LLR_CACHE_MAX
#define LLR_CACHE_MAX MMAX
#endif
#if LLR_CACHE_MAX < 4 || LLR_CACHE_MAX > MMAX
#error "LLR_CACHE_MAX must be in [4, MMAX]: 4 is QPSK, MMAX is the largest supported constellation"
#endif

#include "cplx.clh"

// gray_decode: Gray label -> level index (XOR cascade)
static int gray_decode(int lab) {
  int v = lab;
  for (int s = lab >> 1; s != 0; s >>= 1) v ^= s;
  return v;
}
// qam_delta: axis spacing delta = sqrt(6/(M-1)) for unit-energy square QAM
static float qam_delta(int M) { return sqrt(6.0f / (float)(M - 1)); }
// pam_amp: per-axis amplitude of a Gray-decoded level index
static float pam_amp(int lab, int L, float delta) {
  return (float)(2 * gray_decode(lab) - (L - 1)) * (0.5f * delta);
}
// cand_d2: squared distance |r - (candidate point + replica offset)|^2
static float cand_d2(float2 r, int iI, int iQ, int L, float delta, float2 off) {
  float dr = r.x - off.x - pam_amp(iI, L, delta);
  float di = r.y - off.y - pam_amp(iQ, L, delta);
  return dr * dr + di * di;
}
// label_bit: b-th MSB-first bit of a symbol (in-phase half, then quadrature)
static int label_bit(int iI, int iQ, int b, int mh) {
  int v = (b < mh) ? iI : iQ;
  int pos = (b < mh) ? b : b - mh;
  return (v >> (mh - 1 - pos)) & 1;
}

// qam_map_k: bits[nsymtot][m] (MSB-first, I-axis first) -> QAM symbols s[nsymtot]
__kernel void qam_map_k(__global const uchar* restrict bits, int m, int nsymtot,
                        __global float2* restrict s) {
  int q = get_global_id(0);
  if (q >= nsymtot) return;
  if ((1 << m) > MMAX) { s[q] = (float2)(NAN, NAN); return; }

  int mh = m >> 1, L = 1 << mh, iI = 0, iQ = 0;
  float delta = qam_delta(1 << m);
  __global const uchar* b = bits + (size_t)q * m;
  for (int j = 0; j < mh; ++j) iI = (iI << 1) | b[j];
  for (int j = 0; j < mh; ++j) iQ = (iQ << 1) | b[mh + j];
  s[q] = (float2)(pam_amp(iI, L, delta), pam_amp(iQ, L, delta));
}

// tx_combine_k: x = Frf[Nt x NRF] * Fbb[NRF x T] * s[T x nsym], digital stage shared via __local
//   work-item: (antenna, symbol), gid(0) = a < Nt, gid(1) = sym < nsym
#ifndef TXC_LOCAL_MAX
#define TXC_LOCAL_MAX 1024
#endif

__kernel void tx_combine_k(__global const float2* restrict Frf, int Nt, int NRF,
                           __global const float2* restrict Fbb, int T,
                           __global const float2* restrict s, int nsym,
                           __global float2* restrict x) {
  int a = get_global_id(0), sym = get_global_id(1);
  const int LM = (int)get_local_size(0);
  const int LS = (int)get_local_size(1);
  const int tile = NRF * LS;

  if (LM * LS < 2 || tile > TXC_LOCAL_MAX) {
    if (a >= Nt || sym >= nsym) return;
    float2 acc0 = (float2)(0.0f, 0.0f);
    for (int r = 0; r < NRF; ++r) {
      float2 bb = (float2)(0.0f, 0.0f);
      __global const float2* fbb = Fbb + (size_t)r * T;
      for (int t = 0; t < T; ++t)
        bb = cadd(bb, cmul(fbb[t], s[(size_t)t * nsym + sym]));
      acc0 = cadd(acc0, cmul(Frf[(size_t)a * NRF + r], bb));
    }
    x[(size_t)a * nsym + sym] = acc0;
    return;
  }

  const int a0 = (int)get_group_id(0) * LM;
  const int s0 = (int)get_group_id(1) * LS;
  const int la = (int)get_local_id(0), ls = (int)get_local_id(1);
  __local float2 bbt[TXC_LOCAL_MAX];            // bbt[l*NRF + r] = bb[r][s0+l]

  for (int e = la * LS + ls; e < tile; e += LM * LS) {
    const int l = e / NRF, r = e - l * NRF;
    float2 bb = (float2)(0.0f, 0.0f);
    if (s0 + l < nsym) {
      __global const float2* fbb = Fbb + (size_t)r * T;
      for (int t = 0; t < T; ++t)
        bb = cadd(bb, cmul(fbb[t], s[(size_t)t * nsym + s0 + l]));
    }
    bbt[e] = bb;
  }
  barrier(CLK_LOCAL_MEM_FENCE);

  if (a >= Nt || sym >= nsym) return;           // AFTER the barrier: safe
  float2 acc = (float2)(0.0f, 0.0f);
  __local const float2* bbrow = bbt + (size_t)(sym - s0) * NRF;
  for (int r = 0; r < NRF; ++r)
    acc = cadd(acc, cmul(Frf[(size_t)a * NRF + r], bbrow[r]));
  x[(size_t)a * nsym + sym] = acc;
}

// channel_apply_k: y = H[Nr x Nt] * x + n (flat-fading channel + noise)
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

// rx_combine_k: z = W^H[Nr x d] * y / diag_b per user (diag_b = NULL skips the divide)
//   work-item: (stream k = u*d + j, symbol)
__kernel void rx_combine_k(__global const float2* restrict W, int U, int Nr,
                           int d, __global const float2* restrict y, int nsym,
                           __global const float2* restrict diag_b,
                           __global float2* restrict z) {
  int k = get_global_id(0), sym = get_global_id(1);
  if (k >= U * d || sym >= nsym) return;

  int u = k / d, j = k - u * d;
  __global const float2* w = W + (size_t)u * Nr * d;
  float2 acc = (float2)(0.0f, 0.0f);
  for (int r = 0; r < Nr; ++r)
    acc = cadd(acc, cmulc(w[(size_t)r * d + j], y[(size_t)r * nsym + sym]));
  z[(size_t)k * nsym + sym] = diag_b ? cdiv(acc, diag_b[k]) : acc;
}

// modulo_fold_k: THP receive modulo per axis: out = in - period*floor((in + period/2)/period)
//   work-item: one symbol, gid(0) = q
__kernel void modulo_fold_k(__global const float2* restrict in, int nsymtot,
                            float period, __global float2* restrict out) {
  int q = get_global_id(0);
  if (q >= nsymtot) return;

  float2 v = in[q];
  out[q] = (float2)(cmod_axis(v.x, period), cmod_axis(v.y, period));
}

// replica_sweep: one sweep over the (2*radius+1)^2*M replica set; mode 0 = subset minima, mode 1 = log-sum
static void replica_sweep(float2 r, int L, float delta, int radius, float period,
                          int b, int mh, int mode, float inv_n0, float* dmin,
                          float* S) {
  for (int lr = -radius; lr <= radius; ++lr) {
    for (int li = -radius; li <= radius; ++li) {
      float2 off = (float2)(period * (float)lr, period * (float)li);
      for (int iI = 0; iI < L; ++iI)
        for (int iQ = 0; iQ < L; ++iQ) {
          int bit = label_bit(iI, iQ, b, mh);
          float dd = cand_d2(r, iI, iQ, L, delta, off);
          if (mode) S[bit] += exp((dmin[bit] - dd) * inv_n0);
          else dmin[bit] = fmin(dmin[bit], dd);
        }
    }
  }
}

// softdemod_llr_k: exact log-MAP soft demod (parallel branch): z, n0, m, grp -> llr
//   llr_b = (d2min_0 - d2min_1)/n0 + log S1 - log S0, stabilized (no exp overflow / log 0)
__kernel void softdemod_llr_k(__global const float2* restrict z, int nsymtot,
                              int m, int grp, float n0,
                              __global float* restrict llr) {
  int q = get_global_id(0), g = get_global_id(1);
  int M = 1 << m, mh = m >> 1, L = 1 << mh;
  if (q >= nsymtot || g * grp >= m) return;
  if (M > MMAX || grp < 1) {
    for (int b = 0; b < m; ++b) llr[(size_t)q * m + b] = NAN;
    return;
  }

  float2 r = z[q];
  float delta = qam_delta(M), inv_n0 = 1.0f / n0;

  for (int b = g * grp; b < min(g * grp + grp, m); ++b) {
    float dmin[2] = {INFINITY, INFINITY}, S[2] = {0.0f, 0.0f};
    if (M <= LLR_CACHE_MAX) {
      float d2[LLR_CACHE_MAX];
      for (int iI = 0; iI < L; ++iI)
        for (int iQ = 0; iQ < L; ++iQ)
          d2[iI * L + iQ] = cand_d2(r, iI, iQ, L, delta,
                                    (float2)(0.0f, 0.0f));
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
      replica_sweep(r, L, delta, 0, 0.0f, b, mh, 0, inv_n0, dmin, S);
      replica_sweep(r, L, delta, 0, 0.0f, b, mh, 1, inv_n0, dmin, S);
    }
    llr[(size_t)q * m + b] =
        (dmin[0] - dmin[1]) * inv_n0 + log(S[1]) - log(S[0]);
  }
}

// softdemod_folded_k: THP soft demod over the (2*radius+1)^2*M replica set; radius 0 == softdemod_llr_k
//   work-item: (symbol, bit group), gid(0) = q, gid(1) = g
__kernel void softdemod_folded_k(__global const float2* restrict z, int nsymtot,
                                 int m, int grp, int radius, float period,
                                 float n0, __global float* restrict llr) {
  int q = get_global_id(0), g = get_global_id(1);
  int M = 1 << m, mh = m >> 1, L = 1 << mh;
  if (q >= nsymtot || g * grp >= m) return;
  if (M > MMAX || grp < 1 || radius < 0) {
    for (int b = 0; b < m; ++b) llr[(size_t)q * m + b] = NAN;
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

// gmi_k: per-stream GMI = m - mean_sym sum_b log2(1 + exp(-(2c_b-1)*llr_b)); not clamped, gmi <= 0 = outage
//   work-item: one stream, gid(0) = k < K
__kernel void gmi_k(__global const float* restrict llr,
                    __global const uchar* restrict bits, int m, int nsym,
                    int K, __global float* restrict gmi) {
  int k = get_global_id(0);
  if (k >= K) return;

  float acc = 0.0f, comp = 0.0f;      // comp: the low bits the fp32 add drops
  for (int s = 0; s < nsym; ++s) {
    size_t o = ((size_t)k * nsym + s) * m;
    for (int b = 0; b < m; ++b) {
      float t = (1.0f - 2.0f * (float)bits[o + b]) * llr[o + b];
      float y = fmax(t, 0.0f) + log1p(exp(-fabs(t))) - comp;
      float sum = acc + y;
      comp = (sum - acc) - y;
      acc = sum;
    }
  }
  gmi[k] = (float)m - acc * (1.4426950408889634f / (float)nsym);
}
