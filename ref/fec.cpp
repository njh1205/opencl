//-----------------------------------------------------------------------------
// ref/fec.cpp -- the FEC layer's fp64 host reference. fec.h carries the WHY,
// the sionna.phy.fec mapping, the conventions and the SOURCES for every
// standard constant; this file is only the arithmetic.
//
// Nothing here is OpenCL, nothing here allocates a device, and no number in here
// is quoted from a table that this file does not also derive: d_free is searched,
// the puncture rate is counted out of the pattern, the CRC polynomial's hex form
// is expanded and asserted against its term list, and the two external CRC check
// values are recomputed from the LFSR rather than hardcoded as expectations.
//-----------------------------------------------------------------------------
#include "fec.h"

#include <algorithm>
#include <climits>
#include <cmath>
#include <cstring>

namespace hpocl {
namespace fec {
namespace {

// Parity of the low 32 bits. Written out rather than using __builtin_parity so
// this reference has no compiler-builtin dependency: it is the text a future
// device port would have to reproduce, and a builtin hides the bit order.
int parity32(unsigned v) {
  v ^= v >> 16;
  v ^= v >> 8;
  v ^= v >> 4;
  v ^= v >> 2;
  v ^= v >> 1;
  return static_cast<int>(v & 1u);
}

int gcd(int a, int b) { while (b) { const int t = a % b; a = b; b = t; } return a; }

// 256-entry byte tables, one per CRC width, built once from crc_bitwise so the
// table cannot drift from the definition it accelerates.
struct CrcTables {
  uint32_t t24[256];
  uint32_t t16[256];
  CrcTables() {
    uint8_t b[8];
    for (int x = 0; x < 256; ++x) {
      for (int i = 0; i < 8; ++i) b[i] = static_cast<uint8_t>((x >> (7 - i)) & 1);
      t24[x] = crc_bitwise(b, 8, CrcId::kCrc24A, 0);
      t16[x] = crc_bitwise(b, 8, CrcId::kCrc16, 0);
    }
  }
};
const CrcTables& crc_tables() { static const CrcTables t; return t; }

// S-random acceptance test: position i wants source index c, and every already
// placed position j in the window |i-j| <= s must sit more than s away in value.
// Only j < i needs checking, because the pair (i,j) with i > j is examined when
// the later of the two is placed -- so the constraint is symmetric without
// storing anything.
bool s_ok(const std::vector<int>& perm, int i, int c, int s) {
  const int lo = i - s > 0 ? i - s : 0;
  for (int j = lo; j < i; ++j)
    if (std::abs(c - perm[static_cast<size_t>(j)]) <= s) return false;
  return true;
}

}  // namespace

//============================== CRC ==========================================
int crc_width(CrcId id) {
  switch (id) {
    case CrcId::kCrc24A: return 24;
    case CrcId::kCrc16:  return 16;
    case CrcId::kNone:   return 0;
  }
  return 0;
}

uint32_t crc_poly(CrcId id) {
  // TS 38.212 section 5.1. The hex values are the term lists with the leading
  // D^L dropped: 0x864CFB has bits 23,22,17,16,13,10,9,6,5,4,3,2,1,0 set, i.e.
  // D^24+D^23+D^18+D^17+D^14+D^11+D^10+D^7+D^6+D^5+D^4+D^3+D+1; 0x1021 has bits
  // 12,5,0, i.e. D^16+D^12+D^5+1. runtime/fec_main.cpp re-expands both hex
  // constants into their term lists and fails if they are not the ones quoted in
  // fec.h, so a mistyped constant cannot survive as a self-consistent wrong CRC;
  // and both are pinned from outside by the three published check values in fec.h.
  switch (id) {
    case CrcId::kCrc24A: return 0x864CFBu;
    case CrcId::kCrc16:  return 0x1021u;
    case CrcId::kNone:   return 0u;
  }
  return 0u;
}

uint32_t crc_init(CrcId id) { (void)id; return 0u; }   // TS 38.212: register starts at 0

uint32_t crc_mask(CrcId id) {
  const int w = crc_width(id);
  return w >= 32 ? 0xFFFFFFFFu : ((1u << w) - 1u);
}

uint32_t crc_bitwise(const uint8_t* b, int n, CrcId id, uint32_t init) {
  const int w = crc_width(id);
  if (w == 0 || n <= 0) return 0u;
  const uint32_t poly = crc_poly(id), mask = crc_mask(id);
  uint32_t reg = init & mask;
  for (int i = 0; i < n; ++i) {
    const uint32_t fb = ((reg >> (w - 1)) ^ (b[i] & 1u)) & 1u;
    reg = (reg << 1) & mask;
    if (fb) reg ^= poly;
  }
  return reg;
}

uint32_t crc_table(const uint8_t* b, int n, CrcId id, uint32_t init) {
  const int w = crc_width(id);
  if (w == 0 || n <= 0) return 0u;
  const uint32_t mask = crc_mask(id);
  const uint32_t* tab = (w == 24) ? crc_tables().t24 : crc_tables().t16;
  uint32_t reg = init & mask;
  int i = 0;
  for (; i + 8 <= n; i += 8) {
    uint32_t byte = 0;
    for (int j = 0; j < 8; ++j) byte = (byte << 1) | (b[i + j] & 1u);
    reg = ((reg << 8) & mask) ^ tab[((reg >> (w - 8)) ^ byte) & 0xFFu];
  }
  // Lengths that are not a multiple of 8 are the normal case here (a CRC over an
  // arbitrary number of information bits), so the tail is not an edge case: it
  // finishes with single-bit steps. Comparing crc_table against crc_bitwise on a
  // non-aligned length is what proves this branch right.
  for (; i < n; ++i) {
    const uint32_t fb = ((reg >> (w - 1)) ^ (b[i] & 1u)) & 1u;
    reg = (reg << 1) & mask;
    if (fb) reg ^= crc_poly(id);
  }
  return reg;
}

uint32_t crc_compute(const uint8_t* b, int n, CrcId id) {
  return crc_table(b, n, id, crc_init(id));
}

void crc_to_bits(uint32_t v, int width, uint8_t* out) {
  for (int i = 0; i < width; ++i)
    out[i] = static_cast<uint8_t>((v >> (width - 1 - i)) & 1u);
}

void crc_append(const uint8_t* msg, int A, CrcId id, uint8_t* out) {
  const int w = crc_width(id);
  if (A > 0) std::memcpy(out, msg, static_cast<size_t>(A));
  if (w == 0) return;
  crc_to_bits(crc_compute(msg, A, id), w, out + A);
}

bool crc_check(const uint8_t* codeword, int n, CrcId id) {
  const int w = crc_width(id);
  if (w == 0) return true;
  if (n < w) return false;
  return crc_compute(codeword, n, id) == 0u;
}

//=========================== interleaver =====================================
uint64_t splitmix64(uint64_t* state) {
  uint64_t z = (*state += 0x9E3779B97F4A7C15ULL);
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
  return z ^ (z >> 31);
}

int srandom_spread(int n, int s_requested) {
  if (n < 4) return 0;                    // see the measured table in fec.h
  const int cap = static_cast<int>(std::sqrt(static_cast<double>(n)) / 3.0);
  return std::max(1, std::min(s_requested, cap));
}

Interleaver make_interleaver(IntKind kind, int n, int par, uint64_t seed) {
  Interleaver t;
  t.kind = kind;
  t.n = n;
  t.par = par;
  t.seed = seed;
  if (n <= 0) return t;
  t.perm.resize(static_cast<size_t>(n));
  t.attempts = 1;                        // not a search: one deterministic pass
  if (kind == IntKind::kIdentity) {
    for (int i = 0; i < n; ++i) t.perm[static_cast<size_t>(i)] = i;
    return t;
  }
  if (kind == IntKind::kRowColumn) {
    // Write by rows, read by columns, and drop the padding positions instead of
    // carrying them: an interleaver whose length is not rows*cols still has to be
    // an exact permutation of [0,n), or deinterleave cannot invert it.
    const int cols = std::max(1, std::min(par > 0 ? par : n, n));
    const int rows = (n + cols - 1) / cols;
    t.par = cols;
    int k = 0;
    for (int c = 0; c < cols; ++c)
      for (int r = 0; r < rows; ++r) {
        const int idx = r * cols + c;
        if (idx < n) t.perm[static_cast<size_t>(k++)] = idx;
      }
    return t;
  }
  // S-random: greedy rejection sampling, and a restart ladder around it.
  //   WHY the ladder: a pure greedy walk paints itself into a corner near the END
  //   of the block, where few source indices are still unused and the last S
  //   placed values forbid up to S*(2S+1) of them. Measured with the single-pass
  //   version (1000 draws, then "smallest unused that fits", then "smallest
  //   unused"): n=1000/S=8 -> 0 violations, n=236/S=8 -> 2, n=100/S=8 -> 41, i.e.
  //   the interleaver silently degraded to almost no interleaving exactly on the
  //   short frames the BLER waterfall uses. Restarting the whole walk with the
  //   next seed costs O(n*S) and turns "usually fine" into "found or reported":
  //   the attempt index is stored, so the result is still a pure function of
  //   (n, S, seed) and tools/fec_ref.py reproduces it draw for draw.
  //   Determinism notes that matter for the cross-check: every draw consumes
  //   exactly one splitmix64 output, the fallback scans do not draw, and an
  //   attempt is abandoned the instant a position cannot be filled.
  const int S = srandom_spread(n, par);
  t.par = S;                              // report what was actually used
  std::vector<char> used(static_cast<size_t>(n), 0);
  if (S == 0) {                           // too short for any spread constraint
    for (int i = 0; i < n; ++i) t.perm[static_cast<size_t>(i)] = i;
    return t;
  }
  for (int attempt = 0; attempt < kSRandomMaxAttempts; ++attempt) {
    uint64_t st = seed + static_cast<uint64_t>(attempt);
    std::fill(used.begin(), used.end(), 0);
    bool complete = true;
    for (int i = 0; i < n && complete; ++i) {
      int chosen = -1;
      for (int draw = 0; draw < kSRandomMaxDraws && chosen < 0; ++draw) {
        const int c = static_cast<int>(splitmix64(&st) % static_cast<uint64_t>(n));
        if (used[static_cast<size_t>(c)]) continue;
        if (s_ok(t.perm, i, c, S)) chosen = c;
      }
      if (chosen < 0) { complete = false; break; }
      t.perm[static_cast<size_t>(i)] = chosen;
      used[static_cast<size_t>(chosen)] = 1;
    }
    if (complete) {
      t.attempts = attempt + 1;           // 1-based: "the first try worked"
      t.seed = seed + static_cast<uint64_t>(attempt);
      return t;
    }
  }
  // Budget exhausted: return SOMETHING invertible rather than failing, because
  // the decoder only needs a permutation, and report the damage. The ladder here
  // is random draws -> first fitting unused -> smallest unused, so it always
  // completes, and spread_violations() makes the shortfall a number.
  t.attempts = kSRandomMaxAttempts + 1;
  t.seed = seed;
  uint64_t st = seed;
  std::fill(used.begin(), used.end(), 0);
  for (int i = 0; i < n; ++i) {
    int chosen = -1;
    for (int draw = 0; draw < kSRandomMaxDraws && chosen < 0; ++draw) {
      const int c = static_cast<int>(splitmix64(&st) % static_cast<uint64_t>(n));
      if (used[static_cast<size_t>(c)]) continue;
      if (s_ok(t.perm, i, c, S)) chosen = c;
    }
    if (chosen < 0)
      for (int c = 0; c < n && chosen < 0; ++c)
        if (!used[static_cast<size_t>(c)] && s_ok(t.perm, i, c, S)) chosen = c;
    if (chosen < 0)
      for (int c = 0; c < n && chosen < 0; ++c)
        if (!used[static_cast<size_t>(c)]) chosen = c;
    t.perm[static_cast<size_t>(i)] = chosen;
    used[static_cast<size_t>(chosen)] = 1;
  }
  return t;
}

bool is_permutation(const Interleaver& t) {
  if (static_cast<int>(t.perm.size()) != t.n) return false;
  std::vector<char> seen(static_cast<size_t>(t.n), 0);
  for (int i = 0; i < t.n; ++i) {
    const int v = t.perm[static_cast<size_t>(i)];
    if (v < 0 || v >= t.n || seen[static_cast<size_t>(v)]) return false;
    seen[static_cast<size_t>(v)] = 1;
  }
  return true;
}

int min_spread(const Interleaver& t, int s) {
  if (s < 1 || t.n < 2) return INT_MAX;
  int best = INT_MAX;
  for (int i = 0; i < t.n; ++i)
    for (int j = std::max(0, i - s); j < i; ++j)
      best = std::min(best, std::abs(t.perm[static_cast<size_t>(i)] -
                                     t.perm[static_cast<size_t>(j)]));
  return best;
}

int spread_violations(const Interleaver& t, int s) {
  if (s < 1) return 0;
  int bad = 0;
  for (int i = 0; i < t.n; ++i)
    for (int j = std::max(0, i - s); j < i; ++j)
      if (std::abs(t.perm[static_cast<size_t>(i)] -
                   t.perm[static_cast<size_t>(j)]) <= s) ++bad;
  return bad;
}

void interleave_bits(const uint8_t* in, const Interleaver& t, uint8_t* out) {
  for (int i = 0; i < t.n; ++i) out[i] = in[t.perm[static_cast<size_t>(i)]];
}
void deinterleave_bits(const uint8_t* in, const Interleaver& t, uint8_t* out) {
  for (int i = 0; i < t.n; ++i) out[t.perm[static_cast<size_t>(i)]] = in[i];
}
void interleave_llr(const double* in, const Interleaver& t, double* out) {
  for (int i = 0; i < t.n; ++i) out[i] = in[t.perm[static_cast<size_t>(i)]];
}
void deinterleave_llr(const double* in, const Interleaver& t, double* out) {
  for (int i = 0; i < t.n; ++i) out[t.perm[static_cast<size_t>(i)]] = in[i];
}

//========================= convolutional code ================================
// See SOURCES in fec.h: sionna.phy.fec.conv.polynomial_selector(rate=0.5,
// constraint_length=7) returns ('1011011','1111001'), i.e. (133,171) octal, which
// as K=7 tap masks with the MSB on the current input are 0b1011011 = 0x5B and
// 0b1111001 = 0x79 (5 taps each). runtime/fec_main.cpp re-expands both constants
// into those two binary strings, so a transposed digit cannot pass as a
// self-consistent different code.
//
// This encoder is FEEDFORWARD, so it cannot be catastrophic in the sense that
// term applies to recursive (RSC) encoders: the impulse response is finite, hence
// every nonzero input of finite support produces an output of finite nonzero
// weight. What d_free > 0 actually rests on is the absence of a zero-weight CYCLE
// in the trellis other than the trivial (state 0, u=0) loop, and the bench checks
// that exhaustively (transitive closure over the zero-weight edges, all 64 states)
// rather than trusting the argument above.
const ConvCode kConvK7R12 = {7, {0x5Bu, 0x79u}};

namespace {
struct PunctPat { int period; const uint8_t* pat; };
// Written out explicitly (see SOURCES: these are NOT quoted from a spec table).
// kR23 keeps 3 of every 4; kR34 keeps 4 of every 6 and alternates which of the
// two generators survives, so neither output stream is erased for two periods in
// a row. punct_rate() derives the achieved rate from these arrays by counting.
const uint8_t kPatR12[2] = {1, 1};
const uint8_t kPatR23[4] = {1, 1, 1, 0};
const uint8_t kPatR34[6] = {1, 1, 1, 0, 0, 1};
PunctPat pat_of(PunctId p) {
  switch (p) {
    case PunctId::kR12: return {2, kPatR12};
    case PunctId::kR23: return {4, kPatR23};
    case PunctId::kR34: return {6, kPatR34};
  }
  return {2, kPatR12};
}
}  // namespace

int punct_period(PunctId p) { return pat_of(p).period; }
const uint8_t* punct_pattern(PunctId p) { return pat_of(p).pat; }

int punct_kept(PunctId p) {
  const PunctPat q = pat_of(p);
  int k = 0;
  for (int i = 0; i < q.period; ++i) k += q.pat[i] ? 1 : 0;
  return k;
}

int punct_out_len(int ncoded, PunctId p) {
  const PunctPat q = pat_of(p);
  if (ncoded <= 0) return 0;
  const int full = ncoded / q.period, rem = ncoded % q.period;
  int k = 0;
  for (int i = 0; i < q.period; ++i) k += q.pat[i] ? 1 : 0;
  int extra = 0;
  for (int i = 0; i < rem; ++i) extra += q.pat[i] ? 1 : 0;
  return full * k + extra;
}

void punct_rate(PunctId p, int* num, int* den) {
  // kept/period of the rate-1/2 mother stream survives, so the effective rate is
  // period / (2*kept), reduced. Counted, not typed: change a pattern element and
  // the rate this reports changes with it.
  const int per = punct_period(p), kept = punct_kept(p);
  int n = per, d = 2 * kept;
  const int g = gcd(n, d);
  n /= g; d /= g;
  if (num) *num = n;
  if (den) *den = d;
}

void puncture(const uint8_t* coded, int ncoded, PunctId p, uint8_t* out) {
  const PunctPat q = pat_of(p);
  int k = 0;
  for (int i = 0; i < ncoded; ++i)
    if (q.pat[i % q.period]) out[k++] = coded[i];
}

void depuncture_llr(const double* punc, int npunc, int ncoded, PunctId p,
                    double* out) {
  const PunctPat q = pat_of(p);
  int k = 0;
  for (int i = 0; i < ncoded; ++i) {
    if (q.pat[i % q.period]) out[i] = (k < npunc) ? punc[k++] : 0.0;
    else out[i] = 0.0;      // erasure: the neutral element of sum_b c_b * llr_b
  }
}

TrellisTab build_trellis(const ConvCode& c) {
  TrellisTab tr;
  tr.k = c.k;
  tr.ns = 1 << (c.k - 1);
  const int ns = tr.ns;
  tr.to.assign(static_cast<size_t>(ns) * 2, 0);
  tr.out.assign(static_cast<size_t>(ns) * 4, 0);
  tr.prev.assign(static_cast<size_t>(ns) * 2, 0);
  tr.pin.assign(static_cast<size_t>(ns) * 2, 0);
  for (int s = 0; s < ns; ++s) {
    for (int u = 0; u < 2; ++u) {
      const int w = (u << (c.k - 1)) | s;
      tr.to[static_cast<size_t>(s) * 2 + u] = (u << (c.k - 2)) | (s >> 1);
      tr.out[(static_cast<size_t>(s) * 2 + u) * 2 + 0] =
          static_cast<uint8_t>(parity32(static_cast<unsigned>(w) & c.gen[0]));
      tr.out[(static_cast<size_t>(s) * 2 + u) * 2 + 1] =
          static_cast<uint8_t>(parity32(static_cast<unsigned>(w) & c.gen[1]));
    }
  }
  // Inverse view, indexed by (next state, slot). Slot 0 is the smaller
  // predecessor, which is exactly the Viterbi tie-break order in fec.h.
  const int h = ns / 2;
  for (int n = 0; n < ns; ++n) {
    const int u = n >> (c.k - 2), q = n & (h - 1);
    tr.prev[static_cast<size_t>(n) * 2 + 0] = 2 * q;
    tr.prev[static_cast<size_t>(n) * 2 + 1] = 2 * q + 1;
    tr.pin[static_cast<size_t>(n) * 2 + 0] = u;
    tr.pin[static_cast<size_t>(n) * 2 + 1] = u;
  }
  return tr;
}

int conv_coded_len(int A, const ConvCode& c) {
  return A < 0 ? 0 : 2 * (A + c.k - 1);
}

void conv_encode(const uint8_t* a, int A, const ConvCode& c, uint8_t* out) {
  const int mu = c.k - 1, T = A + mu;
  int s = 0;
  for (int t = 0; t < T; ++t) {
    const int u = (t < A) ? (a[t] & 1) : 0;        // zero tail, K-1 bits
    const int w = (u << (c.k - 1)) | s;
    out[2 * t + 0] = static_cast<uint8_t>(parity32(static_cast<unsigned>(w) & c.gen[0]));
    out[2 * t + 1] = static_cast<uint8_t>(parity32(static_cast<unsigned>(w) & c.gen[1]));
    s = (u << (c.k - 2)) | (s >> 1);
  }
}

int free_distance_dijkstra(const ConvCode& c) {
  const TrellisTab tr = build_trellis(c);
  const int ns = tr.ns;
  const int start = tr.next_state(0, 1);           // the only nonzero first branch
  const int w0 = tr.out_bit(0, 1, 0) + tr.out_bit(0, 1, 1);
  std::vector<int> dist(static_cast<size_t>(ns), INT_MAX);
  std::vector<char> done(static_cast<size_t>(ns), 0);
  dist[static_cast<size_t>(start)] = w0;
  for (int it = 0; it < ns; ++it) {
    int u = -1, bd = INT_MAX;
    for (int s = 0; s < ns; ++s)
      if (!done[s] && dist[s] < bd) { bd = dist[s]; u = s; }
    if (u < 0) break;
    done[static_cast<size_t>(u)] = 1;
    for (int b = 0; b < 2; ++b) {
      const int v = tr.next_state(u, b);
      const int w = tr.out_bit(u, b, 0) + tr.out_bit(u, b, 1);
      const int nd = dist[static_cast<size_t>(u)] + w;
      if (nd < dist[static_cast<size_t>(v)]) dist[static_cast<size_t>(v)] = nd;
    }
  }
  return dist[0] == INT_MAX ? -1 : dist[0];
}

int free_distance_bounded(const ConvCode& c, int branches_max, int* depth_out) {
  const TrellisTab tr = build_trellis(c);
  const int ns = tr.ns;
  const int start = tr.next_state(0, 1);
  const int w0 = tr.out_bit(0, 1, 0) + tr.out_bit(0, 1, 1);
  std::vector<int> cur(static_cast<size_t>(ns), INT_MAX);
  std::vector<int> nxt(static_cast<size_t>(ns), INT_MAX);
  cur[static_cast<size_t>(start)] = w0;
  int best = INT_MAX, bdepth = -1;
  // d = expansions applied so far, so `cur` describes walks of d+1 branches and
  // the loop therefore covers every walk of at most branches_max branches.
  for (int d = 0; d < branches_max; ++d) {
    if (cur[0] < best) { best = cur[0]; bdepth = d + 1; }   // +1: the forced branch
    std::fill(nxt.begin(), nxt.end(), INT_MAX);
    for (int s = 0; s < ns; ++s) {
      if (cur[static_cast<size_t>(s)] == INT_MAX) continue;
      for (int b = 0; b < 2; ++b) {
        const int v = tr.next_state(s, b);
        const int w = tr.out_bit(s, b, 0) + tr.out_bit(s, b, 1);
        const int nd = cur[static_cast<size_t>(s)] + w;
        if (nd < nxt[static_cast<size_t>(v)]) nxt[static_cast<size_t>(v)] = nd;
      }
    }
    cur.swap(nxt);
  }
  if (depth_out) *depth_out = bdepth;
  return best == INT_MAX ? -1 : best;
}

//=============================== Viterbi =====================================
ViterbiOut viterbi_decode(const double* llr_dep, int T, const ConvCode& c) {
  ViterbiOut res;
  const int mu = c.k - 1, A = T - mu;
  if (T <= mu || A <= 0) return res;               // nothing but tail: refuse
  const TrellisTab tr = build_trellis(c);
  const int ns = tr.ns;
  std::vector<double> m(static_cast<size_t>(ns)), m2(static_cast<size_t>(ns));
  std::vector<uint8_t> surv(static_cast<size_t>(T) * static_cast<size_t>(ns));
  for (int s = 0; s < ns; ++s) m[static_cast<size_t>(s)] = (s == 0) ? 0.0 : -INFINITY;
  for (int t = 0; t < T; ++t) {
    const double l0 = llr_dep[2 * t], l1 = llr_dep[2 * t + 1];
    for (int n = 0; n < ns; ++n) {
      const int u = tr.pred_input(n);
      const int s0 = tr.pred(n, 0), s1 = tr.pred(n, 1);
      const double b0 = tr.out_bit(s0, u, 0) * l0 + tr.out_bit(s0, u, 1) * l1;
      const double b1 = tr.out_bit(s1, u, 0) * l0 + tr.out_bit(s1, u, 1) * l1;
      const double c0 = m[static_cast<size_t>(s0)] + b0;
      const double c1 = m[static_cast<size_t>(s1)] + b1;
      const bool take1 = (c1 > c0);                // strict: slot 0 wins ties
      m2[static_cast<size_t>(n)] = take1 ? c1 : c0;
      surv[static_cast<size_t>(t) * ns + static_cast<size_t>(n)] =
          static_cast<uint8_t>(take1 ? s1 : s0);
    }
    m.swap(m2);
  }
  if (!std::isfinite(m[0])) return res;            // unreachable / non-finite LLR
  res.ok = true;
  res.metric = m[0];
  res.info.assign(static_cast<size_t>(A), 0);
  int st = 0;                                      // zero tail: end in state 0
  for (int t = T - 1; t >= 0; --t) {
    const int u = tr.pred_input(st);
    if (t < A) res.info[static_cast<size_t>(t)] = static_cast<uint8_t>(u);
    st = surv[static_cast<size_t>(t) * ns + static_cast<size_t>(st)];
  }
  return res;
}

//=========================== whole-frame helpers =============================
FrameGeom make_geom(int L, CrcId crc, PunctId p, const ConvCode& c) {
  FrameGeom g;
  g.L = L;
  g.crc = crc;
  g.crc_w = crc_width(crc);
  g.A = L + g.crc_w;
  g.k = c.k;
  g.T = g.A + c.k - 1;
  g.ncoded = 2 * g.T;
  g.punct = p;
  g.npunc = punct_out_len(g.ncoded, p);
  return g;
}

void encode_frame(const uint8_t* info, const FrameGeom& g, const ConvCode& c,
                  const Interleaver& in, uint8_t* out, uint8_t* coded) {
  Bits a(static_cast<size_t>(g.A));
  crc_append(info, g.L, g.crc, a.data());
  Bits nc(static_cast<size_t>(g.ncoded));
  conv_encode(a.data(), g.A, c, nc.data());
  if (coded) std::memcpy(coded, nc.data(), static_cast<size_t>(g.ncoded));
  Bits pn(static_cast<size_t>(g.npunc));
  puncture(nc.data(), g.ncoded, g.punct, pn.data());
  if (in.n == g.npunc && in.kind != IntKind::kIdentity)
    interleave_bits(pn.data(), in, out);
  else
    std::memcpy(out, pn.data(), static_cast<size_t>(g.npunc));
}

void decode_frame(const double* llr_int, const FrameGeom& g, const ConvCode& c,
                  const Interleaver& in, uint8_t* info_hat, double* metric,
                  bool* crc_ok, double* llr_dep) {
  std::vector<double> work(static_cast<size_t>(g.npunc));
  if (in.n == g.npunc && in.kind != IntKind::kIdentity)
    deinterleave_llr(llr_int, in, work.data());
  else
    std::memcpy(work.data(), llr_int, sizeof(double) * static_cast<size_t>(g.npunc));
  std::vector<double> dep(static_cast<size_t>(g.ncoded));
  depuncture_llr(work.data(), g.npunc, g.ncoded, g.punct, dep.data());
  if (llr_dep) std::memcpy(llr_dep, dep.data(), sizeof(double) * static_cast<size_t>(g.ncoded));
  const ViterbiOut v = viterbi_decode(dep.data(), g.T, c);
  if (metric) *metric = v.metric;
  if (crc_ok) *crc_ok = v.ok && crc_check(v.info.data(), g.A, g.crc);
  if (v.ok) std::memcpy(info_hat, v.info.data(), static_cast<size_t>(g.L));
  else std::memset(info_hat, 0, static_cast<size_t>(g.L));
}

}  // namespace fec
}  // namespace hpocl
