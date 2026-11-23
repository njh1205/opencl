// fec_bench: the FEC layer, cross-checked three ways where a third way exists.
//
//   numpy oracle (tools/fec_ref.py)     CRC as GF(2)[x] polynomial division and
//                                       as a (width x A) matrix; encoder as a
//                                       generator-matrix multiply; Viterbi as
//                                       batched (B, ns, 2) array arithmetic
//     vs  host C++ (ref/fec.cpp)        CRC as an MSB-first LFSR with a 256-entry
//                                       byte table; encoder as a shift register
//                                       walked one branch at a time; Viterbi as a
//                                       scalar 64-state loop
//     vs  brute force                   exhaustive enumeration of all 2^L
//                                       codewords, run independently on BOTH sides
//                                       for the small frames
//
// There is no device side and no OpenCL: this layer is host-only by design (see
// ref/fec.h "WHAT IT IS NOT" and docs/fec_layer.md). So the gates split into
//
//   PART 1  properties this stack can prove about itself without any oracle:
//           d_free = 10 by two independent searches, CRC residual + the three
//           externally attested check values, exact interleaver inverses,
//           noiseless and punctured loopback at BLER 0, erasures exactly 0.0,
//           and Viterbi == brute force on random LLRs.
//   PART 2  the vector file: numpy vs C++ bit-exact on the permutation, the
//           encoded bits, the decoded bits and the CRC verdict, plus a
//           tolerance-bounded match on the fp64 path metric.
//
// PART 1 is what makes PART 2 meaningful. Two implementations agreeing on a wrong
// code is easy; agreeing while each also reproduces d_free = 10 and an externally
// published CRC check value from its own arithmetic is not.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdarg>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include "fec.h"
#include "hpocl_vecio.h"

using namespace hpocl::fec;

namespace {

double now_s() {
  return std::chrono::duration<double>(
             std::chrono::steady_clock::now().time_since_epoch()).count();
}

// One line per gate, PASS/FAIL plus the number that decided it -- the format
// runtime/rate_main.cpp uses, so `make check` output reads the same throughout.
struct Gates {
  int fails = 0, total = 0;
  void report(const char* name, bool ok, const std::string& detail) {
    ++total;
    fails += !ok;
    printf("  [%s] %-58s %s\n", ok ? "PASS" : "FAIL", name, detail.c_str());
  }
};

char buf_[512];
std::string fmt(const char* f, ...) {
  va_list ap; va_start(ap, f);
  vsnprintf(buf_, sizeof buf_, f, ap);
  va_end(ap);
  return std::string(buf_);
}

// Dumb exhaustive ML search: the third side of the small-frame cross-check. It
// shares nothing with viterbi_decode -- it goes through encode_frame, so a bug in
// the trellis tables, the survivor memory or the traceback cannot hide here. Ties
// take the smallest information index (strict >), which is the enumeration
// analogue of the slot-0 tie-break; tools/fec_ref.py::brute_force_ml uses np.argmax
// and therefore the same rule.
double brute_force_best(const double* llr_int, int L, const ConvCode& c,
                        const Interleaver& in, const FrameGeom& g,
                        uint8_t* info_out) {
  const long long total = 1LL << L;
  std::vector<uint8_t> info((size_t)L), tx((size_t)g.npunc);
  double best = -INFINITY;
  long long besti = 0;
  for (long long idx = 0; idx < total; ++idx) {
    for (int b = 0; b < L; ++b)
      info[(size_t)b] = (uint8_t)((idx >> (L - 1 - b)) & 1);
    encode_frame(info.data(), g, c, in, tx.data(), nullptr);
    double m = 0;
    for (int i = 0; i < g.npunc; ++i) m += (double)tx[(size_t)i] * llr_int[i];
    if (m > best) { best = m; besti = idx; }
  }
  for (int b = 0; b < L; ++b)
    info_out[(size_t)b] = (uint8_t)((besti >> (L - 1 - b)) & 1);
  return best;
}

Bits random_bits(std::mt19937_64& rng, int n) {
  Bits b((size_t)n);
  for (int i = 0; i < n; ++i) b[(size_t)i] = (uint8_t)(rng() & 1u);
  return b;
}

// Turn hard bits into LLRs at a fixed magnitude, the "high SNR" input the loopback
// gates use. amp is finite on purpose: +-inf would make the metric inf-inf=NaN in
// any decoder that ever compares two paths through the same erased positions.
std::vector<double> bits_to_llr(const uint8_t* b, int n, double amp) {
  std::vector<double> v((size_t)n);
  for (int i = 0; i < n; ++i) v[(size_t)i] = b[i] ? amp : -amp;
  return v;
}

//============================================================================
// PART 1a: free distance
//============================================================================
void gate_dfree(Gates& G) {
  const ConvCode c = kConvK7R12;
  printf("[fec_bench] PART 1a  free distance of the (133,171)_8 K=7 rate-1/2 code\n");
  // The generators, re-expanded: '1011011' and '1111001' are the strings
  // sionna.phy.fec.conv.polynomial_selector(rate=0.5, constraint_length=7)
  // returns, i.e. 133 and 171 octal. Printing the expansion is the check that the
  // hex constants in fec.cpp are those strings and not something else with the
  // same popcount.
  char s0[16], s1[16];
  for (int i = 0; i < c.k; ++i) {
    s0[i] = '0' + ((c.gen[0] >> (c.k - 1 - i)) & 1u);
    s1[i] = '0' + ((c.gen[1] >> (c.k - 1 - i)) & 1u);
  }
  s0[c.k] = s1[c.k] = 0;
  G.report("generators expand to the documented octal pair", 
           std::string(s0) == "1011011" && std::string(s1) == "1111001" &&
           c.gen[0] == 0133u && c.gen[1] == 0171u,
           fmt("gen = (%s, %s) = (0%o, 0%o), K = %d", s0, s1, c.gen[0], c.gen[1], c.k));

  // trellis self-consistency: the (next state, slot) predecessor view must be the
  // exact inverse of the (state, input) successor view, or the Viterbi update and
  // the traceback are walking two different trellises.
  const TrellisTab tr = build_trellis(c);
  int bad = 0;
  for (int s = 0; s < tr.ns; ++s)
    for (int u = 0; u < 2; ++u) {
      const int n = tr.next_state(s, u);
      int found = 0;
      for (int slot = 0; slot < 2; ++slot)
        found += (tr.pred(n, slot) == s && tr.pred_input(n) == u);
      bad += (found != 1);
    }
  G.report("trellis predecessor view inverts the successor view", bad == 0,
           fmt("%d states x 2 inputs, %d inconsistencies; u(s=%d) -> s'=%d",
               tr.ns, bad, 0, tr.next_state(0, 1)));

  const int d_dij = free_distance_dijkstra(c);
  int depth40 = -1, depth80 = -1, depth6 = -1, depth7 = -1;
  const int d_b40 = free_distance_bounded(c, 40, &depth40);
  const int d_b80 = free_distance_bounded(c, 80, &depth80);
  const int d_b6 = free_distance_bounded(c, 6, &depth6);
  const int d_b7 = free_distance_bounded(c, 7, &depth7);
  G.report("d_free == 10, Dijkstra vs layered BFS at two budgets",
           d_dij == 10 && d_b40 == 10 && d_b80 == 10 && depth40 == depth80,
           fmt("dijkstra = %d, bounded(40 branches) = %d, bounded(80) = %d, both "
               "attained at %d branches", d_dij, d_b40, d_b80, depth40));
  // The bound is not the binding constraint: 7 branches find it, 6 provably do
  // not. That is what turns "the search was deep enough" from an assumption into a
  // measurement.
  G.report("the branch budget is provably not binding", d_b6 != 10 && d_b7 == 10,
           fmt("bounded(6) = %d (misses it, depth_out %d), bounded(7) = %d at %d "
               "branches", d_b6, depth6, d_b7, depth7));

  // Why 7 branches is enough: the single-impulse input (one 1 then the zero tail)
  // already produces a weight-10 codeword, so d_free is attained by the shortest
  // possible nonzero input. Enumerated here, not asserted from the literature.
  const int A = 7;
  Bits one((size_t)A, 0);
  one[0] = 1;
  std::vector<uint8_t> cw((size_t)conv_coded_len(A, c));
  conv_encode(one.data(), A, c, cw.data());
  int w = 0;
  for (size_t i = 0; i < cw.size(); ++i) w += cw[i];
  G.report("impulse response weight == d_free (explains the 7-branch optimum)",
           w == d_dij,
           fmt("input 1000000 + %d tail bits -> output weight %d, d_free %d",
               c.k - 1, w, d_dij));

  // Non-catastrophic, stated precisely. Zero-weight BRANCHES are normal (31 of the
  // 128 here) and harmless; what would invalidate d_free is a zero-weight CYCLE,
  // because then a nonzero input produces an all-zero output. Transitive closure
  // over the zero-weight edges only -- 64 nodes, so this is exhaustive, not a
  // sample, and it is the property the Dijkstra answer rests on.
  // The trivial (state 0, u=0) loop is excluded: it is a zero-weight cycle, but
  // it is the zero input, not a codeword. Every OTHER zero-weight-only cycle would
  // have to contain a u=1 edge, because a u=0 branch maps s -> s>>1 and so can
  // only cycle at s = 0 -- which makes "no self-reachable state left" exactly the
  // statement "no nonzero input produces a zero-weight output".
  std::vector<char> z(static_cast<size_t>(tr.ns) * tr.ns, 0);
  int nzero = 0, nzero_u1 = 0;
  for (int s = 0; s < tr.ns; ++s)
    for (int u = 0; u < 2; ++u) {
      if (tr.out_bit(s, u, 0) + tr.out_bit(s, u, 1) != 0) continue;
      ++nzero;
      nzero_u1 += u;
      if (s == 0 && u == 0) continue;
      z[(size_t)s * tr.ns + tr.next_state(s, u)] = 1;
    }
  for (int k = 0; k < tr.ns; ++k)
    for (int i = 0; i < tr.ns; ++i)
      if (z[(size_t)i * tr.ns + k])
        for (int j = 0; j < tr.ns; ++j)
          z[(size_t)i * tr.ns + j] = (char)(z[(size_t)i * tr.ns + j] |
                                            z[(size_t)k * tr.ns + j]);
  int cyc = 0;
  for (int s = 0; s < tr.ns; ++s) cyc += z[(size_t)s * tr.ns + s] ? 1 : 0;
  G.report("no zero-weight cycle but the trivial loop (not catastrophic)",
           cyc == 0 && d_dij > 0,
           fmt("%d zero-weight branches of %d (%d of them driven by u=1), %d states "
               "self-reachable over the non-trivial ones, d_free = %d > 0",
               nzero, tr.ns * 2, nzero_u1, cyc, d_dij));
}

//============================================================================
// PART 1b: CRC
//============================================================================
void gate_crc(Gates& G) {
  printf("[fec_bench] PART 1b  CRC-24A / CRC-16 (3GPP TS 38.212 section 5.1)\n");
  // hex <-> term list, so the constant and the polynomial it claims to be cannot
  // drift apart silently.
  const int t24[] = {24, 23, 18, 17, 14, 11, 10, 7, 6, 5, 4, 3, 1, 0};
  const int t16[] = {16, 12, 5, 0};
  auto terms_ok = [](uint32_t poly, int width, const int* t, int nt) {
    std::vector<int> got;
    got.push_back(width);
    for (int e = width - 1; e >= 0; --e)
      if ((poly >> e) & 1u) got.push_back(e);
    if ((int)got.size() != nt) return false;
    for (int i = 0; i < nt; ++i) if (got[i] != t[i]) return false;
    return true;
  };
  G.report("0x864CFB / 0x1021 expand to the TS 38.212 term lists",
           terms_ok(crc_poly(CrcId::kCrc24A), 24, t24, 14) &&
           terms_ok(crc_poly(CrcId::kCrc16), 16, t16, 4),
           fmt("CRC-24A 0x%06X (%d taps), CRC-16 0x%04X (%d taps), init 0x%X both",
               crc_poly(CrcId::kCrc24A), 14, crc_poly(CrcId::kCrc16), 4,
               crc_init(CrcId::kCrc24A)));

  // Externally attested check values for the ASCII string "123456789". These are
  // the only numbers in this file that come from outside the repo; they are used
  // as assertions, never as inputs, and each one pins a different property:
  // XMODEM pins poly+MSB-first+unreflected in the EXACT 3GPP configuration
  // (init 0), IBM-3740 pins the nonzero-init path, OPENPGP pins the 24-bit poly.
  const char* chk = "123456789";
  Bits cbits(72);
  for (int i = 0; i < 9; ++i)
    for (int j = 0; j < 8; ++j)
      cbits[(size_t)(i * 8 + j)] = (uint8_t)((chk[i] >> (7 - j)) & 1);
  const uint32_t g_xmodem = crc_bitwise(cbits.data(), 72, CrcId::kCrc16, 0x0000);
  const uint32_t g_3740 = crc_bitwise(cbits.data(), 72, CrcId::kCrc16, 0xFFFF);
  const uint32_t g_pg = crc_bitwise(cbits.data(), 72, CrcId::kCrc24A, 0xB704CE);
  G.report("check value CRC-16/XMODEM (init 0 = the 3GPP configuration)",
           g_xmodem == 0x31C3u, fmt("0x%04X vs published 0x31C3", g_xmodem));
  G.report("check value CRC-16/IBM-3740 (init 0xFFFF)",
           g_3740 == 0x29B1u, fmt("0x%04X vs published 0x29B1", g_3740));
  G.report("check value CRC-24/OPENPGP (init 0xB704CE, same poly as CRC-24A)",
           g_pg == 0x21CF02u, fmt("0x%06X vs published 0x21CF02", g_pg));
  // The catalogue's residue field is 0x0000 for all three, i.e. running the
  // message followed by its own CRC back through the same register returns zero
  // even with a nonzero init -- which is the property crc_check() relies on.
  std::vector<uint8_t> cat(72 + 24);
  std::memcpy(cat.data(), cbits.data(), 72);
  crc_to_bits(g_pg, 24, cat.data() + 72);
  const uint32_t res24 = crc_bitwise(cat.data(), 96, CrcId::kCrc24A, 0xB704CE);
  std::vector<uint8_t> cat16(72 + 16);
  std::memcpy(cat16.data(), cbits.data(), 72);
  crc_to_bits(g_3740, 16, cat16.data() + 72);
  const uint32_t res16 = crc_bitwise(cat16.data(), 88, CrcId::kCrc16, 0xFFFF);
  G.report("catalogue residue 0 for both nonzero-init variants",
           res24 == 0 && res16 == 0,
           fmt("residue(OPENPGP) = 0x%06X, residue(IBM-3740) = 0x%04X", res24, res16));

  // table-driven vs bitwise LFSR, on aligned AND unaligned lengths: the table
  // path finishes a partial byte with single-bit steps, and that branch is only
  // exercised by a length that is not a multiple of 8.
  std::mt19937_64 rng(0xFEC2026u);
  int mism = 0, ntested = 0;
  for (int n = 1; n <= 300; ++n) {
    Bits b = random_bits(rng, n);
    for (CrcId id : {CrcId::kCrc24A, CrcId::kCrc16}) {
      for (uint32_t init : {0u, 1u, 0xFFFFu, 0xB704CEu, 0xDEADBEEFu}) {
        ++ntested;
        if (crc_table(b.data(), n, id, init & crc_mask(id)) !=
            crc_bitwise(b.data(), n, id, init & crc_mask(id))) ++mism;
      }
    }
  }
  G.report("table-driven == bitwise LFSR on every length 1..300", mism == 0,
           fmt("%d comparisons (2 polys x 5 inits x 300 lengths, %d of them "
               "not a multiple of 8), %d mismatches", ntested, 300 - 300 / 8, mism));

  // residual + single-bit detection + the linearity that makes a CRC a CRC
  int res_bad = 0, det_bad = 0, lin_bad = 0;
  for (int trial = 0; trial < 200; ++trial) {
    const int A = 1 + (int)(rng() % 257);
    for (CrcId id : {CrcId::kCrc24A, CrcId::kCrc16}) {
      const int w = crc_width(id);
      Bits msg = random_bits(rng, A);
      std::vector<uint8_t> cw((size_t)(A + w));
      crc_append(msg.data(), A, id, cw.data());
      if (!crc_check(cw.data(), A + w, id)) ++res_bad;
      for (int flip = 0; flip < A + w; ++flip) {       // EVERY single-bit error
        std::vector<uint8_t> bad = cw;
        bad[(size_t)flip] ^= 1u;
        if (crc_check(bad.data(), A + w, id)) ++det_bad;
      }
      // crc(a ^ b) == crc(a) ^ crc(b) ^ crc(0): the register is GF(2)-linear, so
      // the parity of an XOR is the XOR of the parities up to a constant.
      Bits m2 = random_bits(rng, A);
      std::vector<uint8_t> mix((size_t)A);
      for (int i = 0; i < A; ++i) mix[(size_t)i] = msg[(size_t)i] ^ m2[(size_t)i];
      const uint32_t z = crc_compute(std::vector<uint8_t>((size_t)A, 0).data(), A, id);
      if ((crc_compute(msg.data(), A, id) ^ crc_compute(m2.data(), A, id) ^ z) !=
          crc_compute(mix.data(), A, id)) ++lin_bad;
    }
  }
  G.report("crc(msg || parity) == 0 for random messages of length 1..257",
           res_bad == 0, fmt("%d residuals checked, %d nonzero", 400, res_bad));
  G.report("every single-bit error in the codeword is detected", det_bad == 0,
           fmt("%d flipped-bit codewords checked, %d undetected", 400 * 130, det_bad));
  G.report("the CRC is GF(2)-affine in the message", lin_bad == 0,
           fmt("%d triples checked, %d violations", 400, lin_bad));
}

//============================================================================
// PART 1c: interleaver
//============================================================================
void gate_interleaver(Gates& G) {
  printf("[fec_bench] PART 1c  interleavers (exact inverse + S-random spread)\n");
  std::mt19937_64 rng(0x1234u);
  int inv_bad = 0, perm_bad = 0, viol_bad = 0, worst_att = 0, checked = 0;
  int worst_llr = 0;
  for (int n : {1, 2, 3, 4, 5, 17, 64, 100, 163, 183, 236, 244, 512, 1000, 2048}) {
    for (IntKind k : {IntKind::kIdentity, IntKind::kRowColumn, IntKind::kSRandom}) {
      const Interleaver t = make_interleaver(k, n, k == IntKind::kRowColumn ? 12 : 8,
                                             20261026u + (uint64_t)n);
      ++checked;
      if (!is_permutation(t)) ++perm_bad;
      Bits x((size_t)n);
      for (int i = 0; i < n; ++i) x[(size_t)i] = (uint8_t)(rng() & 1u);
      Bits y((size_t)n), z((size_t)n);
      interleave_bits(x.data(), t, y.data());
      deinterleave_bits(y.data(), t, z.data());
      if (x != z) ++inv_bad;
      // the same on fp64 LLRs, bit for bit: the decoder path permutes LLRs, not
      // bits, and a rounding-free copy is only exact if it really is a copy
      std::vector<double> l((size_t)n), ly((size_t)n), lz((size_t)n);
      for (int i = 0; i < n; ++i) l[(size_t)i] = (double)(int64_t)rng() / 7.0;
      interleave_llr(l.data(), t, ly.data());
      deinterleave_llr(ly.data(), t, lz.data());
      if (!std::equal(l.begin(), l.end(), lz.begin())) ++worst_llr;
      if (k == IntKind::kSRandom) {
        worst_att = std::max(worst_att, t.attempts);
        if (spread_violations(t, t.par) != 0) ++viol_bad;
      }
    }
  }
  G.report("deinterleave(interleave(x)) == x exactly", inv_bad == 0 && worst_llr == 0,
           fmt("%d (length, kind) pairs, lengths 1..2048; %d bit-vector and %d "
               "fp64-LLR round trips failed", checked, inv_bad, worst_llr));
  G.report("every permutation is a permutation", perm_bad == 0,
           fmt("%d checked, %d with a repeated or out-of-range index", checked, perm_bad));
  G.report("S-random spread constraint satisfied at the effective S",
           viol_bad == 0,
           fmt("%d violating pairs across all S-random lengths; worst restart count "
               "%d of the %d budget (S is clamped to floor(sqrt(n)/3), see fec.h)",
               viol_bad, worst_att, kSRandomMaxAttempts));
  // Dense scan of the clamp: every length 4..600 with two seeds, i.e. the whole
  // range of frame sizes this layer is used at (npunc is 158..813 in the shipped
  // vectors and 96..240 in the waterfall). This is what backs the sqrt(n)/3 rule
  // in fec.h -- 15 spot lengths would leave most of the range unmeasured.
  {
    int dv = 0, wa = 0, wn = 0, tested = 0;
    for (int n = 4; n <= 600; ++n)
      for (uint64_t sd : {20261026ull, 7ull}) {
        const Interleaver t = make_interleaver(IntKind::kSRandom, n, 8, sd);
        dv += spread_violations(t, t.par);
        ++tested;
        if (!is_permutation(t)) ++perm_bad;
        if (t.attempts > wa) { wa = t.attempts; wn = n; }
      }
    G.report("S-random clamp holds densely: n = 4..600 x 2 seeds", dv == 0,
             fmt("%d interleavers, %d spread violations, worst restart count %d at "
                 "n=%d (budget %d)", tested, dv, wa, wn, kSRandomMaxAttempts));
  }
  // the clamp is load-bearing, not decorative: the requested S=8 is infeasible at
  // n=100 and the ladder would burn its whole budget there.
  const Interleaver t100 = make_interleaver(IntKind::kSRandom, 100, 8, 7u);
  const Interleaver t244 = make_interleaver(IntKind::kSRandom, 244, 8, 7u);
  G.report("effective S is the clamped one and still legal",
           t100.par == srandom_spread(100, 8) && t244.par == srandom_spread(244, 8) &&
           spread_violations(t100, t100.par) == 0 &&
           spread_violations(t244, t244.par) == 0,
           fmt("n=100: S_req 8 -> S_eff %d, min spread %d, %d attempts; "
               "n=244: S_eff %d, min spread %d, %d attempts",
               t100.par, min_spread(t100, t100.par), t100.attempts,
               t244.par, min_spread(t244, t244.par), t244.attempts));
}

//============================================================================
// PART 1d: loopback, puncture erasures, Viterbi vs brute force
//============================================================================
void gate_loopback(Gates& G) {
  printf("[fec_bench] PART 1d  noiseless loopback, puncture erasures, "
         "Viterbi == brute force\n");
  const ConvCode c = kConvK7R12;
  std::mt19937_64 rng(0xBEEFu);
  const double amp = 8.0;
  const int trials = 40;
  int frames = 0, bit_err = 0, blk_err = 0, crc_miss = 0;
  for (CrcId cr : {CrcId::kNone, CrcId::kCrc16, CrcId::kCrc24A}) {
    for (PunctId p : {PunctId::kR12, PunctId::kR23, PunctId::kR34}) {
      for (int L : {1, 7, 96, 233}) {
        const FrameGeom g = make_geom(L, cr, p, c);
        const Interleaver it = make_interleaver(IntKind::kSRandom, g.npunc, 8,
                                                0xC0FFEEu + (uint64_t)L);
        for (int t = 0; t < trials; ++t) {
          Bits info = random_bits(rng, L);
          std::vector<uint8_t> tx((size_t)g.npunc), ih((size_t)L);
          encode_frame(info.data(), g, c, it, tx.data(), nullptr);
          const std::vector<double> llr = bits_to_llr(tx.data(), g.npunc, amp);
          double met; bool ok;
          decode_frame(llr.data(), g, c, it, ih.data(), &met, &ok, nullptr);
          ++frames;
          for (int i = 0; i < L; ++i) bit_err += (ih[(size_t)i] != info[(size_t)i]);
          blk_err += (ih != info);
          if (cr != CrcId::kNone && !ok) ++crc_miss;
        }
      }
    }
  }
  G.report("noiseless loopback: decoded == input on every frame",
           bit_err == 0 && blk_err == 0,
           fmt("%d frames (3 CRCs x 3 rates x L in {1,7,96,233} x %d), BLER = %d/%d,"
               " bit errors = %d, LLR magnitude %.0f",
               frames, trials, blk_err, frames, bit_err, amp));
  G.report("the CRC verdict agrees with the bit comparison on every frame",
           crc_miss == 0, fmt("%d CRC-bearing frames, %d false alarms", frames, crc_miss));

  // Punctured loopback with the erasure positions inspected, not assumed.
  int er_bad = 0, er_count_bad = 0, keep_bad = 0, pblk = 0, pframes = 0;
  for (PunctId p : {PunctId::kR23, PunctId::kR34}) {
    const int L = 96;
    const FrameGeom g = make_geom(L, CrcId::kCrc16, p, c);
    const Interleaver it = make_interleaver(IntKind::kSRandom, g.npunc, 8, 55u);
    int num = 0, den = 0;
    punct_rate(p, &num, &den);
    for (int t = 0; t < 50; ++t) {
      Bits info = random_bits(rng, L);
      std::vector<uint8_t> tx((size_t)g.npunc), coded((size_t)g.ncoded), ih((size_t)L);
      encode_frame(info.data(), g, c, it, tx.data(), coded.data());
      const std::vector<double> llr = bits_to_llr(tx.data(), g.npunc, amp);
      std::vector<double> dep((size_t)g.ncoded);
      double met; bool ok;
      decode_frame(llr.data(), g, c, it, ih.data(), &met, &ok, dep.data());
      ++pframes;
      pblk += (ih != info);
      // erased positions must be EXACTLY 0.0 (== 0.0, not "small"), kept ones must
      // be the deinterleaved LLR bit for bit, and the counts must match the
      // pattern's kept fraction.
      int nzero = 0;
      for (int i = 0; i < g.ncoded; ++i) {
        const int kept = punct_pattern(p)[i % punct_period(p)];
        if (!kept) { ++nzero; if (!(dep[(size_t)i] == 0.0)) ++er_bad; }
        else if (!(std::fabs(dep[(size_t)i]) == amp)) ++er_bad;
      }
      if (nzero != g.ncoded - g.npunc) ++er_count_bad;
      // the surviving stream is kept/period of the mother code's bits: exactly,
      // when the period divides ncoded, and within one partial period otherwise
      const int per = punct_period(p), kept = punct_kept(p);
      const int64_t lhs = (int64_t)g.npunc * per, rhs = (int64_t)g.ncoded * kept;
      keep_bad += (g.ncoded % per == 0) ? (lhs != rhs)
                                        : (std::llabs(lhs - rhs) >= (int64_t)kept * per);
      (void)num; (void)den;
    }
    printf("    rate %d/%d: ncoded %d -> npunc %d (%d erasures per frame)\n",
           num, den, g.ncoded, g.npunc, g.ncoded - g.npunc);
  }
  G.report("punctured loopback at rate 2/3 and 3/4: BLER 0", pblk == 0,
           fmt("%d frames, %d block errors", pframes, pblk));
  G.report("depunctured erasures are exactly 0.0 and kept LLRs are untouched",
           er_bad == 0,
           fmt("%d depunctured positions checked, %d wrong (0.0 compared with ==, "
               "kept with |x| == %.0f)", pframes * 250, er_bad, amp));
  G.report("erasure count and kept fraction match the pattern",
           er_count_bad == 0 && keep_bad == 0,
           fmt("%d count mismatches, %d kept-fraction mismatches "
               "(npunc*period == ncoded*kept, exact when the period divides)",
               er_count_bad, keep_bad));

  // ---- Viterbi vs exhaustive enumeration ------------------------------------
  // The strongest gate here: on random LLRs the Viterbi optimum must equal the
  // minimum over ALL 2^L codewords, not merely decode correctly. Tolerance is
  // fp64 association order only -- Viterbi sums T branch metrics one trellis step
  // at a time, the enumeration sums npunc terms in index order.
  double worst = 0, worst_rel = 0;
  int bf_bad = 0, bf_bit_bad = 0, bf_trials = 0, bf_nodes = 0;
  for (PunctId p : {PunctId::kR12, PunctId::kR23, PunctId::kR34}) {
    for (int L : {8, 8, 10}) {
      for (int t = 0; t < 24; ++t) {
        const FrameGeom g = make_geom(L, CrcId::kNone, p, c);
        const Interleaver id = make_interleaver(IntKind::kIdentity, g.npunc, 0, 0);
        std::vector<double> llr((size_t)g.npunc);
        const double scale = 0.25 + 4.0 * (double)(rng() % 1000) / 1000.0;
        for (int i = 0; i < g.npunc; ++i) {
          const double u1 = ((double)(rng() >> 11) + 1.0) / 9007199254740993.0;
          const double u2 = ((double)(rng() >> 11) + 1.0) / 9007199254740993.0;
          llr[(size_t)i] = scale * std::sqrt(-2.0 * std::log(u1)) *
                           std::cos(6.283185307179586 * u2);   // Box-Muller N(0,scale^2)
        }
        std::vector<double> dep((size_t)g.ncoded);
        depuncture_llr(llr.data(), g.npunc, g.ncoded, p, dep.data());
        const ViterbiOut v = viterbi_decode(dep.data(), g.T, c);
        Bits bf_info((size_t)L);
        const double bfm = brute_force_best(llr.data(), L, c, id, g,
                                            bf_info.data());
        const double d = std::fabs(v.metric - bfm);
        worst = std::max(worst, d);
        worst_rel = std::max(worst_rel, d / std::max(1.0, std::fabs(bfm)));
        bf_bad += !(d <= 1e-12 * std::max(1.0, std::fabs(bfm)));
        bf_bit_bad += !(v.info == bf_info);
        ++bf_trials;
        bf_nodes += (1 << L);
      }
    }
  }
  G.report("Viterbi metric == exhaustive enumeration over all 2^L codewords",
           bf_bad == 0,
           fmt("%d trials (%d codewords enumerated, L in {8,10}, all 3 rates), worst "
               "|dv| = %.3e absolute, %.3e relative, tol 1e-12 relative",
               bf_trials, bf_nodes, worst, worst_rel));
  G.report("Viterbi decoded bits == the enumerated argmax bits", bf_bit_bad == 0,
           fmt("%d trials, %d disagreements (ties broken to the smaller predecessor "
               "state on both sides)", bf_trials, bf_bit_bad));

  // Degenerate metric: every path scores exactly 0, so the answer is decided by
  // the tie-break rule alone. Both this bench and tools/fec_ref.py must land on
  // the all-zero codeword, which is the smallest index in both orderings.
  {
    const int L = 8;
    const FrameGeom g = make_geom(L, CrcId::kNone, PunctId::kR12, c);
    std::vector<double> zero((size_t)g.ncoded, 0.0);
    const ViterbiOut v = viterbi_decode(zero.data(), g.T, c);
    int nz = 0;
    for (int i = 0; i < L; ++i) nz += v.info[(size_t)i];
    G.report("all-zero LLR: metric 0 and the tie-break picks the all-zero path",
             v.ok && v.metric == 0.0 && nz == 0,
             fmt("metric = %.1f, %d of %d decoded bits set (degenerate: all 2^%d "
                 "paths tie at 0)", v.metric, nz, L, L));
  }
}

//============================================================================
// PART 2: numpy oracle vs C++, bit exact
//============================================================================
void gate_vectors(Gates& G, hpocl::VecReader& rd) {
  printf("[fec_bench] PART 2   numpy oracle vs C++ on build/fec_vectors.bin\n");
  const ConvCode c = kConvK7R12;
  const int ncase = rd.i();
  double worst_met = 0, worst_bf = 0;
  int perm_bad = 0, coded_bad = 0, tx_bad = 0, dec_bad = 0, crc_bad = 0,
      met_bad = 0, att_bad = 0, bfm_bad = 0, bfb_bad = 0;
  for (int ci = 0; ci < ncase; ++ci) {
    const int L = rd.i(), cid = rd.i(), pid = rd.i(), kind = rd.i(),
              par_req = rd.i(), par_eff = rd.i(), att = rd.i(), M = rd.i(),
              ncoded = rd.i(), npunc = rd.i(), has_bf = rd.i();
    int64_t seed = 0;
    rd.raw(&seed, sizeof seed, "int64");
    const double n0 = rd.f(), amp = rd.f(), e_metric = rd.f(), e_bf = rd.f();
    uint8_t e_crc_ok = 0;
    rd.raw(&e_crc_ok, 1, "uint8");
    // Pre-flight, as in runtime/rate_main.cpp: these eleven ints come from the
    // file and size everything below, and every vector file in this repo is
    // argv[1]. A wrong file must fail here, not allocate gigabytes first.
    if (L < 0 || L > 1 << 20 || ncoded < 0 || ncoded > 1 << 21 || npunc < 0 ||
        npunc > ncoded || npunc > 1 << 21) {
      fprintf(stderr, "[fec] case%d: implausible L=%d ncoded=%d npunc=%d -- wrong "
              "vector file? (run: python3 tools/fec_ref.py)\n", ci, L, ncoded, npunc);
      ++G.fails;
      return;
    }
    // info[L] + coded[ncoded] + tx[npunc] + perm[4*npunc] + llr[8*npunc] +
    // dec[L] + bf_info[L]. The first run of this bench asked for one npunc too
    // many and the last case tripped the pre-flight instead of being read.
    const size_t need = (size_t)L * 3 + (size_t)ncoded + 13u * (size_t)npunc;
    if (!rd.take(need, "fec case")) { ++G.fails; return; }
    std::vector<uint8_t> info((size_t)L), coded((size_t)ncoded), tx((size_t)npunc),
        dec((size_t)L), bf_info((size_t)L);
    std::vector<int32_t> perm((size_t)npunc);
    std::vector<double> llr((size_t)npunc);
    rd.by(info); rd.by(coded); rd.by(tx);
    rd.raw(perm.data(), sizeof(int32_t) * (size_t)npunc, "int32[]");
    rd.re(llr); rd.by(dec); rd.by(bf_info);
    if (rd.bad()) { ++G.fails; return; }

    const FrameGeom g = make_geom(L, (CrcId)cid, (PunctId)pid, c);
    const Interleaver it = make_interleaver((IntKind)kind, g.npunc, par_req,
                                            (uint64_t)seed);
    int rnum = 0, rden = 1;
    punct_rate((PunctId)pid, &rnum, &rden);
    // geometry itself is part of the contract
    const bool geom_ok = g.ncoded == ncoded && g.npunc == npunc &&
                         it.par == par_eff && it.attempts == att;
    att_bad += !geom_ok;
    // (1) the permutation, regenerated from the seed rather than read back
    int pd = 0;
    for (int i = 0; i < npunc; ++i) pd += (it.perm[(size_t)i] != perm[(size_t)i]);
    perm_bad += pd;
    // (2) the encoder, from the same information bits
    std::vector<uint8_t> mycoded((size_t)ncoded), mytx((size_t)npunc);
    encode_frame(info.data(), g, c, it, mytx.data(), mycoded.data());
    int cd = 0, td = 0;
    for (int i = 0; i < ncoded; ++i) cd += (mycoded[(size_t)i] != coded[(size_t)i]);
    for (int i = 0; i < npunc; ++i) td += (mytx[(size_t)i] != tx[(size_t)i]);
    coded_bad += cd; tx_bad += td;
    // (3) the decoder, on the oracle's own fp64 LLRs
    std::vector<uint8_t> mydec((size_t)L);
    double met = 0; bool ok = false;
    decode_frame(llr.data(), g, c, it, mydec.data(), &met, &ok, nullptr);
    int dd = 0;
    for (int i = 0; i < L; ++i) dd += (mydec[(size_t)i] != dec[(size_t)i]);
    dec_bad += dd;
    crc_bad += ((ok ? 1 : 0) != (int)e_crc_ok);
    const double dm = std::fabs(met - e_metric);
    worst_met = std::max(worst_met, dm / std::max(1.0, std::fabs(e_metric)));
    met_bad += !(dm <= 1e-12 * std::max(1.0, std::fabs(e_metric)));
    // (4) brute force, recomputed here from the same LLRs
    double bfm = 0; int bd = 0;
    if (has_bf) {
      Bits bfi((size_t)L);
      bfm = brute_force_best(llr.data(), L, c, it, g, bfi.data());
      for (int i = 0; i < L; ++i) bd += (bfi[(size_t)i] != bf_info[(size_t)i]);
      bfb_bad += bd;
      const double db = std::fabs(bfm - e_bf);
      worst_bf = std::max(worst_bf, db / std::max(1.0, std::fabs(e_bf)));
      bfm_bad += !(db <= 1e-12 * std::max(1.0, std::fabs(e_bf)));
    }
    const bool good = geom_ok && pd == 0 && cd == 0 && td == 0 && dd == 0 &&
                      (ok ? 1 : 0) == (int)e_crc_ok &&
                      dm <= 1e-12 * std::max(1.0, std::fabs(e_metric)) && bd == 0 &&
                      bfm_bad == 0;
    // Both strings are materialised BEFORE report() runs. fmt() formats into one
    // static buffer and returns a copy, so passing two of its results as arguments
    // of the same call leaves the first .c_str() pointing at the second's text --
    // argument evaluation order is unspecified, and clang's right-to-left order is
    // the only reason the case label printed correctly at all.
    const std::string nm = fmt("case%-2d%s", ci, good ? "" : " (MISMATCH)");
    const std::string bf = has_bf
        ? fmt(" | bf %d bit diffs, %.3e rel", bd,
              std::fabs(bfm - e_bf) / std::max(1.0, std::fabs(e_bf)))
        : std::string();
    G.report(nm.c_str(), good,
             fmt("L=%-4d crc=%d rate=%d/%d int=%d(S=%d,att %d) M=%-3d amp=%.1f "
                 "n0=%.1e npunc=%-4d | perm %d, coded %d, tx %d, dec %d bit diffs | "
                 "metric %.3e rel | crc_ok %d%s",
                 L, cid, rnum, rden, kind, par_eff, att, M, amp, n0, npunc, pd, cd,
                 td, dd, dm / std::max(1.0, std::fabs(e_metric)), (int)ok,
                 bf.c_str()));
  }
  printf("  [----] %-58s %s\n", "vector-file totals",
         fmt("perm %d, coded %d, tx %d, dec %d, crc %d, geom %d mismatches; worst "
             "metric %.3e rel, worst brute-force metric %.3e rel",
             perm_bad, coded_bad, tx_bad, dec_bad, crc_bad, att_bad, worst_met,
             worst_bf).c_str());
  G.report("numpy vs C++ bit-exact on every case",
           perm_bad == 0 && coded_bad == 0 && tx_bad == 0 && dec_bad == 0 &&
           crc_bad == 0 && att_bad == 0 && met_bad == 0 && bfm_bad == 0 &&
           bfb_bad == 0,
           fmt("%d cases, %d bit-level mismatches in total", ncase,
               perm_bad + coded_bad + tx_bad + dec_bad + bfb_bad));
  if (!rd.expect_done("fec_bench")) ++G.fails;
}

}  // namespace

int main(int argc, char** argv) {
  const double t0 = now_s();
  const char* vec = argc > 1 ? argv[1] : "build/fec_vectors.bin";
  printf("[fec_bench] FEC layer, host-only fp64 (no OpenCL, no device): "
         "ref/fec.cpp vs tools/fec_ref.py vs exhaustive enumeration\n");
  Gates G;
  gate_dfree(G);
  gate_crc(G);
  gate_interleaver(G);
  gate_loopback(G);

  hpocl::VecReader rd(vec);
  if (!rd.ok()) {
    fprintf(stderr, "[fec] %s (run: python3 tools/fec_ref.py)\n", rd.error().c_str());
    return 2;
  }
  gate_vectors(G, rd);

  printf("[fec_bench] %d gates, %d failed, %.2f s wall\n", G.total, G.fails,
         now_s() - t0);
  printf(G.fails ? "[fec_bench] FAIL\n" : "[fec_bench] ALL PASS\n");
  return G.fails ? 1 : 0;
}
