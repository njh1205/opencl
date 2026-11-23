#pragma once
//-----------------------------------------------------------------------------
// ref/fec.h -- the forward-error-correction layer: host-only, fp64, no deps.
//
// WHY THIS FILE EXISTS
//   yang_unified_transceiver (the MART paper this stack implements the control
//   plane of) is a MIMO-BICM model: information bits are FEC-encoded at a code
//   rate R, interleaved, then grouped into q-bit labels of an M = 2^q QAM
//   constellation. Until now this repo carried only the (m, R) *pair* --
//   kernels/mcs_table.clh + kernels/rate.cl turn a measured GMI into a CQI entry
//   -- and README said so out loud: "This is not a channel codec. There is no
//   LDPC/polar/turbo encoder or decoder anywhere in the stack." This is that
//   missing layer: CRC -> convolutional encoder -> puncturer -> interleaver ->
//   soft-decision Viterbi, i.e. exactly the chain the paper's BICM model assumes
//   sits between "information bits" and "q-bit labels", so that the R in the
//   (m, R) table is now a rate this repo can actually deliver bits at.
//
// WHAT IT IS NOT -- read before quoting any number from here
//   * HOST-ONLY. There is no .cl kernel in this layer, nothing was added to the
//     PL, and no __constant budget was spent. It links no OpenCL and needs no
//     device, so `make fec_bench` runs wherever the compiler does. The precoder
//     chain (kernels/datapath.cl) still emits GMI -> MCS and never sees a coded
//     bit; see docs/fec_layer.md for the boundary.
//   * NOT cross-checked against real Sionna. Neither Sionna nor TensorFlow nor
//     PyTorch is installed in this tree and nothing was installed to write this.
//     The names/semantics below are *aligned* to sionna.phy.fec so a later swap
//     is mechanical, but the ground truth used here is only (a) arithmetic this
//     stack can prove about itself -- d_free, CRC residuals, brute-force
//     optimality, exact inverses -- and (b) two externally attested CRC check
//     values (see SOURCES).
//   * ONE code family: K=7 rate-1/2 convolutional, punctured to 2/3 and 3/4.
//     No LDPC, no polar, no turbo, no 5G rate matching, no HARQ, no soft output
//     (Viterbi is hard-decision; an iterative receiver would need BCJR).
//
// MAPPING TO sionna.phy.fec
//   Names verified against the NVlabs/sionna@main source tree (fetched 2026-10-26:
//   src/sionna/phy/fec/__init__.py, conv/__init__.py, conv/utils.py, crc.py,
//   interleaving.py, utils.py, and src/sionna/phy/utils/{metrics,misc}.py).
//   "Aligned" below means same role and same input/output convention, NOT
//   bit-compatible output -- nothing here has been run against Sionna.
//
//     this file                        sionna.phy.fec / sionna.phy.utils
//     -----------------------------    -----------------------------------------
//     crc_append / crc_compute         CRCEncoder            (fec/crc.py)
//     crc_check                        CRCDecoder            (fec/crc.py)
//     conv_encode                      ConvEncoder           (fec/conv/encoding.py)
//     viterbi_decode                   ViterbiDecoder        (fec/conv/decoding.py)
//     build_trellis / TrellisTab       Trellis               (fec/conv/utils.py)
//     kConvK7R12                       polynomial_selector(rate=0.5,
//                                        constraint_length=7) -> ('1011011','1111001')
//     row_column_interleaver           RowColumnInterleaver  (fec/interleaving.py)
//     srandom_interleaver              RandomInterleaver     (fec/interleaving.py)
//                                        + the classic S-random spread constraint
//     deinterleave_*                   Deinterleaver         (fec/interleaving.py)
//     tools/fec_bler.py Eb/N0 -> n0    ebnodb2no             (phy/utils/misc.py)
//     tools/fec_bler.py BER/BLER       compute_ber/compute_bler (phy/utils/metrics.py)
//     -- deliberately absent --        BCJRDecoder, LDPC5G*, Polar5G*, Turbo*,
//                                        Scrambler, llr2mi
//
// CONVENTIONS (every one of these is gated by runtime/fec_main.cpp)
//   * Bits are one per uint8_t, value 0 or 1 -- the same representation
//     ref/modem.h::qam_map consumes, so a coded stream can be handed straight to
//     the modem with no repacking. Bytes are never assumed: CRC input length is
//     an arbitrary bit count, and the tail is handled by single-bit steps.
//   * Trellis state indexing follows sionna.phy.fec.conv.Trellis exactly, so the
//     two agree on what "state 17" means:
//       window  w = (u_t << (K-1)) | s      (K bits, MSB = the CURRENT input)
//       next    s' = (u_t << (K-2)) | (s >> 1)
//       output  (parity(w & gen[0]), parity(w & gen[1])), emitted in that order
//     so gen[] is a K-bit mask whose MSB is the tap on u_t, i.e. the binary
//     string spelling Sionna's polynomial_selector returns.
//   * Zero-tail (tail-bit) termination: K-1 zeros are appended to the
//     information+CRC block before encoding, so a codeword is a walk that starts
//     AND ends in state 0, and n_coded = 2*(A + K - 1). The decoder therefore
//     terminates on state 0 rather than doing an open-end max over states.
//   * LLR sign is ref/modem.h's and tools/modem_common.py's: llr > 0 favours bit
//     value 1, natural log. Punctured (erased) positions carry llr == 0.0
//     exactly, which is the neutral element of the metric and the standard
//     erasure semantics.
//   * Path metric, MAXIMISED:  M(path) = sum over coded bits b of  c_b * llr_b.
//     With llr_b = ln P(b=1|y) - ln P(b=0|y) we have
//     ln P(c_b|y) = c_b*llr_b - softplus(llr_b), and softplus(llr_b) depends on
//     the received sample only, not on the path, so it is a constant offset in
//     the argmax. Hence maximising sum c_b*llr_b over zero-tailed trellis paths
//     IS the maximum-likelihood codeword under the BICM factorisation
//     p(y|c) = prod_b p(y_b|c_b) -- the "max-log" label applies to that
//     factorisation (bit independence within a symbol, which is what the
//     interleaver buys), not to an extra approximation inside the decoder. A
//     BCJR/log-MAP decoder would refine the per-bit a posteriori probabilities,
//     not this argmax.
//   * Viterbi tie-break: among the two predecessors of a next state, the SMALLER
//     state index wins a metric tie (strict > replaces). tools/fec_ref.py applies
//     the identical rule, which is what makes the numpy-vs-C++ decoded bits
//     bit-exact rather than merely equal-in-distribution. With continuous LLRs a
//     tie has probability zero; the rule exists so the two sides cannot diverge
//     on the degenerate inputs (all-zero LLR, +-A hard LLR) the gates do use.
//
// SOURCES for the standard constants (nothing here is invented)
//   * CRC-24A polynomial: 3GPP TS 38.212 section 5.1,
//       g_CRC24A(D) = D^24+D^23+D^18+D^17+D^14+D^11+D^10+D^7+D^6+D^5+D^4+D^3+D+1
//     which expands to 0x864CFB. fec.cpp asserts that expansion bit for bit, and
//     the same 24-bit polynomial is IETF RFC 4880 section 6.1's CRCPOLY, whose
//     published check value pins it externally (next item).
//   * CRC-16 polynomial: 3GPP TS 38.212 section 5.1,
//       g_CRC16(D) = D^16+D^12+D^5+1  ->  0x1021.
//   * TS 38.212 section 5.1 also fixes the LFSR initial condition ("the shift
//     register is set to zero at the start"), which is why crc_init() is 0 for
//     both and why the residual of a codeword is 0 rather than a magic constant.
//   * THREE externally attested check values for the ASCII string "123456789",
//     used ONLY as assertions (never as an implementation input), each pinning a
//     different property. All three quoted from Greg Cook, "Catalogue of
//     parametrised CRC algorithms" (reveng.sourceforge.io/crc-catalogue), class
//     "attested":
//       CRC-16/XMODEM   width 16, poly 0x1021,   init 0x0000,   refin/refout false,
//                       xorout 0, check 0x31C3, residue 0x0000.
//         <- THE ONE THAT MATTERS: these parameters are exactly the configuration
//            TS 38.212 5.1 defines for g_CRC16 (same polynomial, register starts
//            at zero, unreflected, no final XOR), so the deployed path is pinned
//            against a published number and not merely a neighbouring variant.
//       CRC-16/IBM-3740 (alias CCITT-FALSE) width 16, poly 0x1021, init 0xFFFF,
//                       refin/refout false, xorout 0, check 0x29B1, residue 0x0000.
//         <- pins the nonzero-init path, i.e. the init fold-in; attested against
//            AUTOSAR R22-11 SWS_CRCLibrary 7.2.2.1.
//       CRC-24/OPENPGP  width 24, poly 0x864CFB, init 0xB704CE, refin/refout false,
//                       xorout 0, check 0x21CF02, residue 0x000000.
//         <- pins the 24-bit polynomial, which is RFC 4880 section 6.1's CRCPOLY
//            and the same value TS 38.212's g_CRC24A expands to.
//     runtime/fec_main.cpp reproduces all three from its own LFSR, and also the
//     catalogue's residue = 0 for the two nonzero-init variants, which is the
//     property crc_check() rests on.
//   * Generator polynomials (133,171)_8 = ('1011011','1111001') for K=7,
//     rate 1/2: sionna.phy.fec.conv.polynomial_selector(rate=0.5,
//     constraint_length=7), whose own source attributes the table to Moon,
//     "Error Correction Coding" (Wiley). Same code as the one 802.11a/g and
//     IS-95 use, with the two polynomials swapped in some spelling -- the swap
//     only relabels which output bit is called first, and d_free is unaffected.
//   * Puncture patterns are NOT taken from a spec table. They are written out
//     explicitly in fec.cpp and the ONLY claim made about them is the rate they
//     achieve, which punct_rate() derives from the pattern by counting and the
//     bench asserts equals 2/3 and 3/4. (Keeping the two halves of alternating
//     pairs at 3/4 rather than the same half every time is a deliberate choice:
//     it spreads the erasures over both generators.)
//-----------------------------------------------------------------------------
#include <cstdint>
#include <vector>

namespace hpocl {
namespace fec {

using Bits = std::vector<uint8_t>;

//============================== CRC ==========================================
// TS 38.212 section 5.1 lists six generator polynomials; these are the two this
// layer needs -- CRC-24A for a transport-block-sized payload, CRC-16 for the
// short frames the BLER waterfall runs (a 24-bit CRC on a 96-bit payload would
// spend 20% of the frame on parity and distort the measured coding gain).
enum class CrcId : int { kNone = 0, kCrc24A = 1, kCrc16 = 2 };

int      crc_width(CrcId id);        // 24 / 16 / 0
uint32_t crc_poly(CrcId id);         // 0x864CFB / 0x1021 / 0
uint32_t crc_init(CrcId id);         // 0 for both (TS 38.212: register starts at 0)
uint32_t crc_mask(CrcId id);         // (1 << width) - 1

// Two independent implementations of the same LFSR. crc_bitwise shifts one bit at
// a time and is the definition; crc_table consumes 8 bits at a time through a
// 256-entry table and finishes a non-multiple-of-8 length with single-bit steps,
// so the two can be compared on ANY length, not just byte-aligned ones.
uint32_t crc_bitwise(const uint8_t* b, int n, CrcId id, uint32_t init);
uint32_t crc_table  (const uint8_t* b, int n, CrcId id, uint32_t init);
uint32_t crc_compute(const uint8_t* b, int n, CrcId id);   // crc_table, init 0

void crc_to_bits(uint32_t v, int width, uint8_t* out);     // MSB (p_0) first
// out = msg[0..A) ++ crc bits; out has A + crc_width(id) entries.
void crc_append(const uint8_t* msg, int A, CrcId id, uint8_t* out);
// Residual test over a whole codeword: crc of (msg ++ parity) == 0 for init 0.
bool crc_check(const uint8_t* codeword, int n, CrcId id);

//=========================== interleaver =====================================
// Deterministic by construction: RowColumn needs no randomness, S-random is a
// seeded greedy search over a fully specified 64-bit stream (splitmix64), so
// (kind, n, par, seed) pins the permutation exactly and tools/fec_ref.py can
// reproduce it bit for bit in numpy.
enum class IntKind : int { kIdentity = 0, kRowColumn = 1, kSRandom = 2 };

struct Interleaver {
  int n = 0;                                  // length in bits
  IntKind kind = IntKind::kIdentity;
  int par = 0;                                // RowColumn: columns; SRandom: the
                                              //   EFFECTIVE S (srandom_spread())
  uint64_t seed = 0;                          // seed of the attempt that succeeded
  int attempts = 0;                           // restarts used; 1 for every kind
                                              //   that is not a search
  std::vector<int> perm;                      // perm[i] = source index of output i
};
// S-random search budget. attempts > kSRandomMaxAttempts means the target S was
// infeasible at this length and the returned permutation VIOLATES the spread
// constraint -- spread_violations() then reports how many pairs, and the bench
// fails. Both numbers are part of the numpy contract (tools/fec_ref.py asserts
// the same attempt count), so the budget cannot silently differ between sides.
constexpr int kSRandomMaxAttempts = 200;
constexpr int kSRandomMaxDraws = 1000;

// splitmix64 (Sebastiano Vigna's reference stream, public domain): 64-bit state,
// one multiplication-mix per output. Exposed because the numpy oracle has to run
// the SAME stream -- an std::mt19937 or numpy default_rng here would make the two
// sides incomparable without shipping the whole permutation.
uint64_t splitmix64(uint64_t* state);

// Spread actually used for an S-random interleaver of length n when S is asked
// for. The S-random constraint |perm[i]-perm[j]| > S for all |i-j| <= S is not
// satisfiable at every (n, S) -- with n <= S+1 it is vacuously impossible, and
// the greedy search starts failing well before that. Measured largest feasible S
// (0 violations within the 200-restart budget, seed 20261026):
//   n     32  48  64  80 100 128 163 183 236 244 320 512 768 1000 1500 2048
//   Smax   4   5   5   6   6   8   8   8  10  10  10  13  13   15   18   16
// floor(sqrt(n)/3) sits below every one of those. Re-measured densely with S
// requested = 8: 1194 (n, S_eff, seed) triples over n = 4..600 with two seeds give
// 0 spread violations and at most 18 restarts (worst at n = 600), in 0.05 s total;
// n = 1000/2048/4096 need 3/6/19 restarts, also 0 violations. So the clamp is the
// difference between an interleaver that silently stops interleaving and one whose
// spread is asserted -- runtime/fec_main.cpp runs that dense scan as a gate.
// Returns 0 for n < 4, meaning "constraint not applicable, identity permutation".
int srandom_spread(int n, int s_requested);

Interleaver make_interleaver(IntKind kind, int n, int par, uint64_t seed);
bool is_permutation(const Interleaver& in);
// min |perm[i]-perm[j]| over 1 <= |i-j| <= s, and how many such pairs violate
// the S-random constraint (|perm[i]-perm[j]| <= s). Both are reported, not just
// asserted, so a too-aggressive S shows up as a number.
int min_spread(const Interleaver& in, int s);
int spread_violations(const Interleaver& in, int s);

void interleave_bits  (const uint8_t* in, const Interleaver& t, uint8_t* out);
void deinterleave_bits(const uint8_t* in, const Interleaver& t, uint8_t* out);
void interleave_llr   (const double* in, const Interleaver& t, double* out);
void deinterleave_llr (const double* in, const Interleaver& t, double* out);

//========================= convolutional code ================================
struct ConvCode {
  int k = 7;                 // constraint length (register span, incl. u_t)
  unsigned gen[2] = {0, 0};  // K-bit tap masks, MSB = tap on the current input
  int nout() const { return 2; }
};
// (133,171)_8 -> ('1011011','1111001') -> 0x5B, 0x79. See SOURCES.
extern const ConvCode kConvK7R12;

// Explicit puncture patterns over the encoder's output stream (c1,c2,c1,c2,...),
// applied cyclically from index 0. 1 = keep, 0 = erase (decoder sees llr = 0).
enum class PunctId : int { kR12 = 0, kR23 = 1, kR34 = 2 };
int punct_period(PunctId p);
const uint8_t* punct_pattern(PunctId p);
int punct_kept(PunctId p);                       // ones in one period
int punct_out_len(int ncoded, PunctId p);        // kept positions in [0, ncoded)
// Effective rate as a reduced fraction: (1/2) / (kept/period). Derived from the
// pattern by counting, NOT typed in -- that is the whole point of this function.
void punct_rate(PunctId p, int* num, int* den);
void puncture(const uint8_t* coded, int ncoded, PunctId p, uint8_t* out);
// out must hold ncoded doubles; erased positions are written as exactly 0.0.
void depuncture_llr(const double* punc, int npunc, int ncoded, PunctId p, double* out);

// Sionna-shaped trellis tables (see CONVENTIONS for the state indexing).
struct TrellisTab {
  int k = 0, ns = 0;
  std::vector<int>     to;     // [ns*2]  (s,u) -> next state
  std::vector<uint8_t> out;    // [ns*2*2] (s,u) -> the two output bits
  std::vector<int>     prev;   // [ns*2]  (next state, slot) -> predecessor state
  std::vector<int>     pin;    // [ns*2]  (next state, slot) -> input bit u
  int next_state(int s, int u) const { return to[(size_t)s * 2 + (size_t)u]; }
  int out_bit(int s, int u, int j) const {
    return out[((size_t)s * 2 + (size_t)u) * 2 + (size_t)j];
  }
  // slot 0 is the smaller predecessor, which is the tie-break winner.
  int pred(int ns, int slot) const { return prev[(size_t)ns * 2 + (size_t)slot]; }
  int pred_input(int ns) const { return pin[(size_t)ns * 2]; }
};
TrellisTab build_trellis(const ConvCode& c);

// Encode A bits with a zero tail: out holds 2*(A + k - 1) bits.
int  conv_coded_len(int A, const ConvCode& c);
void conv_encode(const uint8_t* a, int A, const ConvCode& c, uint8_t* out);

// Free distance of the zero-tailed code, computed here rather than quoted.
//   free_distance_dijkstra: exact. Dijkstra over the 64-node state graph from
//     to_nodes[0][1] (the only nonzero first branch out of state 0, up to time
//     invariance) to state 0. No depth bound is needed because the graph is
//     finite and every edge weight is a nonnegative integer; a zero-weight cycle
//     would mean the encoder is catastrophic, and the bench prints the smallest
//     nonzero branch weight to show it is not.
//   free_distance_bounded: layered Bellman-Ford over walks of AT MOST
//     branches_max branches (the forced nonzero first branch counts as one), i.e.
//     a second, independent computation of the same number. Running it at
//     branches_max and at branches_max-1 is what turns "the bound was not the
//     binding constraint" from an assumption into a measurement: the smaller
//     budget must miss the optimum. depth_out receives the branch count at which
//     the optimum was attained (-1 if it never was).
int free_distance_dijkstra(const ConvCode& c);
int free_distance_bounded(const ConvCode& c, int branches_max, int* depth_out);

//=============================== Viterbi =====================================
struct ViterbiOut {
  Bits info;              // A bits (information + CRC), tail bits removed
  double metric = 0.0;    // max over zero-tailed paths of sum_b c_b * llr_b
  bool ok = false;        // false if the final state 0 was unreachable
};
// llr_dep holds 2*T depunctured LLRs in trellis order; T = A + k - 1.
ViterbiOut viterbi_decode(const double* llr_dep, int T, const ConvCode& c);

//=========================== whole-frame helpers =============================
// One frame geometry, spelled out once so the encoder, the puncturer, the
// interleaver and the decoder cannot disagree about a length.
struct FrameGeom {
  int L = 0;        // information bits
  CrcId crc = CrcId::kNone;
  int crc_w = 0;    // crc_width(crc)
  int A = 0;        // L + crc_w, the bits the convolutional encoder sees
  int T = 0;        // trellis steps = A + k - 1
  int ncoded = 0;   // 2*T, before puncturing
  PunctId punct = PunctId::kR12;
  int npunc = 0;    // after puncturing; this is what goes to the modem
  int k = 7;        // constraint length of the code in use
};
FrameGeom make_geom(int L, CrcId crc, PunctId p, const ConvCode& c);

// encode_frame: info[L] -> crc -> conv(+zero tail) -> puncture -> interleave.
//   out has g.npunc entries. `coded` (optional, may be null) receives the
//   UNPUNCTURED 2*T-bit encoder output, which is what the cross-check compares.
void encode_frame(const uint8_t* info, const FrameGeom& g, const ConvCode& c,
                  const Interleaver& in, uint8_t* out, uint8_t* coded);
// decode_frame: interleaved LLRs[g.npunc] -> deinterleave -> depuncture ->
//   Viterbi -> CRC check. info_hat[L]; *metric and *crc_ok may be null.
//   llr_dep (optional) receives the depunctured array so a caller can assert the
//   erasure positions are exactly 0.0.
void decode_frame(const double* llr_int, const FrameGeom& g, const ConvCode& c,
                  const Interleaver& in, uint8_t* info_hat, double* metric,
                  bool* crc_ok, double* llr_dep);
}  // namespace fec
}  // namespace hpocl
