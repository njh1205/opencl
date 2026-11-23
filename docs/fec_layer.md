## The FEC layer (`ref/fec.h`, `ref/fec.cpp`, `tools/fec_ref.py`, `tools/fec_bler.py`)

The paper this stack implements (`yang_unified_transceiver`, the MART unified
transceiver) is a **MIMO-BICM** model: information bits are FEC-encoded at a code
rate `R`, interleaved, then grouped into `q`-bit labels of an `M = 2^q` QAM
constellation. Until this layer existed, the repo carried only the *second* half of
that sentence — `kernels/mcs_table.clh` + `kernels/rate.cl` turn a measured GMI into
the highest supportable `(m, R)` pair — and the README said so plainly:

> **This is not a channel codec.** There is no LDPC/polar/turbo encoder or decoder
> anywhere in the stack … If you need real FEC, NVIDIA Sionna has it
> (`sionna.phy.fec`) … neither Sionna nor TensorFlow is installed in this tree, so
> nothing here is cross-checked against it.

That paragraph is now only half true. **There is a channel codec**: CRC-24A and
CRC-16 (3GPP TS 38.212 §5.1), the `(133,171)₈` `K=7` rate-1/2 convolutional encoder
with zero-tail termination, punctured to 2/3 and 3/4, an S-random / row-column
interleaver, and a soft-decision Viterbi decoder. Names and semantics are aligned to
`sionna.phy.fec` so a later swap is mechanical. The second half of the paragraph is
still exactly true: **Sionna is still not installed, nothing was `pip install`ed to
write this, and nothing here has ever been run against Sionna.** The ground truth
used instead is (a) arithmetic this stack can prove about itself and (b) three
externally attested CRC check values. And the layer is **host-only**: no `.cl`, no
PL kernel, no `__constant` budget, no OpenCL device — `make fec_bench` runs where
the ICD does not.

| file | role |
|---|---|
| `ref/fec.h` / `ref/fec.cpp` | the fp64 host reference: CRC, trellis, encoder, puncturer, interleaver, Viterbi, free-distance search |
| `tools/fec_ref.py` | the independent numpy oracle; writes `build/fec_vectors.bin` |
| `runtime/fec_main.cpp` | `fec_bench`: 41 gates, host-only, no OpenCL |
| `tools/fec_bler.py` | the waterfall measurement (NOT a gate); prints BER/BLER/coding gain with error bars |
| `Makefile` | `fec_bench` target, added to `check` and `.PHONY` |

### Mapping to `sionna.phy.fec`

Verified against the `NVlabs/sionna@main` source tree (`src/sionna/phy/fec/__init__.py`,
`conv/__init__.py`, `conv/utils.py`, `crc.py`, `interleaving.py`, and
`src/sionna/phy/utils/{metrics,misc}.py`). "Aligned" means same role and same
input/output convention — **not** bit-compatible output, which nobody has measured.

| this repo | Sionna |
|---|---|
| `fec::crc_append` / `crc_compute` | `CRCEncoder` (`fec/crc.py`) |
| `fec::crc_check` | `CRCDecoder` (`fec/crc.py`) |
| `fec::conv_encode` | `ConvEncoder` (`fec/conv/encoding.py`) |
| `fec::viterbi_decode` | `ViterbiDecoder` (`fec/conv/decoding.py`) |
| `fec::build_trellis` / `TrellisTab` | `Trellis` (`fec/conv/utils.py`) — the state indexing is copied from `Trellis._generate_transitions`: window MSB = current input, `s' = (u << (K-2)) | (s >> 1)` |
| `fec::kConvK7R12` | `polynomial_selector(rate=0.5, constraint_length=7)` → `('1011011','1111001')` = `(133,171)₈`, which Sionna attributes to Moon, *Error Correction Coding* |
| `fec::make_interleaver(kRowColumn,…)` | `RowColumnInterleaver` |
| `fec::make_interleaver(kSRandom,…)` | `RandomInterleaver`, plus the classic S-random spread constraint Sionna's random interleaver does not enforce |
| `fec::deinterleave_*` | `Deinterleaver` |
| `tools/fec_bler.py`'s Eb/N0 → `n0` | `sionna.phy.utils.ebnodb2no` (role only; this file uses this repo's `n0` = `E|noise|²` per complex sample, `Es = 1`) |
| `tools/fec_bler.py`'s BER / BLER counting | `sionna.phy.utils.compute_ber` / `compute_bler` (a block error is any differing element) |
| — deliberately absent — | `BCJRDecoder`, `LDPC5G*`, `Polar5G*`, `Turbo*`, `Scrambler`, `llr2mi` |

Two conventions worth stating because they are the ones a Sionna swap would have to
match: **zero-tail termination** (`K-1` zeros appended, `n_coded = 2·(A + K − 1)`,
decoder terminates on state 0), and the **path metric** `M = Σ_b c_b · llr_b`
**maximised**. Since `ln P(c_b|y) = c_b·llr_b − softplus(llr_b)` and the softplus
term depends on the received sample only, maximising `Σ c_b·llr_b` over zero-tailed
trellis paths *is* the maximum-likelihood codeword under the BICM factorisation
`p(y|c) = Π_b p(y_b|c_b)`. "Max-log" here refers to that factorisation (bit
independence inside a symbol — what the interleaver buys), not to an extra
approximation inside the decoder. Punctured positions carry `llr == 0.0` exactly,
the neutral element of that metric.

### How this `R` relates to `mcs_table.clh`'s `(m, R)`

`kernels/mcs_table.clh` is the 3GPP TS 38.214 Table 5.2.2.1-2 CQI table: index →
`(m, R/1024)`, and `cqi_select` picks the largest entry with `m·R/1024 ≤ GMI`. That
`R` is a **promise about an encoder that the table does not name**. This layer
supplies three concrete encoders and measures what they actually cost:

| code delivered here | `R_eff = L/n_punc` | `m` | `m·R_eff` (bits/channel use) | nearest CQI entry | its `SE = m·R/1024` |
|---|---|---|---|---|---|
| `(133,171)` rate 1/2 | 0.4068 | 2 | 0.8136 | CQI 5 = (2, 449/1024) | 0.8770 |
| rate 2/3 punctured | 0.5444 | 2 | 1.0889 | CQI 6 = (2, 602/1024) | 1.1758 |
| rate 3/4 punctured | 0.6076 | 2 | 1.2152 | CQI 6 = (2, 602/1024) | 1.1758 |
| rate 1/2 | 0.4068 | 4 | 1.6271 | CQI 7 = (4, 378/1024) | 1.4766 |
| rate 2/3 | 0.5444 | 4 | 2.1778 | CQI 9 = (4, 616/1024) | 2.4062 |
| rate 3/4 | 0.6125 | 4 | 2.4500 | CQI 9 = (4, 616/1024) | 2.4062 |
| rate 1/2 | 0.4083 | 6 | 2.4500 | CQI 10 = (6, 466/1024) | 2.7305 |
| rate 2/3 | 0.5444 | 6 | 3.2667 | CQI 11 = (6, 567/1024) | 3.3223 |
| rate 3/4 | 0.6111 | 6 | 3.6667 | CQI 12 = (6, 666/1024) | 3.9023 |

`R_eff` is not exactly `1/2`, `2/3` or `3/4` because a frame is finite: `L`
information bits + 16 CRC bits + 6 tail bits, punctured by a pattern whose period
need not divide `2·(L+22)`. Read the table as: **the CQI table's `R` is the target,
`R_eff` is what a 96–99-bit frame actually achieves, and the two agree to 1–3
percent** — close enough that a `(m, R)` chosen by `cqi_select` can be served by one
of these three encoders, which is the link the paper's model assumes and the repo
did not have. The table is printed by `tools/fec_bler.py` itself (parsed out of
`kernels/mcs_table.clh`, not retyped — one definition of a standard table is the
point of that header).

What is still missing for a real link, and is *not* claimed here: rate matching to
an arbitrary transport-block size, HARQ, an interleaver matched to the QAM labels
(the current one is bit-level and does not do the intra-symbol label protection of
3GPP's bit-interleaver), and any MIMO/fading/THP coupling. `tools/fec_bler.py`
measures **one stream over flat AWGN**; it does not run through
`kernels/datapath.cl`, does not see the precoder, and does not see the THP receive
modulo.

### The gates (`make fec_bench`, 41 of them, all PASS)

Every number below is the bench's own output, not a model of it. `fec_bench` links
neither `runtime/hpocl_runtime.cpp` nor OpenCL and needs no device.

**Part 1a — the code proves itself.** `d_free` is *computed*, three independent ways;
the value 10 is not typed in anywhere.

```
[PASS] d_free == 10, Dijkstra vs layered BFS at two budgets   dijkstra = 10, bounded(40 branches) = 10, bounded(80) = 10, both attained at 7 branches
[PASS] the branch budget is provably not binding              bounded(6) = -1 (misses it, depth_out -1), bounded(7) = 10 at 7 branches
[PASS] impulse response weight == d_free (explains the 7-branch optimum)  input 1000000 + 6 tail bits -> output weight 10, d_free = 10
[PASS] no zero-weight cycle but the trivial loop (not catastrophic)  32 zero-weight branches of 128 (16 of them driven by u=1), 0 states self-reachable over the non-trivial ones, d_free = 10 > 0
```

The three ways are: Dijkstra over the 64-node state graph from `to_nodes[0][1]` to
state 0 (exact, no depth bound needed — a finite graph with non-negative integer
edges); layered Bellman-Ford with an explicit branch budget, run at 40 and 80
branches, both attaining the optimum at 7; and, in the numpy oracle, **exhaustive
enumeration of all 2¹⁶ zero-tailed codewords** through the generator matrix. The
`bounded(6) = -1` line is the one that matters: it shows the budget is not what
produced the answer. And the optimum being attained at 7 branches is explained, not
asserted — the single-impulse input already produces a weight-10 output, so the
shortest possible non-zero input is a `d_free` codeword. The zero-weight-cycle gate
is the hypothesis Dijkstra rests on, checked exhaustively (transitive closure over
the zero-weight edges, all 64 states) rather than sampled.

**Part 1b — CRC.** Three externally attested check values for the ASCII string
`"123456789"`, reproduced from the LFSR; they are the only numbers in this layer
that come from outside the repo, and they are used as assertions, never as inputs.

```
[PASS] check value CRC-16/XMODEM (init 0 = the 3GPP configuration)  0x31C3 vs published 0x31C3
[PASS] check value CRC-16/IBM-3740 (init 0xFFFF)                    0x29B1 vs published 0x29B1
[PASS] check value CRC-24/OPENPGP (init 0xB704CE, same poly as CRC-24A)  0x21CF02 vs published 0x21CF02
[PASS] catalogue residue 0 for both nonzero-init variants           residue(OPENPGP) = 0x000000, residue(IBM-3740) = 0x0000
[PASS] 0x864CFB / 0x1021 expand to the TS 38.212 term lists         CRC-24A 0x864CFB (14 taps), CRC-16 0x1021 (4 taps), init 0x0 both
[PASS] table-driven == bitwise LFSR on every length 1..300          3000 comparisons (2 polys x 5 inits x 300 lengths, 263 of them not a multiple of 8), 0 mismatches
[PASS] crc(msg || parity) == 0 for random messages of length 1..257  400 residuals checked, 0 nonzero
[PASS] every single-bit error in the codeword is detected           52000 flipped-bit codewords checked, 0 undetected
```

Sources: polynomials from 3GPP TS 38.212 §5.1
(`g_CRC24A(D) = D²⁴+D²³+D¹⁸+D¹⁷+D¹⁴+D¹¹+D¹⁰+D⁷+D⁶+D⁵+D⁴+D³+D+1` → `0x864CFB`;
`g_CRC16(D) = D¹⁶+D¹²+D⁵+1` → `0x1021`), with the LFSR started at zero as that
section specifies. Check values from Greg Cook's *Catalogue of parametrised CRC
algorithms* (reveng.sourceforge.io/crc-catalogue), all three entries class
"attested" — CRC-16/XMODEM against the ITU-T V.41 lineage, CRC-16/IBM-3740 against
AUTOSAR R22-11 `SWS_CRCLibrary` §7.2.2.1, CRC-24/OPENPGP against IETF RFC 4880 §6.
**CRC-16/XMODEM is the pin that matters**: its parameters (`poly 0x1021, init 0x0000,
refin/refout false, xorout 0, residue 0x0000`) are exactly the configuration
TS 38.212 §5.1 defines for `g_CRC16`, so the deployed path — not merely a
neighbouring variant — is checked against a published number. The hex↔term-list
identity is re-expanded and asserted in both C++ and numpy, so a mistyped constant
cannot survive as a self-consistent wrong CRC. The oracle additionally cross-checks
its matrix form against its division form on every column, at lengths 1, 7, 8, 37,
100, 257 (i.e. mostly **not** byte-aligned, which is where the table-driven
implementation's partial-byte branch lives).

**Part 1c — interleavers.**

```
[PASS] deinterleave(interleave(x)) == x exactly     45 (length, kind) pairs, lengths 1..2048; 0 bit-vector and 0 fp64-LLR round trips failed
[PASS] every permutation is a permutation           45 checked, 0 with a repeated or out-of-range index
[PASS] S-random clamp holds densely: n = 4..600 x 2 seeds   1194 interleavers, 0 spread violations, worst restart count 18 at n=600 (budget 200)
```

The `sqrt(n)/3` clamp on the requested spread is load-bearing, and the numbers that
show it are in `ref/fec.h`: the largest feasible `S` is 5 at `n=64`, 6 at `n=100`,
8 at `n=163`, 10 at `n=244`, 13 at `n=512`, 15 at `n=1000`. A single-pass greedy
walk asked for `S=8` at `n=100` produced **41 violating pairs** (minimum spread 1,
i.e. no interleaving at all) before the restart ladder existed — that is the bug the
clamp and the ladder fix, and it is why the achieved spread is *measured and printed*
rather than assumed.

**Part 1d — loopback, erasures, and the strongest gate.**

```
[PASS] noiseless loopback: decoded == input on every frame   1440 frames (3 CRCs x 3 rates x L in {1,7,96,233} x 40), BLER = 0/1440, bit errors = 0, LLR magnitude 8
[PASS] punctured loopback at rate 2/3 and 3/4: BLER 0        100 frames, 0 block errors
[PASS] depunctured erasures are exactly 0.0 and kept LLRs are untouched   25000 depunctured positions checked, 0 wrong (0.0 compared with ==, kept with |x| == 8)
[PASS] Viterbi metric == exhaustive enumeration over all 2^L codewords    216 trials (110592 codewords enumerated, L in {8,10}, all 3 rates), worst |dv| = 1.066e-14 absolute, 5.433e-16 relative, tol 1e-12 relative
[PASS] Viterbi decoded bits == the enumerated argmax bits    216 trials, 0 disagreements
[PASS] all-zero LLR: metric 0 and the tie-break picks the all-zero path   metric = 0.0, 0 of 8 decoded bits set (degenerate: all 2^8 paths tie at 0)
```

The enumeration runs through `encode_frame`, so a bug in the trellis tables, the
survivor memory or the traceback cannot hide behind a matching bug in the encoder.
The all-zero-LLR case is the tie-break gate: every one of the 2⁸ paths scores
exactly 0.0, so the answer is decided by the tie rule alone, and both sides must
land on the same codeword. Erasures are compared with `== 0.0`, not `< eps`.

**Part 2 — numpy oracle vs C++, bit-exact.** 12 cases, 6 of them also
brute-force cross-checked, covering `L ∈ {1, 8, 10, 96, 200, 512}`, all three CRC
settings, all three rates, both interleavers, LLRs synthesised directly *and* LLRs
out of a real QPSK/16-QAM/64-QAM logsumexp demodulator, one case deep below the
decoding threshold and one fully degenerate.

```
[PASS] numpy vs C++ bit-exact on every case   12 cases, 0 bit-level mismatches in total
[----] vector-file totals   perm 0, coded 0, tx 0, dec 0, crc 0, geom 0 mismatches; worst metric 2.728e-16 rel, worst brute-force metric 0.000e+00 rel
```

The permutation is **regenerated in C++ from `(kind, n, S, seed)` and compared entry
by entry**, not read back from the file — the splitmix64 stream, the rejection
ladder and the restart count all have to agree, or the comparison would be vacuous.
The bits are compared exactly (not "BER ≈ 0"); the only tolerance in the whole file
is on the fp64 path metric, `1e-12` relative, against a measured worst of
**2.728e-16** (one ulp at `metric ≈ 2.5e5`, from summing in a different association
order). Two bugs in this layer were found by these gates and not by reading: the
row-column interleaver's numpy readout was row-major (the identity permutation —
250 of 252 entries wrong in case 8), and the bench's own pre-flight size asked for
one `npunc` too many, so the last case failed to be read at all.

### Measured waterfall (`tools/fec_bler.py`, 19.7 s)

12 curves (3 modulation orders × {uncoded, 1/2, 2/3, 3/4}), **277 360 frames,
26 875 248 information bits, 12 901 728 symbols, 27 773 block errors**, seed
20261026 (each curve derives its stream from `seed + 1000·(rate index + 1)`, and each
*point* from `(seed, Eb/N0)` so a single row is reproducible without re-running the
sweep that found it). Frame payload `L = 96…99` information bits + CRC-16, chosen so
`n_punc` is a whole number of `m`-bit labels — no padding bit is transmitted and
`R_eff = L/n_punc` is exact. Coded and uncoded frames carry the **same information
payload**, so the BLERs are comparable at equal information content, not merely at
equal symbol count.

`Eb/N0` at `BLER = 1e-2`, and the gain of the coded curve over the uncoded one at
that point (uncoded minus coded, so positive = coding helps):

| `m` | uncoded | rate 1/2 | rate 2/3 | rate 3/4 | gain 1/2 | gain 2/3 | gain 3/4 |
|---|---|---|---|---|---|---|---|
| 2 (QPSK) | 8.480 ± 0.356 | 3.682 ± 0.171 | 4.019 ± 0.157 | 4.491 ± 0.155 | **+4.798 ± 0.395** | +4.461 ± 0.390 | +3.988 ± 0.388 |
| 4 (16-QAM) | 12.197 ± 0.197 | 6.410 ± 0.190 | 7.115 ± 0.108 | 7.491 ± 0.179 | **+5.787 ± 0.274** | +5.082 ± 0.225 | +4.706 ± 0.266 |
| 6 (64-QAM) | 16.371 ± 0.161 | 9.322 ± 0.130 | 10.730 ± 0.479 | 11.341 ± 0.189 | **+7.048 ± 0.207** | +5.641 ± 0.505 | +5.029 ± 0.248 |

All in dB, all at `BLER = 1e-2`, obtained by log-linear interpolation between the two
measured points that bracket the target after a coarse-to-fine search (2 dB probe →
bisection to a ≤ 1 dB bracket → a 5-point final grid spanning bracket ± 1 dB). The
`±` is the endpoints' binomial CI propagated through that interpolation; it does
**not** include the curvature error of assuming `log₁₀(BLER)` linear in `Eb/N0`,
which is why the bracket is narrowed first. The `m=6`, rate-2/3 row has the widest
interval (± 0.479 dB) because its two bracketing points landed close together in
BLER (1.17e-2 at 10.50 dB, 8.30e-3 at 11.00 dB), so the local slope estimate — and
hence the lever arm of the CI — is the noisiest of the twelve.

**Statistical hygiene, printed with every point.** Budget ≤ 6000 frames/point
(probe points ≤ 1200), early stop once 15 block errors are in, batch 512. That makes
the resolution floor `1/6000 = 1.7e-4` and the relative 95 % CI at the early-stop
count `1.96/√15 = 51 %`, i.e. about ± 0.22 dB on a waterfall of slope 1 decade/dB —
quoted in the header of every run so no row can be read as more precise than it is.
Points where zero block errors were seen print `95% < 3/N` (rule of three) and
`FLOOR 1/N`, never `BLER = 0`. Every row also prints frames, information bits,
symbols, block errors *and* bit errors, so the reader can see which counts are too
small to mean anything.

**An independent check of the whole normalisation chain.** Every uncoded row also
prints the *exact* per-bit error rate of Gray-labelled square QAM over flat AWGN,
computed from the per-axis L-PAM decision regions (`pam_gray_ber_exact`, no textbook
approximation anywhere in it). Over the 18 uncoded rows whose bit-error count gives
a relative CI below 15 %, measured/analytic lies in **0.913 … 1.021**, worst
deviation 8.7 % and inside its own error bar. That is what rules out an Es/N0-vs-Eb/N0
factor of two, a wrong `n0/2` split between real dimensions, or a mislabelled
constellation — all of which would leave a plausible-looking waterfall and a
completely wrong gain number. (The same check caught a real bug during development:
the first version hard-decoded `llr < 0` as bit 1, i.e. inverted, and reported
`BER = 1.0` against an analytic `1.8e-7`.)

**Undetected errors.** The CRC verdict is recomputed from the decoded bits and
compared with the decoded parity bits, and any frame that is wrong but still passes
would be printed as `N block errors the CRC MISSED`. Over 277 360 frames and 27 773
block errors: **zero**. A 16-bit CRC on a ~100-bit frame gives an undetected-error
fraction of order 2⁻¹⁶ per erroneous frame, so 27 773 erroneous frames predicts
~0.4 — this run is consistent with that, not evidence of a better CRC.

### What this layer does **not** prove

* **No Sionna cross-check.** Sionna / TensorFlow / PyTorch are not installed here and
  nothing was installed to write this. The mapping table above was read off
  `NVlabs/sionna@main`'s source, and the generator polynomials come from its
  `polynomial_selector`, but no output of this layer has ever been compared against
  Sionna's. A swap still has to be validated.
* **No device, no PL.** There is no `.cl` kernel, no `__constant`, no `Runtime`
  link. The precoder chain (`kernels/datapath.cl`) still ends at GMI → MCS and never
  sees a coded bit; nothing in `mix_precode/` or the scheduling path calls this.
* **Flat AWGN, one stream.** No MIMO, no fading, no THP receive modulo, no
  precoder coupling, no channel estimation error. The waterfall says what the code
  does to a BICM link, not what it does to *this* stack's link.
* **Hard-decision Viterbi only.** No BCJR / log-MAP, so no soft output and no
  iterative receiver; no `llr2mi`-style EXIT analysis.
* **One code family.** No LDPC, no polar, no turbo, no 5G rate matching, no HARQ, no
  interleaver matched to the QAM labels.
* **Puncture patterns are not from a spec.** They are written out explicitly in
  `ref/fec.cpp`; the only claim made about them is the rate they achieve, which
  `punct_rate()` derives by counting and the bench asserts equals 2/3 and 3/4. Their
  *performance* is measured in the waterfall and is not comparable to a specific
  standard's pattern without redoing that measurement.
* **The waterfall is a measurement, not a gate.** `make check` runs `fec_bench`, not
  `fec_bler.py`. A BER is a statistic; a gate has to be a statement that can be
  false. Nothing in `check` depends on the numbers in the table above.

### Running it

```
make fec_bench                 # regenerate the oracle (0.4 s) + 41 gates (0.05 s)
python3 tools/fec_bler.py      # full 12-curve waterfall, 19.7 s on this M4
python3 tools/fec_bler.py --quick --m 4 --rates 2/3
```

`fec_bench` needs no device and no `DEV=`; it also passes clean under
`-fsanitize=address,undefined` (0.30 s, same 41 gates), which is worth knowing
because the vector reader in `runtime/fec_main.cpp` does raw pointer arithmetic over
a byte buffer with sizes read from the file.
