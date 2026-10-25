#pragma once
//-----------------------------------------------------------------------------
// hpocl_modulo.h -- host twin of the centered modulo in kernels/cplx.clh.
//
// production: mix_precode/classical/thp_precoding.centered_modulo_complex
//   fold x onto [-period/2, period/2) with the FLOOR form (not round-half-away),
//   applied per axis. The floor form is what makes the folded receiver match the
//   production datapath exactly at the cell edges.
//
// This used to exist twice on the host (an anonymous lambda inside
// hpref::thp_precoder and modem.cpp's fold_axis) and twice on the device
// (datapath.cl's fold_axis and thp_precoder.cl's mod_tau), under four names and
// two spellings of the same expression. runtime/bench_main.cpp still carries the
// comment from the last time two of those copies disagreed:
//   "a regression check: the fold used to be tau^2 on the kernel"
//
// Templated so the fp64 host reference and the fp32 device path share one
// definition. Bit-exact with both previous host spellings: `period / 2` and
// `0.5 * period` are the same value in binary floating point, and
// `floor(...) * period` equals `period * floor(...)` by commutativity.
//
// fp32 caveat (device): the map is discontinuous at +-period/2, so an input
// within ~1 ulp of an edge can land on the opposite edge -- a full period jump.
// That is inherent to the production definition, not to this implementation.
//-----------------------------------------------------------------------------
#include <cmath>
#include <complex>

namespace hpocl {

template <class T>
inline T cmod_axis(T x, T period) {
  return x - std::floor((x + period / T(2)) / period) * period;
}

template <class T>
inline std::complex<T> cmod(std::complex<T> v, T period) {
  return std::complex<T>(cmod_axis(v.real(), period), cmod_axis(v.imag(), period));
}

}  // namespace hpocl
