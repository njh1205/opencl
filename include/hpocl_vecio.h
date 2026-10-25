#pragma once
//-----------------------------------------------------------------------------
// VecReader: the one bounds-checked reader for the tools/*_ref.py vector files.
//
// Every bench consumed a hand-written copy of this struct, and four of the five
// never compared the read offset against the buffer size:
//   runtime/{modem,rx,pipeline,frontend}_main.cpp  memcpy(v.data(), buf+o, 16*n)
//   runtime/cf_main.cpp                            kept the file length in a
//                                                  member that was never read
// The first field of every vector file is an int32 count followed by the case
// dimensions, and those dimensions size the std::vectors that follow. Passing
// the wrong file (all of them are argv[1], e.g. `modem_bench build/rx_vectors.bin`)
// therefore read a garbage dimension and either allocated gigabytes or walked
// off the end of the buffer -- a segfault, or worse, a plausible-looking FAIL
// that sends you debugging numerics instead of your command line.
//
// runtime/datapath_main.cpp already had the correct version; this is it, shared.
// The on-disk format is unchanged, so tools/*_ref.py need no edits.
//-----------------------------------------------------------------------------
#include <complex>
#include <cstring>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

namespace hpocl {

class VecReader {
 public:
  using dcplx = std::complex<double>;

  explicit VecReader(const char* path) : path_(path ? path : "") {
    std::ifstream f(path_, std::ios::binary);
    if (!f) { bad_ = true; err_ = "cannot open " + path_; return; }
    buf_.assign((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    if (buf_.empty()) { bad_ = true; err_ = "empty file " + path_; }
  }

  bool ok() const { return !bad_; }
  bool bad() const { return bad_; }
  const std::string& error() const { return err_; }
  size_t remaining() const { return buf_.size() - off_; }
  bool done() const { return remaining() == 0; }

  // Report a truncated/mismatched file once, with the offset, and make every
  // later read a no-op so a caller cannot cascade into undefined behaviour.
  bool fail(const char* what, size_t need) {
    if (!bad_) {
      char msg[256];
      std::snprintf(msg, sizeof(msg), "%s: %s needs %zu bytes at offset %zu, file is %zu bytes",
                    path_.c_str(), what, need, off_, buf_.size());
      err_ = msg;
      bad_ = true;
      std::fprintf(stderr, "[vecio] %s\n", msg);
    }
    return false;
  }

  bool take(size_t n, const char* what) {
    if (bad_) return false;
    if (off_ + n > buf_.size()) return fail(what, n);
    return true;
  }

  int i() {
    int32_t v = 0;
    if (take(sizeof v, "int32")) { std::memcpy(&v, buf_.data() + off_, sizeof v); off_ += sizeof v; }
    return v;
  }
  double f() {
    double v = 0;
    if (take(sizeof v, "float64")) { std::memcpy(&v, buf_.data() + off_, sizeof v); off_ += sizeof v; }
    return v;
  }
  void re(std::vector<double>& v) { raw(v.data(), sizeof(double) * v.size(), "float64[]"); }
  void by(std::vector<uint8_t>& v) { raw(v.data(), v.size(), "uint8[]"); }
  void cx(std::vector<dcplx>& v) { raw(v.data(), sizeof(dcplx) * v.size(), "complex128[]"); }
  void raw(void* dst, size_t n, const char* what) {
    if (n == 0) return;
    if (!take(n, what) || !dst) return;
    std::memcpy(dst, buf_.data() + off_, n);
    off_ += n;
  }

  // Trailing bytes mean tools/*_ref.py wrote a field this bench does not read,
  // i.e. the two sides of the binary contract have drifted apart.
  bool expect_done(const char* who) const {
    if (bad_) return false;
    if (remaining() != 0) {
      std::fprintf(stderr, "[%s] %s: %zu trailing bytes -- the field order in the "
                   "tools/*_ref.py writer and this reader disagree\n", who, path_.c_str(),
                   remaining());
      return false;
    }
    return true;
  }

 private:
  std::vector<char> buf_;
  size_t off_ = 0;
  bool bad_ = false;
  std::string err_, path_;
};

}  // namespace hpocl
