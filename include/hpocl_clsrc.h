#pragma once
//-----------------------------------------------------------------------------
// read_cl_source(): read an OpenCL C file and textually inline its
// #include "x.clh" directives, resolved relative to the including file.
//
// Why this exists: some vendor ICDs (Apple's OpenCL, which is what `make check`
// runs on) ignore -I in clBuildProgram options, so a kernel that includes a
// shared header cannot be compiled from source unless the host resolves the
// include itself. hpocl::Runtime::build() always did this; the local helpers in
// runtime/datapath_main.cpp and runtime/chain_main.cpp did not, which is why
// giving the kernels a shared kernels/cplx.clh broke exactly those two benches.
// One implementation instead of three.
//
// A missing include returns an empty string so the caller fails with "cannot
// read <file>" rather than a confusing device-side compile error.
//-----------------------------------------------------------------------------
#include <fstream>
#include <sstream>
#include <string>

namespace hpocl {

inline std::string read_cl_source(const std::string& path) {
  std::ifstream in(path);
  if (!in) return std::string();
  std::stringstream ss;
  ss << in.rdbuf();
  const std::string src = ss.str(), tag = "#include \"";
  const std::string dir = path.substr(0, path.find_last_of('/') + 1);
  std::string out;
  size_t pos = 0;
  while (true) {
    const size_t b = src.find(tag, pos);
    if (b == std::string::npos) { out += src.substr(pos); break; }
    const size_t e = src.find('"', b + tag.size());
    if (e == std::string::npos) { out += src.substr(pos); break; }   // malformed
    out += src.substr(pos, b - pos);
    const std::string inc = read_cl_source(dir + src.substr(b + tag.size(), e - b - tag.size()));
    if (inc.empty()) return std::string();
    out += inc;
    pos = e + 1;
  }
  return out;
}

}  // namespace hpocl
