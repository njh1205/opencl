// tf_bench: OpenCL transformer predictor vs numpy reference (eval semantics).
#include <cstdio>
#include <cmath>
#include <vector>
#include <fstream>
#include <cstring>
#include "hpocl_runtime.h"
#include "transformer_predictor.h"

using namespace hpocl;

int main(int argc, char** argv) {
  const char* wpath = argc > 1 ? argv[1] : "build/tf_weights.bin";
  const char* pref = argc > 2 ? argv[2] : "cpu";
  Runtime rt;
  if (!rt.init(pref)) { fprintf(stderr, "opencl init failed\n"); return 2; }
  if (!rt.load_program({"kernels/tf_ops.cl"})) return 2;
  printf("[tf_bench] device: %s\n", rt.device_name());

  TransformerPredictor tp;
  if (!tp.load(rt, wpath)) return 2;

  std::ifstream f("build/tf_expected.bin", std::ios::binary);
  if (!f) { fprintf(stderr, "missing build/tf_expected.bin\n"); return 2; }
  int T, P, D, H;
  f.read((char*)&T, 4); f.read((char*)&P, 4); f.read((char*)&D, 4); f.read((char*)&H, 4);
  // Dimensions come from the loaded weights, not from literals: the expected
  // file only has to agree with them, which is checked next.
  const TfDims& dm = tp.dims();
  if (T != dm.T || P != dm.P || D != dm.d_model || H != dm.heads) {
    fprintf(stderr, "[tf_bench] tf_expected.bin (T=%d P=%d D=%d H=%d) disagrees with "
            "the weights (T=%d P=%d d_model=%d heads=%d) -- regenerate both with the "
            "same tool\n", T, P, D, H, dm.T, dm.P, dm.d_model, dm.heads);
    return 2;
  }
  std::vector<float> tokens((size_t)T * dm.token_dim), mask(T),
      config(dm.config_dim), snr(P);
  f.read((char*)tokens.data(), 4 * tokens.size());
  f.read((char*)mask.data(), 4 * T);
  f.read((char*)config.data(), 4 * config.size());
  f.read((char*)snr.data(), 4 * P);
  float tau_ref = 0; std::vector<float> sign_ref(P);
  f.read((char*)&tau_ref, 4);
  f.read((char*)sign_ref.data(), 4 * P);

  float tau = 0; std::vector<float> sign(P);
  if (!tp.forward(rt, tokens.data(), mask.data(), config.data(), snr.data(),
                  &tau, sign.data())) { fprintf(stderr, "forward failed\n"); return 2; }
  double et = fabs(tau - tau_ref), es = 0;
  for (int p = 0; p < P; ++p) es = fmax(es, fabs(sign[p] - sign_ref[p]));
  bool ok = et < 2e-3 && es < 2e-3;
  printf("tau: cl=%.6f ref=%.6f err=%.2e | sign_maxerr=%.2e  %s\n",
         tau, tau_ref, et, es, ok ? "PASS" : "FAIL");
  printf("[tf_bench] d_model=%d heads=%d layers=%d T=%d P=%d norm=%s\n",
         dm.d_model, dm.heads, dm.layers, dm.T, dm.P,
         tp.normalizes() ? "applied" : "none");
  printf(ok ? "[tf_bench] PASS\n" : "[tf_bench] FAIL\n");
  return ok ? 0 : 1;
}
