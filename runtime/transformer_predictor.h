#pragma once
//-----------------------------------------------------------------------------
// TransformerPredictor: OpenCL inference graph for
// SpectralSetTransformerThresholdRegressor (see kernels/tf_ops.cl).
// Weight binary layout is defined by tools/export_transformer.py (PyTorch
// checkpoint -> flat fp32) and tools/make_random_tf_weights.py (validation).
//-----------------------------------------------------------------------------
#include <string>
#include <vector>
#include "hpocl_runtime.h"

namespace hpocl {

// Device capacity: kernels/tf_ops.cl sizes its attention/pooling scratch by
// token count (TF_TMAX). Kept here as well so the host can reject an oversized
// header instead of letting the kernel write out of bounds.
constexpr int kTfTMax = 64;
constexpr int kTfHeadsMax = 64;

// Normalization the trained model expects: it consumes (x - mean)/std and emits
// a *normalized* tau, so a deployed graph that only carries weights cannot
// reproduce training semantics. Schema-2 binaries carry these; forward() applies
// them, and normalizes() reports whether it did.
struct TfNorm {
  std::vector<float> feature_mean, feature_std;   // [token_dim]
  std::vector<float> config_mean, config_std;     // [config_dim]
  float target_mean = 0.0f, target_std = 1.0f;
};

struct TfDims {
  int token_dim = 5, config_dim = 10, d_model = 32, heads = 4, layers = 2;
  int T = 32, P = 4;
};

class TransformerPredictor {
 public:
  ~TransformerPredictor();
  bool load(Runtime& rt, const std::string& weights_path);
  // tokens [T][token_dim], mask [T] (1 valid), config [config_dim],
  // snr [P] probe points -> tau (dB), sign_logits [P]
  // tokens [T][token_dim] and config [config_dim] are RAW (unnormalized) when
  // the binary carries a norm block: forward() normalizes them, and returns tau
  // in dB (denormalized). Masked-out token rows are zeroed after normalization,
  // matching tools/transformer_ref.py.
  bool forward(Runtime& rt, const float* tokens, const float* mask,
               const float* config, const float* snr,
               float* tau, float* sign_logits);
  const TfDims& dims() const { return d_; }
  bool normalizes() const { return has_norm_; }
  const TfNorm& norm() const { return norm_; }

 private:
  cl_mem w(size_t idx) const { return wb_[idx]; }
  cl_mem tmp(Runtime& rt, size_t floats);
  void free_tmps();

  TfDims d_;
  TfNorm norm_;
  bool has_norm_ = false;
  std::vector<cl_mem> wb_;      // persistent weight buffers
  std::vector<cl_mem> tmps_;    // per-forward activation buffers
  // Kernel handles are owned by this instance (not process-static): a static
  // cache silently reuses another context's kernels when a second predictor is
  // bound to a different device, which is exactly the teacher/student
  // shadow-mode setup the README advertises.
  std::vector<cl_kernel> k_;
  bool loaded_ = false;
};

}  // namespace hpocl
