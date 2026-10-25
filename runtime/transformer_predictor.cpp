#include "transformer_predictor.h"
#include <cstdio>
#include <cstring>
#include <cmath>
#include <fstream>

namespace hpocl {

// weight buffer indices, in file order
enum WB { TP_W1, TP_B1, TP_W2, TP_B2, CP_W1, CP_B1, CP_W2, CP_B2, LAYER_BASE };
enum TFK { K_DENSE, K_GELU, K_ADDB, K_MULM, K_LN, K_ATTN, K_POOL, K_PACK, K_ADDE, K_N };
static constexpr int kLayerTensors = 10;   // qW qb kW kb vW vb oW ob ln_g ln_b
static constexpr int kAfterLayers = 10;    // pool_w pool_b sh_W sh_b th_W th_b
                                           // sg_W1 sg_b1 sg_W2 sg_b2

TransformerPredictor::~TransformerPredictor() {
  for (auto b : wb_) clReleaseMemObject(b);
  free_tmps();
}

void TransformerPredictor::free_tmps() {
  for (auto b : tmps_) clReleaseMemObject(b);
  tmps_.clear();
}

cl_mem TransformerPredictor::tmp(Runtime& rt, size_t floats) {
  cl_int err = CL_SUCCESS;
  cl_mem b = clCreateBuffer(rt.context(), CL_MEM_READ_WRITE,
                            sizeof(float) * floats, nullptr, &err);
  tmps_.push_back(b);
  return b;
}

// Kernel names in TFK order; resolved once in load() so a missing operator is
// reported there instead of surfacing as a silent all-zero forward().
static const char* const kTfKernelNames[K_N] = {
    "tf_dense", "tf_gelu", "tf_add_broadcast", "tf_mul_mask", "tf_layernorm",
    "tf_attn", "tf_pool", "tf_sign_pack", "tf_add_elem"};

bool TransformerPredictor::load(Runtime& rt, const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) { fprintf(stderr, "[tf] missing weights %s\n", path.c_str()); return false; }

  // Two on-disk schemas, disambiguated by magic (see tools/ckpt_io.py):
  //   0x54464857 v1: 8-int header, weights only, no normalization
  //   0x54464858 v2: 10-int header, weights, then the normalization block
  int32_t magic = 0;
  f.read((char*)&magic, sizeof(magic));
  if (magic == 0x54464858) {
    int32_t h[9];
    f.read((char*)h, sizeof(h));
    if (h[0] != 2) { fprintf(stderr, "[tf] unsupported schema %d\n", h[0]); return false; }
    d_.token_dim = h[1]; d_.config_dim = h[2]; d_.d_model = h[3]; d_.heads = h[4];
    d_.layers = h[5]; d_.T = h[6]; d_.P = h[7]; has_norm_ = h[8] != 0;
  } else if (magic == 0x54464857) {
    int32_t h[7];
    f.read((char*)h, sizeof(h));
    d_.token_dim = h[0]; d_.config_dim = h[1]; d_.d_model = h[2]; d_.heads = h[3];
    d_.layers = h[4]; d_.T = h[5]; d_.P = h[6]; has_norm_ = false;
  } else {
    fprintf(stderr, "[tf] bad magic 0x%08x in %s (expected 0x54464857/0x54464858; "
            "regenerate with tools/export_transformer.py)\n", magic, path.c_str());
    return false;
  }
  if (!f) { fprintf(stderr, "[tf] truncated header in %s\n", path.c_str()); return false; }

  // The header comes from a file: validate it before it sizes any buffer or
  // reaches a kernel whose scratch arrays are fixed at kTfTMax.
  if (!(d_.token_dim > 0 && d_.config_dim > 0 && d_.d_model > 0 && d_.layers >= 0 &&
        d_.heads > 0 && d_.heads <= kTfHeadsMax && d_.d_model % d_.heads == 0 &&
        d_.T > 0 && d_.T <= kTfTMax && d_.P > 0)) {
    fprintf(stderr, "[tf] header dims out of range in %s: token_dim=%d config_dim=%d "
            "d_model=%d heads=%d layers=%d T=%d P=%d (need heads>0, heads|d_model, "
            "heads<=%d, 0<T<=%d, P>0)\n", path.c_str(), d_.token_dim, d_.config_dim,
            d_.d_model, d_.heads, d_.layers, d_.T, d_.P, kTfHeadsMax, kTfTMax);
    return false;
  }
  const size_t D = (size_t)d_.d_model;

  std::vector<size_t> sizes = {D * d_.token_dim, D, D * D, D,
                               D * d_.config_dim, D, D * D, D};
  for (int l = 0; l < d_.layers; ++l)
    for (int i = 0; i < kLayerTensors; ++i)
      sizes.push_back((i % 2 == 1 || i == 7 || i == 8 || i == 9) ? D : D * D);
  // after layers: pool_w[D] pool_b[1] sh_W[D*D] sh_b[D] th_W[D] th_b[1]
  //               sg_W1[D*(D+3)] sg_b1[D] sg_W2[D] sg_b2[1]
  sizes.insert(sizes.end(), {D, 1, D * D, D, D, 1, D * (D + 3), D, D, 1});

  for (auto b : wb_) clReleaseMemObject(b);
  wb_.clear();
  for (size_t i = 0; i < sizes.size(); ++i) {
    const size_t n = sizes[i];
    std::vector<float> v(n);
    f.read((char*)v.data(), sizeof(float) * n);
    if (!f) { fprintf(stderr, "[tf] truncated weights\n"); return false; }
    cl_int err = CL_SUCCESS;
    cl_mem b = clCreateBuffer(rt.context(), CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                              sizeof(float) * n, v.data(), &err);
    if (err != CL_SUCCESS) {
      fprintf(stderr, "[tf] clCreateBuffer for weight %zu failed: %d\n", i, err);
      return false;
    }
    wb_.push_back(b);
  }

  if (has_norm_) {
    auto rdv = [&](std::vector<float>& v, size_t n) {
      v.assign(n, 0.0f);
      f.read((char*)v.data(), sizeof(float) * n);
    };
    rdv(norm_.feature_mean, (size_t)d_.token_dim);
    rdv(norm_.feature_std, (size_t)d_.token_dim);
    rdv(norm_.config_mean, (size_t)d_.config_dim);
    rdv(norm_.config_std, (size_t)d_.config_dim);
    float tm[2] = {0.0f, 1.0f};
    f.read((char*)tm, sizeof(tm));
    norm_.target_mean = tm[0]; norm_.target_std = tm[1];
    for (size_t j = 0; j < norm_.feature_std.size(); ++j)
      if (norm_.feature_std[j] == 0.0f) {
        fprintf(stderr, "[tf] feature_std[%zu] == 0 in %s\n", j, path.c_str()); return false;
      }
    for (size_t j = 0; j < norm_.config_std.size(); ++j)
      if (norm_.config_std[j] == 0.0f) {
        fprintf(stderr, "[tf] config_std[%zu] == 0 in %s\n", j, path.c_str()); return false;
      }
    if (norm_.target_std == 0.0f) {
      fprintf(stderr, "[tf] target_std == 0 in %s\n", path.c_str()); return false;
    }
  }
  if (!f) { fprintf(stderr, "[tf] truncated weights/norm in %s\n", path.c_str()); return false; }
  // Layout drift guard: tensor order lives in tools/ckpt_io.py::tensor_order and
  // is mirrored by the `sizes` list above. Extra bytes mean the two disagreed.
  char extra = 0;
  if (f.read(&extra, 1)) {
    fprintf(stderr, "[tf] %s has trailing bytes after %zu tensors%s -- the layout in "
            "tools/ckpt_io.py and runtime/transformer_predictor.cpp have drifted\n",
            path.c_str(), sizes.size(), has_norm_ ? " + norm block" : "");
    return false;
  }

  // Kernel handles are borrowed from the Runtime registry (owned and released
  // there), so this instance must not clReleaseKernel them.
  k_.assign(K_N, nullptr);
  for (int i = 0; i < K_N; ++i) {
    k_[i] = rt.kernel(kTfKernelNames[i]);
    if (!k_[i]) {
      fprintf(stderr, "[tf] program has no kernel '%s' -- load kernels/tf_ops.cl "
              "before TransformerPredictor::load\n", kTfKernelNames[i]);
      return false;
    }
  }
  loaded_ = true;
  return true;
}

bool TransformerPredictor::forward(Runtime& rt, const float* tokens,
                                   const float* mask, const float* config,
                                   const float* snr, float* tau,
                                   float* sign_logits) {
  if (!loaded_) { fprintf(stderr, "[tf] forward before a successful load\n"); return false; }
  free_tmps();
  const int T = d_.T, D = d_.d_model, H = d_.heads, P = d_.P;
  const int dh = D / H;
  const float sc_scale = 1.0f / std::sqrt((float)dh);
  cl_context cx = rt.context();
  cl_command_queue q = rt.queue();
  // Every OpenCL call below folds its status into `worst`: a failed enqueue used
  // to be invisible, so forward() returned true with zero-filled outputs.
  cl_int worst = CL_SUCCESS;

  // Schema-2 binaries carry the training normalization: the graph consumes
  // (x - mean)/std with masked token rows zeroed, and emits a normalized tau.
  std::vector<float> tokN, cfgN;
  const float* tokIn = tokens;
  const float* cfgIn = config;
  if (has_norm_) {
    tokN.resize((size_t)T * d_.token_dim);
    for (int t = 0; t < T; ++t) {
      const bool valid = mask[t] != 0.0f;
      for (int j = 0; j < d_.token_dim; ++j) {
        const float x = (tokens[(size_t)t * d_.token_dim + j] - norm_.feature_mean[j])
                        / norm_.feature_std[j];
        tokN[(size_t)t * d_.token_dim + j] = valid ? x : 0.0f;
      }
    }
    cfgN.resize(d_.config_dim);
    for (int j = 0; j < d_.config_dim; ++j)
      cfgN[j] = (config[j] - norm_.config_mean[j]) / norm_.config_std[j];
    tokIn = tokN.data();
    cfgIn = cfgN.data();
  }

  auto up = [&](const void* h, size_t n) {
    cl_int e = CL_SUCCESS;
    cl_mem b = clCreateBuffer(cx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                              sizeof(float) * n, const_cast<void*>(h), &e);
    if (e != CL_SUCCESS) {
      fprintf(stderr, "[tf] clCreateBuffer(%zu floats) failed: %d\n", n, e);
      worst |= e;
    }
    tmps_.push_back(b);
    return b;
  };
  auto rd = [&](cl_mem b, void* h, size_t n) {
    worst |= clEnqueueReadBuffer(q, b, CL_TRUE, 0, sizeof(float) * n, h, 0, nullptr, nullptr);
  };
  cl_mem bTok = up(tokIn, (size_t)T * d_.token_dim);
  cl_mem bMask = up(mask, T);
  cl_mem bCfg = up(cfgIn, d_.config_dim);
  cl_mem bSnR = up(snr, P);

  auto dense = [&](cl_mem X, int B, int in, int out, cl_mem W, cl_mem bb, cl_mem Y) {
    cl_kernel kk = k_[K_DENSE];
    clSetKernelArg(kk, 0, sizeof(cl_mem), &X);
    clSetKernelArg(kk, 1, sizeof(cl_mem), &W);
    clSetKernelArg(kk, 2, sizeof(cl_mem), &bb);
    clSetKernelArg(kk, 3, sizeof(int), &B);
    clSetKernelArg(kk, 4, sizeof(int), &in);
    clSetKernelArg(kk, 5, sizeof(int), &out);
    clSetKernelArg(kk, 6, sizeof(cl_mem), &Y);
    size_t g[2] = {(size_t)out, (size_t)B};
    worst |= clEnqueueNDRangeKernel(q, kk, 2, nullptr, g, nullptr, 0, nullptr, nullptr);
  };
  auto gelu = [&](cl_mem X, cl_mem Y, int n) {
    cl_kernel kk = k_[K_GELU];
    clSetKernelArg(kk, 0, sizeof(cl_mem), &X);
    clSetKernelArg(kk, 1, sizeof(int), &n);
    clSetKernelArg(kk, 2, sizeof(cl_mem), &Y);
    size_t g[1] = {(size_t)n};
    worst |= clEnqueueNDRangeKernel(q, kk, 1, nullptr, g, nullptr, 0, nullptr, nullptr);
  };
  auto addb = [&](cl_mem A, cl_mem b, cl_mem Y, int nT) {
    cl_kernel kk = k_[K_ADDB];
    clSetKernelArg(kk, 0, sizeof(cl_mem), &A);
    clSetKernelArg(kk, 1, sizeof(cl_mem), &b);
    clSetKernelArg(kk, 2, sizeof(int), &nT);
    clSetKernelArg(kk, 3, sizeof(int), &D);
    clSetKernelArg(kk, 4, sizeof(cl_mem), &Y);
    size_t g[1] = {(size_t)nT * D};
    worst |= clEnqueueNDRangeKernel(q, kk, 1, nullptr, g, nullptr, 0, nullptr, nullptr);
  };
  auto adde = [&](cl_mem A, cl_mem B, cl_mem Y, int n) {
    cl_kernel kk = k_[K_ADDE];
    clSetKernelArg(kk, 0, sizeof(cl_mem), &A);
    clSetKernelArg(kk, 1, sizeof(cl_mem), &B);
    clSetKernelArg(kk, 2, sizeof(int), &n);
    clSetKernelArg(kk, 3, sizeof(cl_mem), &Y);
    size_t g[1] = {(size_t)n};
    worst |= clEnqueueNDRangeKernel(q, kk, 1, nullptr, g, nullptr, 0, nullptr, nullptr);
  };
  auto mulm = [&](cl_mem A, cl_mem Y, int nT) {
    cl_kernel kk = k_[K_MULM];
    clSetKernelArg(kk, 0, sizeof(cl_mem), &A);
    clSetKernelArg(kk, 1, sizeof(cl_mem), &bMask);
    clSetKernelArg(kk, 2, sizeof(int), &nT);
    clSetKernelArg(kk, 3, sizeof(int), &D);
    clSetKernelArg(kk, 4, sizeof(cl_mem), &Y);
    size_t g[1] = {(size_t)nT * D};
    worst |= clEnqueueNDRangeKernel(q, kk, 1, nullptr, g, nullptr, 0, nullptr, nullptr);
  };
  auto lnorm = [&](cl_mem X, cl_mem g, cl_mem bb, cl_mem Y, int nT) {
    cl_kernel kk = k_[K_LN];
    float eps = 1e-5f;
    clSetKernelArg(kk, 0, sizeof(cl_mem), &X);
    clSetKernelArg(kk, 1, sizeof(cl_mem), &g);
    clSetKernelArg(kk, 2, sizeof(cl_mem), &bb);
    clSetKernelArg(kk, 3, sizeof(int), &nT);
    clSetKernelArg(kk, 4, sizeof(int), &D);
    clSetKernelArg(kk, 5, sizeof(float), &eps);
    clSetKernelArg(kk, 6, sizeof(cl_mem), &Y);
    size_t gl[1] = {(size_t)nT};
    worst |= clEnqueueNDRangeKernel(q, kk, 1, nullptr, gl, nullptr, 0, nullptr, nullptr);
  };
  auto attn = [&](cl_mem Q, cl_mem K, cl_mem V, cl_mem Y) {
    cl_kernel kk = k_[K_ATTN];
    clSetKernelArg(kk, 0, sizeof(cl_mem), &Q);
    clSetKernelArg(kk, 1, sizeof(cl_mem), &K);
    clSetKernelArg(kk, 2, sizeof(cl_mem), &V);
    clSetKernelArg(kk, 3, sizeof(cl_mem), &bMask);
    clSetKernelArg(kk, 4, sizeof(int), &T);
    clSetKernelArg(kk, 5, sizeof(int), &D);
    clSetKernelArg(kk, 6, sizeof(int), &H);
    clSetKernelArg(kk, 7, sizeof(float), &sc_scale);
    clSetKernelArg(kk, 8, sizeof(cl_mem), &Y);
    size_t g[2] = {(size_t)T, (size_t)H};
    worst |= clEnqueueNDRangeKernel(q, kk, 2, nullptr, g, nullptr, 0, nullptr, nullptr);
  };

  // token projection: Linear,GELU,Linear
  cl_mem a1 = tmp(rt, (size_t)T * D), a2 = tmp(rt, (size_t)T * D), a3 = tmp(rt, (size_t)T * D);
  dense(bTok, T, d_.token_dim, D, w(TP_W1), w(TP_B1), a1);
  gelu(a1, a2, T * D);
  dense(a2, T, D, D, w(TP_W2), w(TP_B2), a3);
  // config projection
  cl_mem c1 = tmp(rt, D), c2 = tmp(rt, D), c3 = tmp(rt, D);
  dense(bCfg, 1, d_.config_dim, D, w(CP_W1), w(CP_B1), c1);
  gelu(c1, c2, D);
  dense(c2, 1, D, D, w(CP_W2), w(CP_B2), c3);
  // h = token_proj + config_proj, masked
  cl_mem h0 = tmp(rt, (size_t)T * D), h1 = tmp(rt, (size_t)T * D);
  addb(a3, c3, h0, T);
  mulm(h0, h1, T);

  for (int l = 0; l < d_.layers; ++l) {
    size_t b = LAYER_BASE + (size_t)l * kLayerTensors;
    cl_mem qb = tmp(rt, (size_t)T * D), kb = tmp(rt, (size_t)T * D),
           vb = tmp(rt, (size_t)T * D), ab = tmp(rt, (size_t)T * D),
           ob = tmp(rt, (size_t)T * D), rb = tmp(rt, (size_t)T * D),
           nb = tmp(rt, (size_t)T * D);
    dense(h1, T, D, D, wb_[b + 0], wb_[b + 1], qb);
    dense(h1, T, D, D, wb_[b + 2], wb_[b + 3], kb);
    dense(h1, T, D, D, wb_[b + 4], wb_[b + 5], vb);
    attn(qb, kb, vb, ab);
    dense(ab, T, D, D, wb_[b + 6], wb_[b + 7], ob);
    adde(h1, ob, rb, T * D);
    lnorm(rb, wb_[b + 8], wb_[b + 9], nb, T);
    mulm(nb, h1, T);
  }

  // masked attention pooling -> shared head -> threshold head
  size_t pb = LAYER_BASE + (size_t)d_.layers * kLayerTensors;
  cl_mem g = tmp(rt, D), s1 = tmp(rt, D), s2 = tmp(rt, D), t1 = tmp(rt, 1);
  cl_kernel kp = k_[K_POOL];
  float poolb = 0.f;
  rd(wb_[pb + 1], &poolb, 1);
  clSetKernelArg(kp, 0, sizeof(cl_mem), &h1);
  clSetKernelArg(kp, 1, sizeof(cl_mem), &wb_[pb + 0]);
  clSetKernelArg(kp, 2, sizeof(float), &poolb);
  clSetKernelArg(kp, 3, sizeof(cl_mem), &bMask);
  clSetKernelArg(kp, 4, sizeof(int), &T);
  clSetKernelArg(kp, 5, sizeof(int), &D);
  clSetKernelArg(kp, 6, sizeof(cl_mem), &g);
  { size_t gl[1] = {1}; worst |= clEnqueueNDRangeKernel(q, kp, 1, nullptr, gl, nullptr, 0, nullptr, nullptr); }
  dense(g, 1, D, D, wb_[pb + 2], wb_[pb + 3], s1);
  gelu(s1, s2, D);
  dense(s2, 1, D, 1, wb_[pb + 4], wb_[pb + 5], t1);
  rd(t1, tau, 1);
  if (has_norm_) *tau = (*tau) * norm_.target_std + norm_.target_mean;   // -> dB

  // sign-condition head over P probe SNR points
  float scale = 1.f;
  for (int p = 0; p < P; ++p) scale = fmaxf(scale, fabsf(snr[p]));
  cl_mem pk = tmp(rt, (size_t)P * (D + 3)), q1 = tmp(rt, (size_t)P * D),
         q2 = tmp(rt, (size_t)P * D), q3 = tmp(rt, P);
  cl_kernel kk = k_[K_PACK];
  clSetKernelArg(kk, 0, sizeof(cl_mem), &s2);
  clSetKernelArg(kk, 1, sizeof(cl_mem), &bSnR);
  clSetKernelArg(kk, 2, sizeof(int), &P);
  clSetKernelArg(kk, 3, sizeof(int), &D);
  clSetKernelArg(kk, 4, sizeof(float), &scale);
  clSetKernelArg(kk, 5, sizeof(cl_mem), &pk);
  { size_t gl[2] = {(size_t)P, (size_t)D + 3};
    worst |= clEnqueueNDRangeKernel(q, kk, 2, nullptr, gl, nullptr, 0, nullptr, nullptr); }
  dense(pk, P, D + 3, D, wb_[pb + 6], wb_[pb + 7], q1);
  gelu(q1, q2, P * D);
  dense(q2, P, D, 1, wb_[pb + 8], wb_[pb + 9], q3);
  rd(q3, sign_logits, P);

  worst |= clFinish(q);
  if (worst != CL_SUCCESS) {
    fprintf(stderr, "[tf] forward: OpenCL error %d\n", worst);
    return false;
  }
  return true;
}

}  // namespace hpocl
