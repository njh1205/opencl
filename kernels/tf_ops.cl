//-----------------------------------------------------------------------------
// operator family: tf_*  -- inference operators for the spectral set
// transformer threshold predictor (SpectralSetTransformerThresholdRegressor
// in mix_precode/ai/learn_svd_ucd_threshold.py).
//
//   tokens [T][token_dim]  + config [config_dim]
//     -> token/config projections (Linear,GELU,Linear) + broadcast add
//     -> L x (masked self-attention + LayerNorm) residual blocks
//     -> masked attention pooling -> shared head -> threshold head
//     -> sign-condition head over P probe SNR points
//
// Every operator is a plain data-parallel kernel: on PL they synthesise to
// pipelined MAC arrays; on CPU/GPU OpenCL they are the validation path.
// fp32 only (inference datapath), exactly like the deployed PL build.
//-----------------------------------------------------------------------------
#pragma OPENCL EXTENSION cl_khr_fp64 : disable

// Capacity. tf_attn/tf_pool keep a per-work-item score scratch indexed by the
// token count T, and T arrives from the weight-file header, so it must be
// bounded here. TF_TMAX has to match kTfTMax in
// runtime/transformer_predictor.h -- the host already rejects a larger T in
// load() (and a heads that does not divide d_model), so the guards below are
// defence in depth rather than the primary gate.
#ifndef TF_TMAX
#define TF_TMAX 64
#endif

__kernel void tf_dense(__global const float* restrict X,   // [B][in]
                       __global const float* restrict W,   // [out][in]
                       __global const float* restrict b,   // [out]
                       int B, int in, int out,
                       __global float* restrict Y)         // [B][out]
{
  int o = get_global_id(0);
  int bb = get_global_id(1);
  if (o >= out || bb >= B) return;
  float acc = b[o];
  __global const float* w = W + (size_t)o * in;
  __global const float* x = X + (size_t)bb * in;
  for (int i = 0; i < in; ++i) acc += w[i] * x[i];
  Y[(size_t)bb * out + o] = acc;
}

__kernel void tf_gelu(__global const float* restrict X, int n,
                      __global float* restrict Y) {
  int i = get_global_id(0);
  if (i >= n) return;
  float x = X[i];
  Y[i] = 0.5f * x * (1.0f + erf(x * 0.70710678118654752f));
}

__kernel void tf_add_broadcast(__global const float* restrict A,  // [T][d]
                               __global const float* restrict b,  // [d]
                               int T, int d,
                               __global float* restrict Y) {
  int i = get_global_id(0);
  if (i >= T * d) return;
  Y[i] = A[i] + b[i % d];
}

__kernel void tf_mul_mask(__global const float* restrict A,   // [T][d]
                          __global const float* restrict m,   // [T]
                          int T, int d,
                          __global float* restrict Y) {
  int i = get_global_id(0);
  if (i >= T * d) return;
  Y[i] = A[i] * m[i / d];
}

__kernel void tf_layernorm(__global const float* restrict X,  // [T][d]
                           __global const float* restrict g,  // [d]
                           __global const float* restrict bb, // [d]
                           int T, int d, float eps,
                           __global float* restrict Y) {
  int t = get_global_id(0);
  if (t >= T) return;
  __global const float* x = X + (size_t)t * d;
  float mu = 0.f;
  for (int i = 0; i < d; ++i) mu += x[i];
  mu /= (float)d;
  float va = 0.f;
  for (int i = 0; i < d; ++i) { float e = x[i] - mu; va += e * e; }
  va /= (float)d;
  float rs = 1.0f / sqrt(va + eps);
  for (int i = 0; i < d; ++i)
    Y[(size_t)t * d + i] = (x[i] - mu) * rs * g[i] + bb[i];
}

// masked self-attention, one work-item per (query, head):
//   out[q][h*dh:(h+1)*dh] = sum_k softmax_c(qk/sqrt(dh) + mask) * V[k][h slice]
__kernel void tf_attn(__global const float* restrict Q,   // [T][d]
                      __global const float* restrict K,   // [T][d]
                      __global const float* restrict V,   // [T][d]
                      __global const float* restrict m,   // [T] 1=valid
                      int T, int d, int heads, float sc_scale,
                      __global float* restrict Y)         // [T][d]
{
  int q = get_global_id(0);
  int h = get_global_id(1);
  if (q >= T || h >= heads || T > TF_TMAX || heads <= 0 || d % heads != 0) return;
  int dh = d / heads;
  __global const float* qv = Q + (size_t)q * d + h * dh;
  float sc[TF_TMAX];
  float mx = -1e30f;
  for (int k = 0; k < T; ++k) {
    if (m[k] < 0.5f) { sc[k] = -1e30f; continue; }
    __global const float* kv = K + (size_t)k * d + h * dh;
    float acc = 0.f;
    for (int i = 0; i < dh; ++i) acc += qv[i] * kv[i];
    sc[k] = acc * sc_scale;                   // 1/sqrt(d_head)
    mx = fmax(mx, sc[k]);
  }
  float sum = 0.f;
  for (int k = 0; k < T; ++k) { sc[k] = exp(sc[k] - mx); sum += sc[k]; }
  float inv = 1.0f / (sum > 0.f ? sum : 1.f);
  for (int i = 0; i < dh; ++i) {
    float acc = 0.f;
    for (int k = 0; k < T; ++k)
      acc += (sc[k] * inv) * V[(size_t)k * d + h * dh + i];
    Y[(size_t)q * d + h * dh + i] = acc;
  }
}

// masked attention pooling: global = sum_t softmax_t(pool(h_t)) * h_t
// Single work-item by construction: the host launches gws = {1} (see
// runtime/transformer_predictor.cpp), and the gid guard below makes a wider
// launch a no-op instead of a wrong reduction. Two serial passes (scores, then
// the weighted sum); on PL this becomes one small FSM-driven reduce unit.
__kernel void tf_pool(__global const float* restrict H,   // [T][d]
                      __global const float* restrict w,   // [d]
                      float bp,
                      __global const float* restrict m,   // [T]
                      int T, int d,
                      __global float* restrict Y)         // [d]
{
  if (get_global_id(0) != 0 || T > TF_TMAX) return;
  float sc[TF_TMAX];
  float mx = -1e30f;
  for (int t = 0; t < T; ++t) {
    if (m[t] < 0.5f) { sc[t] = -1e30f; continue; }
    __global const float* h = H + (size_t)t * d;
    float s = bp;
    for (int i = 0; i < d; ++i) s += w[i] * h[i];
    sc[t] = s; mx = fmax(mx, s);
  }
  float sum = 0.f;
  for (int t = 0; t < T; ++t) { sc[t] = exp(sc[t] - mx); sum += sc[t]; }
  float inv = 1.0f / (sum > 0.f ? sum : 1.f);
  for (int i = 0; i < d; ++i) {
    float acc = 0.f;
    for (int t = 0; t < T; ++t) acc += (sc[t] * inv) * H[(size_t)t * d + i];
    Y[i] = acc;
  }
}

// sign-condition head input pack: [P][d+3] = [shared_state, snr, snr^2, 1]
__kernel void tf_sign_pack(__global const float* restrict state,  // [d]
                           __global const float* restrict snr,    // [P]
                           int P, int d, float scale,
                           __global float* restrict Y)            // [P][d+3]
{
  int p = get_global_id(0);
  int i = get_global_id(1);
  if (p >= P || i >= d + 3) return;
  float v;
  if (i < d) v = state[i];
  else {
    float s = snr[p] / scale;
    v = (i == d) ? s : (i == d + 1 ? s * s : 1.0f);
  }
  Y[(size_t)p * (d + 3) + i] = v;
}

__kernel void tf_add_elem(__global const float* restrict A,
                          __global const float* restrict B, int n,
                          __global float* restrict Y) {
  int i = get_global_id(0);
  if (i >= n) return;
  Y[i] = A[i] + B[i];
}
