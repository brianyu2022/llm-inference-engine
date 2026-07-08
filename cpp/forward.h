// Stage 2b: the GPT-2 forward pass in C++, using Apple's Accelerate BLAS
// (cblas_sgemm) for every matmul. Same math as python/model.py, validated to
// produce the same tokens. No KV-cache yet — that's Stage 3.
#pragma once

#include <Accelerate/Accelerate.h>

#include <cmath>
#include <string>
#include <vector>

#include "weights.h"

// C(M x N) = alpha * A(M x K) @ op(B) + beta * C, row-major, contiguous.
// If transB, B is stored (N x K) row-major.
inline void gemm(const float* A, const float* B, float* C, int M, int N, int K,
                 bool transB, float alpha = 1.0f, float beta = 0.0f) {
    cblas_sgemm(CblasRowMajor, CblasNoTrans, transB ? CblasTrans : CblasNoTrans,
                M, N, K, alpha, A, K, B, transB ? K : N, beta, C, N);
}

// Y(T x N) = X(T x K) @ W(K x N) + bias(N). W is a GPT-2 Conv1D weight (in x out).
// fp32 weights go through BLAS; int8 weights use a weight-only quantized kernel
// that reads 1-byte weights and dequantizes on the fly (4x less memory traffic).
inline void linear(const float* X, const Tensor& W, const Tensor& bias, int T,
                   std::vector<float>& Y) {
    int K = W.rows(), N = W.cols();
    Y.resize(static_cast<size_t>(T) * N);

    if (W.dtype == 0) {
        gemm(X, W.data.data(), Y.data(), T, N, K, /*transB=*/false);
    } else {
        // int8: W is stored transposed (N x K); Y[t,n] = scale[n] * <X[t], q[n]>.
        const int8_t* q = W.qdata.data();
        const float* s = W.scale.data();
        for (int t = 0; t < T; ++t) {
            const float* xrow = X + static_cast<size_t>(t) * K;
            float* yrow = Y.data() + static_cast<size_t>(t) * N;
            for (int n = 0; n < N; ++n) {
                const int8_t* wq = q + static_cast<size_t>(n) * K;
                float acc = 0.0f;
                for (int k = 0; k < K; ++k) acc += xrow[k] * static_cast<float>(wq[k]);
                yrow[n] = acc * s[n];
            }
        }
    }

    for (int t = 0; t < T; ++t)
        for (int j = 0; j < N; ++j) Y[static_cast<size_t>(t) * N + j] += bias.data[j];
}

inline void layernorm(const float* x, const Tensor& g, const Tensor& b, int T, int C,
                      std::vector<float>& y, float eps = 1e-5f) {
    y.resize(static_cast<size_t>(T) * C);
    for (int t = 0; t < T; ++t) {
        const float* row = x + static_cast<size_t>(t) * C;
        float mean = 0.0f;
        for (int i = 0; i < C; ++i) mean += row[i];
        mean /= C;
        float var = 0.0f;
        for (int i = 0; i < C; ++i) {
            float d = row[i] - mean;
            var += d * d;
        }
        var /= C;
        float inv = 1.0f / std::sqrt(var + eps);
        float* out = y.data() + static_cast<size_t>(t) * C;
        for (int i = 0; i < C; ++i) out[i] = (row[i] - mean) * inv * g.data[i] + b.data[i];
    }
}

inline void gelu_(std::vector<float>& x) {
    const float c = std::sqrt(2.0f / static_cast<float>(M_PI));
    for (float& v : x) v = 0.5f * v * (1.0f + std::tanh(c * (v + 0.044715f * v * v * v)));
}

// Multi-head causal self-attention. x_ln is (T x C); writes (T x C) into `out`.
inline void attention(const float* x_ln, const Model& m, int layer, int T,
                      std::vector<float>& out) {
    int C = m.cfg.n_embd, H = m.cfg.n_head, hd = C / H;
    std::string p = "h." + std::to_string(layer);

    std::vector<float> qkv;
    linear(x_ln, m.get(p + ".attn.c_attn.weight"), m.get(p + ".attn.c_attn.bias"), T, qkv);  // (T, 3C)

    // Split the packed qkv into contiguous (T x C) q, k, v.
    std::vector<float> q((size_t)T * C), k((size_t)T * C), v((size_t)T * C);
    for (int t = 0; t < T; ++t) {
        const float* row = qkv.data() + (size_t)t * 3 * C;
        std::copy(row, row + C, q.data() + (size_t)t * C);
        std::copy(row + C, row + 2 * C, k.data() + (size_t)t * C);
        std::copy(row + 2 * C, row + 3 * C, v.data() + (size_t)t * C);
    }

    std::vector<float> attn_out((size_t)T * C, 0.0f);
    std::vector<float> scores((size_t)T * T);
    float scale = 1.0f / std::sqrt(static_cast<float>(hd));

    for (int h = 0; h < H; ++h) {
        const float* Qh = q.data() + h * hd;  // head slice; row stride is C
        const float* Kh = k.data() + h * hd;
        const float* Vh = v.data() + h * hd;

        // scores(T x T) = scale * Qh(T x hd) @ Kh^T. lda = ldb = C (strided heads).
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, T, T, hd, scale, Qh, C, Kh, C,
                    0.0f, scores.data(), T);

        // Causal mask + row softmax (only over j <= i).
        for (int i = 0; i < T; ++i) {
            float* srow = scores.data() + (size_t)i * T;
            float maxv = -INFINITY;
            for (int j = 0; j <= i; ++j) maxv = srow[j] > maxv ? srow[j] : maxv;
            float sum = 0.0f;
            for (int j = 0; j <= i; ++j) {
                float e = std::exp(srow[j] - maxv);
                srow[j] = e;
                sum += e;
            }
            for (int j = 0; j <= i; ++j) srow[j] /= sum;
            for (int j = i + 1; j < T; ++j) srow[j] = 0.0f;
        }

        // out_h(T x hd) = scores(T x T) @ Vh(T x hd), written into column block h*hd.
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, T, hd, T, 1.0f, scores.data(), T,
                    Vh, C, 0.0f, attn_out.data() + h * hd, C);
    }

    linear(attn_out.data(), m.get(p + ".attn.c_proj.weight"), m.get(p + ".attn.c_proj.bias"), T, out);
}

inline void mlp(const float* x_ln, const Model& m, int layer, int T, std::vector<float>& out) {
    std::string p = "h." + std::to_string(layer);
    std::vector<float> h1;
    linear(x_ln, m.get(p + ".mlp.c_fc.weight"), m.get(p + ".mlp.c_fc.bias"), T, h1);  // (T, 4C)
    gelu_(h1);
    linear(h1.data(), m.get(p + ".mlp.c_proj.weight"), m.get(p + ".mlp.c_proj.bias"), T, out);
}

// Full forward pass. Returns the logits (size vocab) for the LAST position only —
// that's all generation needs, and it skips the big vocab projection on the rest.
inline std::vector<float> forward_last_logits(const Model& m, const std::vector<int>& ids) {
    int T = static_cast<int>(ids.size()), C = m.cfg.n_embd, V = m.cfg.vocab_size;
    const Tensor& wte = m.get("wte.weight");  // (V, C)
    const Tensor& wpe = m.get("wpe.weight");  // (n_ctx, C)

    std::vector<float> x((size_t)T * C);
    for (int t = 0; t < T; ++t) {
        const float* te = wte.data.data() + (size_t)ids[t] * C;
        const float* pe = wpe.data.data() + (size_t)t * C;
        for (int i = 0; i < C; ++i) x[(size_t)t * C + i] = te[i] + pe[i];
    }

    std::vector<float> ln, att, ff;
    for (int l = 0; l < m.cfg.n_layer; ++l) {
        std::string p = "h." + std::to_string(l);
        layernorm(x.data(), m.get(p + ".ln_1.weight"), m.get(p + ".ln_1.bias"), T, C, ln);
        attention(ln.data(), m, l, T, att);
        for (size_t i = 0; i < x.size(); ++i) x[i] += att[i];

        layernorm(x.data(), m.get(p + ".ln_2.weight"), m.get(p + ".ln_2.bias"), T, C, ln);
        mlp(ln.data(), m, l, T, ff);
        for (size_t i = 0; i < x.size(); ++i) x[i] += ff[i];
    }
    layernorm(x.data(), m.get("ln_f.weight"), m.get("ln_f.bias"), T, C, ln);

    // logits(1 x V) = last_hidden(1 x C) @ wte^T
    const float* last = ln.data() + (size_t)(T - 1) * C;
    std::vector<float> logits(V);
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, 1, V, C, 1.0f, last, C, wte.data.data(), C,
                0.0f, logits.data(), V);
    return logits;
}
