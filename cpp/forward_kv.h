// Stage 3: KV-cache. Instead of recomputing K/V for the whole sequence every
// token, we store each layer's keys/values and, on each new token, compute K/V
// for just that token and attend against the cached history.
//
// forward_step() handles BOTH regimes with one code path:
//   - prefill: pass the whole prompt (M = prompt length, cache empty)
//   - decode:  pass one token   (M = 1, cache holds the history)
#pragma once

#include "forward.h"  // gemm, linear, layernorm, gelu_, mlp, Accelerate

struct KVCache {
    int n_layer, C, n_ctx, len = 0;              // len = positions currently cached
    std::vector<std::vector<float>> k, v;        // per layer: (n_ctx x C), row = position

    explicit KVCache(const Config& cfg)
        : n_layer(cfg.n_layer), C(cfg.n_embd), n_ctx(cfg.n_ctx),
          k(cfg.n_layer), v(cfg.n_layer) {
        for (int l = 0; l < n_layer; ++l) {
            k[l].resize(static_cast<size_t>(n_ctx) * C);  // preallocate so pointers stay valid
            v[l].resize(static_cast<size_t>(n_ctx) * C);
        }
    }
};

// Process M new tokens against the cache, append their K/V, and return the logits
// (size vocab) for the LAST new token.
inline std::vector<float> forward_step(const Model& m, const std::vector<int>& new_ids,
                                       KVCache& cache) {
    int M = static_cast<int>(new_ids.size());
    int C = m.cfg.n_embd, H = m.cfg.n_head, hd = C / H, V = m.cfg.vocab_size;
    int pos0 = cache.len;        // global position of the first new token
    int Lnew = pos0 + M;         // total cached length after this step
    if (Lnew > cache.n_ctx)
        throw std::runtime_error("sequence length exceeds context window (n_ctx)");
    const Tensor& wte = m.get("wte.weight");
    const Tensor& wpe = m.get("wpe.weight");

    // Embed the new tokens at their absolute positions.
    std::vector<float> x((size_t)M * C);
    for (int i = 0; i < M; ++i)
        embed_token(wte, wpe, new_ids[i], pos0 + i, C, x.data() + (size_t)i * C);

    std::vector<float> ln, qkv, q((size_t)M * C), attn_out((size_t)M * C),
        scores((size_t)M * Lnew), proj, ff;
    float scale = 1.0f / std::sqrt(static_cast<float>(hd));

    for (int l = 0; l < m.cfg.n_layer; ++l) {
        std::string p = "h." + std::to_string(l);

        layernorm(x.data(), m.get(p + ".ln_1.weight"), m.get(p + ".ln_1.bias"), M, C, ln);
        linear(ln.data(), m.get(p + ".attn.c_attn.weight"), m.get(p + ".attn.c_attn.bias"), M, qkv);

        // Keep q for the new tokens; write the new k/v straight into the cache.
        for (int i = 0; i < M; ++i) {
            const float* row = qkv.data() + (size_t)i * 3 * C;
            std::copy(row, row + C, q.data() + (size_t)i * C);
            std::copy(row + C, row + 2 * C, cache.k[l].data() + (size_t)(pos0 + i) * C);
            std::copy(row + 2 * C, row + 3 * C, cache.v[l].data() + (size_t)(pos0 + i) * C);
        }

        for (int h = 0; h < H; ++h) {
            const float* Qh = q.data() + h * hd;             // (M x hd),   ld = C
            const float* Kh = cache.k[l].data() + h * hd;    // (Lnew x hd), ld = C
            const float* Vh = cache.v[l].data() + h * hd;    // (Lnew x hd), ld = C

            // scores(M x Lnew) = scale * Qh @ Kh^T
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, M, Lnew, hd, scale, Qh, C, Kh, C,
                        0.0f, scores.data(), Lnew);

            // Causal mask + softmax: new token i (global pos0+i) attends to j in [0, pos0+i].
            for (int i = 0; i < M; ++i) {
                float* srow = scores.data() + (size_t)i * Lnew;
                int last = pos0 + i;
                float maxv = -INFINITY;
                for (int j = 0; j <= last; ++j) maxv = srow[j] > maxv ? srow[j] : maxv;
                float sum = 0.0f;
                for (int j = 0; j <= last; ++j) {
                    float e = std::exp(srow[j] - maxv);
                    srow[j] = e;
                    sum += e;
                }
                for (int j = 0; j <= last; ++j) srow[j] /= sum;
                for (int j = last + 1; j < Lnew; ++j) srow[j] = 0.0f;
            }

            // out_h(M x hd) = scores @ Vh, into column block h*hd (ld = C).
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, hd, Lnew, 1.0f, scores.data(),
                        Lnew, Vh, C, 0.0f, attn_out.data() + h * hd, C);
        }

        linear(attn_out.data(), m.get(p + ".attn.c_proj.weight"), m.get(p + ".attn.c_proj.bias"), M, proj);
        for (size_t i = 0; i < x.size(); ++i) x[i] += proj[i];

        layernorm(x.data(), m.get(p + ".ln_2.weight"), m.get(p + ".ln_2.bias"), M, C, ln);
        mlp(ln.data(), m, l, M, ff);
        for (size_t i = 0; i < x.size(); ++i) x[i] += ff[i];
    }
    cache.len = Lnew;

    layernorm(x.data(), m.get("ln_f.weight"), m.get("ln_f.bias"), M, C, ln);
    const float* last = ln.data() + (size_t)(M - 1) * C;
    std::vector<float> logits(V);
    logits_from_hidden(wte, last, C, V, logits.data());
    return logits;
}
