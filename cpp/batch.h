// Stage 5: batched decoding for continuous batching. We decode one token for each
// of B concurrent sequences per step. The big linear projections (qkv, c_proj,
// mlp) are batched (M = B) — one matmul serves all sequences, amortizing the
// weight reads that dominate decode. Attention is per-sequence, since each
// sequence has its own KV cache at its own length.
#pragma once

#include "forward.h"     // linear (dtype-aware: fp32 BLAS or int8 SDOT), layernorm, mlp
#include "forward_kv.h"  // KVCache

// One decode step for B sequences. toks[b] is sequence b's current token; the
// step appends its K/V to cache[b], attends, and writes the next token to
// next_tok[b].
inline void batch_decode_step(const Model& m, const std::vector<int>& toks,
                              std::vector<KVCache*>& caches, std::vector<int>& next_tok) {
    const int B = static_cast<int>(toks.size());
    const int C = m.cfg.n_embd, H = m.cfg.n_head, hd = C / H, V = m.cfg.vocab_size;
    const Tensor& wte = m.get("wte.weight");
    const Tensor& wpe = m.get("wpe.weight");
    const float scale = 1.0f / std::sqrt(static_cast<float>(hd));

    std::vector<int> pos(B);
    for (int b = 0; b < B; ++b) {
        pos[b] = caches[b]->len;  // this token's position, per sequence
        if (pos[b] + 1 > caches[b]->n_ctx)
            throw std::runtime_error("sequence length exceeds context window (n_ctx)");
    }

    // Embed the B tokens (each at its own position) -> x (B, C).
    std::vector<float> x(static_cast<size_t>(B) * C);
    for (int b = 0; b < B; ++b)
        embed_token(wte, wpe, toks[b], pos[b], C, x.data() + static_cast<size_t>(b) * C);

    std::vector<float> ln, qkv, attn(static_cast<size_t>(B) * C), proj, ff;

    for (int l = 0; l < m.cfg.n_layer; ++l) {
        std::string p = "h." + std::to_string(l);

        layernorm(x.data(), m.get(p + ".ln_1.weight"), m.get(p + ".ln_1.bias"), B, C, ln);
        linear(ln.data(), m.get(p + ".attn.c_attn.weight"), m.get(p + ".attn.c_attn.bias"), B, qkv);  // (B, 3C)

        // Attention is independent per sequence, so fan the sequences across cores.
        // (The linears above/below are already batched into single matmuls.) Each
        // task touches only its own cache row and its own slice of `attn`.
        const float* qkvp = qkv.data();
        float* attnp = attn.data();
        KVCache* const* cptr = caches.data();
        const int* posp = pos.data();
        dispatch_apply(B, dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_DEFAULT, 0), ^(size_t bs) {
            int b = static_cast<int>(bs);
            const float* row = qkvp + static_cast<size_t>(b) * 3 * C;
            const float* qb = row;              // query (C)
            const float* kb = row + C;
            const float* vb = row + 2 * C;
            KVCache* cache = cptr[b];
            // Append this token's K/V to sequence b's cache at position pos[b].
            std::copy(kb, kb + C, cache->k[l].data() + static_cast<size_t>(posp[b]) * C);
            std::copy(vb, vb + C, cache->v[l].data() + static_cast<size_t>(posp[b]) * C);

            const int Lb = posp[b] + 1;         // attend over [0, pos[b]]
            float* outb = attnp + static_cast<size_t>(b) * C;
            std::vector<float> sc(Lb);
            for (int h = 0; h < H; ++h) {
                const float* qh = qb + h * hd;
                const float* Kc = cache->k[l].data() + h * hd;  // rows stride C
                const float* Vc = cache->v[l].data() + h * hd;
                float smax = -INFINITY;
                for (int j = 0; j < Lb; ++j) {
                    const float* kj = Kc + static_cast<size_t>(j) * C;
                    float d = 0.0f;
                    for (int e = 0; e < hd; ++e) d += qh[e] * kj[e];
                    d *= scale;
                    sc[j] = d;
                    if (d > smax) smax = d;
                }
                float sum = 0.0f;
                for (int j = 0; j < Lb; ++j) { sc[j] = std::exp(sc[j] - smax); sum += sc[j]; }
                float inv = 1.0f / sum;
                float* oh = outb + h * hd;
                for (int e = 0; e < hd; ++e) oh[e] = 0.0f;
                for (int j = 0; j < Lb; ++j) {
                    float w = sc[j] * inv;
                    const float* vj = Vc + static_cast<size_t>(j) * C;
                    for (int e = 0; e < hd; ++e) oh[e] += w * vj[e];
                }
            }
        });

        linear(attn.data(), m.get(p + ".attn.c_proj.weight"), m.get(p + ".attn.c_proj.bias"), B, proj);
        for (size_t i = 0; i < x.size(); ++i) x[i] += proj[i];

        layernorm(x.data(), m.get(p + ".ln_2.weight"), m.get(p + ".ln_2.bias"), B, C, ln);
        mlp(ln.data(), m, l, B, ff);
        for (size_t i = 0; i < x.size(); ++i) x[i] += ff[i];
    }
    for (int b = 0; b < B; ++b) caches[b]->len = pos[b] + 1;

    layernorm(x.data(), m.get("ln_f.weight"), m.get("ln_f.bias"), B, C, ln);

    // Greedy next token per sequence.
    next_tok.assign(B, 0);
    std::vector<float> logits(V);
    for (int b = 0; b < B; ++b) {
        const float* xb = ln.data() + static_cast<size_t>(b) * C;
        logits_from_hidden(wte, xb, C, V, logits.data());
        int best = 0;
        float bv = logits[0];
        for (int i = 1; i < V; ++i)
            if (logits[i] > bv) { bv = logits[i]; best = i; }
        next_tok[b] = best;
    }
}
