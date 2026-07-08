// Stage 5: a continuous-batching scheduler + throughput/latency benchmark.
//   serve <weights> <num_requests> <max_new> <max_batch> <id0> [id1 ...]
//
// It runs `num_requests` generation requests (all using the given prompt) through
// a decoder that keeps up to `max_batch` sequences in flight at once. As soon as
// a sequence finishes, a waiting one is admitted and prefilled — that's what makes
// it *continuous* batching rather than static. Prints aggregate throughput and
// per-request latency percentiles to stderr, and the first request's output ids
// to stdout (for correctness checks).
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <string>
#include <vector>

#include "batch.h"

static double now() {
    return std::chrono::duration<double>(
               std::chrono::high_resolution_clock::now().time_since_epoch())
        .count();
}

static int argmax(const std::vector<float>& v) {
    int best = 0;
    float bv = v[0];
    for (int i = 1; i < static_cast<int>(v.size()); ++i)
        if (v[i] > bv) { bv = v[i]; best = i; }
    return best;
}

struct Seq {
    KVCache cache;
    std::vector<int> gen;
    int next_tok = 0;
    int produced = 0;
    int max_new;
    double start = 0.0;
    Seq(const Config& cfg, int mx) : cache(cfg), max_new(mx) {}
};

int main(int argc, char** argv) {
    if (argc < 6) {
        fprintf(stderr, "usage: %s <weights> <num_requests> <max_new> <max_batch> <id0> [id1 ...]\n",
                argv[0]);
        return 1;
    }
    std::string path = argv[1];
    int R = std::atoi(argv[2]);
    int max_new = std::atoi(argv[3]);
    int B = std::atoi(argv[4]);
    std::vector<int> prompt;
    for (int i = 5; i < argc; ++i) prompt.push_back(std::atoi(argv[i]));

    Model m = load_model(path);

    std::vector<std::unique_ptr<Seq>> active;
    std::vector<double> latencies;
    std::vector<int> sample_out;  // first finished request's tokens, for validation
    long total_tokens = 0;
    int admitted = 0;

    auto finish = [&](std::unique_ptr<Seq>& s) {
        latencies.push_back(now() - s->start);
        total_tokens += s->produced;
        if (sample_out.empty()) sample_out = s->gen;
    };

    // Admit + prefill new requests until the batch is full or none are left.
    auto admit = [&]() {
        while (static_cast<int>(active.size()) < B && admitted < R) {
            auto s = std::make_unique<Seq>(m.cfg, max_new);
            s->start = now();
            std::vector<float> logits = forward_step(m, prompt, s->cache);
            int first = argmax(logits);
            s->gen.push_back(first);
            s->next_tok = first;
            s->produced = 1;
            ++admitted;
            if (s->produced >= max_new)
                finish(s);
            else
                active.push_back(std::move(s));
        }
    };

    double t0 = now();
    admit();
    while (!active.empty()) {
        std::vector<int> toks;
        std::vector<KVCache*> caches;
        toks.reserve(active.size());
        caches.reserve(active.size());
        for (auto& s : active) {
            toks.push_back(s->next_tok);
            caches.push_back(&s->cache);
        }

        std::vector<int> next_tok;
        batch_decode_step(m, toks, caches, next_tok);

        std::vector<std::unique_ptr<Seq>> keep;
        for (size_t i = 0; i < active.size(); ++i) {
            auto& s = active[i];
            s->gen.push_back(next_tok[i]);
            s->next_tok = next_tok[i];
            s->produced++;
            if (s->produced >= s->max_new)
                finish(s);
            else
                keep.push_back(std::move(s));
        }
        active.swap(keep);
        admit();  // continuous: backfill freed slots immediately
    }
    double wall = now() - t0;

    std::sort(latencies.begin(), latencies.end());
    auto pct = [&](double p) {
        if (latencies.empty()) return 0.0;
        size_t idx = static_cast<size_t>(p * (latencies.size() - 1));
        return latencies[idx] * 1e3;  // ms
    };
    double mean = 0.0;
    for (double l : latencies) mean += l;
    mean = latencies.empty() ? 0.0 : mean / latencies.size() * 1e3;

    fprintf(stderr,
            "batch=%d requests=%d max_new=%d | %.2fs | %ld tokens | "
            "%.1f tok/s | latency mean %.0fms p50 %.0fms p95 %.0fms\n",
            B, R, max_new, wall, total_tokens, total_tokens / wall, mean, pct(0.50), pct(0.95));

    for (size_t i = 0; i < sample_out.size(); ++i)
        printf("%d%s", sample_out[i], i + 1 < sample_out.size() ? " " : "");
    printf("\n");
    return 0;
}
