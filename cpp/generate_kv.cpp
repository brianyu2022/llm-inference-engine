// Stage 3 CLI: greedy generation with a KV-cache. Same interface as generate.cpp
//   generate_kv <weights> <n_new> <id0> [id1 ...]
// Splits the work into prefill (process the prompt, fill the cache) and decode
// (one token at a time using the cache), and reports both timings separately.
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "forward_kv.h"

static int argmax(const std::vector<float>& v) {
    int best = 0;
    float bv = v[0];
    for (int i = 1; i < static_cast<int>(v.size()); ++i)
        if (v[i] > bv) { bv = v[i]; best = i; }
    return best;
}

int main(int argc, char** argv) {
    if (argc < 4) {
        fprintf(stderr, "usage: %s <weights> <n_new> <id0> [id1 ...]\n", argv[0]);
        return 1;
    }
    std::string path = argv[1];
    int n_new = std::atoi(argv[2]);
    std::vector<int> prompt;
    for (int i = 3; i < argc; ++i) prompt.push_back(std::atoi(argv[i]));

    Model m = load_model(path);
    KVCache cache(m.cfg);
    fprintf(stderr, "loaded %s (prompt %zu tokens)\n", path.c_str(), prompt.size());

    using clock = std::chrono::high_resolution_clock;
    std::vector<int> generated;

    // Prefill: process the whole prompt in one shot, filling the cache.
    auto t0 = clock::now();
    std::vector<float> logits = forward_step(m, prompt, cache);
    auto t1 = clock::now();
    generated.push_back(argmax(logits));

    // Decode: one token at a time, each attending to the cached history.
    for (int n = 1; n < n_new; ++n) {
        logits = forward_step(m, {generated.back()}, cache);
        generated.push_back(argmax(logits));
    }
    auto t2 = clock::now();

    double prefill_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    double decode_ms = std::chrono::duration<double, std::milli>(t2 - t1).count();
    int dec = n_new - 1;
    fprintf(stderr, "prefill: %zu tok in %.1f ms | decode: %d tok in %.1f ms = %.1f tok/s\n",
            prompt.size(), prefill_ms, dec, decode_ms, dec > 0 ? dec / (decode_ms / 1e3) : 0.0);

    for (size_t i = 0; i < generated.size(); ++i)
        printf("%d%s", generated[i], i + 1 < generated.size() ? " " : "");
    printf("\n");
    return 0;
}
