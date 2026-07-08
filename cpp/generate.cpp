// Stage 2b CLI: greedy generation in C++.
//   generate <weights> <n_new> <id0> [id1 ...]
// Reads prompt token ids as args, prints the generated ids to stdout (one line,
// space-separated). Logs go to stderr. Tokenization stays in Python (run_cpp.py).
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "forward.h"

int main(int argc, char** argv) {
    if (argc < 4) {
        fprintf(stderr, "usage: %s <weights> <n_new> <id0> [id1 ...]\n", argv[0]);
        return 1;
    }
    std::string path = argv[1];
    int n_new = std::atoi(argv[2]);
    std::vector<int> ids;
    for (int i = 3; i < argc; ++i) ids.push_back(std::atoi(argv[i]));

    Model m = load_model(path);
    fprintf(stderr, "loaded %s (prompt %zu tokens)\n", path.c_str(), ids.size());

    std::vector<int> generated;
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int n = 0; n < n_new; ++n) {
        std::vector<float> logits = forward_last_logits(m, ids);
        int best = 0;
        float bv = logits[0];
        for (int i = 1; i < static_cast<int>(logits.size()); ++i)
            if (logits[i] > bv) { bv = logits[i]; best = i; }
        ids.push_back(best);
        generated.push_back(best);
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    double dt = std::chrono::duration<double>(t1 - t0).count();
    fprintf(stderr, "[%d tokens in %.2fs = %.1f tok/s]\n", n_new, dt, n_new / dt);

    for (size_t i = 0; i < generated.size(); ++i)
        printf("%d%s", generated[i], i + 1 < generated.size() ? " " : "");
    printf("\n");
    return 0;
}
