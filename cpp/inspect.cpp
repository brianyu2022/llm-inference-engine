// Stage 2a sanity check: load the exported weights in C++ and print the config,
// shapes, and parameter count. Should match python/inspect_weights.py — minus the
// causal-mask buffers we dropped in export (so ~124.4M params, not 137M).
#include <cstdio>
#include <map>
#include <regex>
#include <string>

#include "weights.h"

int main(int argc, char** argv) {
    std::string path = (argc > 1) ? argv[1] : "weights/gpt2.bin";
    Model m = load_model(path);

    printf("=== config ===\n");
    printf("  n_layer %d  n_head %d  n_embd %d  n_ctx %d  vocab %d\n",
           m.cfg.n_layer, m.cfg.n_head, m.cfg.n_embd, m.cfg.n_ctx, m.cfg.vocab_size);

    // Sort names and hide repeated blocks 1..n-1 so the output stays readable.
    std::map<std::string, const Tensor*> sorted;
    size_t total = 0;
    for (const auto& kv : m.tensors) {
        sorted[kv.first] = &kv.second;
        total += kv.second.numel();
    }

    std::regex layer(R"(\bh\.(\d+)\.)");
    printf("\n=== tensors (non-block + block 0) ===\n");
    for (const auto& [name, t] : sorted) {
        std::smatch mm;
        if (std::regex_search(name, mm, layer) && mm[1] != "0") continue;
        std::string shp;
        for (size_t d = 0; d < t->shape.size(); ++d) {
            shp += std::to_string(t->shape[d]);
            if (d + 1 < t->shape.size()) shp += ",";
        }
        printf("  %-34s (%s)\n", name.c_str(), shp.c_str());
    }

    printf("\n%zu tensors, total parameters: %zu\n", m.tensors.size(), total);
    return 0;
}
