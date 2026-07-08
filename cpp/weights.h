// Loader for the flat GPT-2 weight format written by python/export_weights.py.
// Format v2 adds a per-tensor dtype so weights can be fp32 or int8. v1 files
// (no dtype field, all fp32) still load.
#pragma once

#include <cstdint>
#include <fstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

struct Tensor {
    std::vector<int> shape;        // logical shape (K, N) for weight matrices
    int dtype = 0;                 // 0 = fp32; 1 = int8 transposed (N,K) per-column; 2 = int8 row-major (R,C) per-row
    std::vector<float> data;       // fp32 values                    (dtype 0)
    std::vector<int8_t> qdata;     // int8 values: (N,K) dtype 1, (R,C) dtype 2
    std::vector<float> scale;      // per-column (dtype 1) or per-row (dtype 2) scales

    size_t numel() const {
        size_t n = 1;
        for (int d : shape) n *= static_cast<size_t>(d);
        return n;
    }
    int rows() const { return shape.at(0); }  // K (input dim)
    int cols() const { return shape.at(1); }  // N (output dim)
};

struct Config {
    int n_layer, n_head, n_embd, n_ctx, vocab_size;
};

struct Model {
    Config cfg{};
    std::unordered_map<std::string, Tensor> tensors;

    const Tensor& get(const std::string& name) const {
        auto it = tensors.find(name);
        if (it == tensors.end()) throw std::runtime_error("missing tensor: " + name);
        return it->second;
    }
};

template <typename T>
static T read_pod(std::ifstream& f) {
    T v;
    f.read(reinterpret_cast<char*>(&v), sizeof(T));
    if (!f) throw std::runtime_error("unexpected EOF while reading header");
    return v;
}

inline Model load_model(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot open " + path + " (did you run export_weights.py?)");

    char magic[4];
    f.read(magic, 4);
    if (std::string(magic, 4) != "GPT2") throw std::runtime_error("bad magic in " + path);

    Model m;
    int version = read_pod<int32_t>(f);
    m.cfg.n_layer    = read_pod<int32_t>(f);
    m.cfg.n_head     = read_pod<int32_t>(f);
    m.cfg.n_embd     = read_pod<int32_t>(f);
    m.cfg.n_ctx      = read_pod<int32_t>(f);
    m.cfg.vocab_size = read_pod<int32_t>(f);

    int n_tensors = read_pod<int32_t>(f);
    for (int i = 0; i < n_tensors; ++i) {
        int name_len = read_pod<int32_t>(f);
        std::string name(name_len, '\0');
        f.read(name.data(), name_len);

        Tensor t;
        t.dtype = (version >= 2) ? read_pod<int32_t>(f) : 0;

        int ndim = read_pod<int32_t>(f);
        t.shape.resize(ndim);
        for (int d = 0; d < ndim; ++d) t.shape[d] = read_pod<int32_t>(f);

        if (t.dtype == 0) {
            t.data.resize(t.numel());
            f.read(reinterpret_cast<char*>(t.data.data()),
                   static_cast<std::streamsize>(t.numel() * sizeof(float)));
        } else {  // int8: numel int8 values, then scales (per-column for dtype 1, per-row for dtype 2)
            t.qdata.resize(t.numel());
            f.read(reinterpret_cast<char*>(t.qdata.data()),
                   static_cast<std::streamsize>(t.qdata.size()));
            int n_scale = (t.dtype == 1) ? t.shape[1] : t.shape[0];  // cols vs rows
            t.scale.resize(n_scale);
            f.read(reinterpret_cast<char*>(t.scale.data()),
                   static_cast<std::streamsize>(n_scale * sizeof(float)));
        }
        if (!f) throw std::runtime_error("EOF while reading tensor " + name);

        m.tensors.emplace(std::move(name), std::move(t));
    }
    return m;
}
