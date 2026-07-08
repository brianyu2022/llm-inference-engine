// Loader for the flat GPT-2 weight format written by python/export_weights.py.
// Deliberately tiny: read the header, then read each tensor's floats straight
// into a vector. No JSON, no dependencies.
#pragma once

#include <cstdint>
#include <fstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

struct Tensor {
    std::vector<int> shape;
    std::vector<float> data;

    size_t numel() const {
        size_t n = 1;
        for (int d : shape) n *= static_cast<size_t>(d);
        return n;
    }
    // Convenience for 2-D weight matrices.
    int rows() const { return shape.at(0); }
    int cols() const { return shape.at(1); }
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
    (void)read_pod<int32_t>(f);  // version
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

        int ndim = read_pod<int32_t>(f);
        Tensor t;
        t.shape.resize(ndim);
        for (int d = 0; d < ndim; ++d) t.shape[d] = read_pod<int32_t>(f);

        t.data.resize(t.numel());
        f.read(reinterpret_cast<char*>(t.data.data()),
               static_cast<std::streamsize>(t.numel() * sizeof(float)));
        if (!f) throw std::runtime_error("EOF while reading tensor " + name);

        m.tensors.emplace(std::move(name), std::move(t));
    }
    return m;
}
