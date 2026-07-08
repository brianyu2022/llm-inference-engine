# LLM Inference Engine

> Working name — rename before you push to GitHub (ideas: `nanoserve`, `tinfer`, `ferrite`).

A from-scratch LLM inference engine. Starts as a readable NumPy reference that runs
**GPT-2** with no PyTorch, then becomes a fast C++ engine with a KV-cache,
quantization, continuous batching, and (stretch) hand-written Metal/SIMD kernels.

The point isn't "it generates text" — that's the easy part. The point is the
**performance engineering**: making it fast and measuring it honestly.

## Why this exists

Built as a deep-dive into how large language models actually run on hardware:
memory bandwidth, cache behavior, quantization, and the arithmetic intensity of
attention. Every optimization is benchmarked against a baseline.

## Roadmap

- [x] **Stage 1 — Correct forward pass (Python/NumPy).** Loads real GPT-2 124M
      weights and generates text; validated token-for-token against the reference
      greedy continuation. Baseline benchmarked in `benchmarks/`.
- [x] **Stage 2 — C++ engine.** Full forward pass in C++ on Accelerate BLAS,
      validated token-for-token against the NumPy reference. **~76 tok/s vs. 10
      tok/s NumPy (~8×)** — from killing interpreter overhead and only projecting
      the last position's logits, before any KV-cache.
- [x] **Stage 3 — KV-cache.** Prefill/decode split; cache each layer's K/V and
      attend against history instead of recomputing. **243 tok/s vs. 19 tok/s
      non-cached at 256 tokens (~13×)**, with flat per-token latency.
- [~] **Stage 4 — int8 quantization.** Weights 4× smaller (whole model 498 → 243
      MB on disk) at **+0.85% perplexity**, greedy output unchanged. BUT the
      naive scalar kernel is **10× slower** than fp32 BLAS — it reads less data
      but is single-threaded and unvectorized, so it wastes the bandwidth win
      and the cores. Realizing the speedup needs a SIMD + multithreaded int8
      kernel (Stage 6). *Great lesson: quantization only helps if the kernel
      exploits it.*
- [ ] **Stage 5 — Continuous batching + serving API.** Metric: throughput and
      p50/p95 under concurrent load.
- [ ] **Stage 6 — Custom SIMD/threaded kernels + roofline.** First target: make
      the int8 kernel actually beat fp32.

## Status

**Stage 4 in progress.** fp32 C++ engine with a KV-cache does **243 tok/s**
decode (vs. 19 non-cached, 10 NumPy), validated token-for-token. int8
quantization is implemented and quality-validated (+0.85% perplexity, 2× smaller
on disk) but the naive int8 kernel is bandwidth-underutilized and slower — next
up is a SIMD + multithreaded int8 kernel to actually beat fp32.

### Benchmarks (GPT-2 124M, M4 Pro, greedy, 256 tokens)

| Engine | Decode throughput |
|---|---:|
| NumPy reference | ~10 tok/s |
| C++ (no cache) | ~19 tok/s |
| C++ + KV-cache | **~243 tok/s** |

### Running the C++ engine

```sh
python python/export_weights.py     # one-time: writes weights/gpt2.bin
make -C cpp                          # build ./cpp/build/{inspect,generate,generate_kv}
python python/run_cpp.py             # validate both C++ engines vs NumPy

# generate directly (prompt as token ids; tokenize with python/tokenizer.py)
./cpp/build/generate_kv weights/gpt2.bin 256 <id0> <id1> ...
```

## Quickstart (Stage 1)

```sh
# from the repo root
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python python/download_weights.py   # fetch GPT-2 124M weights (~500 MB) into weights/
python python/inspect_weights.py     # print the model's tensors — your map of GPT-2
python python/generate.py            # generate text (greedy by default)
```

Correctness check: with the default prompt and greedy decoding, GPT-2 124M
continues *"Alan Turing theorized that computers would one day become"* with
*" the most powerful machines on the planet."*

## Layout

```
python/        Stage 1 reference implementation (NumPy, no torch)
cpp/           Stage 2+ the real engine
weights/       downloaded model weights (gitignored)
benchmarks/    benchmark results and scripts
```
