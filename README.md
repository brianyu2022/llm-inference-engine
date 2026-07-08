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
- [x] **Stage 4 — int8 quantization + custom kernel.** Weights 2× smaller on
      disk (498 → 243 MB) at **+0.85% perplexity**, greedy output unchanged.
      Custom int8 matmul with **NEON SIMD + GCD multithreading**: 184 tok/s —
      **7× faster than the naive scalar kernel** (26 tok/s), ~0.78× of fp32 BLAS.
      The remaining gap is Apple's **AMX matrix coprocessor** (used by Accelerate
      for fp32 GEMM), which NEON can't match. *Lesson: quantization only helps if
      the kernel exploits it, and you're competing against dedicated matrix HW.*
- [x] **Stage 5 — Continuous batching.** Decode B sequences per step — batching
      the linear projections while attending per-sequence — and admit/prefill
      waiting requests as slots free. Aggregate throughput scales **272 → 511
      tok/s (1.88×) from batch 1 → 16**, with the expected latency tradeoff.
- [x] **Stage 6 — W8A8 + SDOT integer kernel.** Dynamic per-row int8 activations
      + ARM `SDOT` (int8·int8 → int32). **~302 tok/s — 1.21× faster than fp32
      BLAS**, beating Apple's AMX path, at +4.0% perplexity. *The full arc:
      naive 26 → NEON weight-only 184 → W8A8 302 tok/s.*
- [x] **Roofline analysis.** Decode is 0.5–1.0 FLOP/byte — firmly memory-bound.
      fp32 hits 44% of the M4 Pro's ~273 GB/s peak, int8 26%. The fp32 logits
      projection (`wte`, 154 MB/token) dominates int8 traffic → quantizing the
      embedding table is the next win. (`python/roofline.py`)

## Status

**All core stages (1–6) done.** From-scratch GPT-2: NumPy reference → C++ engine
→ KV-cache (243 tok/s) → int8 quantization → **custom W8A8 + SDOT kernel that
beats Apple's fp32 BLAS by 1.21×** → **continuous batching** (1.88× throughput at
batch 16). Everything validated token-for-token against the reference. Polish
remaining: roofline analysis, bigger models, a writeup.

### Benchmarks (GPT-2 124M, M4 Pro, greedy, 256 tokens)

| Engine | Decode throughput | vs fp32 |
|---|---:|---:|
| NumPy reference | ~10 tok/s | |
| C++ (no cache) | ~19 tok/s | |
| C++ + KV-cache (fp32 BLAS) | **~249 tok/s** | 1.0× |
| C++ + KV-cache + int8, naive kernel | ~26 tok/s | 0.10× |
| C++ + KV-cache + int8, NEON+threads (weight-only) | ~184 tok/s | 0.78× |
| **C++ + KV-cache + W8A8 SDOT** | **~302 tok/s** | **1.21×** |

int8 model is 243 MB on disk (vs 498 MB fp32). Quality cost: +0.85% perplexity
weight-only, +4.0% for W8A8 (weights + activations).

### Continuous batching (GPT-2 124M int8, 32 requests × 64 tokens)

| max batch | aggregate tok/s | speedup | p50 latency | p95 latency |
|---:|---:|---:|---:|---:|
| 1 | 272 | 1.00× | 234 ms | 236 ms |
| 4 | 353 | 1.30× | 699 ms | 723 ms |
| 8 | 437 | 1.61× | 1114 ms | 1167 ms |
| 16 | 511 | 1.88× | 1862 ms | 1979 ms |

Throughput scales with batch size (the linear projections are batched across
sequences); per-request latency rises — the classic serving tradeoff. Scaling is
sublinear because attention is still per-sequence (what PagedAttention fixes).

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
