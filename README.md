# LLM Inference Engine

**A from-scratch GPT-2 inference engine in C++ — with a hand-written int8 kernel
that runs 1.21× faster than Apple's Accelerate BLAS.**

No PyTorch, no ML framework. The forward pass, KV-cache, quantization, custom
SIMD kernels, and a continuous-batching server are all implemented from the
weights up and validated token-for-token against a reference. Built and
benchmarked on an Apple M4 Pro.

📄 **[Read the full writeup →](WRITEUP.md)** — the engineering story and what each
optimization taught (prefill vs. decode, memory-bound kernels, AMX vs. NEON vs.
integer SDOT, the batching throughput/latency tradeoff).

---

## Highlights

- **From-scratch forward pass** — GPT-2 implemented first in NumPy, then C++,
  reading raw safetensors with no framework. Validated token-for-token against
  the reference greedy output.
- **A custom int8 kernel that beats the vendor** — W8A8 quantization with an ARM
  **NEON + `SDOT`** matmul, multithreaded with Grand Central Dispatch:
  **1.21× faster than Apple's fp32 BLAS**, at a 2–4× smaller model.
- **KV-cache** with a prefill/decode split — **13× faster decode** and flat
  per-token latency.
- **Continuous batching** — batches the linear projections across concurrent
  requests and fans attention across cores: **1.92× throughput** at batch 16,
  with p50/p95 latency reporting.
- **Profiling-driven** — a roofline analysis proves decode is memory-bound and
  drove every optimization; quantization quality is measured with perplexity.
- **Architecture-general** — runs any GPT-2 size (124M / 355M / 774M / 1.5B) from
  the model config with no code changes.

## Benchmarks

GPT-2 124M, Apple M4 Pro, greedy decode.

**Decode throughput — the optimization journey**

| Engine | tok/s | vs. fp32 BLAS |
|---|---:|---:|
| NumPy reference | ~10 | |
| C++, no KV-cache | ~19 | |
| C++ + KV-cache (fp32 BLAS) | ~249 | 1.00× |
| + int8, naive scalar kernel | ~26 | 0.10× |
| + int8, NEON + multithreaded | ~184 | 0.74× |
| **+ int8, W8A8 + SDOT kernel** | **~302** | **1.21×** |

**Model size & quality**

| Build | Size | Perplexity vs. fp32 |
|---|---:|---:|
| fp32 | 498 MB | — |
| int8 (W8A8) | 243 MB | +4.0% |
| int8 + quantized embedding table | 128 MB | +4.1% |

*(Weight-only int8 — fp32 activations — costs just +0.85%; the W8A8 kernel trades
~+4% perplexity for the SDOT speedup. Greedy output is unchanged on test prompts.)*

**Continuous batching** (32 requests × 64 tokens, int8)

| Max batch | Aggregate tok/s | Speedup | p50 | p95 |
|---:|---:|---:|---:|---:|
| 1 | 302 | 1.00× | 210 ms | 213 ms |
| 4 | 382 | 1.26× | 648 ms | 673 ms |
| 8 | 494 | 1.63× | 982 ms | 1030 ms |
| 16 | 581 | 1.92× | 1638 ms | 1738 ms |

**Scaling** (runs larger models with no code changes)

| Model | Params | fp32 decode | int8 decode | int8 size |
|---|---:|---:|---:|---:|
| GPT-2 | 124M | 249 tok/s | 302 tok/s | 243 MB |
| GPT-2 medium | 355M | 91 tok/s | 110 tok/s | 514 MB |

## How it works

```
safetensors ──► flat binary weight format ──► C++ engine
 (Hugging Face)   (export_weights.py, GGUF-like)   │
                                                    ├─ embed
                                                    ├─ N × (multi-head attention + MLP)   ← KV-cache on decode
                                                    └─ logits                              ← int8 SDOT kernels
```

The matmuls use Apple's Accelerate BLAS for fp32 and a hand-written NEON/`SDOT`
kernel for int8. Tokenization uses `tiktoken` (Python) around the C++ core.

## Quickstart

```sh
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python python/download_weights.py                    # GPT-2 124M weights
python python/export_weights.py --int8 --quant-wte   # -> weights/gpt2-int8w.bin
make -C cpp                                          # build the engine
python python/run_cpp.py                             # validate C++ vs. NumPy reference
```

Generate, benchmark, and serve:

```sh
python python/bench_quant.py    # fp32 vs. int8 decode + quality
python python/bench_serve.py    # continuous-batching throughput and latency
python python/roofline.py       # memory-bandwidth roofline analysis
```

## Repository layout

```
python/   NumPy reference, weight export/quantization, tokenizer,
          and benchmarks (quantization, batching, roofline, perplexity)
cpp/      C++ engine: weight loader, forward pass, KV-cache,
          int8 NEON/SDOT kernels, and the continuous-batching server
weights/  downloaded weights + exported binaries (gitignored)
```

## Future work

- Match Apple's AMX matrix coprocessor with a hand-tuned kernel, or a Metal GPU backend
- PagedAttention, so batched throughput scales closer to linearly
- A modern architecture (Llama: RoPE, RMSNorm, SwiGLU, grouped-query attention)
