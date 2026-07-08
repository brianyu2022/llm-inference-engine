# Building an LLM Inference Engine From Scratch

A from-scratch GPT-2 inference engine — no PyTorch, no ML framework — that ends up
**beating Apple's own optimized BLAS** with a hand-written int8 kernel, and serves
concurrent requests with continuous batching. This is the story of how it was
built and what each optimization taught.

Everything is benchmarked on an **Apple M4 Pro**, GPT-2 124M, greedy decoding.

---

## Why build this

Most people who list "LLMs" on a resume have called an API or `model.generate()`.
I wanted to understand what actually happens when a language model runs on
hardware — the memory traffic, the arithmetic intensity, the difference between
processing a prompt and generating a token — and to be able to make it fast
myself. So I built the whole thing from the weights up and optimized it with
measurements at every step.

---

## The arc, in one number

Decode throughput for GPT-2 124M, from the first working version to the last:

| Version | tok/s | notes |
|---|---:|---|
| NumPy reference | ~10 | correct, no framework |
| C++ + Accelerate BLAS | ~76 | kill interpreter overhead |
| C++ + KV-cache | ~243 | stop recomputing history |
| C++ + KV-cache + W8A8 int8 kernel | **~302** | **beats fp32 BLAS 1.21×** |

Plus continuous batching for **1.88× aggregate throughput** at batch 16.

---

## A correct forward pass (NumPy)

I implemented GPT-2's forward pass in pure NumPy: tokenizer, embeddings,
multi-head causal attention, the tanh-GELU MLP, LayerNorm, and weight-tied
logits — reading the real 124M weights straight from safetensors (no torch).

**Validation:** greedy decoding is deterministic, so matching GPT-2's known
continuation token-for-token proves correctness:

> *Alan Turing theorized that computers would one day become* **the most powerful
> machines on the planet.**

**Gotcha learned:** GPT-2 uses OpenAI `Conv1D` layers, which store weights already
transposed vs. `nn.Linear` — so the op is `x @ W`, no transpose. Getting this
wrong is the classic from-scratch bug.

Baseline: **~10 tok/s**, and it got *slower* with sequence length.

## A C++ engine

Ported the forward pass to C++ on Apple's Accelerate BLAS (`cblas_sgemm`). To make
loading trivial I designed a small flat binary weight format (the same idea as
llama.cpp's GGUF).

Result: **~76 tok/s, ~8× over NumPy** — validated token-for-token against the
reference. The insight: at decode the bottleneck wasn't the matmuls (BLAS does
those in both), it was **Python interpreter overhead** and wastefully projecting
*every* position to the 50k vocab instead of just the last one.

## KV-cache

To generate token *N*, the naive engine reprocesses all *N* tokens, recomputing
keys/values it already computed. The KV-cache stores each layer's K/V; each step
computes K/V for just the new token and attends over the cached history. I split
the code into **prefill** (process the prompt, fill the cache) and **decode** (one
token at a time).

Result: **~243 tok/s vs. ~19 non-cached at 256 tokens (~13×)**, with per-token
latency now *flat* instead of growing. This is the single biggest optimization,
and it's algorithmic — no BLAS trick gives it to you.

**The mental model this built:** prefill is *compute-bound* (one big parallel
matmul), decode is *memory-bound* (stream all the weights to make one token).
They are different problems.

## int8 quantization

Quantized the big linear weights to int8 (per-output-column symmetric). Measured
the quality cost properly with **perplexity**: **+0.85%** weight-only. Model
shrinks 498 → 243 MB.

But the naive int8 kernel was **10× *slower*** than fp32 BLAS. Why? It read 4× less
data but ran single-threaded and unvectorized, while Accelerate uses all cores and
SIMD. **Reading less data doesn't help if you can't keep the machine busy.**

## A custom kernel that beats BLAS

Two steps to fix it:
1. **NEON SIMD + multithreading** (GCD `dispatch_apply`): 26 → 184 tok/s (7×), but
   still 0.78× of fp32. The gap is Apple's **AMX matrix coprocessor**, which
   Accelerate uses for fp32 GEMM and NEON can't match.
2. **W8A8 + SDOT**: also quantize activations (per-row, dynamic) and use ARM's
   `SDOT` integer dot-product instruction — true int8·int8→int32, 16 MACs per
   instruction, no float conversion in the inner loop.

Result: **~302 tok/s — 1.21× faster than Apple's fp32 BLAS**, at 2× smaller model
and +4.0% perplexity (greedy output unchanged). The full arc:

| int8 kernel | tok/s | vs fp32 |
|---|---:|---:|
| naive scalar | 26 | 0.10× |
| NEON + threads | 184 | 0.78× |
| **W8A8 + SDOT** | **302** | **1.21×** |

## Roofline analysis

| model | bytes/token | FLOP/byte | tok/s | achieved GB/s | % of ~273 GB/s peak |
|---|---:|---:|---:|---:|---:|
| fp32 | 494 MB | 0.50 | 241 | 119 | 44% |
| W8A8 int8 | 239 MB | 1.03 | 295 | 71 | 26% |

Decode sits far below the M4 Pro's FLOP:byte knee → **memory-bound**, which is
*why* int8 wins. There's still headroom (not saturating bandwidth), and the fp32
logits projection (`wte`, 154 MB/token) dominates int8's traffic — so quantizing
the embedding table is the clear next win.

### Acting on the roofline: quantizing `wte`

So I did it — per-row int8 for the tied embedding/logits table. One catch found by
measurement: quantizing the *activation* into the logits (W8A8) wrecked output
quality, because the logits feed an argmax over 50k classes and are
precision-sensitive. Keeping the activation fp32 (weight-only int8) fixed it. The
result: the model **nearly halved (232 → 121 MB)** at **+0.01% perplexity**, with
a **modest ~5% decode gain**. Less than a pure-bandwidth model predicts — because,
same as the linears, the int8 logits kernel runs on NEON while fp32 BLAS taps AMX.
The roofline pointed the right direction; the payoff showed up mostly as size, and
the speed ceiling is again AMX.

## Continuous batching

A scheduler keeps up to *B* sequences in flight and admits/prefills waiting
requests as slots free. The trick: **batch the linear projections across
sequences** (one matmul serves all *B*), while doing attention per-sequence
against each one's cache.

| max batch | aggregate tok/s | speedup | p50 | p95 |
|---:|---:|---:|---:|---:|
| 1 | 302 | 1.00× | 210 ms | 213 ms |
| 8 | 494 | 1.63× | 982 ms | 1030 ms |
| 16 | 581 | 1.92× | 1638 ms | 1738 ms |

Throughput scales with batch; per-request latency rises — the classic serving
tradeoff. The per-sequence attention is fanned across cores with GCD (a ~14% peak
bump); scaling is still sublinear because at large batch the batched linears
saturate too — which is what PagedAttention/FlashAttention address.

---

## Scaling

The engine reads all dimensions from the model config, so it runs any GPT-2 size
with no code changes. Verified on **GPT-2 medium (355M)** — fp32 and int8 produce
identical greedy output:

| model | params | fp32 decode | int8 decode | int8 size |
|---|---:|---:|---:|---:|
| gpt2 | 124M | 243 tok/s | 302 tok/s | 243 MB |
| gpt2-medium | 355M | 91 tok/s | 110 tok/s | 514 MB |

The ~1.2× int8 speedup holds; bigger models are more bandwidth-bound, so
quantization matters more as you scale up.

## What I'd do next

- **A hand-written AMX / Metal matmul** — the recurring speed ceiling above is
  Apple's AMX coprocessor; matching it (or moving to the GPU) is the next frontier.
- **PagedAttention** — batch attention across sequences to make batching scale
  closer to linearly.
- **A Metal GPU backend** for the matmuls.
- **A modern architecture** (Llama: RoPE, RMSNorm, SwiGLU, GQA).

## What this project demonstrates

Not "I used an inference engine" but "I built one, and I can tell you exactly
where every microsecond goes" — prefill vs. decode, compute- vs. memory-bound,
AMX vs. NEON vs. integer SDOT, and the throughput/latency tradeoff of batching.
Every optimization here was driven by a measurement, and every result was
validated against a reference.
