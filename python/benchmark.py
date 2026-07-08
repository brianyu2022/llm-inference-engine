"""Benchmark the Stage 1 NumPy engine and record the BASELINE.

Two numbers that matter for an inference engine:
  1. Forward latency vs sequence length — shows how cost grows with context.
  2. Decode throughput (tok/s) — what a user actually feels, token by token.

Every later stage (C++, KV-cache, quantization, kernels) is measured against
the table this writes to benchmarks/baseline_python.md.

Rule: measure before you optimize.
"""

import platform
import time
from pathlib import Path

import numpy as np

from model import GPT2

OUT = Path(__file__).resolve().parent.parent / "benchmarks" / "baseline_python.md"
SEQ_LENS = [8, 32, 128, 256, 512]
DECODE_PROMPT_LEN = 16
DECODE_NEW_TOKENS = 32


def bench_forward(model: GPT2, T: int, reps: int = 5) -> float:
    """Best-case seconds for one forward pass over a length-T sequence."""
    ids = np.random.default_rng(0).integers(0, 50257, size=T)
    model.forward(ids)  # warm up (page in weights, let BLAS settle)
    return min(_time(lambda: model.forward(ids)) for _ in range(reps))


def bench_decode(model: GPT2) -> tuple[float, float]:
    """Generate DECODE_NEW_TOKENS greedily; return (tok/s, first-vs-last per-token ms)."""
    ids = list(np.random.default_rng(1).integers(0, 50257, size=DECODE_PROMPT_LEN))
    first = last = 0.0
    t0 = time.perf_counter()
    for i in range(DECODE_NEW_TOKENS):
        dt = _time(lambda: model.forward(np.array(ids)))
        ids.append(0)  # dummy next token — we're timing the forward, not sampling
        if i == 0:
            first = dt
        last = dt
    total = time.perf_counter() - t0
    return DECODE_NEW_TOKENS / total, (first * 1e3, last * 1e3)


def _time(fn) -> float:
    t0 = time.perf_counter()
    fn()
    return time.perf_counter() - t0


def main() -> None:
    print("loading model ...")
    model = GPT2()

    lines = []
    lines.append("| seq_len | ms/forward | tok/s (prefill) |")
    lines.append("|--------:|-----------:|----------------:|")
    print("\nforward latency vs sequence length:")
    for T in SEQ_LENS:
        dt = bench_forward(model, T)
        row = f"| {T} | {dt * 1e3:.1f} | {T / dt:.1f} |"
        lines.append(row)
        print(f"  T={T:>4}  {dt * 1e3:>8.1f} ms  ({T / dt:>7.1f} tok/s prefill)")

    tok_s, (first_ms, last_ms) = bench_decode(model)
    print(f"\ndecode throughput: {tok_s:.1f} tok/s "
          f"(per-token {first_ms:.0f}ms -> {last_ms:.0f}ms as context grows)")

    header = (
        f"# Baseline — Stage 1 (NumPy)\n\n"
        f"- machine: {platform.machine()} / {platform.system()} "
        f"(NumPy {np.__version__}, BLAS via the platform default)\n"
        f"- decode throughput: **{tok_s:.1f} tok/s** "
        f"({DECODE_NEW_TOKENS} tokens, prompt {DECODE_PROMPT_LEN})\n"
        f"- per-token latency grew {first_ms:.0f}ms -> {last_ms:.0f}ms as context grew "
        f"(the O(T^2) recompute we kill in Stage 3)\n\n"
        f"## Forward latency\n\n"
    )
    OUT.write_text(header + "\n".join(lines) + "\n")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
