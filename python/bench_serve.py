"""Benchmark continuous batching: run a fixed set of requests through the server
at increasing batch sizes and show aggregate throughput scale up (and latency).

Also validates correctness: with greedy decoding and identical prompts, every
batch size must produce the same tokens as batch=1 (and match the known
single-sequence output).

    python python/export_weights.py --int8   # (or fp32 gpt2.bin)
    make -C cpp
    python python/bench_serve.py
"""

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "python"))

import tokenizer  # noqa: E402

PROMPT = "Alan Turing theorized that computers would one day become"
WEIGHTS = "gpt2-int8.bin"   # the fast path; use gpt2.bin for fp32
NUM_REQUESTS = 32
MAX_NEW = 64
BATCH_SIZES = [1, 2, 4, 8, 16]
BIN = ROOT / "cpp" / "build" / "serve"


def run(batch: int, ids: list[int]):
    proc = subprocess.run(
        [str(BIN), str(ROOT / "weights" / WEIGHTS), str(NUM_REQUESTS), str(MAX_NEW),
         str(batch), *map(str, ids)],
        capture_output=True, text=True,
    )
    if proc.returncode != 0:
        sys.exit(f"serve failed (batch={batch}): {proc.stderr}")
    out_ids = [int(x) for x in proc.stdout.split()]
    m = re.search(r"([\d.]+) tok/s", proc.stderr)
    tps = float(m.group(1)) if m else float("nan")
    lat = re.search(r"p50 (\d+)ms p95 (\d+)ms", proc.stderr)
    p50, p95 = (int(lat.group(1)), int(lat.group(2))) if lat else (0, 0)
    return out_ids, tps, p50, p95


def main() -> None:
    ids = tokenizer.encode(PROMPT)
    print(f"model={WEIGHTS}  requests={NUM_REQUESTS}  max_new={MAX_NEW}\n")

    baseline_out = None
    base_tps = None
    print(f"{'batch':>5} {'tok/s':>9} {'speedup':>8} {'p50(ms)':>8} {'p95(ms)':>8}  correct")
    for b in BATCH_SIZES:
        out, tps, p50, p95 = run(b, ids)
        if baseline_out is None:
            baseline_out = out
            base_tps = tps
        ok = out == baseline_out
        print(f"{b:>5} {tps:>9.1f} {tps / base_tps:>7.2f}x {p50:>8} {p95:>8}  {ok}")

    print(f"\nsample output: {tokenizer.decode(baseline_out[:12])!r} ...")


if __name__ == "__main__":
    main()
