"""Compare the fp32 and int8 C++ engines: output quality (does int8 change the
greedy tokens?), decode throughput, and on-disk size.

Requires both weights/gpt2.bin and weights/gpt2-int8.bin, and the built engine:
    python python/export_weights.py           # gpt2.bin (fp32)
    python python/export_weights.py --int8     # gpt2-int8.bin
    make -C cpp
"""

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "python"))

import tokenizer  # noqa: E402

PROMPT = "Alan Turing theorized that computers would one day become"
BIN = ROOT / "cpp" / "build" / "generate_kv"


def run(weights: str, n: int, ids: list[int]):
    proc = subprocess.run(
        [str(BIN), str(ROOT / "weights" / weights), str(n), *map(str, ids)],
        capture_output=True, text=True,
    )
    if proc.returncode != 0:
        sys.exit(f"{weights} failed: {proc.stderr}")
    out_ids = [int(x) for x in proc.stdout.split()]
    m = re.search(r"=\s*([\d.]+)\s*tok/s", proc.stderr)
    return out_ids, (float(m.group(1)) if m else float("nan"))


def main() -> None:
    ids = tokenizer.encode(PROMPT)

    print("== quality (greedy, 8 tokens) ==")
    fp32_ids, _ = run("gpt2.bin", 8, ids)
    int8_ids, _ = run("gpt2-int8.bin", 8, ids)
    print(f"  fp32: {tokenizer.decode(fp32_ids)!r}")
    print(f"  int8: {tokenizer.decode(int8_ids)!r}")
    print(f"  identical tokens: {fp32_ids == int8_ids}")

    print("\n== decode throughput (256 tokens) ==")
    _, fp32_tps = run("gpt2.bin", 256, ids)
    _, int8_tps = run("gpt2-int8.bin", 256, ids)
    print(f"  fp32: {fp32_tps:.1f} tok/s")
    print(f"  int8: {int8_tps:.1f} tok/s   ({int8_tps / fp32_tps:.2f}x)")

    print("\n== on-disk size ==")
    for w in ("gpt2.bin", "gpt2-int8.bin"):
        print(f"  {w:16} {(ROOT / 'weights' / w).stat().st_size / 1e6:.0f} MB")


if __name__ == "__main__":
    main()
