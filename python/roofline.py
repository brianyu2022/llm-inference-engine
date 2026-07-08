"""Roofline analysis of decode. Decode is a chain of matrix-vector products (one
new token), so each weight is read once and used for ~2 FLOPs — an arithmetic
intensity well below the hardware's FLOP:byte ratio, i.e. firmly memory-bound.

This computes, per generated token: bytes of weights streamed, FLOPs, and
arithmetic intensity; then runs the engine to get tok/s and reports the achieved
memory bandwidth as a fraction of the M4 Pro's ~273 GB/s peak.
"""

import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "python"))
import tokenizer  # noqa: E402

PEAK_BW_GB_S = 273.0  # Apple M4 Pro unified-memory bandwidth (published spec)
GEN_TOKENS = 256
PROMPT = "Alan Turing theorized that computers would one day become"


def streamed(cfg: dict, int8: bool):
    """Weight elements/bytes read per decode token (the memory-bound part)."""
    C, L, V = cfg["n_embd"], cfg["n_layer"], cfg["vocab_size"]
    per_layer = 12 * C * C          # c_attn(3C^2)+c_proj(C^2)+c_fc(4C^2)+mlp_c_proj(4C^2)
    lin_elems = per_layer * L
    wte_elems = V * C               # logits projection reads the whole embedding table
    lin_bytes = lin_elems * (1 if int8 else 4)
    wte_bytes = wte_elems * 4       # wte stays fp32 in both engines
    return lin_bytes + wte_bytes, lin_elems + wte_elems


def measure_tok_s(weights: str, ids: list[int]) -> float:
    proc = subprocess.run(
        [str(ROOT / "cpp" / "build" / "generate_kv"), str(ROOT / "weights" / weights),
         str(GEN_TOKENS), *map(str, ids)],
        capture_output=True, text=True,
    )
    m = re.search(r"=\s*([\d.]+)\s*tok/s", proc.stderr)
    return float(m.group(1)) if m else float("nan")


def main() -> None:
    cfg = json.loads((ROOT / "weights" / "config.json").read_text())
    ids = tokenizer.encode(PROMPT)

    print(f"decode roofline (peak memory bandwidth assumed {PEAK_BW_GB_S:.0f} GB/s)\n")
    print(f"{'model':>12} {'bytes/tok':>10} {'FLOP/byte':>10} {'tok/s':>8} "
          f"{'GB/s':>7} {'% peak':>7}")
    for label, weights, int8 in [("fp32", "gpt2.bin", False), ("W8A8 int8", "gpt2-int8.bin", True)]:
        wbytes, params = streamed(cfg, int8)
        flops = 2 * params
        intensity = flops / wbytes
        tps = measure_tok_s(weights, ids)
        achieved_gb_s = wbytes * tps / 1e9
        print(f"{label:>12} {wbytes / 1e6:>9.0f}M {intensity:>10.2f} {tps:>8.1f} "
              f"{achieved_gb_s:>7.0f} {100 * achieved_gb_s / PEAK_BW_GB_S:>6.0f}%")

    print("\nArithmetic intensity << ~10 FLOP/byte (the M4 Pro's FLOP:bandwidth knee),")
    print("so decode is memory-bound: fewer weight bytes -> more tokens/sec, which is")
    print("why int8 wins. Note the fp32 logits (wte, ~154 MB/token) dominate int8's")
    print("traffic -> quantizing the embedding table is the next bandwidth win.")


if __name__ == "__main__":
    main()
