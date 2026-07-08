"""Load the raw GPT-2 weights and print the config, the tensor shapes, and the
total parameter count.

This is your *map* of the model: every tensor printed here is something the
forward pass we build next will consume. Run it once you've downloaded weights.
"""

import json
import re
from pathlib import Path

from safetensors.numpy import load_file

WEIGHTS = Path(__file__).resolve().parent.parent / "weights"
_LAYER = re.compile(r"\bh\.(\d+)\.")  # matches the per-block tensors, e.g. h.0., h.11.


def main() -> None:
    cfg = json.loads((WEIGHTS / "config.json").read_text())
    print("=== config ===")
    for k in ("n_layer", "n_head", "n_embd", "n_positions", "vocab_size"):
        print(f"  {k:12} {cfg.get(k)}")

    tensors = load_file(str(WEIGHTS / "model.safetensors"))

    total = 0
    n_layers = 0
    print("\n=== tensors (showing all non-block tensors + block 0 only) ===")
    for name in sorted(tensors):
        t = tensors[name]
        total += t.size
        m = _LAYER.search(name)
        if m is not None:
            n_layers = max(n_layers, int(m.group(1)) + 1)
        # Print everything except repeated transformer blocks 1..n-1, so the
        # output stays readable — block 0 is representative of them all.
        if m is None or m.group(1) == "0":
            print(f"  {name:34} {str(tuple(t.shape)):18} {t.dtype}")

    print(f"\ndetected {n_layers} transformer blocks (each identical in shape to block 0)")
    print(f"total parameters: {total:,}")


if __name__ == "__main__":
    main()
