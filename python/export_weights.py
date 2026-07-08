"""Stage 2 prep: export GPT-2 weights to a simple flat binary our C++ engine can
load with no JSON/safetensors parsing.

This mirrors what real engines do (llama.cpp's GGUF): pick a dead-simple on-disk
format so the C++ loader is a few lines and loads fast.

Format (all little-endian):
    magic  'G','P','T','2'
    int32  version (=1)
    int32  n_layer, n_head, n_embd, n_ctx, vocab_size   (config, in that order)
    int32  n_tensors
    then per tensor:
        int32              name_len
        name_len bytes     name (ascii)
        int32              ndim
        ndim * int32       dims
        prod(dims) * f32   data (row-major)

We drop the per-block causal-mask buffers (h.N.attn.bias) — the C++ engine builds
its own mask, same as the NumPy version.
"""

import json
import re
import struct
from pathlib import Path

import numpy as np
from safetensors.numpy import load_file

WEIGHTS = Path(__file__).resolve().parent.parent / "weights"
OUT = WEIGHTS / "gpt2.bin"
MASK_BUFFER = re.compile(r"h\.\d+\.attn\.bias$")  # the (1,1,1024,1024) causal mask, not a real weight


def main() -> None:
    cfg = json.loads((WEIGHTS / "config.json").read_text())
    tensors = load_file(str(WEIGHTS / "model.safetensors"))
    keep = {k: v for k, v in tensors.items() if not MASK_BUFFER.search(k)}

    with open(OUT, "wb") as f:
        f.write(b"GPT2")
        f.write(struct.pack("<i", 1))  # version
        for key in ("n_layer", "n_head", "n_embd", "n_positions", "vocab_size"):
            f.write(struct.pack("<i", cfg[key]))
        f.write(struct.pack("<i", len(keep)))
        for name, arr in keep.items():
            arr = np.ascontiguousarray(arr, dtype="<f4")
            name_bytes = name.encode("ascii")
            f.write(struct.pack("<i", len(name_bytes)))
            f.write(name_bytes)
            f.write(struct.pack("<i", arr.ndim))
            for d in arr.shape:
                f.write(struct.pack("<i", int(d)))
            f.write(arr.tobytes())

    dropped = len(tensors) - len(keep)
    size_mb = OUT.stat().st_size / 1e6
    print(f"wrote {OUT}")
    print(f"  {len(keep)} tensors ({dropped} mask buffers dropped), {size_mb:.0f} MB")


if __name__ == "__main__":
    main()
