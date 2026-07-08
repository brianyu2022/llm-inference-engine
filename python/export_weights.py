"""Export GPT-2 weights to our flat binary format for the C++ engine.

Format v2 (little-endian):
    magic  'G','P','T','2'
    int32  version (=2)
    int32  n_layer, n_head, n_embd, n_ctx, vocab_size
    int32  n_tensors
    per tensor:
        int32            name_len
        name_len bytes   name (ascii)
        int32            dtype           # 0 = fp32, 1 = int8
        int32            ndim
        ndim * int32     dims            # logical (K, N) for weight matrices
        if dtype == 0:   prod(dims) * f32     data (row-major)
        if dtype == 1:   N*K * int8           data (TRANSPOSED to (N, K))
                         N   * f32            per-column scales

Run with --int8 to quantize the big per-layer linear weights (writes
gpt2-int8.bin); without it, everything is fp32 (writes gpt2.bin). We drop the
per-block causal-mask buffers (h.N.attn.bias) either way.
"""

import argparse
import json
import re
import struct
from pathlib import Path

import numpy as np
from safetensors.numpy import load_file

from quantize import quantize_int8, should_quantize

WEIGHTS = Path(__file__).resolve().parent.parent / "weights"
MASK_BUFFER = re.compile(r"h\.\d+\.attn\.bias$")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt2", help="gpt2 | gpt2-medium | gpt2-large | gpt2-xl")
    ap.add_argument("--int8", action="store_true", help="quantize the big linear weights to int8")
    args = ap.parse_args()

    src = WEIGHTS if args.model == "gpt2" else WEIGHTS / args.model
    cfg = json.loads((src / "config.json").read_text())
    tensors = load_file(str(src / "model.safetensors"))
    keep = {k: v for k, v in tensors.items() if not MASK_BUFFER.search(k)}
    out_path = WEIGHTS / f"{args.model}{'-int8' if args.int8 else ''}.bin"

    n_quant = 0
    with open(out_path, "wb") as f:
        f.write(b"GPT2")
        f.write(struct.pack("<i", 2))  # version 2
        for key in ("n_layer", "n_head", "n_embd", "n_positions", "vocab_size"):
            f.write(struct.pack("<i", cfg[key]))
        f.write(struct.pack("<i", len(keep)))

        for name, arr in keep.items():
            name_bytes = name.encode("ascii")
            f.write(struct.pack("<i", len(name_bytes)))
            f.write(name_bytes)

            if args.int8 and should_quantize(name, arr):
                q, scale = quantize_int8(np.ascontiguousarray(arr, dtype=np.float32))  # q (K,N), scale (N,)
                q_t = np.ascontiguousarray(q.T)  # (N, K): each output column contiguous
                f.write(struct.pack("<i", 1))    # dtype int8
                f.write(struct.pack("<i", arr.ndim))
                for d in arr.shape:              # logical (K, N)
                    f.write(struct.pack("<i", int(d)))
                f.write(q_t.tobytes())
                f.write(np.ascontiguousarray(scale, dtype="<f4").tobytes())
                n_quant += 1
            else:
                a = np.ascontiguousarray(arr, dtype="<f4")
                f.write(struct.pack("<i", 0))    # dtype fp32
                f.write(struct.pack("<i", a.ndim))
                for d in a.shape:
                    f.write(struct.pack("<i", int(d)))
                f.write(a.tobytes())

    size_mb = out_path.stat().st_size / 1e6
    print(f"wrote {out_path}")
    print(f"  {len(keep)} tensors ({n_quant} int8), {size_mb:.0f} MB")


if __name__ == "__main__":
    main()
