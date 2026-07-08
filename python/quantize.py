"""int8 weight quantization for GPT-2.

Symmetric, per-output-column quantization: for a weight W (K x N), each output
column n gets its own scale s[n] = max|W[:, n]| / 127, so W[:, n] ~= q[:, n]*s[n]
with q in int8. Then y = x @ W becomes  y[n] = s[n] * sum_k x[k] * q[k, n].

This is the exact scheme the C++ engine will use. Here we also "fake-quantize"
(quantize then dequantize back to fp32) so we can measure the quality cost in the
NumPy reference before writing the fast integer kernel.
"""

import re

import numpy as np

# The big per-layer linear weights we quantize (the memory-bandwidth hogs).
# 1-D tensors (biases, LayerNorm) and the embedding table stay fp32.
QUANT_PATTERNS = [
    re.compile(r"h\.\d+\.attn\.c_attn\.weight$"),
    re.compile(r"h\.\d+\.attn\.c_proj\.weight$"),
    re.compile(r"h\.\d+\.mlp\.c_fc\.weight$"),
    re.compile(r"h\.\d+\.mlp\.c_proj\.weight$"),
]


def should_quantize(name: str, arr: np.ndarray) -> bool:
    return arr.ndim == 2 and any(p.search(name) for p in QUANT_PATTERNS)


def quantize_int8(W: np.ndarray):
    """W (K, N) fp32 -> (q int8 (K, N), scale fp32 (N,)), per-column symmetric."""
    maxabs = np.max(np.abs(W), axis=0)  # (N,)
    scale = np.where(maxabs > 0, maxabs / 127.0, 1.0).astype(np.float32)
    q = np.clip(np.round(W / scale), -127, 127).astype(np.int8)
    return q, scale


def dequantize_int8(q: np.ndarray, scale: np.ndarray) -> np.ndarray:
    return q.astype(np.float32) * scale  # broadcast over the N columns


def fake_quantize_model(weights: dict) -> dict:
    """Copy of the weights with the big linears round-tripped through int8.
    Used to measure the quality impact in the fp32 NumPy reference."""
    out = dict(weights)
    for name, arr in weights.items():
        if should_quantize(name, arr):
            q, s = quantize_int8(arr)
            out[name] = dequantize_int8(q, s)
    return out
