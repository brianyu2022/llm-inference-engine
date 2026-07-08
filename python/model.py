"""Stage 1: GPT-2 forward pass, from scratch, in NumPy.

No PyTorch. We read the raw safetensors and implement every operation ourselves:
LayerNorm, the Conv1D "linear", multi-head causal attention, the GELU MLP, and
the final tied projection to vocab logits.

Shapes use these names:
    T = sequence length (number of tokens)
    C = n_embd = 768 (model width)
    H = n_head = 12, and head_dim = C // H = 64
    V = vocab_size = 50257
"""

import json
from pathlib import Path

import numpy as np
from safetensors.numpy import load_file

WEIGHTS_DIR = Path(__file__).resolve().parent.parent / "weights"


# --- primitive ops -----------------------------------------------------------

def gelu(x: np.ndarray) -> np.ndarray:
    """GPT-2's GELU is the tanh approximation ('gelu_new'). Matching this exactly
    matters — the plain erf GELU gives slightly different logits."""
    return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x**3)))


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numerically stable softmax (subtract the max before exp)."""
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)


def layer_norm(x: np.ndarray, weight: np.ndarray, bias: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    """LayerNorm over the last axis. np.var uses population variance (ddof=0),
    which is what PyTorch's LayerNorm uses."""
    mean = np.mean(x, axis=-1, keepdims=True)
    var = np.var(x, axis=-1, keepdims=True)
    return weight * (x - mean) / np.sqrt(var + eps) + bias


def linear(x: np.ndarray, weight: np.ndarray, bias: np.ndarray) -> np.ndarray:
    """GPT-2's Conv1D stores weight as (in, out), so this is just x @ W + b with
    NO transpose — see the shapes from inspect_weights.py."""
    return x @ weight + bias


# --- transformer block -------------------------------------------------------

def attention(x: np.ndarray, w: dict, p: str, n_head: int) -> np.ndarray:
    """Multi-head causal self-attention. x is (T, C); returns (T, C)."""
    T, C = x.shape
    head_dim = C // n_head

    # Project to Q, K, V in one matmul, then split. c_attn maps C -> 3C.
    qkv = linear(x, w[f"{p}.attn.c_attn.weight"], w[f"{p}.attn.c_attn.bias"])  # (T, 3C)
    q, k, v = np.split(qkv, 3, axis=-1)                                        # each (T, C)

    # Reshape (T, C) -> (H, T, head_dim) so each head attends independently.
    def split_heads(t):
        return t.reshape(T, n_head, head_dim).transpose(1, 0, 2)
    q, k, v = split_heads(q), split_heads(k), split_heads(v)                   # (H, T, head_dim)

    # Scaled dot-product scores, then mask out the future (causal).
    scores = q @ k.transpose(0, 2, 1) / np.sqrt(head_dim)                      # (H, T, T)
    causal = np.tril(np.ones((T, T), dtype=bool))
    scores = np.where(causal, scores, -np.inf)
    weights = softmax(scores, axis=-1)                                         # (H, T, T)

    out = weights @ v                                                         # (H, T, head_dim)
    out = out.transpose(1, 0, 2).reshape(T, C)                                # merge heads -> (T, C)
    return linear(out, w[f"{p}.attn.c_proj.weight"], w[f"{p}.attn.c_proj.bias"])


def mlp(x: np.ndarray, w: dict, p: str) -> np.ndarray:
    """Position-wise feed-forward: C -> 4C -> GELU -> C."""
    h = linear(x, w[f"{p}.mlp.c_fc.weight"], w[f"{p}.mlp.c_fc.bias"])          # (T, 4C)
    h = gelu(h)
    return linear(h, w[f"{p}.mlp.c_proj.weight"], w[f"{p}.mlp.c_proj.bias"])   # (T, C)


def block(x: np.ndarray, w: dict, i: int, n_head: int) -> np.ndarray:
    """One transformer block, pre-norm with residual connections."""
    p = f"h.{i}"
    x = x + attention(layer_norm(x, w[f"{p}.ln_1.weight"], w[f"{p}.ln_1.bias"]), w, p, n_head)
    x = x + mlp(layer_norm(x, w[f"{p}.ln_2.weight"], w[f"{p}.ln_2.bias"]), w, p)
    return x


# --- model -------------------------------------------------------------------

class GPT2:
    def __init__(self, weights_dir: Path = WEIGHTS_DIR):
        cfg = json.loads((weights_dir / "config.json").read_text())
        self.n_head = cfg["n_head"]
        self.n_layer = cfg["n_layer"]
        self.n_ctx = cfg["n_positions"]
        self.w = load_file(str(weights_dir / "model.safetensors"))

    def forward(self, input_ids: np.ndarray) -> np.ndarray:
        """input_ids: 1-D array of token ids (length T). Returns logits (T, V)."""
        input_ids = np.asarray(input_ids)
        T = input_ids.shape[0]
        if T > self.n_ctx:
            raise ValueError(f"sequence length {T} exceeds context window {self.n_ctx}")

        wte = self.w["wte.weight"]   # (V, C) token embeddings
        wpe = self.w["wpe.weight"]   # (n_ctx, C) position embeddings

        # Embed: look up each token, add the position embedding for its slot.
        x = wte[input_ids] + wpe[np.arange(T)]                                 # (T, C)

        for i in range(self.n_layer):
            x = block(x, self.w, i, self.n_head)

        x = layer_norm(x, self.w["ln_f.weight"], self.w["ln_f.bias"])          # (T, C)
        logits = x @ wte.T                                                    # (T, V) — weight tying
        return logits
