"""Stage 1 CLI: prompt in, generated text out.

    python python/generate.py --prompt "Alan Turing theorized that computers would one day become" --n 8

With the default greedy decoding (temperature 0), GPT-2 124M should continue that
exact prompt with:  " the most powerful machines on the planet."
That's our correctness check — if you get that, the forward pass is right.

Note: this recomputes the full sequence for every new token (no KV-cache yet).
That's O(T^2) work and it's the whole motivation for Stage 3.
"""

import argparse
import time

import numpy as np

from model import GPT2, softmax
import tokenizer


def sample_next(logits: np.ndarray, temperature: float, top_k: int | None, rng: np.random.Generator) -> int:
    """Pick the next token id from the last-position logits (shape (V,))."""
    if temperature == 0.0:
        return int(np.argmax(logits))

    logits = logits / temperature
    if top_k is not None:
        # Keep only the top-k logits; mask the rest to -inf before softmax.
        kth = np.partition(logits, -top_k)[-top_k]
        logits = np.where(logits < kth, -np.inf, logits)
    probs = softmax(logits)
    return int(rng.choice(len(probs), p=probs))


def generate(model: GPT2, ids: list[int], n_new: int, temperature: float, top_k: int | None, seed: int) -> list[int]:
    rng = np.random.default_rng(seed)
    ids = list(ids)
    for _ in range(n_new):
        logits = model.forward(np.array(ids))     # (T, V)
        next_id = sample_next(logits[-1], temperature, top_k, rng)
        ids.append(next_id)
    return ids


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", default="Alan Turing theorized that computers would one day become")
    ap.add_argument("--n", type=int, default=8, help="number of new tokens to generate")
    ap.add_argument("--temperature", type=float, default=0.0, help="0 = greedy/deterministic")
    ap.add_argument("--top-k", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    print("loading model ...")
    model = GPT2()

    prompt_ids = tokenizer.encode(args.prompt)
    print(f"prompt: {args.prompt!r}  ({len(prompt_ids)} tokens)")

    t0 = time.perf_counter()
    out_ids = generate(model, prompt_ids, args.n, args.temperature, args.top_k, args.seed)
    dt = time.perf_counter() - t0

    completion = tokenizer.decode(out_ids[len(prompt_ids):])
    print(f"\n{args.prompt}{completion}")
    print(f"\n[{args.n} tokens in {dt:.2f}s  =  {args.n / dt:.1f} tok/s]")


if __name__ == "__main__":
    main()
