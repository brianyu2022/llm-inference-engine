"""Measure the quality cost of int8 quantization, using the NumPy reference
(which produces full-sequence logits, unlike the decode-only C++ engine).

Perplexity = exp(mean negative log-likelihood of each token given the prior
tokens). Lower is better. We compare fp32 weights vs int8-quantized weights on a
fixed passage, and report the model-size reduction too.
"""

import numpy as np

import tokenizer
from model import GPT2
from quantize import fake_quantize_model, quantize_int8, should_quantize

# A fixed passage to score (teacher-forced). ~200 tokens of ordinary English.
TEXT = (
    "The history of computing stretches back long before the electronic computer. "
    "For centuries, people built mechanical aids to calculation, from the abacus to "
    "the geared calculators of the seventeenth century. In the nineteenth century, "
    "Charles Babbage designed the Analytical Engine, a machine that could in principle "
    "be programmed with punched cards, and Ada Lovelace wrote what many consider the "
    "first algorithm intended for such a machine. The modern era began in the twentieth "
    "century, when Alan Turing formalized the idea of computation and showed that a "
    "single universal machine could carry out any calculation that any other machine "
    "could perform. During the Second World War, engineers built the first large-scale "
    "electronic computers, and in the decades that followed, transistors and integrated "
    "circuits made machines smaller, faster, and vastly more affordable, until computers "
    "reached nearly every desk, pocket, and home on the planet."
)


def perplexity(model: GPT2, ids: list[int]) -> float:
    logits = model.forward(np.array(ids))       # (T, V)
    logits = logits[:-1]                          # position t predicts token t+1
    targets = np.array(ids[1:])
    m = logits.max(axis=-1, keepdims=True)
    logsumexp = m.squeeze(-1) + np.log(np.exp(logits - m).sum(axis=-1))
    chosen = logits[np.arange(len(targets)), targets]
    nll = -(chosen - logsumexp).mean()
    return float(np.exp(nll))


def main() -> None:
    ids = tokenizer.encode(TEXT)
    model = GPT2()

    ppl_fp32 = perplexity(model, ids)

    # Size accounting for just the tensors we quantize.
    fp32_bytes = int8_bytes = 0
    for name, arr in model.w.items():
        if should_quantize(name, arr):
            fp32_bytes += arr.nbytes
            q, s = quantize_int8(arr)
            int8_bytes += q.nbytes + s.nbytes
    total_fp32 = sum(a.nbytes for a in model.w.values())

    model.w = fake_quantize_model(model.w)
    ppl_int8 = perplexity(model, ids)

    print(f"passage: {len(ids)} tokens\n")
    print(f"perplexity fp32 : {ppl_fp32:.4f}")
    print(f"perplexity int8 : {ppl_int8:.4f}")
    print(f"quality cost    : {100 * (ppl_int8 - ppl_fp32) / ppl_fp32:+.2f}% perplexity\n")
    print(f"quantized weights: {fp32_bytes / 1e6:.0f} MB fp32 -> {int8_bytes / 1e6:.0f} MB int8 "
          f"({fp32_bytes / int8_bytes:.1f}x smaller)")
    print(f"whole model      : {total_fp32 / 1e6:.0f} MB -> "
          f"{(total_fp32 - fp32_bytes + int8_bytes) / 1e6:.0f} MB")


if __name__ == "__main__":
    main()
