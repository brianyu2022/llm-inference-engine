"""GPT-2 byte-pair-encoding tokenizer.

For now this wraps tiktoken (OpenAI's fast, correct BPE) so we can focus Stage 1
on the forward pass. Reimplementing BPE from scratch is a great later deep-dive —
we can swap this module out without touching the model.
"""

import tiktoken

_enc = tiktoken.get_encoding("gpt2")


def encode(text: str) -> list[int]:
    return _enc.encode(text)


def decode(ids: list[int]) -> str:
    return _enc.decode(ids)
