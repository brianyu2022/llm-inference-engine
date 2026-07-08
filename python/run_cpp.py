"""Validate the C++ engines against the NumPy reference: same prompt, greedy
decoding, compared token-for-token. Checks both the plain engine and the
KV-cached engine, and prints each one's throughput.

Tokenization stays in Python — we encode here, pass token ids to the C++ binary,
and decode its output. This is also how you'd actually drive the engine.
"""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "python"))

import tokenizer  # noqa: E402
from model import GPT2  # noqa: E402
from generate import generate as numpy_generate  # noqa: E402

PROMPT = "Alan Turing theorized that computers would one day become"
N = 8
ENGINES = ["generate", "generate_kv"]
WEIGHTS = ROOT / "weights" / "gpt2.bin"


def run_engine(name: str, ids: list[int]) -> list[int]:
    binary = ROOT / "cpp" / "build" / name
    proc = subprocess.run(
        [str(binary), str(WEIGHTS), str(N), *map(str, ids)],
        capture_output=True, text=True,
    )
    sys.stderr.write(f"[{name}] " + proc.stderr.strip().replace("\n", "\n         ") + "\n")
    if proc.returncode != 0:
        sys.exit(f"{name} failed (exit {proc.returncode})")
    return [int(x) for x in proc.stdout.split()]


def main() -> None:
    ids = tokenizer.encode(PROMPT)

    ref_full = numpy_generate(GPT2(), ids, N, 0.0, None, 0)
    ref_ids = ref_full[len(ids):]
    print(f"\nprompt   : {PROMPT!r}")
    print(f"reference: {tokenizer.decode(ref_ids)!r}  {ref_ids}\n")

    all_ok = True
    for name in ENGINES:
        out_ids = run_engine(name, ids)
        ok = out_ids == ref_ids
        all_ok &= ok
        status = "PASS" if ok else "FAIL"
        print(f"{status}  {name:12} {tokenizer.decode(out_ids)!r}")

    print("\nAll engines match the NumPy reference." if all_ok else "\nMismatch detected.")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
