"""Download GPT-2 weights + config from Hugging Face.

    python python/download_weights.py                 # 124M (default)
    python python/download_weights.py gpt2-medium     # 355M
    python python/download_weights.py gpt2-large      # 774M
    python python/download_weights.py gpt2-xl         # 1.5B

All GPT-2 sizes share one architecture, so the engine runs any of them with no
code changes. We grab the raw safetensors and read them with numpy (no torch).
"""

import sys
from pathlib import Path

from huggingface_hub import hf_hub_download

REPOS = {
    "gpt2": "openai-community/gpt2",
    "gpt2-medium": "openai-community/gpt2-medium",
    "gpt2-large": "openai-community/gpt2-large",
    "gpt2-xl": "openai-community/gpt2-xl",
}
WEIGHTS = Path(__file__).resolve().parent.parent / "weights"


def dest_dir(model: str) -> Path:
    # gpt2 keeps the top-level layout; larger models get their own subdir so
    # nothing collides (HF always names the files model.safetensors/config.json).
    return WEIGHTS if model == "gpt2" else WEIGHTS / model


def main() -> None:
    model = sys.argv[1] if len(sys.argv) > 1 else "gpt2"
    if model not in REPOS:
        sys.exit(f"unknown model {model!r}; choose from {list(REPOS)}")
    out = dest_dir(model)
    out.mkdir(parents=True, exist_ok=True)
    for fname in ("model.safetensors", "config.json"):
        print(f"downloading {model}/{fname} ...")
        path = hf_hub_download(repo_id=REPOS[model], filename=fname, local_dir=str(out))
        print(f"  -> {path}")
    print(f"\ndone. next: python python/export_weights.py --model {model} [--int8]")


if __name__ == "__main__":
    main()
