"""Stage 1: download GPT-2 (124M) weights + config from Hugging Face.

We deliberately grab the raw `model.safetensors` and read it with numpy later,
so our inference engine never depends on PyTorch. This is the only script that
touches the network.
"""

from pathlib import Path

from huggingface_hub import hf_hub_download

REPO = "openai-community/gpt2"  # the original 124M GPT-2
FILES = ("model.safetensors", "config.json")
OUT = Path(__file__).resolve().parent.parent / "weights"


def main() -> None:
    OUT.mkdir(exist_ok=True)
    for fname in FILES:
        print(f"downloading {fname} from {REPO} ...")
        path = hf_hub_download(repo_id=REPO, filename=fname, local_dir=str(OUT))
        print(f"  -> {path}")
    print("\ndone. next: python python/inspect_weights.py")


if __name__ == "__main__":
    main()
