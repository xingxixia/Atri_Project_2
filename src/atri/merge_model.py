from __future__ import annotations

import argparse

from transformers import AutoTokenizer

from atri.config import load_config, project_path
from atri.train_model import merge_lora


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge a trained Atri LoRA adapter into a standalone model.")
    parser.add_argument("--config", default="configs/training.json")
    parser.add_argument("--lora-dir", default=None)
    args = parser.parse_args()

    train_config = load_config(args.config)
    model_config = train_config["model"]
    lora_dir = project_path(args.lora_dir or model_config["output_dir"])

    if not lora_dir.exists():
        raise FileNotFoundError(f"LoRA directory not found: {lora_dir}")

    tokenizer = AutoTokenizer.from_pretrained(lora_dir, trust_remote_code=True)
    merge_lora(model_config, lora_dir, tokenizer)


if __name__ == "__main__":
    main()
