from __future__ import annotations

import argparse
import os
import shutil
import json
import subprocess
import sys
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorForSeq2Seq, Trainer, TrainingArguments

from atri.config import load_yaml, project_path
from atri.prompts import build_system_prompt


def resolve_base_model(model_config: dict) -> str:
    return os.environ.get("ATRI_BASE_MODEL_PATH") or model_config.get("base_model_path") or model_config["base_model"]


def ensure_training_data(config_path: str, character_path: str, train_jsonl: Path) -> None:
    if train_jsonl.exists():
        return
    subprocess.run(
        [sys.executable, "-m", "atri.clean_text", "--config", config_path, "--character", character_path],
        check=True,
    )
    subprocess.run(
        [sys.executable, "-m", "atri.curate_data", "--config", config_path, "--character", character_path],
        check=True,
    )


def validate_training_data(config_path: str, character_path: str) -> None:
    subprocess.run(
        [sys.executable, "-m", "atri.validate_data", "--config", config_path, "--character", character_path],
        check=True,
    )


def load_jsonl(path: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    if not rows:
        raise ValueError(f"No training rows found in {path}")
    return rows


def as_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def make_tokenize_func(tokenizer, system_prompt: str, max_length: int):
    def tokenize(example: dict[str, str]) -> dict[str, list[int]]:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": example["instruction"] + example.get("input", "")},
        ]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        response = f"{example['output']}{tokenizer.eos_token or ''}"

        prompt_tokens = tokenizer(prompt, add_special_tokens=False)
        response_tokens = tokenizer(response, add_special_tokens=False)

        prompt_ids = prompt_tokens["input_ids"]
        prompt_mask = prompt_tokens["attention_mask"]
        response_ids = response_tokens["input_ids"]
        response_mask = response_tokens["attention_mask"]

        if len(response_ids) >= max_length:
            response_ids = response_ids[: max_length - 1] + [tokenizer.eos_token_id]
            response_mask = response_mask[:max_length]
            prompt_ids = []
            prompt_mask = []

        available_prompt_len = max_length - len(response_ids)
        if len(prompt_ids) > available_prompt_len:
            prompt_ids = prompt_ids[-available_prompt_len:]
            prompt_mask = prompt_mask[-available_prompt_len:]

        input_ids = prompt_ids + response_ids
        attention_mask = prompt_mask + response_mask
        labels = [-100] * len(prompt_ids) + response_ids

        if all(label == -100 for label in labels):
            raise ValueError(f"Training sample has no response labels after truncation: {example}")

        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

    return tokenize


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Atri LoRA model.")
    parser.add_argument("--config", default="configs/training.json")
    parser.add_argument("--character", default="configs/character.json")
    parser.add_argument("--merge", action="store_true", help="Merge LoRA into a full model after training. This can need more memory.")
    parser.add_argument("--skip-validate", action="store_true", help="Skip data validation before training.")
    args = parser.parse_args()

    train_config = load_yaml(args.config)
    character_config = load_yaml(args.character)
    system_prompt = build_system_prompt(character_config)

    model_config = train_config["model"]
    data_config = train_config["data"]
    lora_config_raw = train_config["lora"]
    training_raw = train_config["training"]

    train_jsonl = project_path(data_config["train_jsonl"])
    ensure_training_data(args.config, args.character, train_jsonl)
    if not args.skip_validate:
        validate_training_data(args.config, args.character)
    rows = load_jsonl(train_jsonl)

    base_model = resolve_base_model(model_config)
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.float16 if model_config.get("torch_dtype") == "float16" else "auto",
        device_map="auto",
        trust_remote_code=True,
    )

    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        target_modules=lora_config_raw["target_modules"],
        r=int(lora_config_raw["r"]),
        lora_alpha=int(lora_config_raw["alpha"]),
        lora_dropout=float(lora_config_raw["dropout"]),
    )
    model = get_peft_model(model, peft_config)
    model.enable_input_require_grads()
    model.print_trainable_parameters()

    dataset = Dataset.from_list(rows)
    tokenized = dataset.map(
        make_tokenize_func(tokenizer, system_prompt, int(training_raw["max_length"])),
        remove_columns=dataset.column_names,
    )

    output_dir = project_path(model_config["output_dir"])
    args_train = TrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=int(training_raw["per_device_train_batch_size"]),
        gradient_accumulation_steps=int(training_raw["gradient_accumulation_steps"]),
        logging_steps=int(training_raw["logging_steps"]),
        num_train_epochs=float(training_raw["num_train_epochs"]),
        save_strategy=training_raw["save_strategy"],
        learning_rate=float(training_raw["learning_rate"]),
        gradient_checkpointing=as_bool(training_raw["gradient_checkpointing"]),
        dataloader_num_workers=0,
        fp16=as_bool(training_raw["fp16"]),
        remove_unused_columns=False,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=args_train,
        train_dataset=tokenized,
        tokenizer=tokenizer,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True, pad_to_multiple_of=8),
    )

    trainer.train()
    trainer.model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    save_run_metadata(output_dir, args.config, args.character, system_prompt)
    print(f"LoRA saved to: {output_dir}")

    if not args.merge:
        print("Skipping merge. Use --merge later if you need a standalone merged model.")
        return

    merge_lora(model_config, output_dir, tokenizer)


def save_run_metadata(output_dir: Path, config_path: str, character_path: str, system_prompt: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(project_path(config_path), output_dir / "training.config.json")
    shutil.copy2(project_path(character_path), output_dir / "character.config.json")
    (output_dir / "system_prompt.txt").write_text(system_prompt, encoding="utf-8")


def merge_lora(model_config: dict, lora_dir: Path, tokenizer) -> None:
    merged_dir = project_path(model_config["merged_dir"])
    base_model_path = resolve_base_model(model_config)
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    merged_model = PeftModel.from_pretrained(base_model, model_id=str(lora_dir)).merge_and_unload()
    merged_model.save_pretrained(merged_dir)
    tokenizer.save_pretrained(merged_dir)
    print(f"Merged model saved to: {merged_dir}")


if __name__ == "__main__":
    main()
