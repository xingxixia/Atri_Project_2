from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from threading import Thread
from typing import Iterable

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

from atri.config import project_path
from atri.guardrails import clean_generated_text, deterministic_reply, is_low_quality_reply
from atri.prompts import build_system_prompt


@dataclass
class AtriChatSession:
    tokenizer: object
    model: object
    character_config: dict
    generation_config: dict
    bad_words_ids: list[list[int]] | None
    max_history_messages: int = 12
    history: list[dict[str, str]] = field(default_factory=list)

    @property
    def system_prompt(self) -> str:
        return build_system_prompt(self.character_config)

    def reply(self, user_text: str) -> str:
        user_text = str(user_text).strip()
        fixed = deterministic_reply(user_text, self.character_config)
        if fixed is not None:
            self._append_turn(user_text, fixed)
            return fixed

        messages = [{"role": "system", "content": self.system_prompt}]
        messages.extend(self.history[-self.max_history_messages :])
        messages.append({"role": "user", "content": user_text})

        raw_reply = self._generate(messages)
        reply = clean_generated_text(raw_reply, self.character_config)
        if is_low_quality_reply(reply):
            reply = f"{self.character_config['character']['owner_title']}，我刚才有点没说清楚。请再说一遍，我会认真回答。"

        self._append_turn(user_text, reply)
        return reply

    def stream_reply(self, user_text: str) -> Iterable[str]:
        reply = self.reply(user_text)
        yield reply

    def import_gradio_history(self, history) -> list[dict[str, str]]:
        messages: list[dict[str, str]] = []
        for item in history or []:
            if isinstance(item, dict):
                role = item.get("role", "user")
                content = item.get("content", "")
                if content:
                    messages.append({"role": str(role), "content": str(content)})
            elif isinstance(item, (list, tuple)) and len(item) == 2:
                user_text, assistant_text = item
                if user_text:
                    messages.append({"role": "user", "content": str(user_text)})
                if assistant_text:
                    messages.append({"role": "assistant", "content": str(assistant_text)})
        return messages[-self.max_history_messages :]

    def reply_with_external_history(self, user_text: str, history) -> str:
        old_history = self.history
        self.history = self.import_gradio_history(history)
        try:
            return self.reply(user_text)
        finally:
            self.history = old_history

    def _append_turn(self, user_text: str, reply: str) -> None:
        self.history.append({"role": "user", "content": user_text})
        self.history.append({"role": "assistant", "content": reply})
        self.history = self.history[-self.max_history_messages :]

    def _generate(self, messages: list[dict[str, str]]) -> str:
        input_ids = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        ).to(self.model.device)
        attention_mask = torch.ones_like(input_ids, device=self.model.device)

        streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)
        kwargs = dict(
            input_ids=input_ids,
            attention_mask=attention_mask,
            streamer=streamer,
            max_new_tokens=int(self.generation_config["max_new_tokens"]),
            do_sample=True,
            temperature=float(self.generation_config["temperature"]),
            top_p=float(self.generation_config["top_p"]),
            top_k=int(self.generation_config["top_k"]),
            repetition_penalty=float(self.generation_config["repetition_penalty"]),
        )
        if self.bad_words_ids:
            kwargs["bad_words_ids"] = self.bad_words_ids

        thread = Thread(target=self.model.generate, kwargs=kwargs)
        thread.start()
        return "".join(token for token in streamer)


def load_tokenizer_and_model(model_path: str, base_model: str | None = None, adapter_path: str | None = None):
    model_path_obj = Path(model_path)
    looks_like_windows_path = "\\" in model_path or ":" in model_path
    looks_like_relative_path = model_path.startswith(".")
    if not adapter_path and not model_path_obj.exists() and (looks_like_relative_path or looks_like_windows_path):
        raise FileNotFoundError(
            f"Model path does not exist: {model_path}. Train first with `python -m atri.train_model`, "
            "or pass a Hugging Face model id with --model-path for prompt-only testing."
        )

    if adapter_path:
        if not base_model:
            raise ValueError("base_model is required when adapter_path is provided")
        tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
        base = AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )
        model = PeftModel.from_pretrained(base, model_id=adapter_path)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model.eval()
    return tokenizer, model


def create_session(train_config: dict, character_config: dict, model_path: str | None = None, adapter_path: str | None = None) -> AtriChatSession:
    model_config = train_config["model"]
    default_lora_path = project_path(model_config["output_dir"])
    default_merged_path = project_path(model_config["merged_dir"])

    if model_path is None and adapter_path is None and default_lora_path.exists():
        selected_model_path = model_config["base_model"]
        adapter_path = str(default_lora_path)
    else:
        selected_model_path = model_path or str(default_merged_path)

    try:
        tokenizer, model = load_tokenizer_and_model(
            selected_model_path,
            base_model=selected_model_path if adapter_path else model_config["base_model"],
            adapter_path=adapter_path,
        )
    except ImportError as exc:
        raise RuntimeError("Missing dependency. Run `pip install -r requirements.txt` first.") from exc
    except OSError as exc:
        raise RuntimeError(
            f"Failed to load model: {selected_model_path}. If you have not trained yet, "
            "run with `--model-path Qwen/Qwen2.5-3B-Instruct` for prompt-only testing."
        ) from exc
    bad_words = character_config.get("forbidden_outputs", [])
    bad_words_ids = tokenizer(bad_words, add_special_tokens=False).input_ids if bad_words else None
    return AtriChatSession(
        tokenizer=tokenizer,
        model=model,
        character_config=character_config,
        generation_config=train_config["generation"],
        bad_words_ids=bad_words_ids,
    )
