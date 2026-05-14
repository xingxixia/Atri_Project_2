from __future__ import annotations

import argparse

import gradio as gr

from atri.chat_core import create_session
from atri.config import load_yaml, project_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Atri Gradio chat.")
    parser.add_argument("--config", default="configs/training.json")
    parser.add_argument("--character", default="configs/character.json")
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--adapter-path", default=None)
    args = parser.parse_args()

    train_config = load_yaml(args.config)
    character_config = load_yaml(args.character)
    model_path = args.model_path
    try:
        session = create_session(train_config, character_config, model_path=model_path, adapter_path=args.adapter_path)
    except Exception as exc:
        raise SystemExit(f"启动失败：{exc}") from exc

    def predict(message, history):
        yield session.reply_with_external_history(str(message), history)

    gr.ChatInterface(
        fn=predict,
        title="Atri Project 2",
        examples=["你是谁？", "我是谁？", "我今天有点累。", "随便说点什么。"],
        cache_examples=False,
    ).launch(inbrowser=True, share=False)


if __name__ == "__main__":
    main()
