from __future__ import annotations

import argparse

from atri.chat_core import create_session
from atri.config import load_yaml, project_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Chat with Atri in terminal.")
    parser.add_argument("--config", default="configs/training.json")
    parser.add_argument("--character", default="configs/character.json")
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--adapter-path", default=None)
    args = parser.parse_args()

    train_config = load_yaml(args.config)
    character_config = load_yaml(args.character)
    model_path = args.model_path

    print(f"Loading model: {model_path or 'default LoRA/merged model'}")
    try:
        session = create_session(train_config, character_config, model_path=model_path, adapter_path=args.adapter_path)
    except Exception as exc:
        print(f"启动失败：{exc}")
        return
    print("Atri is ready. Type exit to quit.")

    while True:
        user_text = input("\n星锡丅: ").strip()
        if not user_text:
            continue
        if user_text.lower() in {"exit", "quit", "退出"}:
            print("Atri: 星锡丅先生要休息了吗？好的，我会乖乖等你回来。")
            break

        reply = session.reply(user_text)
        print(f"Atri: {reply}")


if __name__ == "__main__":
    main()
