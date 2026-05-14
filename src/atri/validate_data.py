from __future__ import annotations

import argparse
import json

from atri.config import load_config, project_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate cleaned Atri training data.")
    parser.add_argument("--config", default="configs/training.json")
    parser.add_argument("--character", default="configs/character.json")
    args = parser.parse_args()

    train_config = load_config(args.config)
    character_config = load_config(args.character)
    data_config = train_config["data"]
    data_path = project_path(data_config["train_jsonl"])
    forbidden = set(character_config.get("forbidden_outputs", []))
    forbidden.update({"夏生", "夏森", "小夏", "Natsuo", "User:", "用户：", "ATRI："})

    if not data_path.exists():
        raise FileNotFoundError(f"Training jsonl not found: {data_path}. Run `python -m atri.clean_text` first.")

    total = 0
    bad_rows: list[tuple[int, str, str]] = []
    with data_path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            total += 1
            item = json.loads(line)
            merged = f"{item.get('input', '')}\n{item.get('output', '')}"
            for word in forbidden:
                if word and word in merged:
                    bad_rows.append((line_no, word, item.get("output", "")))
                    break

    print(f"Rows checked: {total}")
    if bad_rows:
        print(f"Bad rows: {len(bad_rows)}")
        for line_no, word, output in bad_rows[:20]:
            print(f"- line {line_no}, forbidden={word!r}, output={output[:120]}")
        raise SystemExit(1)

    print("Data validation passed.")


if __name__ == "__main__":
    main()
