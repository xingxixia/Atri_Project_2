from __future__ import annotations

import argparse
import json
import re

from atri.clean_text import build_identity_examples, dedupe_examples, is_bad_training_text
from atri.config import load_config, project_path


def load_jsonl(path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def curate_row(row: dict, fixes: dict, data_config: dict) -> dict | None:
    instruction = row["instruction"].strip()
    output = row["output"].strip()
    merged = instruction + "\n" + output

    if any(token in merged for token in fixes.get("drop_if_contains", [])):
        return None

    if any(re.search(pattern, merged) for pattern in fixes.get("drop_if_regex", [])):
        return None

    output = fixes.get("rewrite_outputs", {}).get(output, output)

    if not looks_like_portable_dialogue(instruction, output):
        return None

    if is_bad_training_text(instruction, 1, int(data_config["max_source_chars"])):
        return None
    if is_bad_training_text(output, int(data_config["min_target_chars"]), int(data_config["max_target_chars"])):
        return None

    if instruction == output:
        return None

    return {
        "instruction": instruction,
        "input": row.get("input", ""),
        "output": output,
        "source": row.get("source", "curated"),
    }


def build_supplemental_examples(fixes: dict, data_config: dict) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for item in fixes.get("supplemental_examples", []):
        instruction = item["instruction"].strip()
        output = item["output"].strip()
        if not looks_like_portable_dialogue(instruction, output):
            continue
        if is_bad_training_text(instruction, 1, int(data_config["max_source_chars"])):
            continue
        if is_bad_training_text(output, int(data_config["min_target_chars"]), int(data_config["max_target_chars"])):
            continue
        rows.append(
            {
                "instruction": instruction,
                "input": item.get("input", ""),
                "output": output,
                "source": item.get("source", "supplemental"),
            }
        )
    return rows


def build_tone_examples(path, data_config: dict) -> list[dict[str, str]]:
    if not path.exists():
        return []
    rows: list[dict[str, str]] = []
    with path.open("r", encoding="utf-8") as f:
        items = json.load(f)
    for item in items:
        instruction = item["instruction"].strip()
        output = item["output"].strip()
        if not looks_like_portable_dialogue(instruction, output):
            continue
        if is_bad_training_text(instruction, 1, int(data_config["max_source_chars"])):
            continue
        if is_bad_training_text(output, int(data_config["min_target_chars"]), int(data_config["max_target_chars"])):
            continue
        rows.append(
            {
                "instruction": instruction,
                "input": item.get("input", ""),
                "output": output,
                "source": item.get("source", "tone_authored"),
            }
        )
    return rows


def looks_like_portable_dialogue(instruction: str, output: str) -> bool:
    merged = instruction + "\n" + output
    if len(instruction) > 60 or len(output) > 80:
        return False
    if instruction in {"……", "…………", "嗯", "哦"}:
        return False
    if output in {"怎么了？", "什么？", "星锡丅先生……", "……"}:
        return False
    scene_words = [
        "27天", "这座岛", "逃离", "管理员", "伊甸", "诗菜", "乃音子",
        "最后的命令", "证据", "销毁", "内衣", "亲亲", "亲过",
    ]
    if any(word in merged for word in scene_words):
        return False
    fact_words = [
        "之前的主人", "上一个主人", "以前的主人", "主人捡", "被主人", "主人看日志",
        "记忆", "存储断片", "日志", "笔记", "命令", "型号", "制造", "工厂", "研究院",
        "仿生人", "仿生", "真正的", "真相", "假话", "撒谎", "恋人", "交往", "告白",
        "喜欢你", "喜欢上", "无法回复", "母亲", "老太婆", "最后", "休眠", "沉睡",
        "床上", "睡觉时", "抱着你睡", "拉到床上", "强硬地", "身体擦汗", "温暖你",
        "伦理规定", "义肢", "脚", "腿", "成绩优秀", "特招生",
    ]
    if any(word in merged for word in fact_words):
        return False
    risky_words = ["死", "伤害", "紧急求助"]
    if any(word in output for word in risky_words) and "好想死" not in instruction:
        return False
    style_markers = [
        "高性能", "学习完毕", "人类", "机器人", "数据", "功能", "模块", "维护", "性能",
        "哼哼", "嗯哼", "呣", "真是的", "为什么拒绝", "不准", "才不是", "派上用场",
        "星锡丅先生", "Atri",
    ]
    if not any(marker in merged for marker in style_markers):
        return False
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply manual curation to cleaned Atri dialogue data.")
    parser.add_argument("--config", default="configs/training.json")
    parser.add_argument("--character", default="configs/character.json")
    parser.add_argument("--fixes", default="configs/manual_fixes.json")
    parser.add_argument("--tone-corpus", default="configs/atri_tone_corpus.json")
    parser.add_argument("--include-game-dialogue", action="store_true", help="Also include strictly filtered original dialogue flavor lines.")
    parser.add_argument("--preview", type=int, default=12)
    args = parser.parse_args()

    train_config = load_config(args.config)
    character_config = load_config(args.character)
    fixes = load_config(args.fixes)
    data_config = train_config["data"]

    cleaned_path = project_path(data_config["cleaned_jsonl"])
    train_path = project_path(data_config["train_jsonl"])

    rows = build_identity_examples(character_config)
    tone_rows = build_tone_examples(project_path(args.tone_corpus), data_config)
    # This project trains Atri's tone/personality, not original plot facts. Keep the
    # authored current-relationship corpus dominant and use original lines as flavor.
    for _ in range(5):
        rows.extend(tone_rows)
    if args.include_game_dialogue:
        for row in load_jsonl(cleaned_path):
            curated = curate_row(row, fixes, data_config)
            if curated is not None:
                rows.append(curated)
    rows.extend(build_supplemental_examples(fixes, data_config))

    rows = dedupe_examples(rows)
    write_jsonl(train_path, rows)

    print(f"Curated rows: {len(rows)} -> {train_path}")
    for row in rows[: max(args.preview, 0)]:
        print("---")
        print("Q:", row["instruction"])
        print("A:", row["output"])


if __name__ == "__main__":
    main()
