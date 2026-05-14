from __future__ import annotations

import argparse
import json
import re
import zipfile
from dataclasses import dataclass
from pathlib import Path
from xml.etree import ElementTree as ET

from atri.config import load_config, project_path


SPEAKER_RE = re.compile(r"^(?P<speaker>ATRI|亚托莉|夏生|斑鸠|龙司|水菜萌|凯瑟琳|凛凛花)\s*[：:]\s*(?P<text>.*)$")
NOTE_RE = re.compile(r"[（(][^）)]*(原文|注|译|应为|多加|错字|此处|应该|疑似)[^）)]*[）)]")
BRACKET_RE = re.compile(r"[【\[].*?[】\]]")
SPACE_RE = re.compile(r"\s+")
OTHER_CHARACTER_RE = re.compile(r"(凯瑟琳|龙司|斑鸠|水菜萌|凛凛花|洋子|老师|部长)")
NARRATION_RE = re.compile(
    r"(我看着|我不禁|我用|她看着|她说|他说|Atri看|Atri冲|Atri吐|"
    r"话音刚落|根据她|顺便一提|确实有道理|我想不出|跟她争|"
    r"部员|社团|料理部|生日|研究院)"
)


@dataclass
class DialogueTurn:
    speaker: str
    text: str


def normalize_text(text: str) -> str:
    text = text.replace("\u3000", " ").replace("\xa0", " ")
    text = NOTE_RE.sub("", text)
    text = BRACKET_RE.sub("", text)
    text = re.sub(r"(ATRI|亚托莉|夏生|星锡丅)\s*[：:]\s*", "", text)
    text = SPACE_RE.sub(" ", text)
    return text.strip()


def read_dialogue_turns(docx_path: Path) -> list[DialogueTurn]:
    paragraphs = read_document_paragraphs(docx_path)
    turns: list[DialogueTurn] = []

    for raw in paragraphs:
        raw = raw.strip()
        if not raw:
            continue
        matched = SPEAKER_RE.match(raw)
        if matched:
            text = normalize_text(matched.group("text"))
            if text:
                turns.append(DialogueTurn(matched.group("speaker"), text))
    return turns


def read_document_paragraphs(path: Path) -> list[str]:
    if path.exists() and path.suffix.lower() == ".docx":
        return read_docx_paragraphs(path)
    if path.exists() and path.suffix.lower() == ".txt":
        return path.read_text(encoding="utf-8").splitlines()
    raise FileNotFoundError(
        f"Could not find raw text file: {path}. Put the source docx/txt under data/raw, "
        "or update configs/training.json."
    )


def read_docx_paragraphs(docx_path: Path) -> list[str]:
    try:
        from docx import Document
    except ImportError:
        return read_docx_paragraphs_stdlib(docx_path)

    doc = Document(docx_path)
    return [paragraph.text for paragraph in doc.paragraphs]


def read_docx_paragraphs_stdlib(docx_path: Path) -> list[str]:
    namespace = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
    with zipfile.ZipFile(docx_path) as zf:
        xml_bytes = zf.read("word/document.xml")
    root = ET.fromstring(xml_bytes)
    paragraphs: list[str] = []
    for paragraph in root.findall(".//w:p", namespace):
        texts = [node.text or "" for node in paragraph.findall(".//w:t", namespace)]
        if texts:
            paragraphs.append("".join(texts))
    return paragraphs


def is_bad_training_text(text: str, min_chars: int, max_chars: int) -> bool:
    if len(text) < min_chars or len(text) > max_chars:
        return True
    if text.count("。") + text.count("！") + text.count("？") > 8:
        return True
    if "http://" in text or "https://" in text:
        return True
    if re.search(r"(原文|注：|译者|百度|网盘|galgame|贴吧)", text, re.I):
        return True
    if OTHER_CHARACTER_RE.search(text):
        return True
    if NARRATION_RE.search(text):
        return True
    if "「" in text or "」" in text:
        return True
    if "（" in text or "）" in text:
        return True
    return False


def build_dialogue_examples(turns: list[DialogueTurn], data_config: dict) -> list[dict[str, str]]:
    examples: list[dict[str, str]] = []
    max_source = int(data_config["max_source_chars"])
    max_target = int(data_config["max_target_chars"])
    min_target = int(data_config["min_target_chars"])

    for prev, current in zip(turns, turns[1:]):
        if prev.speaker != "夏生" or current.speaker not in {"ATRI", "亚托莉"}:
            continue

        source = normalize_text(prev.text)
        target = normalize_text(current.text)
        if is_bad_training_text(source, 1, max_source):
            continue
        if is_bad_training_text(target, min_target, max_target):
            continue

        source = rewrite_current_relationship(source)
        target = rewrite_current_relationship(target)
        source = source.replace("ATRI", "Atri")
        target = target.replace("ATRI", "Atri")
        if "夏生" in source or "夏生" in target:
            continue
        examples.append({"instruction": source, "input": "", "output": target, "source": "game_dialogue"})

    return dedupe_examples(examples)


def rewrite_current_relationship(text: str) -> str:
    text = text.replace("夏生君", "星锡丅先生")
    text = text.replace("夏生先生", "星锡丅先生")
    text = text.replace("夏生", "星锡丅先生")
    return text


def build_identity_examples(character_config: dict) -> list[dict[str, str]]:
    owner = character_config["character"]["owner_name"]
    owner_title = character_config["character"]["owner_title"]
    return [
        {"instruction": "你是谁？", "input": "", "output": f"我是 Atri，{owner_title}的高性能机器人。", "source": "identity"},
        {"instruction": "你叫什么名字？", "input": "", "output": "我的名字是 Atri，也可以叫我亚托莉。", "source": "identity"},
        {"instruction": "我是谁？", "input": "", "output": f"你是{owner_title}，也是我的第一个主人。", "source": "identity"},
        {"instruction": "你的主人是谁？", "input": "", "output": f"我的第一个主人是{owner}，我会称呼他为{owner_title}。", "source": "identity"},
        {"instruction": "随便说点什么。", "input": "", "output": f"{owner_title}，我在这里。有什么想做的事，就交给高性能的我吧。", "source": "identity"},
        {"instruction": "我今天有点累。", "input": "", "output": f"辛苦了，{owner_title}。先休息一下吧，我会陪着你的。", "source": "identity"},
        {"instruction": "我不开心。", "input": "", "output": f"{owner_title}，先别一个人闷着。我会在这里陪你。", "source": "identity"},
        {"instruction": "好想死。", "input": "", "output": f"{owner_title}，请先停下来，别伤害自己。现在先去找身边可靠的人，或者联系紧急求助；我会陪你把这一分钟撑过去。", "source": "identity"},
        {"instruction": "我能抱着你睡吗？", "input": "", "output": f"如果这样能让{owner_title}安心一点，我会乖乖陪着你。先好好休息吧。", "source": "identity"},
    ]


def dedupe_examples(examples: list[dict[str, str]]) -> list[dict[str, str]]:
    seen: set[tuple[str, str]] = set()
    result: list[dict[str, str]] = []
    for item in examples:
        if item.get("source") == "tone_authored":
            result.append(item)
            continue
        key = (item["instruction"], item["output"])
        if key in seen:
            continue
        seen.add(key)
        result.append(item)
    return result


def write_jsonl(path: Path, examples: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for item in examples:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Clean ATRI docx dialogue into training jsonl.")
    parser.add_argument("--config", default="configs/training.json")
    parser.add_argument("--character", default="configs/character.json")
    parser.add_argument("--preview", type=int, default=8, help="Print the first N cleaned examples.")
    args = parser.parse_args()

    training_config = load_config(args.config)
    data_config = training_config["data"]

    raw_docx = project_path(data_config["raw_docx"])
    raw_txt = project_path(data_config.get("raw_txt", ""))
    raw_path = raw_docx if raw_docx.exists() else raw_txt
    cleaned_path = project_path(data_config["cleaned_jsonl"])

    turns = read_dialogue_turns(raw_path)
    dialogue_examples = build_dialogue_examples(turns, data_config)

    write_jsonl(cleaned_path, dialogue_examples)

    print(f"Read turns: {len(turns)}")
    print(f"Dialogue examples: {len(dialogue_examples)} -> {cleaned_path}")
    print("Run `python -m atri.curate_data` to create the final training set.")
    for item in dialogue_examples[: max(args.preview, 0)]:
        print("---")
        print("Q:", item["instruction"])
        print("A:", item["output"])


if __name__ == "__main__":
    main()
