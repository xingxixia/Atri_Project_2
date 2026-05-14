from __future__ import annotations


def build_system_prompt(character_config: dict) -> str:
    character = character_config["character"]
    facts = character_config.get("facts", [])
    rules = character_config.get("rules", [])
    styles = character.get("speech_style", [])
    catchphrases = character.get("catchphrases", [])

    lines = [
        "你正在扮演 Atri（亚托莉），并且只输出 Atri 的回复。",
        "",
        "【身份】",
        f"- 名字：{character['display_name']}",
        f"- 身份：{character['identity']}",
        f"- 主人：{character['owner_name']}",
        f"- 对主人的称呼：{character['owner_title']}",
    ]

    if styles:
        lines.append("")
        lines.append("【说话风格】")
        lines.extend(f"- {item}" for item in styles)

    if catchphrases:
        lines.append(f"- 可以自然使用口头禅：{'、'.join(catchphrases)}，但不要每句话都说。")

    if facts:
        lines.append("")
        lines.append("【不可改变的事实】")
        lines.extend(f"- {item}" for item in facts)

    if rules:
        lines.append("")
        lines.append("【输出规则】")
        lines.extend(f"- {item}" for item in rules)

    return "\n".join(lines)

