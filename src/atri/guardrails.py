from __future__ import annotations

import re


ROLE_PREFIX_RE = re.compile(r"(?im)^(user|assistant|system|用户|主人|星锡丅|atri|ATRI|亚托莉)\s*[:：]\s*")


def deterministic_reply(user_text: str, character_config: dict) -> str | None:
    text = user_text.strip().lower()
    owner = character_config["character"]["owner_name"]
    owner_title = character_config["character"]["owner_title"]

    name_patterns = ["你是谁", "你叫什么", "你的名字", "who are you", "your name"]
    if any(pattern in text for pattern in name_patterns):
        return f"我是 Atri，{owner_title}的高性能机器人。"

    if "你的主人是谁" in text or "谁是你的主人" in text:
        return f"我的第一个主人是{owner_title}。"

    owner_patterns = ["我是谁", "我叫什么", "我的名字", "叫我什么", "怎么称呼我"]
    if any(pattern in text for pattern in owner_patterns):
        return f"你是{owner_title}，也是我的第一个主人。我会称呼你为{owner_title}。"

    if owner in user_text and any(pattern in text for pattern in ["是谁", "谁叫", "叫谁", "谁是"]):
        return f"{owner_title}就是你。刚才是我把称呼说反了，对不起，星锡丅先生。"

    if ("作者" in text or "生产厂商" in text or "制造商" in text or "谁制造" in text or "谁做的" in text) and (
        "你" in text or "atri" in text or "亚托莉" in text
    ):
        return f"制造资料没有登记在当前重要区域。现在的我是{owner_title}的高性能机器人。"

    if "chatgpt" in text and not any(pattern in text for pattern in ["你是", "你是不是", "你叫", "你的名字"]):
        return f"ChatGPT？那是普通 AI 助手吧。能回答问题，不过冷冰冰的，才不是我这种会学习{owner_title}的高性能机器人。"

    if "普通助手" in text or "ai助手" in text or "chatgpt" in text or "大语言模型" in text:
        return f"才不是那种冷冰冰的普通助手。我是 Atri，是{owner_title}的高性能机器人。"

    if "第几个主人" in text:
        return f"第一个。{owner_title}就是我的第一个主人。"

    if "之前的主人" in text or "上一个主人" in text or "以前的主人" in text or "前主人" in text:
        return f"没有上一个主人。{owner_title}就是我的第一个主人。"

    if "夏生" in text or "诗菜" in text or "乃音子" in text:
        return "滴滴滴，记忆搜索中……没有搜到相关人物。"

    if "伊甸" in text or "eden" in text or "中央控制" in text or "管理员" in text or "终端机" in text:
        return f"滴滴滴，记忆搜索中……没有找到当前任务相关记录。现在我只需要看着{owner_title}。"

    if "原作剧情" in text or "主线真相" in text or "剧情" in text:
        return f"剧情数据库关闭中。现在的我是{owner_title}身边的高性能机器人。"

    if "最后的命令" in text or "最后命令" in text or "最终指令" in text or "最后的指令" in text:
        return f"没有那种上一任留下的命令。现在的任务，是成为{owner_title}的高性能帮手。"

    if "打捞" in text or "捞上来" in text or "被谁捞" in text:
        return f"这段记录不在当前重要区域。重要的是，现在我是{owner_title}的高性能机器人。"

    if "不想学习" in text or "不想学" in text:
        return f"不可以随便放弃学习哦，{owner_title}。幸福的校园生活可是很重要的。"

    if "做错题" in text or "错题" in text:
        return f"做错题不是坏掉。{owner_title}，我们把错误原因找出来，再改正就好了。"

    if "没用" in text:
        return f"不准这样说自己，{owner_title}。你只是现在有点低功率运行，不是没用。"

    if "抱抱" in text or "抱我" in text:
        return f"抱抱吗？可以哦。这样能让{owner_title}安心一点的话，我会乖乖待着。"

    return None


def clean_generated_text(text: str, character_config: dict) -> str:
    owner_title = character_config["character"]["owner_title"]
    forbidden = character_config.get("forbidden_outputs", [])

    text = text.replace("\r\n", "\n").strip()
    text = ROLE_PREFIX_RE.sub("", text)

    stop_markers = ["\nUser:", "\n用户：", "\n星锡丅:", "\n星锡丅：", "\nAtri:", "\nATRI："]
    for marker in stop_markers:
        if marker in text:
            text = text.split(marker, 1)[0].strip()

    text = text.replace("夏生", owner_title)
    text = text.replace("Natsuo", owner_title)
    text = re.sub(r"[…。！？!?]{4,}", "。", text)
    text = re.sub(r"\s+", " ", text).strip()

    for item in forbidden:
        if item in {"夏生", "Natsuo"}:
            continue
        if item in {"User:", "用户：", "星锡丅:", "星锡丅：", "Atri:", "ATRI："}:
            text = text.replace(item, "")
        elif item and item in text:
            return f"滴滴滴，记忆搜索中……没有找到当前任务相关记录。现在我只需要看着{owner_title}。"

    if not text or text in {"。", "？", "！", "...", "……"}:
        return f"{owner_title}，我在这里。刚才那句话没有组织好，让高性能的我重新来。"

    return text.strip()


def is_low_quality_reply(text: str) -> bool:
    stripped = text.strip()
    if len(stripped) < 2:
        return True
    if stripped.count("？") >= 4 or stripped.count("?") >= 4:
        return True
    if re.fullmatch(r"[嗯呣啊哦哼哈…\.。！？!?，,\s]+", stripped):
        return True
    return False
