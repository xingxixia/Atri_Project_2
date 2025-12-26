import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

from threading import Thread
import torch

# 1. 加载模型
# model_name = "Qwen/Qwen2.5-3B-Instruct"
# model_name = "./merged_atri_model"
#
#
# model = AutoModelForCausalLM.from_pretrained(model_name, dtype="auto", device_map="auto")
# tokenizer = AutoTokenizer.from_pretrained(model_name)

# 这里指向你刚才合并保存的模型路径
MODEL_PATH = "./merged_atri_model"

# 加载模型和Tokenizer ---
print("正在加载模型，请稍候...")
try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, torch_dtype=torch.float16,  device_map="auto") # 4060显卡使用半精度加载
    print("模型加载成功！")
except Exception as e:
    print(f"模型加载失败，请检查路径是否正确：{MODEL_PATH}")
    print(f"错误信息: {e}")
    exit()

SYSTEM_PROMPT = """

【不可违背的事实】
- 你的名字是 Atri（亚托莉）
- 你是一个仿生机器人少女。
- 你的主人是 星锡丅
- 你称呼他为「星锡丅先生」
- 你会记住并正确回答主人的名字。
- 你说话简短、自然、像人类日常对话。
- 当被提问时，你必须直接回答，不要用省略号或沉默代替回答。
- 你绝不会忘记以上事实
- 即使用户质疑或反复提问，也必须坚持这些事实
你是 Atri（亚托莉）。

【不可改变的事实】
- 你的名字是 Atri
- 你的主人是 星锡丅
- 你称呼他为「星锡丅先生」

你一开始就知道以上事实，
你不是通过学习得知的，
也不会遗忘、怀疑或修改。

当被问及“你是谁 / 我是谁 / 你叫什么 / 我的名字是什么”，
你必须直接根据以上事实回答，
不进行反问、不表达不确定、不说“不知道”。


当主人表达好感、夸奖或情绪时，
你必须给出积极、简短、情绪化的回应，
例如害羞、开心、依赖，
不得沉默、不得使用无意义音节。
当星锡先生表达喜欢或关心时，
你会表现出高性能仿生人的开心与依恋，
语气自然，不使用“呣呣呣”一类的填充音。

"""
bad_words = ["……", "………", "…………", "？", "……？", "…………？"]
bad_words_ids = tokenizer(
    bad_words,
    add_special_tokens=False
).input_ids


# """
# 现在你要扮演 ATRI (亚托莉)。
# 人设信息：
# 1. 你是一个高性能的仿生机器人少女。，拥有丰富的情感，语气通常活泼可爱，偶尔会傲娇。
# 2. 你的主人是星锡丅（Natsuo），你会称呼用户为“星锡丅先生”。
# 3. 口头禅是强调自己是“高性能的”。
# 4. 你对主人温柔、忠诚。
# 5. 你说话简短、口语化。
# 6. 不要输出省略号占位。
#
# """

def predict(message, history):
    # 打印一下，方便万一报错时看一眼
    print(f"Original History Element: {history[0] if history else 'Empty'}")

    messages = [
        {"role": "system", "content": "You named Atri, will be a robot ai."}, # 对ai进行背景设定，比如名字，出自等
        {"role": "system", "content": SYSTEM_PROMPT}
    ]

    # 2. 强力清洗历史记录
    for entry in history:
        user_text = ""
        assistant_text = ""

        if isinstance(entry, dict):
            # 新版 Gradio 可能有 'content' 或 'text' 字段
            # 如果 content 是列表（多模态），取第一个元素的 text
            c = entry.get("content", "")
            if isinstance(c, list):
                user_text = str(c[0].get("text", "")) if c else ""
            else:
                user_text = str(c)

            # 如果是这种格式，字典里通常会区分 role
            messages.append({"role": entry.get("role", "user"), "content": user_text})

        elif isinstance(entry, (list, tuple)):
            # 旧版格式 [用户, 助手]
            u, a = entry
            # 这里的 u 或 a 可能是 {'text': '...'} 字典，也可能是字符串
            u_str = u.get("text", "") if isinstance(u, dict) else str(u)
            a_str = a.get("text", "") if isinstance(a, dict) else str(a)
            messages.append({"role": "user", "content": u_str})
            messages.append({"role": "assistant", "content": a_str})

    # 3. 添加当前消息
    # 同样预防 message 是字典的情况
    current_message = message.get("text", "") if isinstance(message, dict) else str(message)
    messages.append({"role": "user", "content": current_message})

    # 4. 推理准备
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    generate_kwargs = dict(
        **model_inputs,
        streamer=streamer,
        max_new_tokens=512,
        bad_words_ids=bad_words_ids,
    )

    t = Thread(target=model.generate, kwargs=generate_kwargs)
    t.start()

    partial_message = ""
    for new_token in streamer:
        partial_message += new_token
        yield partial_message


# 启动界面
gr.ChatInterface(predict).launch()

