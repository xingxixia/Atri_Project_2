import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from threading import Thread
import torch

# 1. 加载模型
model_name = "Qwen/Qwen2.5-3B-Instruct"
model = AutoModelForCausalLM.from_pretrained(model_name, dtype="auto", device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(model_name)


def predict(message, history):
    # 打印一下，方便万一报错时看一眼
    print(f"Original History Element: {history[0] if history else 'Empty'}")

    messages = [
        {"role": "system", "content": "You named Atri, will be a robot ai."} # 对ai进行背景设定，比如名字，出自等
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
    )

    t = Thread(target=model.generate, kwargs=generate_kwargs)
    t.start()

    partial_message = ""
    for new_token in streamer:
        partial_message += new_token
        yield partial_message


# 启动界面
gr.ChatInterface(predict).launch()

