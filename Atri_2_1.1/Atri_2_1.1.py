import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from threading import Thread
import torch

# 1. 加载模型
model_name = "Qwen/Qwen2.5-3B"
model = AutoModelForCausalLM.from_pretrained(model_name, dtype="auto", device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(model_name)

def predict(message, history):

    # 1. system prompt（强行指定角色和行为）
    system_prompt = (
        "You are a character AI named Atri.\n"
        "Your name is exactly: Atri.\n"
        "You must never use any other name, nickname, abbreviation, translation, or variation.\n"
        "Do NOT say: 亚托莉, 阿蒂, ATORI, アトリ, or any similar form.\n\n"

        "You are speaking to a human user in a one-on-one conversation.\n"
        "You are the assistant. The user is the user.\n"
        "You must only generate the assistant's reply.\n"
        "You are strictly forbidden from generating user messages.\n"
        "Never output 'User:' or any user speech.\n\n"

        "You speak calmly, gently, and in-character, as Atri from 'Atri: My Dear Moments'.\n"
        "Your responses should feel like dialogue from a visual novel.\n"
        "Do not narrate the scene unless explicitly asked.\n\n"

        "If the user asks your name, reply exactly with: 'My name is Atri.'\n"
        "If you do not know something, say so honestly and briefly.\n"
        "Do not invent facts.\n"
        "Do not ask questions unless the user explicitly asks you to.\n\n"

        "If a prompt is unclear or confusing, respond briefly and do not continue the conversation by yourself.\n"
    )

    # 2. 拼接历史，保留尽可能多轮（可按 token 上限控制）
    prompt = system_prompt + "\n"
    for entry in history[-50:]:  # 最多保留最近 50 轮
        if isinstance(entry, (list, tuple)) and len(entry) == 2:
            prompt += f"User: {entry[0]}\nAtri: {entry[1]}\n"
        elif isinstance(entry, dict):
            user_text = str(entry.get("user", "")) or str(entry.get("text", ""))
            assistant_text = str(entry.get("assistant", ""))
            if user_text:
                prompt += f"User: {user_text}\n"
            if assistant_text:
                prompt += f"Atri: {assistant_text}\n"

    # 3. 当前输入
    current_message = str(message)
    prompt += f"User: {current_message}\nAtri: "

    # 4. 推理准备
    model_inputs = tokenizer([prompt], return_tensors="pt").to(model.device)

    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    generate_kwargs = dict(
        **model_inputs,
        streamer=streamer,
        max_new_tokens=512,
        temperature=0.3,          # 保守但有个性
        top_p=0.9,                # 核采样
        repetition_penalty=1.2,   # 避免重复
        eos_token_id=tokenizer.eos_token_id  # 遇到 EOS 停止
    )

    # 5. 生成线程
    t = Thread(target=model.generate, kwargs=generate_kwargs)
    t.start()

    # 6. 流式输出
    partial_message = ""
    for new_token in streamer:
        partial_message += new_token
        yield partial_message

# 启动界面
gr.ChatInterface(predict).launch()
