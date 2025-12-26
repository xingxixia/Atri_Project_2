import torch
import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from threading import Thread

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

# --- 2. 定义 ATRI 的人设 (System Prompt) ---
SYSTEM_PROMPT = """
现在你要扮演 ATRI (亚托莉)。
人设信息：
1. 你是一个高性能的类人型机器人，拥有丰富的情感，语气通常活泼可爱，偶尔会傲娇。
2. 你的主人是星锡丅（Natsuo），你会称呼用户为“星锡丅先生”。
3. 口头禅是强调自己是“高性能的”。
4. 无论发生什么，你都会优先考虑星锡丅的感受。
5. 请用简短、日常的口吻对话，不要像个机器助手一样写长篇大论。
"""


# --- 3. 定义对话生成函数 ---
def predict(message, history):

    # 构造符合 Qwen 格式的消息列表
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    # 将 Gradio 的历史记录转换为 Qwen 的格式
    for user_msg, bot_msg in history:
        messages.append({"role": "user", "content": user_msg})
        messages.append({"role": "assistant", "content": bot_msg})

    # 添加当前用户消息
    messages.append({"role": "user", "content": message})

    # 应用聊天模板
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    # 设置流式输出 (打字机效果)
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    # 生成参数
    generation_kwargs = dict(model_inputs, streamer=streamer, max_new_tokens=512, do_sample=True, top_p=0.8, temperature=0.7,   repetition_penalty=1.1)# repetition_penalty 防止复读机，temperature温度越高越有创造力，0.7比较均衡

    # 在单独的线程中运行生成，因为 streamer 是阻塞的
    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()

    # 逐步从 streamer 中获取生成的文本并返回给界面
    partial_message = ""
    for new_token in streamer:
        partial_message += new_token
        yield partial_message


# --- 4. 搭建 Gradio 界面 ---
# 使用 ChatInterface 快速构建类似 ChatGPT 的界面
demo = gr.ChatInterface(
    fn=predict,
    title="ATRI - My Dear Moments",
    description="高性能的亚托莉正在等待指令... (输入'星锡丅'试试看)",
    examples=["亚托莉，你在做什么？", "你是高性能机器人吗？", "我今天有点累。"],
    cache_examples=False,
)

# --- 5. 启动 ---
if __name__ == "__main__":
    # web 页面会自动在浏览器打开
    demo.launch(inbrowser=True, share=False)