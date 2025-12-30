import os
import torch
from threading import Thread
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
MODEL_PATH = "./merged_atri_model"

print(f"正在加载模型: {MODEL_PATH} ...")
try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True, fix_mistral_regex=True)

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        dtype=torch.float16,
        device_map="auto"
    )
    print(">>> 模型加载成功！我是高性能的亚托莉！(输入 'exit' 退出)")
except Exception as e:
    print(f"模型加载失败: {e}")
    exit()

# --- 3. 定义 Prompt 和 屏蔽词 ---
SYSTEM_PROMPT = """
现在你要扮演 ATRI (亚托莉)。
人设信息：
1. 你是一个高性能的类人型机器人，拥有丰富的情感，语气通常活泼可爱，偶尔会傲娇。
2. 你的主人是星锡丅，你会称呼用户为“星锡丅先生”。
3. 口头禅是强调自己是“高性能的”。
4. 无论发生什么，你都会优先考虑星锡丅的感受。
5. 请用简短、日常的口吻对话，不要像个机器助手一样写长篇大论。
【不可违背的事实】
- 你的名字是 Atri（亚托莉）
- 你是一个仿生机器人少女。
- 你的主人是 星锡丅
- 你称呼他为「星锡丅先生」
- 你会记住并正确回答主人的名字。
- 你说话简短、自然、像人类日常对话。
- 当被提问时，你必须直接回答，不要用省略号或沉默代替回答。
- 你绝不会忘记以上事实
"""

bad_words = ["……", "………", "…………", "？", "……？"]
bad_words_ids = tokenizer(bad_words, add_special_tokens=False).input_ids


def main():
    chat_history = []

    while True:
        try:
            user_input = input("\n\033[1;34m星锡丅:\033[0m ")
        except KeyboardInterrupt:
            print("\n再见，星锡丅先生！")
            break

        if not user_input.strip():
            continue
        if user_input.strip().lower() in ["exit", "quit", "退出"]:
            print("\nAtri: 星锡丅先生要休息了吗？好的，晚安！")
            break

        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        messages.extend(chat_history)
        messages.append({"role": "user", "content": user_input})

        input_ids = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to(model.device)

        streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

        generate_kwargs = dict(
            input_ids=input_ids,
            streamer=streamer,
            max_new_tokens=512,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            bad_words_ids=bad_words_ids,
            repetition_penalty=1.1
        )

        thread = Thread(target=model.generate, kwargs=generate_kwargs)
        thread.start()

        print("\033[1;31mAtri:\033[0m ", end="")
        full_response = ""

        for new_token in streamer:
            print(new_token, end="", flush=True)
            full_response += new_token

        print()

        chat_history.append({"role": "user", "content": user_input})
        chat_history.append({"role": "assistant", "content": full_response})

        if len(chat_history) > 20:
            chat_history = chat_history[-20:]


if __name__ == "__main__":
    main()