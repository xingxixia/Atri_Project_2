import os
import json
import torch
import gc
import pandas as pd
from docx import Document
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForSeq2Seq
from peft import LoraConfig, get_peft_model, TaskType, PeftModel

# --- 全局配置 ---
# 设置国内镜像
#os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"


MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"

DATA_PATH = "data/ATRI-my dear moments.docx"
JSONL_PATH = "atri1.jsonl"
OUTPUT_DIR = "./output"
FINAL_LORA_PATH = "./atri_lora_final"
MERGED_MODEL_PATH = "./merged_atri_model"

keywords = ['夏生', 'ATRI']

def clear_cache():
    #清理显存和内存
    print("\n[System]正在清理显存...")
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("[System]显存清理完毕。\n")

def prepare_data():
    # 读取docx并生成jsonl文件
    if not os.path.exists(DATA_PATH):
        print(f"错误：找不到文件 {DATA_PATH}")
        return False

    print("正在处理数据...")
    doc = Document(DATA_PATH)
    all_lines = [p.text.strip() for p in doc.paragraphs if p.text.strip()]

    instructions = []
    outputs = []
    current_speaker = None
    current_chunk = []

    for line in all_lines:
        if line.startswith("夏生："):
            if current_speaker == "ATRI" and current_chunk:
                outputs.append(current_chunk)
                current_chunk = []
            current_speaker = "夏生"
            current_chunk.append(line)
        elif line.startswith("ATRI："):
            if current_speaker == "夏生" and current_chunk:
                instructions.append(current_chunk)
                current_chunk = []
            current_speaker = "ATRI"
            current_chunk.append(line)

    # 处理最后一个块
    if current_chunk:
        if current_speaker == "夏生":
            instructions.append(current_chunk)
        elif current_speaker == "ATRI":
            outputs.append(current_chunk)

    # 确保长度一致
    min_len = min(len(instructions), len(outputs))

    datas = []
    # 跳过第一条可能过长的数据
    for i in range(1, min_len):
        chunk_instructions = instructions[i]
        chunk_outputs = outputs[i]

        # 去除角色标签
        instruction_str = "".join([text[len(keywords[0]) + 1:] for text in chunk_instructions])
        output_str = "".join([text[len(keywords[1]) + 1:] for text in chunk_outputs])

        new_data = {"instruction": instruction_str, "input": "", "output": output_str}
        datas.append(new_data)

    # 写入文件
    with open(JSONL_PATH, "w", encoding="utf-8") as f:
        for item in datas:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"数据处理完成，生成 {len(datas)} 条对话数据。")
    return True

def train():
    print("=== TRAINING START ===")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 加载 Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

    # Qwen 的 pad_token 处理（防止报错）
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # 加载模型
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype="auto", device_map="auto")

    # LoRA 配置
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        r=64,
        lora_alpha=128,
        lora_dropout=0.1,
    )
    model = get_peft_model(model, lora_config)
    model.enable_input_require_grads()
    model.print_trainable_parameters()

    df = pd.read_json(JSONL_PATH, lines=True)
    ds = Dataset.from_pandas(df)

    def process_func(example):
        MAX_LENGTH = 512

        instruction = tokenizer(
            f"<|im_start|>system\n现在你要扮演的是--ATRI<|im_end|>\n"
            f"<|im_start|>user\n{example['instruction'] + example['input']}<|im_end|>\n"
            f"<|im_start|>assistant\n",
            add_special_tokens=False
        )
        response = tokenizer(
            f"{example['output']}<|im_end|>",
            add_special_tokens=False
        )

        input_ids = instruction["input_ids"] + response["input_ids"] + [tokenizer.pad_token_id]
        attention_mask = instruction["attention_mask"] + response["attention_mask"] + [0]
        # label掩码：-100 表示计算 loss 时忽略该位置
        labels = [-100] * len(instruction["input_ids"]) + response["input_ids"] + [-100]

        if len(input_ids) > MAX_LENGTH:
            input_ids = input_ids[:MAX_LENGTH]
            attention_mask = attention_mask[:MAX_LENGTH]
            labels = labels[:MAX_LENGTH]

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }

    tokenized_ds = ds.map(process_func, remove_columns=ds.column_names)

    args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        logging_steps=10,
        num_train_epochs=3,
        save_strategy="epoch",
        learning_rate=1e-4,
        gradient_checkpointing=True,
        dataloader_num_workers=0,
        fp16=True,  # 开启混合精度
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized_ds,
        tokenizer=tokenizer,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
    )

    trainer.train()

    # 保存LoRA
    trainer.model.save_pretrained(FINAL_LORA_PATH)
    tokenizer.save_pretrained(FINAL_LORA_PATH)
    print(f"LoRA权重已保存至: {FINAL_LORA_PATH}")

    # 释放显存
    del model, trainer
    clear_cache()

def test_and_merge():
    print("=== 开始测试与合并阶段 ===")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    base_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float16, device_map="auto")

    # 注意：加载的是 train() 阶段保存的 FINAL_LORA_PATH
    print(f"正在加载 LoRA: {FINAL_LORA_PATH}")
    model = PeftModel.from_pretrained(base_model, model_id=FINAL_LORA_PATH)
    model.eval()

    # 推理测试
    prompt_text = "你是谁"
    print(f"测试提问: {prompt_text}")

    # 推理时也要保持 System Prompt 一致，且放在 system role 里
    messages = [
        {"role": "system", "content": "现在你要扮演的是--ATRI"},
        {"role": "user", "content": prompt_text}
    ]

    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    inputs = tokenizer([text], return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=512, do_sample=True, top_k=50, temperature=0.7)
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, outputs)
        ]
        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        print(f"ATRI 回复: {response}")

    print("正在合并并保存模型（这可能需要几分钟）...")
    merged_model = model.merge_and_unload()
    merged_model.save_pretrained(MERGED_MODEL_PATH)
    tokenizer.save_pretrained(MERGED_MODEL_PATH)
    print(f"合并完成！完整模型位于: {MERGED_MODEL_PATH}")

    del model, base_model, merged_model
    clear_cache()

if __name__ == "__main__":
    if prepare_data():
        train()
        test_and_merge()# 测试与合并