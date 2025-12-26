import os
import json
from docx import Document

def read_docx_to_jsonl(docx_path, jsonl_path):
    """
    Read a .docx file, extract lines with roles '夏生' (user) and 'ATRI' (assistant),
    remove the role prefixes, and save as JSONL with fields 'instruction' and 'output'.
    """
    doc = Document(docx_path)
    data = []
    user_role = "夏生"
    assistant_role = "ATRI"
    for para in doc.paragraphs:
        text = para.text.strip()
        if not text:
            continue
        # Check for user and assistant lines
        # Expect format like "夏生: question" or "ATRI: answer"
        if text.startswith(f"{user_role}") or text.startswith(f"{assistant_role}"):
            # Split on first colon (Chinese or English)
            if "：" in text:
                role, content = text.split("：", 1)
            elif ":" in text:
                role, content = text.split(":", 1)
            else:
                continue
            role = role.strip()
            content = content.strip()
            if role == user_role:
                # Start a new conversation entry
                data.append({"instruction": content, "output": ""})
            elif role == assistant_role and data:
                # Assign assistant response to last user
                data[-1]["output"] = content

    # Remove entries without assistant response
    data = [item for item in data if item.get("output")]
    # Save to JSONL
    os.makedirs(os.path.dirname(jsonl_path), exist_ok=True)
    with open(jsonl_path, 'w', encoding='utf-8') as f:
        for item in data:
            json_line = json.dumps(item, ensure_ascii=False)
            f.write(json_line + "\n")

# Example usage
docx_file = os.path.join("data", "atri1.docx")      # Path to input DOCX file
jsonl_file = os.path.join("data", "atri1.jsonl")    # Path for output JSONL data
read_docx_to_jsonl(docx_file, jsonl_file)

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model

def load_model_with_lora(model_name):
    """
    Load the base model and tokenizer, then apply LoRA to the model.
    """
    print(f"Loading model {model_name}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,   # use fp16 for memory efficiency
        device_map="auto"           # automatic device placement
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # Enable gradient checkpointing for memory efficiency:contentReference[oaicite:3]{index=3}
    model.gradient_checkpointing_enable()
    # LoRA configuration: low-rank adaption on attention matrices:contentReference[oaicite:4]{index=4}
    peft_config = LoraConfig(
        r=16,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]
    )
    model = get_peft_model(model, peft_config)
    return model, tokenizer

# Example usage
model_name = "Qwen/Qwen2.5-3B-Instruct"
model, tokenizer = load_model_with_lora(model_name)

from torch.utils.data import Dataset
from transformers import TrainingArguments, Trainer, default_data_collator

class ChatDataset(Dataset):
    def __init__(self, encodings):
        self.encodings = encodings
    def __len__(self):
        return len(self.encodings["input_ids"])
    def __getitem__(self, idx):
        return {k: v[idx] for k, v in self.encodings.items()}

def prepare_train_dataset(jsonl_path, tokenizer):
    """
    Load JSONL data and create input_ids, attention_masks, and labels for training.
    Labels for the user part are masked with -100 to not compute loss on them.
    """
    data = []
    system_message = "You are ATRI, an attentive assistant."
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            obj = json.loads(line)
            user_txt = obj["instruction"]
            assistant_txt = obj["output"]
            # Format prompt using Qwen chat tokens:contentReference[oaicite:8]{index=8}:contentReference[oaicite:9]{index=9}
            prompt = (
                "<|im_start|>system\n" + system_message + "<|im_end|>\n"
                "<|im_start|>user\n" + user_txt + "<|im_end|>\n"
                "<|im_start|>assistant\n" + assistant_txt + "<|im_end|>"
            )
            data.append(prompt)

    # Tokenize all examples
    encodings = tokenizer(
        data,
        return_tensors='pt',
        padding=True,
        truncation=True
    )
    input_ids = encodings["input_ids"]
    attention_mask = encodings["attention_mask"]
    labels = input_ids.clone()

    # Mask out everything up to the assistant response
    assistant_tag = "<|im_start|>assistant"
    for i, prompt in enumerate(data):
        # find where assistant response begins in characters
        assistant_pos = prompt.find(assistant_tag) + len(assistant_tag)
        # find token index corresponding to that character position
        offsets = tokenizer(prompt, return_offsets_mapping=True, add_special_tokens=False)["offset_mapping"]
        token_index = 0
        for idx, (start, _) in enumerate(offsets):
            if start >= assistant_pos:
                token_index = idx
                break
        # mask all tokens before assistant part
        labels[i, :token_index] = -100

    encodings["labels"] = labels
    return ChatDataset(encodings)

# Prepare dataset and training arguments
train_dataset = prepare_train_dataset(jsonl_file, tokenizer)
training_args = TrainingArguments(
    output_dir=os.path.join("output"),
    per_device_train_batch_size=1,     # small batch size due to GPU memory
    gradient_accumulation_steps=4,     # accumulate to simulate larger batch
    learning_rate=2e-4,
    num_train_epochs=3,
    logging_steps=20,
    save_total_limit=2,
    fp16=True                          # use mixed precision
)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    data_collator=default_data_collator
)

# Train the model
trainer.train()

def generate_atri_response(model, tokenizer, user_input, max_new_tokens=256):
    """
    Generate a response from the fine-tuned model given a user prompt.
    """
    system_message = "You are ATRI, an attentive assistant."
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_input}
    ]
    # Apply Qwen chat template:contentReference[oaicite:11]{index=11}
    prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer([prompt_text], return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=max_new_tokens)
    # Extract only the newly generated tokens (remove prompt)
    generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
    response = tokenizer.decode(generated_ids, skip_special_tokens=True)
    return response

# Example inference
user_prompt = "今天的天气怎么样？"
print("User:", user_prompt)
print("ATRI:", generate_atri_response(model, tokenizer, user_prompt))

from peft import PeftModel

def save_merged_model(peft_model, tokenizer, merged_dir):
    """
    Merge LoRA weights into the base model and save the merged model.
    """
    # Merge LoRA adapter into base model:contentReference[oaicite:13]{index=13}
    merged_model = peft_model.merge_and_unload()
    # Save merged model and tokenizer
    os.makedirs(merged_dir, exist_ok=True)
    merged_model.save_pretrained(merged_dir)
    tokenizer.save_pretrained(merged_dir)

# Save PEFT adapter (optional) and merged model
adapter_dir = os.path.join("output", "lora_adapter")
merged_dir = os.path.join("output", "merged_model")
model.save_pretrained(adapter_dir)       # saves only LoRA adapter weights
tokenizer.save_pretrained(adapter_dir)
save_merged_model(model, tokenizer, merged_dir)
