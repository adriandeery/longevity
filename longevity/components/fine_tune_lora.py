# scripts/fine_tune_lora.py
# Example LoRA fine-tune script for a small generator model (distilgpt2).
# This is intentionally small and illustrative so it can run on limited GPUs using PEFT.
import argparse
import json
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from peft import get_peft_config, LoraConfig, get_peft_model, prepare_model_for_kbit_training
import torch

parser = argparse.ArgumentParser()
parser.add_argument("--data", required=True, help="jsonl with prompt/response pairs")
parser.add_argument("--output", default="./models/lora_biomed")
parser.add_argument("--model", default="distilgpt2")
args = parser.parse_args()

# Load data
examples = []
with open(args.data, "r", encoding='utf-8') as f:
    for line in f:
        examples.append(json.loads(line))
ds = Dataset.from_list(examples)

tokenizer = AutoTokenizer.from_pretrained(args.model)
tokenizer.pad_token = tokenizer.eos_token

def tokenize_fn(ex):
    return tokenizer(ex["prompt"], truncation=True, max_length=512, padding="max_length")

ds = ds.map(lambda x: tokenize_fn(x), batched=False)
ds.set_format(type="torch", columns=["input_ids","attention_mask"])

# Load model with 8-bit prepare if desired
model = AutoModelForCausalLM.from_pretrained(args.model)
model = prepare_model_for_kbit_training(model)

peft_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["c_attn","q_attn","v_attn"] if hasattr(model, "transformer") else ["attn"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, peft_config)

training_args = TrainingArguments(
    output_dir=args.output,
    per_device_train_batch_size=2,
    num_train_epochs=3,
    logging_steps=10,
    learning_rate=2e-4,
    fp16=False,
    save_total_limit=2,
    remove_unused_columns=False
)

def collate_fn(batch):
    input_ids = torch.stack([b["input_ids"] for b in batch])
    attention_mask = torch.stack([b["attention_mask"] for b in batch])
    labels = input_ids.clone()
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=ds,
    data_collator=collate_fn
)
trainer.train()
model.save_pretrained(args.output)
print("Saved LoRA model to", args.output)
