import os

import torch

from datasets import load_dataset
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from trl import SFTTrainer
from peft import LoraConfig

import argparse
from utils import prompt_template

os.environ['HF_TOKEN'] = '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'
os.environ['HF_HOME']='/scratch/pp2959/hpml-project/hf_home_at5373'
device = torch.device('cuda:0')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--quantize', action='store_true', help='Quantize the model')
    parser.add_argument('--save-dir', type=str, required=True, help='Path to store weights')
    parser.add_argument('--max-new-tokens', type=int, default=32, help='Max newly generated tokens by LLM')
    parser.add_argument('--lora-rank', type=int, default=32, help='r in LoRA')
    args = parser.parse_args()
    print(args)

    lora_config = LoraConfig(
        r=args.lora_rank,
        target_modules=["q_proj", "o_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj", "lm_head"],
        task_type="CAUSAL_LM",
    )


    llm_id = "google/gemma-2b-it"
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    llm_tokenizer = AutoTokenizer.from_pretrained(llm_id, padding=True, truncation=8192)
    if args.quantize:
        llm = AutoModelForCausalLM.from_pretrained(llm_id, quantization_config=bnb_config, device_map={"":0})
    else:
        llm = AutoModelForCausalLM.from_pretrained(llm_id, device_map={"":0})

    data = load_dataset("foodvisor-nyu/labeled-food-ingredients", split="train")
    data = data.map(lambda samples: llm_tokenizer(samples["prompt"]), batched=True)
    data = data.train_test_split(test_size=0.2, stratify_by_column="int_label", seed=2024)

    def formatting_func(example):
        text = f"{example['prompt']}\n\nAnswer: {example['class']}<eos>"
        # \nReason: {example['reason']}
        return [text]

    sample_ing = data["test"][1]["ingredient"]
    sample_answer = formatting_func(data["test"][1])

    text = prompt_template.format(ing=sample_ing)
    inputs = llm_tokenizer(text, return_tensors="pt").to(device)

    outputs = llm.generate(**inputs, max_new_tokens=32)
    print("\nllm response", llm_tokenizer.decode(outputs[0], skip_special_tokens=True))
    print("\nsample answer", sample_answer[0])

    print("==== Training ====")

    trainer = SFTTrainer(
        model=llm,
        train_dataset=data["train"],
        args=transformers.TrainingArguments(
            per_device_train_batch_size=1,
            gradient_accumulation_steps=4,
            warmup_steps=2,
            max_steps=10,
            learning_rate=2e-4,
            fp16=True,
            logging_steps=1,
            output_dir="outputs",
            optim="paged_adamw_8bit"
        ),
        peft_config=lora_config,
        formatting_func=formatting_func,
        eval_dataset=data["test"]
    )
    trainer.train()
    trainer.save_model(args.save_dir)

    text = prompt_template.format(ing=sample_ing)
    inputs = llm_tokenizer(text, return_tensors="pt").to(device)

    outputs = llm.generate(**inputs, max_new_tokens=32)
    print("\nllm response", llm_tokenizer.decode(outputs[0], skip_special_tokens=True))
    print("\nsample answer", sample_answer[0])
