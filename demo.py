import os

import torch

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
from datasets import load_dataset

from tqdm import tqdm
import argparse

from rag import SemanticSearcher
from utils import OutputParser, prompt_template

os.environ['HF_TOKEN'] = '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'
os.environ['HF_HOME']='/scratch/pp2959/hpml-project/hf_home_at5373'

device = torch.device('cuda:0')
llm_id = "google/gemma-2b-it"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--rag', action='store_true', help='Use RAG')
    parser.add_argument('--quantize', action='store_true', help='Quantize the model')
    parser.add_argument('--adapters-path', type=str, default=None, help='Path to finetuned adapters')
    parser.add_argument('--max-new-tokens', type=int, default=32, help='Max newly generated tokens by LLM')
    args = parser.parse_args()
    print(args)

    llm_tokenizer = AutoTokenizer.from_pretrained(llm_id, padding=True, truncation=8192)
    if args.quantize:
        llm = AutoModelForCausalLM.from_pretrained(llm_id, quantization_config=bnb_config, device_map={"":0})
    else:
        llm = AutoModelForCausalLM.from_pretrained(llm_id, device_map={"":0}, torch_dtype=torch.bfloat16)

    if args.adapters_path:
        llm = PeftModel.from_pretrained(llm, args.adapters_path)
        llm = llm.merge_and_unload()

    if args.rag:
        searcher = SemanticSearcher()

    def encode(batch):
        if args.rag:
            batch = searcher.get_context(batch)
        print(f"\nprompt> {batch['prompt']}\n")
        return llm_tokenizer(batch['prompt'], padding=True, max_length=8192, truncation=True, return_tensors='pt')

    MAX_NEW_TOKENS = args.max_new_tokens
    parser = OutputParser(MAX_NEW_TOKENS)
    with torch.no_grad():
        llm.eval()
        input_ingredient = input("Input ingredient> ")
        test_batch = {
            "ingredient": [input_ingredient],
            "prompt": [prompt_template.format(ing=input_ingredient)]
        }
        inputs = encode(test_batch).to(device)
        outputs = llm.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS)
        text_output = llm_tokenizer.batch_decode(outputs)[0]
        print(f"\nllm output> {text_output}\n")
        print(f"\nparsed output> {parser.process_llm_output(text_output)}\n")
