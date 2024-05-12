import os

import torch

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
from datasets import load_dataset

from tqdm import tqdm
import argparse
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

from rag import SemanticSearcher
from utils import OutputParser

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
    parser.add_argument('--batch-size', type=int, default=8, help='Minibatch size for inference')
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

    data = load_dataset("foodvisor-nyu/labeled-food-ingredients", split="train")
    data = data.train_test_split(test_size=0.2, stratify_by_column="int_label", seed=2024)
    if args.rag:
        searcher = SemanticSearcher()

    def encode(batch):
        if args.rag:
            batch = searcher.get_context(batch)
        return llm_tokenizer(batch['prompt'], padding=True, max_length=8192, truncation=True, return_tensors='pt')
    test_data = data["test"]
    y_test = test_data["class"]
    test_data.set_transform(encode)

    BATCH_SIZE = args.batch_size
    MAX_NEW_TOKENS = args.max_new_tokens

    preds = []
    parser = OutputParser(MAX_NEW_TOKENS)
    with torch.no_grad():
        llm.eval()
        for idx in tqdm(range(0, len(test_data), BATCH_SIZE)):
            inputs = test_data[idx:(idx + BATCH_SIZE)].to(device)
            outputs = llm.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS)
            preds.extend(
                map(parser.process_llm_output, llm_tokenizer.batch_decode(outputs))
            )

    labels=['harmful', 'not harmful', 'controversial']
    y_pred = preds

    # 'weighted' to handle label imbalance
    f1 = f1_score(y_test, y_pred, average='weighted', labels=labels)
    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred, labels=labels)

    print("Accuracy:", accuracy)
    print("F1 Score:", f1)
    print("Confusion Matrix:\n", cm)
