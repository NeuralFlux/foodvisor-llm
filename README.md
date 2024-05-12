# foodvisor-llm
A deep learning course project that evaluates and fine-tunes Google Gemma on food ingredient data, offered with retrieval-augmented generation.

## Usage
```bash
python demo.py --rag --quantize --adapters-path="path/to/finetuned/weights" --batch-size=32 --max-new-tokens=32
```
```bash
python inference.py --rag --quantize --adapters-path="path/to/finetuned/weights" --batch-size=32 --max-new-tokens=32
```
```bash
python finetune.py --quantize --save-dir="path/to/checkpoint" --lora-rank=32 --max-new-tokens=32
```
