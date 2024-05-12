# foodvisor-llm
A deep learning course project that evaluates and fine-tunes Google Gemma on food ingredient data, offered with retrieval-augmented generation.
Published 🤗 datasets [Labeled Ingredients](https://huggingface.co/datasets/foodvisor-nyu/labeled-food-ingredients) and [Wikipedia Article Embeddings](https://huggingface.co/datasets/foodvisor-nyu/mxbai-wikipedia-ingredients-embed).

## Milestones Completed 
- [x] Scraped and Parsed over 65,618 Wiki Articles related to Food and Ingredients using Multi-Processing on 4 CPUs.
- [x] Using the mixed-bread model to tokenize and generate embeddings.Divide the articles into chunks of size ~ 300 tokens/chunk on average, resulting in 235,197 chunks in total.
- [x] Generate Embeddings using DDP with 2 RTX-8000s using mxbai and index these embeddings with FAISS. We have a vector DB with ~ 235,190 embeddings related to food.
- [x] Obtained 3000 unique ingredients from USDA and their labels and reasons from ChatGPT 4.0
- [x] Engineered an instruction and response dataset with prompts
- [x] Used semantic search to provide more context to LLM
- [x] Fine-tuned Gemma 2B using LoRA and QLoRA (4bit NF to store and Brain Float to compute, with paged AdamW 8-bit optim)

## Repository description and structure 

The repository includes the following files:

* demo.py:
This file employs model quantization and enhances contextuality using Retrieval-Augmented Generation (RAG). The script determines whether to activate the RAG pipeline and quantization features. When quantization is enabled, it utilizes the BitsAndBytes library for efficient model quantization, which includes activating 4-bit loading and specifying quantization and computation data types. Additionally, if a path to fine-tuned adapters is provided, the script loads them using a PeftModel wrapper and integrates them into the primary model weights. If the RAG feature is enabled, it initializes a SemanticSearcher, potentially altering inputs or retrieving extra context for them. The encoding function processes batches of text data by obtaining additional context with RAG, followed by tokenization using the model's tokenizer. The main execution block sets up output parsing, processes user inputs in real-time, tokenizes these inputs, feeds them into the model to generate predictions, converts the generated tokens back into text, and prepares the textual output for display.
    
* finetuning.py:
The file optimizes a causal language model leveraging PyTorch and the Hugging Face Transformers library, incorporating optional configurations for model quantization and LoRA. If quantization is required, it is configured using the BitsAndBytesConfig. The script initializes by loading a tokenizer and the model from Hugging Face's Model Hub. When quantization is activated, the model is adjusted accordingly. It processes inputs through the model to generate text and decodes the resulting output. Additionally, the script establishes a trainer for Supervised Fine-Tuning (SFT) using designated training parameters and initiates the training process. Upon completion, the trained model is saved to the specified directory.

* inference.py:
The script establishes a configurable text processing pipeline utilizing PyTorch and Hugging Face Transformers, incorporating model quantization and semantic search to assess model performance across metrics such as accuracy and F1 score. When Retrieval-Augmented Generation (RAG) is activated, a SemanticSearcher is initialized to enhance data processing with additional context. The script includes a function to encode data batches using the tokenizer; if RAG is enabled, this function modifies the batch data via the SemanticSearcher. The script processes test data in batches through the model, gathering predictions. An OutputParser handles the model outputs. Finally, the script calculates and displays the accuracy, F1 score, and confusion matrix using sklearn.metrics to evaluate the model's performance on the test data.

* rag.py:
The code defines a SemanticSearcher class that uses a Transformer model to generate text embeddings and perform semantic searches on a pre-embedded Wikipedia database with Faiss for efficient retrieval. It enhances ingredient-related queries with contextual information from Wikipedia to assist in applications like safety assessments.
    
* utils.py:
This file has an Output parser class with a constructor to define a search window in the text processing method and a method to process and classify the text. 


## Usage
Tested on Python 3.12
```bash
pip install -r requirements.txt
```
```bash
python demo.py --rag --quantize --adapters-path="path/to/finetuned/weights" --batch-size=32 --max-new-tokens=32
```
```bash
python inference.py --rag --quantize --adapters-path="path/to/finetuned/weights" --batch-size=32 --max-new-tokens=32
```
```bash
python finetune.py --quantize --save-dir="path/to/checkpoint" --lora-rank=32 --max-new-tokens=32
```
## Results
![image](https://github.com/NeuralFlux/foodvisor-llm/assets/116789851/c11e099a-27e8-49b3-96c1-3b2109707ef1)

![image](https://github.com/NeuralFlux/foodvisor-llm/assets/116789851/132721e1-febd-4de8-9f54-de13e4ff53ab)




