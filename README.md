# foodvisor-llm
A deep learning course project that evaluates and fine-tunes Google Gemma on food ingredient data, offered with retrieval-augmented generation.

## Milestones Completed 
*Scraped and Parsed over 65,618 Wiki Articles related to Food and Ingredients using Multi-Processing on 4 CPUs, saved as pickle files.

*Using the mixed-bread model to tokenize and generate embeddings.
* Divide the articles into chunks of size ~ 300 tokens/chunk on average, resulting in 235,197 chunks in total.
*Generate Embeddings using DDP with 2 RTX-8000s using mxbai and index these embeddings with FAISS.
* We have a vector DB with ~ 235,190 embeddings related to food
* Obtained 3000 unique ingredients from USDA
*Obtained labels and reasons from ChatGPT 4.0
*Engineered an instruction and response dataset with prompts
*Used semantic search to provide more context to LLM
*Fine-tuned Gemma 2B using LoRA and QLoRA (4bit NF to store and Brain Float to compute, with paged AdamW 8-bit optim)

## Repository description and structure 

The repository includes the following files:

* demo.py:
The file utilizes model quantization and contextual enhancement through Retrieval-Augmented Generation (RAG). It utilizes command-line arguments to customize settings, processes user inputs, and generates text responses, incorporating semantic context where applicable.
    
* finetuning.py:
The file fine-tunes a causal language model using PyTorch and the Hugging Face Transformers library, with optional model quantization and LoRA configurations. It processes text data, conducts a training session with specific hyperparameters, and evaluates the model's performance by generating responses to prompts, all configurable through command-line arguments.
    
* inference.py:
The code sets up a configurable text processing pipeline using PyTorch and Hugging Face Transformers, integrating model quantization and semantic search to evaluate model performance on a dataset with metrics like accuracy and F1 score. It allows for dynamic feature toggling and parameter adjustment through command-line arguments.
    
* rag.py:
The code defines a SemanticSearcher class that uses a Transformer model to generate text embeddings and perform semantic searches on a pre-embedded Wikipedia database with Faiss       for efficient retrieval. It enhances ingredient-related queries with contextual information from Wikipedia to assist in applications like safety assessments.
    
* utils.py:
This file has an Output parser class with a constructor to define a search window in the text processing method and a method to process text and classify the text. 


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
## Results
With Fine-Tuning:
![image](https://github.com/NeuralFlux/foodvisor-llm/assets/116789851/c11e099a-27e8-49b3-96c1-3b2109707ef1)

Without Fine-Tuning:
![image](https://github.com/NeuralFlux/foodvisor-llm/assets/116789851/52173858-4f14-4a7d-a623-5b364f9300c7)

