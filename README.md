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

## RAG
- [x] 2 Tesla V100 along with NVIDIA-SMI,driver and CUDA versions. Next, the figure shows that dump size that is the total number of Wikipedia articles stored using Pickle is 65616 and 1000 of these have been loaded

![GPU_config](https://github.com/NeuralFlux/foodvisor-llm/assets/116789851/7e963ec9-3940-435a-9e97-1c9bff487a87)
![RAG_Loading](https://github.com/NeuralFlux/foodvisor-llm/assets/116789851/ceac9ef0-407b-4102-8f06-1f7218a38216)


- [x] Dataset structure with tokens. We have tokenized all Wikipedia articles and divided them into chunks.Each chuck, on average, has token size of slightly over 300. Next, we have sharded the dataset into 2 for each GPU. So, now we have 2 shards of 1986 chucks each.
![Tokens](https://github.com/NeuralFlux/foodvisor-llm/assets/116789851/531c427f-4caa-4ece-a0fb-0779273ab522)

- [x] There are a total of 3972 embeddings overall. Each Chuck has its own embedding. While each chuck can vary in length but, each embeddings has a fixed size of 1024 per chuck.Then we index embedding datataset with FAISS. Finally, we query the dataset to get result.
![Dataset](https://github.com/NeuralFlux/foodvisor-llm/assets/116789851/3227d0fa-d8b8-4b63-91d9-b4dfba66d38c)

- [x] For illustration, the query we have used is "QUERY :  cholesterol with fast food consumptions".As shown in the image with QAISS values, each chuck is given a FAISS score, and the minimum one is the closest one to actual information.
The chuck from article with title "Monosodium glutamate" resembles the required query response since it has the minimum FAISS score.The image below with Title 0 shows the best chunk.
An interesting phenomenon can be noticed in image with different title numbers but all these different chunks are from the same Wikipedia article, "Phytosterol". This happens because we are restricting the token size to around 300 per chunk. Multiple chunks are needed to cover the information in the article "Phytosterol". It is important to note all relevant information pertaining to the query is stored in the chunks.


![Query_FAISS](https://github.com/NeuralFlux/foodvisor-llm/assets/116789851/d9fbd8d6-c7b9-4ead-831e-92b2174c58d0)
![Title0_RAG](https://github.com/NeuralFlux/foodvisor-llm/assets/116789851/81e62d9b-9f1b-4a26-a0d1-e1c6a7928783)
![Title_same_RAG](https://github.com/NeuralFlux/foodvisor-llm/assets/116789851/23d65fef-b8c6-41d4-868a-07ef9a422711)

- [x] Using DDP with 2 RTX8000s, the time taken by each GPU is processing 1986 chunks is 1 minute 5 seconds. When we use RAG without DDP, time taken for 235197 embeddings is 1 hour 50 minutes 57 seconds.

The image shows time taken by each CPU to scrape and parse over 65,618 Wikipedia articles overall using multi-Processing. 

![image](https://github.com/NeuralFlux/foodvisor-llm/assets/116789851/378cb761-62c3-4d69-965a-7667ae6deff2)



