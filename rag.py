import os

import numpy as np
import torch

from transformers import AutoTokenizer, AutoModel
import datasets
from datasets import load_dataset, Dataset

from tqdm import tqdm

os.environ['HF_TOKEN'] = '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'
os.environ['HF_HOME']='/scratch/pp2959/hpml-project/hf_home_at5373'

device = torch.device('cuda:0')

class SemanticSearcher():
    def __init__(self) -> None:
        embed_model_name = "mixedbread-ai/mxbai-embed-large-v1"
        self.embed_tokenizer = AutoTokenizer.from_pretrained(embed_model_name)
        self.embed_model = AutoModel.from_pretrained(embed_model_name).to(device)

        self.wiki_db_embedded = datasets.load_from_disk("../rag-pipeline/wiki_db_embedded.hf")
        self.wiki_db_embedded.add_faiss_index(column = "embeddings")

    def _cls_pooling(self, model_output):
        return model_output.last_hidden_state[:, 0]

    def _get_embeddings(self, text_list):
        encoded_input = self.embed_tokenizer(
            text_list, padding=True, truncation=True, return_tensors="pt"
        ).to(device)
        model_output = self.embed_model(**encoded_input)
        return self._cls_pooling(model_output)
    
    def get_context(self, batch):
        queries = list(map(lambda ingredient: f"{ingredient} health effects", batch["ingredient"]))
        query_embedding = self._get_embeddings(queries).detach().cpu()
        query_embedding = np.asarray(query_embedding)

        _, samples = self.wiki_db_embedded.get_nearest_examples_batch(
                "embeddings", query_embedding, k = 4
        )

        # FIXME: hacky iterative join
        context_prompts = [
            batch["prompt"][idx] + f"\nContext: {'. '.join(samples[idx]['text'])}"\
            for idx in range(len(batch["prompt"]))
        ]

        return Dataset.from_dict({"prompt": context_prompts})
