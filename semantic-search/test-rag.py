import mwclient  # for downloading example Wikipedia articles
import mwparserfromhell  # for splitting Wikipedia articles into sections
import openai  # for generating embeddings
import os  # for environment variables
import pandas as pd  # for DataFrames to store article sections and embeddings
import re  # for cutting <ref> links out of Wikipedia articles
import tiktoken  # for counting tokens

import pickle

import faiss

from datasets import load_dataset
import datasets

from transformers import AutoTokenizer, AutoModel
import torch


model_name = "mixedbread-ai/mxbai-embed-large-v1"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

device = torch.device('cuda:0')
model.to(device)

def cls_pooling(model_output):
    return model_output.last_hidden_state[:, 0]

def get_embeddings(text_list):
    encoded_input = tokenizer(
        text_list, padding=True, truncation=True, return_tensors="pt"
    )
    # print("embedding_input : ")
    # print(encoded_input)
    encoded_input = {k: v.to(device) for k, v in encoded_input.items()}
    model_output = model(**encoded_input)
    return cls_pooling(model_output)

def map_embeddings(chunks):
    encoded_input = {k: torch.tensor(v).to(device) for k, v in chunks.items()}
    embeddings = model(**encoded_input).last_hidden_state[:,0]
    return embeddings.detach().cpu().numpy()[0]


wiki_db_embedded = datasets.load_from_disk("wiki_db_embedded.hf")

print(wiki_db_embedded)

wiki_db_embedded.add_faiss_index(column = "embeddings")

import numpy as np

query = 'sea foods that are good for bodybuilding'

query_embedding = get_embeddings(query).detach().cpu()
query_embedding = np.asarray(query_embedding)

print(query_embedding.shape)

scores, samples = wiki_db_embedded.get_nearest_examples(
        "embeddings", query_embedding, k = 10
)

print(scores)
print(samples['title'])
print(" ")
for sample in samples['text']:
    print(sample)
    print("---------------------------------------------")


