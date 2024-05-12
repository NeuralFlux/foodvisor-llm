import mwclient  # for downloading example Wikipedia articles
import mwparserfromhell  # for splitting Wikipedia articles into sections
import openai  # for generating embeddings
import os  # for environment variables
import pandas as pd  # for DataFrames to store article sections and embeddings
import re  # for cutting <ref> links out of Wikipedia articles
import tiktoken  # for counting tokens
import gc  # for garbage collections

import pickle

import faiss

from datasets import load_dataset # huggingface datasets library to work with the dataset
import datasets# for huggingface datasets

from transformers import AutoTokenizer, AutoModel# for loading the models using huggingface transformers api
import torch # pytorch !

wiki_dumps = []
# specify the number of dumps in the above directory.
num_of_dumps = 4
# Spicify directory path to the parsed wiki articles.
dump_names = "dumps_1/wiki_dump_out_"

# load the pickle dumps 
for i in range(num_of_dumps):
    dbfile = open(f'{dump_names}{i}.pickle', 'rb')
    wiki_dumps.extend(pickle.load(dbfile))
    dbfile.close()

print("dump size : ", len(wiki_dumps))

# convert the dumps to huggingface datasets api
wiki_db = datasets.Dataset.from_list(wiki_dumps)

# specify the mixedbread embedding model to use
model_name = "mixedbread-ai/mxbai-embed-large-v1"
# tokenizer from huggingface's transformers library
tokenizer = AutoTokenizer.from_pretrained(model_name)
# pytorch's model 
model = AutoModel.from_pretrained(model_name)

print("model")
device = torch.device('cuda:0')
model.to(device)

# specify the average max taken size per chunk.
MAX_TOKENS = 300
wikipedia_strings = []

# this mapping function splits long articles into chunks maintaining context
def add_token_lens(x):

    token_checkpoint = 0
    collect_chunks = []
    collect_chunk_lens = []

    # iterate throught every character
    for i, char in enumerate(x['text']):
        # whenever we encounter a '.' indicating the end of a sentence .. 
        if char == '.':
            # .. we tokenize the whole sentences and ..
            tokens = tokenizer(
                [x['text'][token_checkpoint: i + 1]], padding = True, truncation = False, return_tensors = "pt"
            )
            # .. check for it lenght, if there are more than 300 tokens then we append the chunks, and mark our checkpoint. 
            tokens_len = tokens['input_ids'].shape[1]
            if tokens_len > 300:
                collect_chunks.append( (token_checkpoint, i + 1) )
                collect_chunk_lens.append( tokens_len )
                token_checkpoint = i + 1
    
    # in case the chunks fall short in the end, we take the rest of the sentences as it is. 
    if token_checkpoint < len(x['text']):
        tokens = tokenizer(
                x['text'][token_checkpoint: len(x['text'])], padding = True, truncation = False, return_tensors = "pt"
        )
        tokens_len = tokens['input_ids'].shape[1]
        collect_chunks.append( (token_checkpoint,  len(x['text'])) )
        collect_chunk_lens.append( tokens_len )
    
    # finally we iterate through our chunks and tokenize them. 
    tokens = tokenizer(
        [x['text'][i:j] for i, j in collect_chunks],
        padding = True, truncation = True, return_tensors = "pt"
    )
    # store the chunks along with their tokens
    x['text'] = [x['text'][i:j] for i, j in collect_chunks]
    x['tokenized_chunks'] = tokens
    x['tokenized_chunks_lens'] = collect_chunk_lens
    return x

# filter out articles that are less than 100 characters in length.
wiki_db = wiki_db.filter(lambda x: len(x['text']) > 100)

# map the above tokenizer function on top of our dataset.
wiki_db = wiki_db.map(add_token_lens)
print(wiki_db)

# Since we end with multiple chunks per wikipedia articles, we need to flatten the \
# the dataset over chunks.
wiki_db_extended = []
_id = 0

for x in wiki_db:
  for i, tokens_len in enumerate(x['tokenized_chunks_lens']):
    wiki_db_extended.extend([{
      '_id': _id,
      'title': x['title'],
      'site': x['site'],
      'text': x['text'][i],
      'tokens': {key: [value[i]] for key, value in x['tokenized_chunks'].items()},
      'tokens_len': tokens_len
    }])
    _id += 1

# serialize the flattened dataset to huggingface's dataset. 
wiki_db_flattened = datasets.Dataset.from_list(wiki_db_extended)

# clear the previous datasets using python's garbage collector.
del wiki_db
del wiki_db_extended
gc.collect()

print(wiki_db_flattened)

idx = 0

# check the chunks on how they look.
print("\n title : \n")
print(wiki_db_flattened[idx]['title'])
print("\n site : \n")
print(wiki_db_flattened[idx]['site'])
print("\n text : \n")
print(wiki_db_flattened[idx]['text'])
print("\n tokens_len : \n")
print(wiki_db_flattened[idx]['tokens_len'])
print("\n tokens : \n")
print(wiki_db_flattened[idx]['tokens'])

# cls pooling function to get the embedding.
def cls_pooling(model_output):
    return model_output.last_hidden_state[:, 0]

# embedding function to generate embedding for a given string ( query )
def get_embeddings(text_list):
    encoded_input = tokenizer(
        text_list, padding=True, truncation=True, return_tensors="pt"
    )
    # print("embedding_input : ")
    # print(encoded_input)
    encoded_input = {k: v.to(device) for k, v in encoded_input.items()}
    model_output = model(**encoded_input)
    return cls_pooling(model_output)

# embedding mapper function which will be used to generate our embeddings using the flattened and tokenized dataset
def map_embeddings(chunks):
    encoded_input = {k: torch.tensor(v).to(device) for k, v in chunks.items()}
    embeddings = model(**encoded_input).last_hidden_state[:,0]
    return embeddings.detach().cpu().numpy()[0]

# generate the embeddings ( this portion of the code is a major bottleneck )
wiki_db_embedded = wiki_db_flattened.map(lambda x: {'embeddings' : map_embeddings(x['tokens'])})
print(wiki_db_embedded)

# save our generate embeddings as a checkpoint.
wiki_db_embedded.save_to_disk("wiki_db_embedded.hf")

# index out embeddings with faiss which will be used as our vector database.
wiki_db_embedded.add_faiss_index(column = "embeddings")


# testing our vector lookup... 
import numpy as np

# specify an example query
query = 'cholesterol with fast food consumptions'

# generate the embeddings for our query.
query_embedding = get_embeddings(query).detach().cpu()
query_embedding = np.asarray(query_embedding)

print(query_embedding.shape)

# perform the lookup of the query with our vector db.
scores, samples = wiki_db_embedded.get_nearest_examples(
        "embeddings", query_embedding, k = 10
)

# print the retrieved chunks and evaluate the quality of the generated query.
print(scores)
print(samples['title'])
print(samples['text'])




