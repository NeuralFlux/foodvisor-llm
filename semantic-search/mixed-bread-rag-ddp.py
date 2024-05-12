import mwclient  # for downloading example Wikipedia articles
import mwparserfromhell  # for splitting Wikipedia articles into sections
import openai  # for generating embeddings
import os  # for environment variables
import pandas as pd  # for DataFrames to store article sections and embeddings
import re  # for cutting <ref> links out of Wikipedia articles
import tiktoken  # for counting tokens
import gc # for garbage collections

import torch.multiprocessing as mp # for multiprocessing
from torch.nn.parallel import DistributedDataParallel as DDP # for DDP to embed using 2 or more GPUs
from torch.distributed import init_process_group, destroy_process_group # To create and destroy DDP processes

import pickle # to pickle files

import faiss # to index embedings using faiss

from datasets import load_dataset # huggingface datasets library to work with the dataset
import datasets # for huggingface datasets

from transformers import AutoTokenizer, AutoModel # for loading the models using huggingface transformers api
import torch # pytorch !

# Spicify directory path to the parsed wiki articles.
dump_names = "dumps_1/wiki_dump_out_" 
# specify the number of dumps in the above directory.
num_of_dumps = 4


# DDP Setup script.
def ddp_setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    init_process_group(backend = "gloo", rank = rank, world_size = world_size)


# The Mapper function to chunk down long wiki articles based on AVERAGE token length.
MAX_TOKENS = 300
def add_token_lens(x):

    token_checkpoint = 0
    collect_chunks = []
    collect_chunk_lens = []

    for i, char in enumerate(x['text']):
        if char == '.':
            tokens = tokenizer(
                [x['text'][token_checkpoint: i + 1]], padding = True, truncation = False, return_tensors = "pt"
            )
            tokens_len = tokens['input_ids'].shape[1]
            if tokens_len > 300:
                collect_chunks.append( (token_checkpoint, i + 1) )
                collect_chunk_lens.append( tokens_len )
                token_checkpoint = i + 1

    if token_checkpoint < len(x['text']):
        tokens = tokenizer(
                x['text'][token_checkpoint: len(x['text'])], padding = True, truncation = False, return_tensors = "pt"
        )
        tokens_len = tokens['input_ids'].shape[1]
        collect_chunks.append( (token_checkpoint,  len(x['text'])) )
        collect_chunk_lens.append( tokens_len )

    tokens = tokenizer(
        [x['text'][i:j] for i, j in collect_chunks],
        padding = True, truncation = True, return_tensors = "pt"
    )
    x['text'] = [x['text'][i:j] for i, j in collect_chunks]
    x['tokenized_chunks'] = tokens
    x['tokenized_chunks_lens'] = collect_chunk_lens

    return x
 
# Main function to generate embeddings using pytorch DDP.
def main(rank, world_size, dataset, size):
    
    ddp_setup(rank, world_size)

    model_name = "mixedbread-ai/mxbai-embed-large-v1"
    model = AutoModel.from_pretrained(model_name, force_download = True)
    
    model = model.to(rank)
    model.eval()
    model = DDP(model, device_ids = [rank])
   
    ds_shard = dataset.shard(
        num_shards=world_size,
        index=rank,
        contiguous=True,
    )

    def map_embeddings(chunks):
        encoded_input = {k: torch.tensor(v).to(rank) for k, v in chunks.items()}
        embeddings = model(**encoded_input).last_hidden_state[:,0]
        return embeddings.detach().cpu().numpy()[0]
    
    wiki_db_embedded = ds_shard.take(size).map(lambda x: {'embeddings' : map_embeddings(x['tokens'])})

    print(wiki_db_embedded)

    wiki_db_embedded.save_to_disk(f"wiki_db_embedded_rank_{rank}.hf")
    
    torch.distributed.barrier()
    destroy_process_group()

if __name__ == "__main__":

    world_size = torch.cuda.device_count()
    rank = 0

    wiki_dumps = []
    # 0 -> 0, 1
    # 1 -> 2, 3 
    for i in range(4):
        dbfile = open(f'{dump_names}{i}.pickle', 'rb')
        wiki_dumps.extend(pickle.load(dbfile))
        dbfile.close()
    
    print("\ndump size : ", len(wiki_dumps))
    
    wiki_db = datasets.Dataset.from_list(wiki_dumps[:1000])
    
    model_name = "mixedbread-ai/mxbai-embed-large-v1"
    tokenizer = AutoTokenizer.from_pretrained(model_name, torch_dtype=torch.bfloat16)
    model = AutoModel.from_pretrained(model_name, force_download = True)
    
    model = model.to(rank)
   
    wiki_db = wiki_db.filter(lambda x: len(x['text']) > 100)
    wiki_db = wiki_db.map(add_token_lens)
    
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
    
    wiki_db_flattened = datasets.Dataset.from_list(wiki_db_extended)
    
    del wiki_db
    del wiki_db_extended
    gc.collect()
    
    print("\nFLATTENED DATASET : ")
    print(wiki_db_flattened)
    
    idx = 0
    
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


    size = int(wiki_db_flattened.shape[0] / world_size)
    mp.spawn(main, args = ( world_size, wiki_db_flattened, size, ), nprocs = world_size, join = True)

    wiki_db_embedded = datasets.concatenate_datasets( [ datasets.load_from_disk(f"wiki_db_embedded_rank_{i}.hf") for i in range(world_size)] )
    print("\nEMBEDDED DATASET : ")
    print(wiki_db_embedded)
   
    wiki_db_embedded.add_faiss_index(column = "embeddings")


    import numpy as np

    def cls_pooling(model_output):
        return model_output.last_hidden_state[:, 0]
    
    def get_embeddings(text_list):
        encoded_input = tokenizer(
            text_list, padding=True, truncation=True, return_tensors="pt"
        )
        # print("embedding_input : ")
        # print(encoded_input)
        encoded_input = {k: v.to(rank) for k, v in encoded_input.items()}
        model_output = model(**encoded_input)
        return cls_pooling(model_output)
   
    query = 'cholesterol with fast food consumptions'
    print("QUERY : ", query)
    
    query_embedding = get_embeddings(query).detach().cpu()
    query_embedding = np.asarray(query_embedding)
    
    print(query_embedding.shape)
   
    k = 10
    scores, samples = wiki_db_embedded.get_nearest_examples(
            "embeddings", query_embedding, k = k
    )
    
    print("FAISS SCORES : ", scores)
    for i in range(k):
        print(" ")
        print("- ", i, " Title : ", samples['title'][i])
        print("- ", samples['text'][i])
#    
#    
#    
#   

#    world_size = 2
#
#    wiki_dumps = []
#    num_of_dumps = 4
#    dump_names = "dumps_1/wiki_dump_out_"
#    
#    for i in range(num_of_dumps):
#        dbfile = open(f'{dump_names}{i}.pickle', 'rb')
#        wiki_dumps.extend(pickle.load(dbfile))
#        dbfile.close()
#    
#    print("dump size : ", len(wiki_dumps))
#    
#    wiki_db = datasets.Dataset.from_list(wiki_dumps[:100])
#    
#    model_name = "mixedbread-ai/mxbai-embed-large-v1"
#    tokenizer = AutoTokenizer.from_pretrained(model_name, torch_dtype=torch.bfloat16)
#    model = AutoModel.from_pretrained(model_name, force_download = True)
#    
#    model = model.to(rank)
#    ddp_model = DDP(model, device_ids = [rank])
#    
#    print("model")
#    # device = torch.device('cuda:0')
#    # model.to(device)
#    
#    ddp_setup(0, 2)
#    # split sections into chunks
#    MAX_TOKENS = 300
#    wikipedia_strings = []
#    
#    def add_token_lens(x):
#    
#        token_checkpoint = 0
#        collect_chunks = []
#        collect_chunk_lens = []
#    
#        for i, char in enumerate(x['text']):
#            if char == '.':
#                tokens = tokenizer(
#                    [x['text'][token_checkpoint: i + 1]], padding = True, truncation = False, return_tensors = "pt"
#                )
#                tokens_len = tokens['input_ids'].shape[1]
#                if tokens_len > 300:
#                    collect_chunks.append( (token_checkpoint, i + 1) )
#                    collect_chunk_lens.append( tokens_len )
#                    token_checkpoint = i + 1
#    
#        if token_checkpoint < len(x['text']):
#            tokens = tokenizer(
#                    x['text'][token_checkpoint: len(x['text'])], padding = True, truncation = False, return_tensors = "pt"
#            )
#            tokens_len = tokens['input_ids'].shape[1]
#            collect_chunks.append( (token_checkpoint,  len(x['text'])) )
#            collect_chunk_lens.append( tokens_len )
#    
#        tokens = tokenizer(
#            [x['text'][i:j] for i, j in collect_chunks],
#            padding = True, truncation = True, return_tensors = "pt"
#        )
#        x['text'] = [x['text'][i:j] for i, j in collect_chunks]
#        x['tokenized_chunks'] = tokens
#        x['tokenized_chunks_lens'] = collect_chunk_lens
#        return x
#    
#    wiki_db = wiki_db.filter(lambda x: len(x['text']) > 100)
#    
#    wiki_db = wiki_db.map(add_token_lens)
#    print(wiki_db)
#    
#    wiki_db_extended = []
#    _id = 0
#    
#    for x in wiki_db:
#      for i, tokens_len in enumerate(x['tokenized_chunks_lens']):
#        wiki_db_extended.extend([{
#          '_id': _id,
#          'title': x['title'],
#          'site': x['site'],
#          'text': x['text'][i],
#          'tokens': {key: [value[i]] for key, value in x['tokenized_chunks'].items()},
#          'tokens_len': tokens_len
#        }])
#        _id += 1
#    
#    wiki_db_flattened = datasets.Dataset.from_list(wiki_db_extended)
#    
#    del wiki_db
#    del wiki_db_extended
#    gc.collect()
#    
#    print(wiki_db_flattened)
#    
#    idx = 0
#    
#    print("\n title : \n")
#    print(wiki_db_flattened[idx]['title'])
#    print("\n site : \n")
#    print(wiki_db_flattened[idx]['site'])
#    print("\n text : \n")
#    print(wiki_db_flattened[idx]['text'])
#    print("\n tokens_len : \n")
#    print(wiki_db_flattened[idx]['tokens_len'])
#    print("\n tokens : \n")
#    print(wiki_db_flattened[idx]['tokens'])
#    
#    
#    def cls_pooling(model_output):
#        return model_output.last_hidden_state[:, 0]
#    
#    def get_embeddings(text_list):
#        encoded_input = tokenizer(
#            text_list, padding=True, truncation=True, return_tensors="pt"
#        )
#        # print("embedding_input : ")
#        # print(encoded_input)
#        encoded_input = {k: v.to(device) for k, v in encoded_input.items()}
#        model_output = model(**encoded_input)
#        return cls_pooling(model_output)
#    
#    
#    def map_embeddings(chunks):
#        encoded_input = {k: torch.tensor(v).to(device) for k, v in chunks.items()}
#        embeddings = model(**encoded_input).last_hidden_state[:,0]
#        return embeddings.detach().cpu().numpy()[0]
#    
#     
#    
#    wiki_db_embedded = wiki_db_flattened.map(lambda x: {'embeddings' : map_embeddings(x['tokens'])})
#    print(wiki_db_embedded)
#    
#    rank = torch.distributed.get_rank()
#    world_size = torch.distributed.get_world_size()
#    
#    print('rank : ', rank)
#    print('world size : ', world_size)
#    
#    ds_shard = wiki_db_flattened.shard(num_shards = world_size,
#                            index = rank,
#                            contiguous = True)
#    
#    print(ds_shard)
#    ds_shard = ds_shard.map(lambda x: {'embeddings' : map_embeddings(x['tokens'])})
#    torch.distributed.barrier()
#    
#    print(ds_shard)
#    
#    wiki_db_embedded.save_to_disk("wiki_db_embedded_LLM2VEC.hf")
#    
#    wiki_db_embedded.add_faiss_index(column = "embeddings")
#    
#    import numpy as np
#    
#    query = 'cholesterol with fast food consumptions'
#    
#    query_embedding = get_embeddings(query).detach().cpu()
#    query_embedding = np.asarray(query_embedding)
#    
#    print(query_embedding.shape)
#    
#    scores, samples = wiki_db_embedded.get_nearest_examples(
#            "embeddings", query_embedding, k = 10
#    )
#    
#    print(scores)
#    print(samples['title'])
#    print(samples['text'])
#    
#    
#    
#
