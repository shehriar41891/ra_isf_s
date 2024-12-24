import json
import logging
import re
import string

import os
import argparse
import csv
import json
import logging
import pickle
import time
import glob
from pathlib import Path

import numpy as np
import torch
import transformers

from openai import OpenAI
from config import args
from contriever_config import c_args
from collections import Counter
from utils import write_json, print_now, load_data, print_exp, mkpath
# from source.model.llama2_predict import predict, model_init
from source.model.gpt_predict import predict,model_init

from transformers import LlamaTokenizer, LlamaForCausalLM, AutoConfig

from retrieval_contriever.passage_retrieval import embed_queries, index_encoded_data, add_embeddings, validate, add_passages, add_hasanswer, load_data
from source.arch.passage_relevance.pr import Passage_Relevance_Model
from source.arch.self_knowledge.sk import Self_Knowledge_Model
from source.arch.task_decomposition.td import Task_Decomposition_Model
from transformers import AutoModel, AutoTokenizer

import retrieval_contriever.src.index
import retrieval_contriever.src.contriever
from retrieval_contriever.src.data import load_passages
from agents.skm_agents import run_skm_agent
from agents.tdm_agent import run_tdm_agent
from source.model.llm_model import get_llm
from source.arch.passage_relevance.similarirty_splitting import split_text_by_words,compute_cosine_similarity
import os 

def load_passages_id_map():
    embeddings_dir = r'D:\rs-isf\ra-isf\fiass_store'
    index_path = os.path.join(embeddings_dir, "index.faiss")
    passage_map_path = os.path.join(embeddings_dir, "passage_id_map.pkl")
    
    # Initialize Indexer
    index = retrieval_contriever.src.index.Indexer(768, 0, 8)
    
    # Check if the index file exists
    if os.path.exists(index_path):
        print('The index file exists, loading...')
        index.deserialize_from(embeddings_dir)
    else:
        print('Index file not found. Indexing passages...')
        input_paths = glob.glob('./data/wikipedia_embeddings/*')
        input_paths = sorted(input_paths)
        print(f"Input paths: {input_paths}")
        start_time_indexing = time.time()
        index_encoded_data(index, input_paths, c_args.indexing_batch_size)
        print(f"Indexing time: {time.time() - start_time_indexing:.1f} seconds")
        index.serialize(embeddings_dir)
        print('Indexing completed and saved.')

    # Check if the passage ID map exists
    if os.path.exists(passage_map_path):
        print('Passage ID map exists, loading...')
        with open(passage_map_path, 'rb') as f:
            passage_id_map = pickle.load(f)
    else:
        print('Passage ID map not found. Generating...')
        passages = load_passages('./data/psgs_w100.tsv')
        passage_id_map = {x["id"]: x for x in passages}
        with open(passage_map_path, 'wb') as f:
            pickle.dump(passage_id_map, f)
        print('Passage ID map generated and saved.')

    return passage_id_map, index


load_passages_id_map()