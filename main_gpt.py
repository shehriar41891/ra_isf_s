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
from dotenv import load_dotenv
from agents.direct_query_llm import direct_relevant_answer
from agents.context_query_llm import relevant_answer
from similarity_module.filtering import chunk_paragraphs_with_overlap,filter_chunks_by_similarity
from similarity_module.realtime_searching import Googlesearch
import litellm
# from data_loading import load_passages_id_map

# openai_api_key = os.getenv('OPENAI_API')
# os.environ['OPENAI_API_KEY'] = openai_api_key

import tempfile
litellm.set_verbose=True
# Ensure tempfile uses the desired directory
tempfile.tempdir = 'D:/rs-isf/ra-isf'
print(f"Python temp directory: {tempfile.gettempdir()}")


# llm_model = ChatOpenAI()

def load_dataset(data_path):
    dataset = list()
    with open(data_path, 'r', encoding='UTF-8') as f:
        for idx, line in enumerate(f):
            datas = json.loads(line)
            dataset.append(datas)
    return dataset

def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s.strip()))))

def f1_score(prediction, ground_truth):
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1

def mean_pooling(token_embeddings, mask):
    token_embeddings = token_embeddings.masked_fill(~mask[..., None].bool(), 0.)
    sentence_embeddings = token_embeddings.sum(dim=1) / mask.sum(dim=1)[..., None]
    return sentence_embeddings
# embeddings = mean_pooling(outputs[0], inputs['attention_mask'])


def load_contriever():
    model_name = "facebook/contriever-msmarco"  # Using the model ID from Hugging Face
    print(f"Loading model from: {model_name}")
    model = AutoModel.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer

# def load_passages_id_map():
#     index = retrieval_contriever.src.index.Indexer(768, 0, 8)
#     # index all passages
#     input_paths = glob.glob('./data/wikipedia_embeddings/*')
#     input_paths = sorted(input_paths)
#     print('The input path is ',input_paths)
#     # embeddings_dir = os.path.dirname(input_paths[0])
#     embeddings_dir = r'D:\rs-isf\ra-isf\fiass_store'
#     index_path = os.path.join(embeddings_dir, "index.faiss")
#     print('Index path: ',index_path)
#     if os.path.exists(index_path):
#         print('The path exists')
#         index.deserialize_from(embeddings_dir)
#     else:
#         print(f"Indexing passages from files {input_paths}")
#         start_time_indexing = time.time()
#         print('Single path:',input_paths)
#         index_encoded_data(index, input_paths, c_args.indexing_batch_size)
#         print(f"Indexing time: {time.time()-start_time_indexing:.1f} s.")
#         if True:
#             # import random
#             # random_filename = f"index_{random.randint(0, 99)}.faiss"
#             # index_file_path = os.path.join(embeddings_dir, random_filename)

#             # print(f"We are saving the files to: {index_file_path}")
#             index.serialize(embeddings_dir)
#         else:
#             print('Not saving it')


    # load passages
    print('Now we started loading the passages................')
    passages = load_passages('./data/psgs_w100.tsv')
    passage_id_map = {x["id"]: x for x in passages}
    return passage_id_map, index

def beam_retrieve(input, contriever_model, contriever_tokenizer):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Ensure input is properly formatted
    queries = [" ".join(input[0]) + " " + " ".join(input[1])] if isinstance(input[0], list) and isinstance(input[1], list) else [str(input[0]) + " " + str(input[1])]
    
    # Get embeddings of queries
    questions_embedding = embed_queries(c_args, queries, contriever_model, contriever_tokenizer)

    # get top k results
    top_ids_and_scores = index.search_knn(questions_embedding, 10)

    print('The top ids retrieved are',top_ids_and_scores)

    m_docs = []
    m_scores = []
    print('Passages length:',len(passage_id_map))
    for i in top_ids_and_scores:
        i = str(i)
        docs = [passage_id_map[i]]
        m_docs.append(docs)
    
    # m_docs = []
    # m_scores = []
    # for i, score in enumerate(top_ids_and_scores):
    #     docs = [passage_id_map[doc_id] for doc_id in score[0]]
    #     print(docs)
    #     scores = [str(s) for s in score[1]]  # corrected indexing
    #     m_docs.append(docs)
    #     m_scores.append(scores)
    
    print(m_docs)

    return m_docs, m_scores

def gpt_mdoel_init():
    # contriever, contriever_tokenizer = load_contriever()
    base_model = model_init(args.base_model_path)
    sk_model, sk_tokenizer = model_init(args.self_knowledge_model_path)
    pr_model, pr_tokenizer = model_init(args.passage_relevance_model_path)
    td_model, td_tokenizer = model_init(args.task_decomposition_model_path)
    print('Sk_model is ',sk_model)
    return Self_Knowledge_Model(sk_model, sk_tokenizer), Passage_Relevance_Model(pr_model, pr_tokenizer), Task_Decomposition_Model(td_model, td_tokenizer)

outfile = '/content/drive/MyDrive/ra-isf2/output_skm.txt'
def write_output_to_file(output, outfile):
    """Append the model output to the specified output file."""
    with open(outfile, 'a') as f:
        f.write(output + '\n') 

def problem_solving(input, iter, SKM, PRM, TDM, contriever, contriever_tokenizer): #passage_id_map, index
    if iter > args.iteration_max_time:
        return "not know"
    
    print('The real input in our case is ',input)
    print('input[0]:',input[0],'input[1]:',input[1])
    result = run_skm_agent(input)
    result = str(result)
    print('The result is ',result)
    if result.strip() in  ['know','know.','Know','Know.']:
        print('Model knows the answer',result)
        prompt = input
        print('The prompt is ',prompt)
        answer = direct_relevant_answer(prompt)
        answer = str(answer)
        print('The answer to the question is ',answer)
        
        return answer


    # Perform beam retrieval (commented for simplicity)
    m_docs, m_scores = beam_retrieve(input, contriever, contriever_tokenizer)
    r_docs = []
    texts = [item[0]['text'] for item in m_docs]
    
    print(texts)
    
    user_query = input[1]
    #splitting the data into chunks 
    chunks = filter_chunks_by_similarity(texts,user_query)
    # filtering chunks based on similarity score 
    print(chunks)
    similar_chunks = filter_chunks_by_similarity(chunks,input[1])
    print(similar_chunks)
    
    if similar_chunks:
        print('Entered into similar chunks........')
        answer = relevant_answer(similar_chunks,user_query)
        
        return answer
    else:
        print('Entered into google chunks........')
        search_results = Googlesearch(user_query)
        answer = relevant_answer(search_results,user_query)
        
        return answer
    


def run_gpt(dataset, SKM, PRM, TDM, contriever, contriever_tokenizer): #passage_id_map, index
    answer_set = list()
    i = 0
    for idx, data in enumerate(dataset):
        try:
            print('The question asked is ', data)
            input = data['question']
            ans = problem_solving(input, 0, SKM, PRM, TDM, contriever, contriever_tokenizer)
            print('Answer we get',ans)
            answer_set.append(ans)
            if i == 10:
                break
            i = i + 1
        except Exception as e:
            print(f"Error occurred while processing question {idx}: {e}")
    
    print('Answer set we get',answer_set)
            
    return answer_set

    # with open(args.output_path, 'a', encoding = "UTF-8") as f:
    #     for idx, ans in enumerate(answer_set):
    #         f.write(json.dumps(ans) + '\n')
            
 

if __name__ == '__main__':
    # Load the base model and tokenizer
    # base_model, tokenizer = model_init(args.base_model_path)
    
    # Load the Contriever model and tokenizer
    contriever, contriever_tokenizer = load_contriever()
    
    # Load the dataset
    dataset = load_dataset(args.data_path)
    
    print(f"Dataset size: {len(dataset)}")
    print(f"First few entries: {dataset[:5]}")

    
    # Load the passage ID map and index
    # passage_id_map, index = load_passages_id_map()
    
    # Initialize submodels: Self-Knowledge, Passage-Relevance, Task-Decomposition
    SKM, PRM, TDM = gpt_mdoel_init()
    
    # Call run_gpt with all required arguments
    answer = run_gpt(dataset, SKM, PRM, TDM, contriever, contriever_tokenizer) #passage_id_map, index
    
    print('The final answers are: ',answer)
