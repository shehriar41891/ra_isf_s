import json
import re
import glob
import random
import string
from collections import Counter
import numpy as np
import torch
import time 
from config import args
from contriever_config import c_args
from transformers import AutoModel, AutoTokenizer
from retrieval_contriever.passage_retrieval import embed_queries, index_encoded_data, add_embeddings, validate, add_passages, add_hasanswer, load_data
import retrieval_contriever.src.index
from retrieval_contriever.src.data import load_passages

from agents.direct_query_llm import direct_relevant_answer
from agents.skm_agents import run_skm_agent
from agents.tdm_agent import run_tdm_agent
from evaluation_st.st_direct_llm import yes_no_answer
from evaluation_st.st_query_llm import yes_no_answer_with_context
from answers_qa.trivia_Qa import random_sample
from answers_qa.st_qa import examples_data

# from agents.context_query_llm import relevant_answer
from similarity_module.filtering import split_array_into_chunks,filter_chunks_by_similarity_openai
from similarity_module.realtime_searching import Googlesearch
import os 
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, Process
from dotenv import load_dotenv
import os 
import litellm
# from test.test_queries import spotifyQueries
from langchain.llms import OpenAI

litellm.set_verbose=True

load_dotenv()

openai_api = os.getenv('OPENAI_API')
# os.environ['OPENAI_API'] = openai_api
os.environ['MODEL_NAME'] = 'gpt-3.5-turbo'

litellm.api_key = openai_api

openai_Api_key = os.getenv('OPENAI_API_KEY')
os.environ['MODEL_NAME'] = 'gpt-3.5-turbo'
os.environ['OPENAI_API_KEY'] = openai_Api_key

llm = OpenAI(api_key = openai_Api_key)

# print(direct_relevant_answer('Who is babar azam'))


#####################################Query Evaluator Agent########################################

Query_Evaluator = Agent(
role='Evaluating the query',
goal="""
Evaluate the given query {query} and based on your knowledge, determine whether you have an idea or understanding of the query. 
If you have any slightest knowledge or reasonable idea about the query, answer with 'know'. 
If the query is entirely outside your knowledge domain or you have no idea, answer with 'not know'.
""",
verbose=True,
memory=True,
backstory=(
"""You are a Query Evaluator who is an expert in evaluating queries and determining whether you have some knowledge 
or understanding about the topic. Answer with 'know' if you have any knowledge, otherwise answer 'not know'."""
),
allow_delegation=True
)

Query_Evaluation = Task(
description=(
    """Analyze the given query: {query} and based on your knowledge or understanding, answer 'know' 
    if you have any idea about it. If you have no idea or it is outside your knowledge domain, answer 'not know'.
    """
),
expected_output="'know' or 'not know' based on your knowledge",
agent=Query_Evaluator,
allow_delegation=True
)


def compile_skm_agent(query):
    crew = Crew(
    agents=[Query_Evaluator],
    tasks=[Query_Evaluation],
    verbose=True,
    process=Process.sequential,
    debug=True,
    max_iterations=5
    )
    
    result = crew.kickoff(inputs={'query' : query})
    
    return result


############################################ Conextual Answers # ########################################
import os
import pprint
from dotenv import load_dotenv
from langchain_community.utilities import GoogleSerperAPIWrapper
from crewai import Agent, Task, Crew, Process
import os
import litellm
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# from models.embedding_model import get_embedding_model
# from pinecone.grpc import PineconeGRPC as Pinecone


# Enable verbose mode for LiteLLM
litellm.set_verbose = True

# Load environment variables
load_dotenv()
openai_api = os.getenv('OPENAI_API')
pinecone_api_key = os.getenv('PINECONE_API_KEY')

os.environ['OPENAI_API_KEY'] = openai_api
os.environ['MODEL_NAME'] = 'gpt-3.5-turbo'


Short_Responder = Agent(
    role='Short answer giver',
    goal=(
        """Analyze the array of documents: {text} based on the user query: {query}. Extract **only** the exact
        word or phrase that directly answers the query, with **no supporting phrases or extra words**. 
        If the answer is not found in the provided context, use your own knowledge to give a **single word 
        or precise phrase** as the response, with no supporting details or elaboration.
        For example, if the query asks about the capital of Pakistan, the answer should be 'Islamabad' only, 
        nothing else."""
    ),
    verbose=True,
    memory=True,
    backstory=(
        """You are an expert document evaluator, skilled in extracting precise answers
        from text based on a query, with no added elaboration. You are also capable of providing
        concise responses from your own knowledge when the information is not found in the documents."""
    ),
    allow_delegation=True,
)

# Define the evaluation task
Short_answering = Task(
    description=(
        """Analyze the provided array of documents `{text}` to extract the exact word or phrase 
        that directly answers the user query `{query}`. Avoid any additional context, 
        explanations, or supporting phrases. If the answer is not present in the documents, 
        provide a response from your own knowledge as a single word or precise phrase."""
    ),
    expected_output="A single word or phrase that directly answers the query, with no additional text.",
    agent=Short_Responder,
    allow_delegation=True,
)


# Create the crew
crew = Crew(
    agents=[Short_Responder],
    tasks=[Short_answering],
    verbose=True,
    process=Process.sequential,
    debug=True,
    max_iterations=5,
)

def relevant_answer(results, user_query):
    result = crew.kickoff(inputs={'text': results, 'query': user_query})
    return result


################################################################################################
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


def load_passages_id_map():
    index = retrieval_contriever.src.index.Indexer(768, 0, 8)
    # index all passages
    input_paths = glob.glob('./data/wikipedia_embeddings/*')
    input_paths = sorted(input_paths)
    print('The input path is ',input_paths)
    # embeddings_dir = os.path.dirname(input_paths[0])
    embeddings_dir = r'D:\rs-isf\ra-isf\fiass_store'
    index_path = os.path.join(embeddings_dir, "index.faiss")
    print('Index path: ',index_path)
    if os.path.exists(index_path):
        print('The path exists')
        index.deserialize_from(embeddings_dir)
    else:
        print(f"Indexing passages from files {input_paths}")
        start_time_indexing = time.time()
        print('Single path:',input_paths)
        index_encoded_data(index, input_paths, c_args.indexing_batch_size)
        print(f"Indexing time: {time.time()-start_time_indexing:.1f} s.")
        if True:
            # import random
            # random_filename = f"index_{random.randint(0, 99)}.faiss"
            # index_file_path = os.path.join(embeddings_dir, random_filename)

            # print(f"We are saving the files to: {index_file_path}")
            index.serialize(embeddings_dir)
        else:
            print('Not saving it')


    # load passages
    print('Now we started loading the passages................')
    passages = load_passages('./data/psgs_w100.tsv')
    passage_id_map = {x["id"]: x for x in passages}
    return passage_id_map, index


def beam_retrieve(input, contriever_model, contriever_tokenizer,passage_id_map,index):
    print('We entered into beam_retriever')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    print('The queries in beam retriever are',input)
    
    # Get embeddings of queries
    questions_embedding = embed_queries(c_args, input, contriever_model, contriever_tokenizer)

    # get top k results
    top_ids_and_scores = index.search_knn(questions_embedding, 20)

    print('The top ids retrieved are',top_ids_and_scores)

    m_docs = []
    m_scores = []
    print('Passages length:',len(passage_id_map))
    for i in top_ids_and_scores:
        print(i)
        i = str(i)
        print("Type of i:", type(i))
        if i not in passage_id_map:
            print(f"Missing passage ID in map: {i}")
        else:
            docs = passage_id_map[i]
            print('The document is ',docs)
            m_docs.append(docs['text'])
    
    # m_docs = []
    # m_scores = []
    # for i, score in enumerate(top_ids_and_scores):
    #     docs = [passage_id_map[doc_id] for doc_id in score[0]]
    #     print(docs)
    #     scores = [str(s) for s in score[1]]  # corrected indexing
    #     m_docs.append(docs)
    #     m_scores.append(scores)
    
    print('The m_docs is ',m_docs)

    return m_docs, m_scores


# def gpt_mdoel_init():
#     # contriever, contriever_tokenizer = load_contriever()
#     base_model = model_init(args.base_model_path)
#     sk_model, sk_tokenizer = model_init(args.self_knowledge_model_path)
#     pr_model, pr_tokenizer = model_init(args.passage_relevance_model_path)
#     td_model, td_tokenizer = model_init(args.task_decomposition_model_path)
#     print('Sk_model is ',sk_model)
#     return Self_Knowledge_Model(sk_model, sk_tokenizer), Passage_Relevance_Model(pr_model, pr_tokenizer), Task_Decomposition_Model(td_model, td_tokenizer)

outfile = '/content/drive/MyDrive/ra-isf2/output_skm.txt'
def write_output_to_file(output, outfile):
    """Append the model output to the specified output file."""
    with open(outfile, 'a') as f:
        f.write(output + '\n') 

def problem_solving(input, iter, contriever, contriever_tokenizer,passage_id_map,index): #passage_id_map, index
    print('we entered into problem solving')
    if iter > args.iteration_max_time:
        return "not know"
    
    print('The real input in our case is ',input)
    try:
        result = compile_skm_agent(input)
        print("Agent result:", result)
    except Exception as e:
        print("Error while calling the agent:", str(e))

    result = str(result)
    print('The result is ',result)
    if result.strip() in  ['know','know.','Know','Know.']:
        print('Model knows the answer',result)
        prompt = input
        print('The prompt is ',prompt)
        answer = yes_no_answer(prompt)
        answer = str(answer)
        print('The answer to the question is ',answer)
        
        return answer
    else:
        print('Model has no idea about it ')


        # Perform beam retrieval (commented for simplicity)
        m_docs, m_scores = beam_retrieve(input, contriever, contriever_tokenizer,passage_id_map,index)
        r_docs = []
        
        user_query = input[1]
        print('The betrayed user query is ',user_query)
        #splitting the data into chunks 
        chunks = split_array_into_chunks(m_docs)
        # filtering chunks based on similarity score 
        print('The chunks we get are',chunks)
        similar_chunks = filter_chunks_by_similarity_openai(chunks,user_query)
        print('The similar chunks we get are',similar_chunks)
        
        if similar_chunks:
            print('Entered into similar chunks........')
            answer = yes_no_answer_with_context(similar_chunks, input)
            
            print('The answer from similar chunks are',answer)
            
            return str(answer)
        else:
            print('Could not find similar chunks ')
            # print('Entered into google chunks........')
            # search_results = Googlesearch(user_query)
            # answer = relevant_answer(search_results,user_query)
            
            # return answer
        
            # use tdm agent to split the question into sub questions 
            sub_question = run_tdm_agent( input)
            
            questions = re.split(r'\d+\.\s', sub_question)[1:]  # Split and ignore the first empty string
            questions = [q.strip() for q in questions if q]  # Clean up whitespace and empty elements
            
            sub_qas = []
            for i, sub_query in enumerate(questions, start=1):
                print(f"Sub Question {i}: {sub_query}")
                
                sub_answer = problem_solving(input, iter, contriever, contriever_tokenizer,passage_id_map,index)
                sub_qa = [sub_query, sub_answer]
                sub_qas.append(sub_qa)
            sub_str = ""
            for idx, sub_qa in enumerate(sub_qas):
                sub_str = sub_str + "\nsub_question " + idx + ": " + sub_qa[0]
                sub_str = sub_str + "\nsub_question " + idx + ": " + sub_qa[1]
            sub_str = sub_str + "\nBase on the sub-question answer. Provide a one-word or one-phrase answer to the original question without supporting details."
            print(sub_str)
            answer = llm.predict(args, input[0] + sub_str + input[1])
            print('The answer from TDM branch is ',answer)
            return str(answer)
            
            
        
    


import json

def run_gpt(dataset, contriever, contriever_tokenizer, passage_id_map, index): 
    print('We entered into the run_gpt branch')
    answer_set = list()
    i = 0

    # Loop through the dataset
    for idx, data in enumerate(dataset):
        try:
            print('The question asked is ', data)
            print('question no.',i)
            input_data = data['input']
            ans = problem_solving(input_data, 0, contriever, contriever_tokenizer, passage_id_map, index)
            print('Answer we get', ans)
            
            # Append the answer to the answer_set with both question and answer
            element = {
                'question': data['input'],  # Save the question
                'answer': ans,                 # Save the answer
                'expected_answer': data['target'].split(".")[0]  # Save the expected answer if exists
            }
            output_file_path = "./answers_qa/output_st.json"
            with open(output_file_path, 'a', encoding="UTF-8") as f:
                f.write(json.dumps(element) + '\n')
            answer_set.append(element)
            
            i = i+1

        except Exception as e:
            print(f"Error occurred while processing question {idx}: {e}")
    
    print('Answer set we get', answer_set)

    # # Save the questions, answers, and expected answers to a file
    # output_file_path = "./answers_qa/output_hotpot.json"
    # with open(output_file_path, 'a', encoding="UTF-8") as f:
    #     for item in answer_set:
    #         f.write(json.dumps(item) + '\n')
            
    return answer_set


            
 

if __name__ == '__main__':
    # Load the base model and tokenizer
    # base_model, tokenizer = model_init(args.base_model_path)
    
    # Load the Contriever model and tokenizer
    contriever, contriever_tokenizer = load_contriever()
    
    # Load the dataset
    dataset = load_dataset(args.data_path)
    
    ###loading hotpot_qa###########
    file_path = "./dataset/StrategyQA/StrategyQA.json"

    # Read the JSON file
    with open(file_path, "r") as file:
        data = json.load(file)
        
    examples_data = data.get("examples", [])

    # Randomly sample 500 items
    random_sample = random.sample(examples_data, 600)
    
    # random_sample = random.sample(dataset, 100)
    print(f"Dataset size: {len(dataset)}")
    print(f"First few entries: {dataset[:5]}")

    
    # Load the passage ID map and index
    passage_id_map, index = load_passages_id_map()
    print('passage numbers',len(passage_id_map))
    from itertools import islice

        
    # for element in random_sample[:5]:
    #     print('question:',element['question'],'answer:',element['answer'])
    # Initialize submodels: Self-Knowledge, Passage-Relevance, Task-Decomposition
    # SKM, PRM, TDM = gpt_mdoel_init()
    
    # Call run_gpt with all required arguments
    answer = run_gpt(random_sample, contriever, contriever_tokenizer,passage_id_map,index) #passage_id_map, index
    
    print('The final answers are: ',answer)
