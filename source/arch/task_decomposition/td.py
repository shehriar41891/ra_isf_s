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
import torch

import numpy as np
import torch
import transformers
from agents.tdm_agent import run_tdm_agent

class Task_Decomposition_Model():
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.query_list = list()
    def decompose(self, context, query):
        print('We have to enter into TDM branch')
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # inputs = self.tokenizer(" ".join(context) + " " + query, return_tensors="pt").to(device)
        # generate_ids = self.model.generate(**inputs, max_length=512, temperature=0)
        # generate_ids = generate_ids[0][len(inputs["input_ids"][0]):-1]
        # result = self.tokenizer.decode(generate_ids)
        # print('Th result from TDM branch is ',result)
        # try:
        #     data = json.loads(result)
        #     for idx, q in data['query']:
        #         self.query_list.append(q)
        # except json.JSONDecodeError:
        #     print(f"Invalid format on TDM query: json_string: {result}")
        
        result  = run_tdm_agent(query=query)
        
        print(result)
        
        return result
