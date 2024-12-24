import json
import logging
import re
import string

import torch
import os
import argparse
import csv
import json
import logging
import pickle
import time
import glob

import numpy as np
import torch
import transformers
import requests
from agents.skm_agents import run_skm_agent


class Self_Knowledge_Model():
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    # def find_known(self, context, query):
    #     print('Here we start with SRM module')
    #     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #     inputs = self.tokenizer(" ".join(context) + " " + query, return_tensors="pt").to(device)
    #     generate_ids = self.model.generate(**inputs, max_length=512, temperature=0)
    #     generate_ids = generate_ids[0][len(inputs["input_ids"][0]):-1]
    #     result = self.tokenizer.decode(generate_ids)
    #     print('The result from SKM is ',result)
    #     if result == "know":
    #         return True
    #     elif result == "unknow":
    #         return False
    #     else:
    #         print(f"Invalid output on SKM query: {inputs}")
    #         return False
        
    def find_known(self,query):
        print('Here we start with SRM module')
        print('The query is ',query)
        query_res,query_analysis = run_skm_agent(query)
        
        print('The Query Response is ',query_res)
        
        if query_analysis == "know":
            return True
        elif query_analysis == "unknow":
            return False
        else:
            print(f'Invalid analysis type {query_analysis}')
            return False