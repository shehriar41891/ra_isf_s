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

import numpy as np
import torch
import transformers
from crewai import Agent, Task, Crew, Process


class Passage_Relevance_Model():
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def find_relevance(self, context, query, passage):
        print('We have to enter into PRM module')
        inputs = self.tokenizer(context + query + "\nPassage: " + passage, return_tensors="pt").to('cuda')
        generate_ids = self.model.generate(**inputs, max_length=512, temperature=0)
        generate_ids = generate_ids[0][len(inputs["input_ids"][0]):-1]
        result = self.tokenizer.decode(generate_ids)
        print('Result from PRM is ',result)
        if result == "relevance":
            return True
        elif result == "irrelevance":
            return False
        else:
            print(f"Invalid output on PRM query: {context + query}")
            return False

    def find_relevance(documents,query):
        Evaluator = Agent(
        role='Evaluation of the documents',
        goal="""
        Evaluate the given 10 documents {documents} based on the query: {query} and take the top 3 out of 
        them which you think contain the answer to the query: {query}
        """,
        verbose=True,
        memory=True,
        backstory=(
        """You are an document evaluator who looks carefully on the documents and query and take top 3
        most matched documents with query"""
        ),
        allow_delegation=True
        )
        
        Evaluation = Task(
        description=(
            """Analyze the documents {documents} based on the query {query}. Take top 3 documents
            from given 10 documents which you think contains the answer to the {query}
            rank them from most relevant to less relevant        
            """
        ),
        expected_output="Top 3 relevant documents to the given query",
        agent=Evaluator,
        allow_delegation=True
        )
        
        crew = Crew(
        agents=[Evaluator],
        tasks=[Evaluation],
        verbose=True,
        process=Process.sequential,
        debug=True,
        max_iterations=2
        )
        
        result = crew.kickoff(inputs={'documents': documents,'query' : query})
        
        return result
    
        
