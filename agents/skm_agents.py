from crewai import Agent, Task, Crew, Process
from dotenv import load_dotenv
import os 
import litellm
# from test.test_queries import spotifyQueries

litellm.set_verbose=True

load_dotenv()

openai_api = os.getenv('OPENAI_API')
# os.environ['OPENAI_API'] = openai_api
os.environ['MODEL_NAME'] = 'gpt-3.5-turbo'

litellm.api_key = openai_api

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



def run_skm_agent(query):
    result = compile_skm_agent(query)
    result = str(result)