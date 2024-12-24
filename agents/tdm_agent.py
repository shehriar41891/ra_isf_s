from langchain import OpenAI
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, Process
import os

load_dotenv()
# Initialize the OpenAI LLM
api_key = os.getenv('OPENAI_API')
if api_key is None:
    raise ValueError("API Key not found in environment")
os.environ['OPENAI_API_KEY'] = api_key

llm = OpenAI(api_key=api_key, temperature=0)

# Define the Query Decomposer Agent
Query_decomposer = Agent(
    role='Decomposing the query',
    goal="""You will be given a long and complicated query: {query}. You need to decompose that query into smaller, understandable subqueries that are short and simple.""",
    verbose=True,
    memory=False,  # Try setting memory=False to simplify
    backstory="You are a query decomposer, expert at creating small subqueries from complex queries.",
    allow_delegation=False  # Disabling delegation for simplicity
)

Query_decomposition = Task(
    description="""Decompose the given long and complicated query: {query} into only 3 small and easy subqueries.""",
    expected_output="3 Small and easy subqueries for long and complicated query",
    agent=Query_decomposer,
    allow_delegation=False  # Ensuring no delegation is allowed
)

def run_tdm_agent(query):
    crew = Crew(
        agents=[Query_decomposer],
        tasks=[Query_decomposition],
        verbose=True,
        process=Process.sequential,
        debug=True,  # Debugging enabled to trace issues
        max_iterations=2
    )
    result = crew.kickoff(inputs={'query': query})
    result = str(result)
    return result

query = "What are the economic, social, and environmental impacts of climate change, and how can these be mitigated?"

result = run_tdm_agent(query)

result = str(result)

import re
questions = re.split(r'\d+\.\s', result)[1:]  # Split and ignore the first empty string
questions = [q.strip() for q in questions if q]  # Clean up whitespace and empty elements

# Print the extracted questions
for i, question in enumerate(questions, start=1):
    print(f"Question {i}: {question}")