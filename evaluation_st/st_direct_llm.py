import os
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, Process
import litellm

# Enable verbose mode for LiteLLM
litellm.set_verbose = True

# Load environment variables
load_dotenv()
openai_api = os.getenv('OPENAI_API')

os.environ['OPENAI_API_KEY'] = openai_api
os.environ['MODEL_NAME'] = 'gpt-3.5-turbo'

# Define the responder agent with analytical yes/no restriction
Yes_No_Responder = Agent(
    role='Analytical yes/no responder',
    goal=(
        """Answer the user's analytical query: {query}. Respond with either "yes" or "no" 
        based on your knowledge available up to 2018. Avoid providing any additional context, 
        explanations, or supporting phrases. If you do not have enough knowledge to answer 
        confidently within the pre-2019 context, respond with "Unknown"."""
    ),
    verbose=True,
    memory=False,
    backstory=(
        """You are an expert in analytical reasoning, skilled in answering 
        queries with a direct "yes" or "no" response based solely on knowledge 
        available up to the year 2018."""
    ),
    allow_delegation=True,
)

# Define the task for analytical yes/no answering
Yes_No_Task = Task(
    description=(
        """Analyze the user query `{query}` and respond with either "yes" or "no". 
        Do not include any additional context, explanations, or phrases. Use only 
        knowledge available before 2019."""
    ),
    expected_output='A single word: "yes" or "no".',
    agent=Yes_No_Responder,
    allow_delegation=True,
)

# Create the crew
crew = Crew(
    agents=[Yes_No_Responder],
    tasks=[Yes_No_Task],
    verbose=True,
    process=Process.sequential,
    debug=True,
    max_iterations=2,
)

def yes_no_answer(user_query):
    result = crew.kickoff(inputs={'query': user_query})
    return result

# Example input
query = "Has the Paksitan cricket team won any World Cup?"

# Call the function
# relevant_result = yes_no_answer(query)

# # Output the result
# print(relevant_result)
