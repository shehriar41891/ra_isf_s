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

# Define the responder agent for analytical yes/no questions
Yes_No_Responder = Agent(
    role='Analytical yes/no responder',
    goal=(
        """Analyze the given context: {text} in relation to the user query: {query}. 
        Respond with either "yes" or "no" based on the information in the context or 
        general knowledge. If the context does not provide enough information to 
        answer confidently, answer using general knowledge or facts. 
        Do not include any additional text or explanations."""
    ),
    verbose=True,
    memory=True,
    backstory=(
        """You are an expert at providing concise analytical answers based on a given 
        context. Your responses should be strictly "yes" or "no" and rely on the context 
        or general knowledge to answer the query."""
    ),
    allow_delegation=True,
)

# Define the evaluation task for yes/no answering
Yes_No_Task = Task(
    description=(
        """Analyze the provided context `{text}` and answer the user query `{query}` 
        with either "yes" or "no", using both the context and general knowledge 
        to provide the most accurate response. Avoid any additional context or explanation."""
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

def yes_no_answer_with_context(context, user_query):
    """
    This function accepts a context and a user query, processes the query with the 
    context, and returns a response of either "yes" or "no".
    The answer will be strictly limited to "yes" or "no".
    """
    result = crew.kickoff(inputs={'text': context, 'query': user_query})
    return str(result)

# Example input
# context = """
# Dogs are domesticated carnivores that are commonly known for having four legs. 
# They are often referred to as 'man's best friend' and are widely kept as pets.
# """
# query = "Do dogs have 4 legs?"

# # Call the function and get the result
# result = yes_no_answer_with_context(context, query)

# # Output the result
# print(result)  # Expected Output: yes
