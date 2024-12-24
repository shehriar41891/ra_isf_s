from langchain import LLMChain, OpenAI
import os 
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, Process
import litellm

load_dotenv()

api_key = os.getenv('OPENAI_API')

os.environ['OPENAI_API_KEY'] = api_key
os.environ['MODEL_NAME'] = 'gpt-3.5-turbo'
os.environ['LITELLM_LOG'] = 'DEBUG'
# litellm.set_verbose=True

# Initialize the OpenAI LLM
llm = OpenAI(api_key = api_key, temperature=0)

print(llm)

# Define the responder agent with pre-2019 knowledge restriction
Short_Responder = Agent(
    role='Short answer giver',
    goal=(
        """Answer the user's query: {query}. Extract **only** the exact
        word or phrase that directly answers the query, with **no supporting phrases or extra words**.
        **Important:** The knowledge you base your answers on must not exceed the year 2018.
        If you do not have enough knowledge to answer confidently within the pre-2019 context, respond with 'Unknown'."""
    ),
    verbose=True,
    memory=False,
    backstory=(
        """You are an expert answer provider, skilled in extracting precise answers
        from knowledge and queries available only until 2018. Your answers are short and exact."""
    ),
    allow_delegation=True,
)

# Define the task with the temporal restriction
Short_answering = Task(
    description=(
        """Analyze the user query `{query}` and provide the exact word or phrase 
        that directly answers the query. Avoid any additional context, 
        explanations, or supporting phrases. Only use knowledge from before 2019."""
    ),
    expected_output="A single word or phrase that directly answers the query, with no additional text.",
    agent=Short_Responder,
    allow_delegation=True,
)

# Create the crew
def direct_relevant_answer(user_query): 
    crew = Crew(
    agents=[Short_Responder],
    tasks=[Short_answering],
    verbose=True,
    process=Process.sequential,
    debug=True,
    max_iterations=5,
    )
    result = crew.kickoff(inputs={'query': user_query})
    
    print('final result from agent file',result)
    return result

# Example input
query = "where did they film hot tub time machine?"

# Call the function
relevant_result = direct_relevant_answer(query)

# Output the result
print(relevant_result)