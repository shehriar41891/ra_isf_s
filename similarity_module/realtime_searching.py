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
# import wikipediaapi


# Enable verbose mode for LiteLLM
litellm.set_verbose = True

# Load environment variables
load_dotenv()
openai_api = os.getenv('OPENAI_API')
pinecone_api_key = os.getenv('PINECONE_API_KEY')

os.environ['OPENAI_API_KEY'] = openai_api
os.environ['MODEL_NAME'] = 'gpt-3.5-turbo'

# Configure LiteLLM with API key
litellm.api_key = openai_api
# embedding_model = get_embedding_model()

SERPER_API_KEY = os.getenv('SERPER_API_KEY')

os.environ["SERPER_API_KEY"] = SERPER_API_KEY

def Googlesearch(query):
    search = GoogleSerperAPIWrapper()
    result = search.run(query+'answer in context of before 2019')
    
    return result


# print(Googlesearch('Who is Babar Azam'))