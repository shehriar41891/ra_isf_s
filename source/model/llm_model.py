from langchain_openai import ChatOpenAI
import os
from dotenv import load_dotenv

def get_llm():
    # Load the environment variables
    load_dotenv()
    # Retrieve OpenAI API key from environment variables
    openai_api_key = os.getenv("OPENAI_API")
        
    if not openai_api_key:
        raise EnvironmentError("OpenAI API key is not set in your environment variables.")
    
    # Return an instance of ChatOpenAI instead of OpenAI for chat models
    return ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.7, api_key=openai_api_key)

llm = get_llm()