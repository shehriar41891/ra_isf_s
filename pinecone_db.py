import os
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from langchain_community.embeddings import OpenAIEmbeddings
import random
# Load environment variables
load_dotenv()

pinecone_api_key = os.getenv('PINECONE_API_KEY')
pinecone_environment = os.getenv('PINECONE_ENVIRONMENT') 
openai_api_key = os.getenv('OPENAI_API')


pc = Pinecone(api_key=pinecone_api_key)
embedding_model = OpenAIEmbeddings(api_key = openai_api_key)

index_name = "ra-isf" 

# Check if the index exists, create if not
if index_name not in pc.list_indexes().names():
    print(f"Index '{index_name}' not found. Creating a new index...")
    pc.create_index(
        name=index_name,
        dimension=1536,  
        metric="cosine",  
        spec=ServerlessSpec(cloud="aws", region=pinecone_environment) 
    )

# Connect to the index
index = pc.Index(name=index_name)

print(f"Connected to Pinecone index: {index_name}")

#method to upsert the data 
def upsert_data(query, answer, threshold=0.9):
    # Check for existing similar queries
    query_embeddings = embedding_model.embed_query(query)
    response = index.query(
        vector=query_embeddings,
        top_k=1,  # Check for the most similar existing query
        include_metadata=True
    )
    
    if response["matches"] and response["matches"][0]["score"] >= threshold:
        print("Similar query already exists. Skipping upsert.")
        return  
    
    # Proceed with upsertion if no similar query exists
    vector = [
        {
            'id': str(random.randint(0, 10000)),  
            'values': query_embeddings,
            'metadata': {'answer': answer}
        }
    ]
    index.upsert(vector)
    print("Query successfully upserted.")

    

def compare_query(query, bench_mark=0.75):
    query_embeddings = embedding_model.embed_query(query)
    
    # Query Pinecone index for top matches
    response = index.query(
        vector=query_embeddings,
        top_k=3,
        include_values=False,
        include_metadata=True
    )
    
    # Ensure matches are present
    if not response.matches:
        print("No similar data found in the query")
        return []

    similar_data = []
    for match in response.matches:
        if match["score"] >= bench_mark:
            similar_data.append(match["metadata"]["answer"])
    
    if similar_data:
        return similar_data
    else:
        print("No similar data found with a score above the benchmark")
        return []