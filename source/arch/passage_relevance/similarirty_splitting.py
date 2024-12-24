from sklearn.metrics.pairwise import cosine_similarity
import os
import sys
import numpy as np
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from langchain.embeddings import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

openi_api_key = os.getenv('OPENAI_API')

print(openi_api_key)
os.environ['OPENI_API'] = openi_api_key

embedding_model = OpenAIEmbeddings()

def split_text_by_words(text, chunk_size, chunk_overlap=0):
    """
    Splits the input text into chunks of specified length based on words with optional overlap.

    Parameters:
    - text (str): The input text to split.
    - chunk_size (int): The approximate maximum number of words in each chunk.
    - chunk_overlap (int): The number of overlapping words between chunks.

    Returns:
    - List[str]: A list of text chunks.
    """
    if chunk_size <= 0:
        raise ValueError("Chunk size must be greater than 0.")
    if chunk_overlap < 0:
        raise ValueError("Chunk overlap cannot be negative.")

    words = text.split()  # Split the text into words
    chunks = []
    start = 0

    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])  # Join the selected words into a chunk
        chunks.append(chunk)

        # Move the starting index for the next chunk, considering word overlap
        start = end - chunk_overlap

    return chunks


def compute_cosine_similarity(subparts, query, embedding_function):
    """
    Computes cosine similarity between a query and each subpart of text.
    
    Args:
        subparts (list of str): The list of text subparts.
        query (str): The query string.
        embedding_function (callable): A function to generate embeddings for the text. 
                                        Must accept a string and return a vector (list/ndarray).
    
    Returns:
        list of tuple: A list of tuples where each tuple contains the subpart and its cosine similarity score.
    """
    # Generate embeddings for the subparts and the query
    subpart_embeddings = [embedding_function(subpart) for subpart in subparts]
    query_embedding = embedding_function(query)

    # Ensure all embeddings have the same size
    max_length = max(len(vec) for vec in subpart_embeddings + [query_embedding])
    subpart_embeddings = [np.pad(vec, (0, max_length - len(vec))) for vec in subpart_embeddings]
    query_embedding = np.pad(query_embedding, (0, max_length - len(query_embedding)))

    # Compute cosine similarities
    similarities = cosine_similarity([query_embedding], subpart_embeddings)[0]
    
    # Pair subparts with their similarity scores
    results = [(subparts[i], similarities[i]) for i in range(len(subparts))]
    
    # Sort by similarity score in descending order
    results.sort(key=lambda x: x[1], reverse=True)
    
    return results


# Integrated Usage Example
if __name__ == "__main__":
    # Dummy embedding function (replace with an actual embedding generator)
    def dummy_embedding(text):
        return [ord(c) for c in text][:10]  # Mock embedding based on ASCII values (for demonstration)

    # Sample long text
    text = """Once past due, service can be disrupted. It's important to contact Financial 
    Services so a payment arrangement can be considered.
    """
    
    # Query
    query = "What to do when service is disrupted?"

    # Step 1: Split the text into chunks
    chunks = split_text_by_words(text, chunk_size=14, chunk_overlap=4)

    # Step 2: Compute cosine similarity with the query
    results = compute_cosine_similarity(chunks, query, dummy_embedding)
    
    # Output the results
    for subpart, similarity in results:
        print(f"Subpart: '{subpart}'\nSimilarity: {similarity:.4f}\n")
