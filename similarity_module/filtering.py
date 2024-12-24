from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from dotenv import load_dotenv
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import openai
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from langchain.embeddings import OpenAIEmbeddings

load_dotenv()

openai_api_key = os.getenv('OPENAI_API')

embedding_model = OpenAIEmbeddings(api_key = openai_api_key)

def recursive_text_splitter(text, chunk_size=30, overlap=9):
    """
    Recursively splits text into chunks of a specified size with overlap.

    Args:
        text (str): The input text to split.
        chunk_size (int): The size of each chunk (in words).
        overlap (int): The number of overlapping words between consecutive chunks.

    Returns:
        list: A list of text chunks.
    """
    words = text.split()
    if len(words) <= chunk_size:
        return [text]  # Base case: return the text if it fits in one chunk

    # Create the current chunk and recurse on the remaining text
    current_chunk = " ".join(words[:chunk_size])
    next_text = " ".join(words[chunk_size - overlap:])  # Prepare the remaining text with overlap
    return [current_chunk] + recursive_text_splitter(next_text, chunk_size, overlap)


def split_array_into_chunks(array, chunk_size=30, overlap=9):
    """
    Splits each element of an array into chunks using the recursive text splitter.

    Args:
        array (list): The array of strings to process.
        chunk_size (int): The size of each chunk (in words).
        overlap (int): The number of overlapping words between consecutive chunks.

    Returns:
        list: A list of all the chunks from all elements of the array.
    """
    chunks = []
    for text in array:
        chunks.extend(recursive_text_splitter(text, chunk_size, overlap))
    return chunks


def get_openai_embedding(text, model="text-embedding-ada-002"):
    """
    Fetches the embedding for a given text using OpenAI's embedding model.

    Args:
        text (str): The input text.
        model (str): The OpenAI embedding model to use.

    Returns:
        np.array: The embedding vector for the input text.
    """
    response = openai.Embedding.create(input=text, model=model)
    return np.array(response['data'][0]['embedding'])

def filter_chunks_by_similarity_openai(chunks, query, benchmark=0.7, model="text-embedding-ada-002"):
    """
    Filters text chunks by calculating cosine similarity against a query using OpenAI embeddings and returns the top 5 based on similarity.

    Args:
        chunks (list): List of text chunks.
        query (str): The query string.
        benchmark (float): The similarity threshold to filter chunks.
        model (str): The OpenAI embedding model to use.

    Returns:
        list: Top 5 chunks that meet the similarity benchmark.
    """
    # Compute the embedding for the query
    query_embedding = embedding_model.embed_query(query)

    # Compute embeddings for all chunks
    chunk_embeddings = [embedding_model.embed_query(chunk) for chunk in chunks]

    # Calculate cosine similarities
    cosine_similarities = cosine_similarity([query_embedding], chunk_embeddings).flatten()

    # Pair chunks with their similarity scores and filter by benchmark
    chunk_similarity_pairs = [(chunks[i], cosine_similarities[i]) for i in range(len(chunks)) if cosine_similarities[i] >= benchmark]

    # Sort by similarity score in descending order and take top 5
    top_chunks = sorted(chunk_similarity_pairs, key=lambda x: x[1], reverse=True)[:5]

    # Extract the chunks (ignoring similarity scores)
    top_chunks = [chunk for chunk, _ in top_chunks]

    return top_chunks


if __name__ == "__main__":
    sample_text = [
        "OpenAI provides state-of-the-art language models. "
        "You can use embeddings to compare text similarities."
        "To highlight the reasoning improvement over GPT-4o, we tested our models on a diverse set of human exams and ML benchmarks. We show that o1 significantly outperforms GPT-4o on the vast majority of these reasoning-heavy tasks. Unless otherwise specified, we evaluated o1 on the maximal test-time compute setting.",
        "In many reasoning-heavy benchmarks, o1 rivals the performance of human experts. Recent frontier models1 do so well on MATH2 and GSM8K that these benchmarks are no longer effective at differentiating models. We evaluated math performance on AIME, an exam designed to challenge the brightest high school math students in America. On the 2024 AIME exams, GPT-4o only solved on average 12% (1.8/15) of problems. o1 averaged 74% (11.1/15) with a single sample per problem, 83% (12.5/15) with consensus among 64 samples, and 93% (13.9/15) when re-ranking 1000 samples with a learned scoring function. A score of 13.9 places it among the top 500 students nationally and above the cutoff for the USA Mathematical Olympiad."
    ]
    query_text = "Who is Babar Azam?"

    # Split the sample text into chunks
    chunks = split_array_into_chunks(sample_text)
    
    print(chunks)

    # Filter the chunks by similarity to the query
    filtered = filter_chunks_by_similarity_openai(chunks, query_text)

    print("Filtered chunks:", filtered)
