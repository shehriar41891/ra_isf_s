import faiss
import numpy as np
import pickle
import os
import glob

# Save the index to a file
def save_index_to_disk(index, save_path):
    faiss.write_index(index, save_path)
    print(f"Index saved to: {save_path}")

# Load the index from a file
def load_index_from_disk(index_path):
    if os.path.exists(index_path):
        index = faiss.read_index(index_path)
        print(f"Index loaded from: {index_path}")
        return index
    else:
        raise FileNotFoundError(f"Index file not found at {index_path}")

# Add embeddings to the index and save
def add_embeddings_and_save(index_path, embedding_files, indexing_batch_size):
    if os.path.exists(index_path):
        index = load_index_from_disk(index_path)  # Load existing index if it exists
    else:
        # Create a new index if it doesn't exist
        embedding_dim = 768  # Match the embedding dimension
        nlist = 100
        m = 8
        nbits = 8
        quantizer = faiss.IndexFlatL2(embedding_dim)  # L2 distance
        index = faiss.IndexIVFPQ(quantizer, embedding_dim, nlist, m, nbits)
        index.nprobe = 10

    for file_path in embedding_files:
        print(f"Loading file: {file_path}")
        with open(file_path, "rb") as fin:
            ids, embeddings = pickle.load(fin)

        embeddings = np.array(embeddings).astype('float32')
        ids = np.array(ids).astype('int64')

        # Train the index if not already trained
        if not index.is_trained:
            print("Training the index...")
            index.train(embeddings)

        # Add embeddings in batches
        for i in range(0, len(embeddings), indexing_batch_size):
            batch_embeddings = embeddings[i:i + indexing_batch_size]
            batch_ids = ids[i:i + indexing_batch_size]
            print(f"Adding batch of size {len(batch_embeddings)}...")
            index.add_with_ids(batch_embeddings, batch_ids)

        # Save the index to disk after processing each file
        save_index_to_disk(index, index_path)

    print("Indexing completed.")

# Query the index after loading it
def query_index(index_path, query_embeddings, top_k=10):
    index = load_index_from_disk(index_path)
    query_embeddings = np.array(query_embeddings).astype('float32')
    distances, indices = index.search(query_embeddings, top_k)
    return distances, indices

# Parameters
index_path = "./fiass_store/index_on_disk.faiss"
input_paths = sorted(glob.glob('./data/wikipedia_embeddings/*'))
indexing_batch_size = 1000000

# Build or load the index and save
add_embeddings_and_save(index_path, input_paths, indexing_batch_size)

# Example query
query_embedding = np.random.rand(1, 768).astype('float32')  # Replace with actual query
distances, indices = query_index(index_path, query_embedding)
print("Nearest neighbors:", indices)
print("Distances:", distances)
