import sentence_transformers
import numpy as np
import time

MODEL = sentence_transformers.SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
# MODEL = sentence_transformers.SentenceTransformer("sentence-transformers/paraphrase-multilingual-mpnet-base-v2")
# Using this model because it returns an embedding vector of 768 which is close to the chunk_size and then also because of the balance performance and query per second

def get_embeddings(texts: list[str]) -> np.ndarray:
    """
    Get embeddings for a list of texts.
    Args:
        texts (list[str]): List of texts to get embeddings for.
    Returns:
        np.ndarray: Embeddings for the input texts.
    """
    
    embeddings = MODEL.encode(texts)

    return embeddings
    

if __name__ == "__main__":
    french = "Les facteurs de risque liés à l'activité pétrolière"
    english = "Risk factors related to oil and gas operations"
    unrelated = "The weather in Tokyo is nice today"

    start_time = time.time()
    vecs = get_embeddings([french, english, unrelated])
    print(f"Vector shape: {vecs.shape}")
    end_time = time.time()
    print(f"Time taken: {end_time - start_time}")

    # Cosine similarity - how close are these vectors to each other?
    from numpy.linalg import norm
    def cosine_similarity(a, b):
        return np.dot(a, b) / (norm(a) * norm(b))
    
    print("Cosine similarity between French and English:", cosine_similarity(vecs[0], vecs[1]))
    print("Cosine similarity between French and Unrelated:", cosine_similarity(vecs[0], vecs[2]))