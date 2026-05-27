import chromadb
import numpy as np
from logging import getLogger

logger = getLogger(__name__)

# client = chromadb.PersistentClient(path="data/vectorstore")

_client = None


def get_client():
    """Get the ChromaDB client, initializing it if it doesn't exist.
    Returns:
        chromadb.PersistentClient: The ChromaDB client.
"""
    global _client
    logger.info("Retrieving ChromaDB client...")
    if _client is None:
        _client = chromadb.PersistentClient(path="data/vectorstore")
    return _client


_collection = None


def get_collection():
    """Get the ChromaDB collection, initializing it if it doesn't exist.
    Returns:
        chromadb.Collection: The ChromaDB collection.
    """
    global _collection
    logger.info("Retrieving ChromaDB collection...")
    if _collection is None:
        _collection = get_client().get_or_create_collection(name="amf_docs")
    return _collection


def add_chunks(chunks: list[dict], embeddings: np.ndarray, batch_size: int = 5000):
    """
    Add chunks to the vector store.
    Args:
        chunks (list[dict]): List of chunks to add. Each chunk should be a dictionary with keys "text", "page_number", "source", and "chunk_index".
        embeddings (np.ndarray): Embeddings for the chunks. Should have the same length as the chunks list.
    """

    logger.info(f"Adding {len(chunks)} chunks to the vector store in batches of {batch_size}...")
    for i in range(0, len(chunks), batch_size):

        logger.info(f"Processing batch {i // batch_size + 1}...")
        batch_chunks = chunks[i:i+batch_size]
        batch_embeddings = embeddings[i:i+batch_size]
        logger.info(f"Adding batch of {len(batch_chunks)} chunks to the vector store...")
        get_collection().add(ids=[f"{chunk['source']}_chunk_{chunk['chunk_index']}" for chunk in batch_chunks],
                    documents=[chunk['text'] for chunk in batch_chunks],
                    embeddings=batch_embeddings.tolist(),
                    metadatas=[{"page_number": chunk['page_number'], 
                                "source": chunk["source"]}
                                for chunk in batch_chunks])
        logger.info(f"Batch {i // batch_size + 1} added to the vector store.")


def load_all_chunks() -> list[dict]:
    """
    Load all chunks from the vector store.
    Returns:
        list[dict]: A list of dictionaries containing the chunks and their metadata.
    """

    logger.info("Loading all chunks from the vector store...")
    results = get_collection().get(include=["documents", "metadatas"])
    logger.info(f"Loaded {len(results['documents'])} chunks from the vector store.")
    logger.info("Returning chunks with their metadata...")

    return [
        {
            "text": document,
            "page_number": metadata["page_number"],
            "source": metadata["source"]
        }
        for document, metadata in zip(results["documents"], results["metadatas"])
    ]


def search(query_embedding: np.ndarray, k: int = 5) -> list[dict]:
    """
    Search for relevant chunks in the vector store based on a query embedding.
    Args:
        query_embedding (np.ndarray): The embedding of the query. Should have the same dimension as the chunk embeddings.
        k (int): The number of relevant chunks to return.
    Returns:
        list[dict]: A list of dictionaries containing the relevant chunks and their metadata.
    """

    results = get_collection().query(query_embeddings=[query_embedding.tolist()], n_results=k)

    output = []

    for document, metadata, distance in zip(results["documents"][0], results["metadatas"][0], results["distances"][0]):

        output.append({
            "text": document,
            "page_number": metadata['page_number'],
            "source": metadata['source'],
            "distance": distance
        })


    return output

if __name__ == "__main__":
    from amf_rag_agent.retrieval.embedder import get_embeddings

    query = "What are TotalEnergies' main risk factors?"
    query_vec = get_embeddings([query])[0]
    results = search(query_vec, k=3)

    for i, r in enumerate(results):
        print(f"\n--- Result {i+1} (distance: {r['distance']:.3f}, page: {r['page_number']}) ---")
        print(r["text"][:300])