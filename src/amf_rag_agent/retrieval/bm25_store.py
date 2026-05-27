from rank_bm25 import BM25Okapi

from logging import getLogger

logger = getLogger(__name__)
_bm25 = None
_chunks = None  # Keeping chunks in memory to return them by index


def build_bm25_index(chunks: list[dict]):
    """
    Build a BM25 index from the provided chunks.
        Args:
            chunks (list[dict]): A list of chunks, where each chunk is a dictionary containing
            at least a 'text' key with the chunk's content.
    """

    logger.info(f"Building BM25 index for {len(chunks)} chunks...")
    tokens = [chunk['text'].lower().split() for chunk in chunks]
    logger.info("BM25 index built successfully.")
    logger.info("Storing chunks in memory for retrieval...")
    global _bm25, _chunks
    _bm25 = BM25Okapi(tokens)
    _chunks = chunks
    logger.info("Chunks stored in memory for retrieval.")


def search_bm25(query: str, k: int = 20) -> list[dict]:
    """
    Search the BM25 index for the most relevant chunks to the query.
        Args:
            query (str): The search query.
            k (int): The number of top results to return.
        Returns:
            list[dict]: A list of the most relevant chunks, each represented as a dictionary.
    """

    if _bm25 is None or _chunks is None:
        logger.error("BM25 index has not been built. Please call build_bm25_index() first.")
        return []

    logger.info(f"Searching BM25 index for query: {query}")
    logger.info("Tokenizing query for BM25 search...")
    tokenized_query = query.lower().split()
    logger.info("Calculating BM25 scores for chunks...")
    scores = _bm25.get_scores(tokenized_query)
    logger.info(f"BM25 scores calculated for {len(scores)} chunks.")
    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    logger.info(f"Top {k} indices identified.")
    logger.info("Returning top relevant chunks...")
    return [_chunks[i] for i in top_indices[:k]]