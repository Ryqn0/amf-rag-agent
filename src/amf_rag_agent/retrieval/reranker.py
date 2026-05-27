from sentence_transformers import CrossEncoder
from logging import getLogger

logger = getLogger(__name__)

_model = None

def get_reranker_model():

    global _model

    if _model is None:

        logger.info("Loading cross-encoder model for reranking...")
        _model = CrossEncoder("cross-encoder/mmarco-mMiniLMv2-L12-H384-v1")
        logger.info("Cross-encoder model loaded.")

    return _model

def rerank_chunks(query: str, chunks: list[dict], top_k: int = 5) -> list[dict]:
    """
    Rerank chunks based on their relevance to the query using the cross-encoder model.
    """

    logger.info(f"Reranking {len(chunks)} chunks for query: {query}")
    pairs = [(query, chunk["text"]) for chunk in chunks]
    logger.info(f"Pairs created for reranking: {len(pairs)}")
    logger.info("Predicting relevance scores for chunks...")
    scores = get_reranker_model().predict(pairs)
    logger.info(f"Relevance scores predicted: {len(scores)}")
    logger.info("Combining chunks with their relevance scores...")
    results = [{"chunk": chunk, "score": score} for chunk, score in zip(chunks, scores)]
    logger.info("Sorting chunks by relevance score...")
    results = sorted(results, key=lambda x: x["score"], reverse=True)
    logger.info(f"Top {top_k} chunks selected after reranking.")
    logger.info("Reranking completed.")

    return [result["chunk"] for result in results[:top_k]]