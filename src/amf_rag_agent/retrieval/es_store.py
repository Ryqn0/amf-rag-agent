import os
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk
import logging

logger = logging.getLogger(__name__)

_client = None

def get_es_client():
    global _client

    logger.info("Connecting to Elasticsearch...")
    if _client is None:

        logger.info("Creating new Elasticsearch client...")
        url = os.getenv("ELASTICSEARCH_URL", "http://localhost:9200")
        _client = Elasticsearch(url)
    return _client


INDEX_NAME = "amf_chunks"

MAPPINGS = {
    "properties": {
        "text": {"type": "text"},
        "source": {"type": "keyword"},
        "page_number": {"type": "integer"},
    }
}


def create_index():
    """Create Elasticsearch index if it doesn't exist."""

    client = get_es_client()

    if not client.indices.exists(index=INDEX_NAME):

        logger.info("Creating Elasticsearch index...")
        client.indices.create(index=INDEX_NAME, mappings=MAPPINGS)


def index_chunks(chunks: list[dict]):
    """
    Index chunks into Elasticsearch.
    Args:
        chunks (list[dict]): A list of chunks, where each chunk is a dictionary containing 'text', 'source', and 'page_number' keys.
    """

    def generate_docs():
        """
        Generate documents for bulk indexing.
         Returns:
             dict: A dictionary representing a document to be indexed in Elasticsearch, containing '_index', 'text', 'source', and 'page_number' keys.
        """

        logger.info(f"Indexing {len(chunks)} chunks into Elasticsearch...")
        for chunk in chunks:

            logger.debug(f"Indexing chunk from source {chunk['source']} page {chunk['page_number']} with text length {len(chunk['text'])} characters.")
            yield {
                "_index": INDEX_NAME,
                "text": chunk['text'],
                "source": chunk['source'],
                "page_number": chunk['page_number'],
            }
        logger.info("All chunks have been processed for indexing.")

    logger.info("Starting bulk indexing of chunks into Elasticsearch...")
    success, _ = bulk(get_es_client(), generate_docs())
    logger.info(f"Bulk indexing completed. Successfully indexed {success} chunks.")


def search_es(query: str, k = 20) -> list[dict]:
    """
    Search for chunks in Elasticsearch.
    Args:
        query (str): The search query.
        k (int): The maximum number of results to return.
    Returns:
        list[dict]: A list of matching chunks.
    """

    logger.info(f"Searching for query: {query}")
    response = get_es_client().search(
        index=INDEX_NAME,
        query={"match": {"text": query}},
        size=k)
    
    logger.info(f"Search completed. Found {response['hits']['total']['value']} total hits. Returning top {k} results.")
    chunks = []
    for result in response["hits"]["hits"]:

        logger.debug(f"Processing search hit with score {result['_score']} from source {result['_source']['source']} page {result['_source']['page_number']} with text length {len(result['_source']['text'])} characters.")
        chunks.append(result["_source"])

    logger.info(f"Returning {len(chunks)} chunks from search results.")
    return chunks