from amf_rag_agent.config import setup_logging
from amf_rag_agent.retrieval.store import load_all_chunks
from amf_rag_agent.retrieval.es_store import create_index, index_chunks

if __name__ == "__main__":

    setup_logging()
    create_index()
    chunks = load_all_chunks()
    index_chunks(chunks)