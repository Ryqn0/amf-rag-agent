from amf_rag_agent.ingestion.parser import parse_pdf
from amf_rag_agent.ingestion.parser_fr import parse_xhtml
from amf_rag_agent.ingestion.chunker import chunk_pages
from amf_rag_agent.retrieval.embedder import get_embeddings
from amf_rag_agent.retrieval.store import add_chunks

import logging

logger = logging.getLogger(__name__)


def run_ingestion_pipeline(file_paths: list[str]):
    """
    Ingests a list of files (PDF or HTML), parses them, chunks the text, generates embeddings, and adds to vector store.
    """

    all_chunks = []

    for file_path in file_paths:

        logger.info(f"Starting ingestion for {file_path}...")
        
        if file_path.endswith(".pdf"):

            logger.info("Parsing PDF...")
            pages = parse_pdf(file_path)

        elif file_path.endswith(".xhtml") or file_path.endswith(".html"):

            logger.info("Parsing HTML...")
            pages = parse_xhtml(file_path)
            
        else:
            
            raise ValueError(f"Unssuported file type: {file_path}")
        
        logger.info(f"Parsed {len(pages)} pages.")
        logger.info("Chunking pages...")
        chunks = chunk_pages(pages)
        all_chunks.extend(chunks)
        logger.info(f"Chunked into {len(chunks)} chunks.")
        
    logger.info("Generating embeddings for chunks...")
    embeddings = get_embeddings([chunk["text"] for chunk in all_chunks])
    logger.info(f"Generated embeddings with shape: {embeddings.shape}")
    logger.info("Adding chunks and embeddings to vector store...")
    add_chunks(all_chunks, embeddings)
    logger.info("Ingestion complete!")

if __name__ == "__main__": 

    from amf_rag_agent.config import setup_logging

    setup_logging()

    run_ingestion_pipeline([
    r"data/raw/totalenergies_universal-registration-document-2025_2026_en.pdf",
    r"data/raw/529900S21EQ1BO4ESM68-2025-12-31-1-fr.xhtml"
])