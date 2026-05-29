from amf_rag_agent.ingestion.parser import parse_pdf
from amf_rag_agent.ingestion.parser_fr import parse_xhtml
from amf_rag_agent.ingestion.chunker import chunk_pages
from amf_rag_agent.retrieval.embedder import get_embeddings
from amf_rag_agent.retrieval.store import add_chunks
import logging
import json
from pathlib import Path
from datetime import datetime
import os


logger = logging.getLogger(__name__)
MANIFEST_PATH = "data/manifest.json"


def load_manifest() -> dict:
    """Load the manifest file which keeps track of ingested files and their metadata.
    Returns:
        dict: The manifest data containing ingested files and their metadata.
    """
    
    logger.info("Loading manifest...")

    if Path(MANIFEST_PATH).exists():
    
        logger.info("Manifest found, loading data...")

        with open(MANIFEST_PATH) as f:
    
            logger.info("Manifest loaded successfully.")
            return json.load(f)
    
    logger.info("No manifest found, starting with empty manifest.")
    return {"ingested_files": []}


def save_manifest(manifest: dict):
    """Save the manifest data to a JSON file.
    Args:
        manifest (dict): The manifest data to save, which includes ingested files and their metadata.
    """
    logger.info("Saving manifest...")

    with open(MANIFEST_PATH, "w") as f:

        logger.info("Manifest saved successfully.")
        json.dump(manifest, f, indent=2)


def run_ingestion_pipeline(file_paths: list[str]):
    """
    Ingests a list of files (PDF or HTML), parses them, chunks the text, generates embeddings, and adds to vector store.
    """

    logger.info("Starting ingestion pipeline...")
    logger.info(f"Loading manifest from {MANIFEST_PATH}...")
    manifest = load_manifest()
    logger.info("Manifest loaded.")
    logger.info("Checking for already ingested files...")
    ingested_files = {f["filename"] for f in manifest["ingested_files"]}
    logger.info(f"Already ingested files: {ingested_files}")

    all_chunks = []

    for file_path in file_paths:

        logger.info(f"Starting ingestion for {file_path}...")
    
        filename = os.path.basename(file_path)

        if filename in ingested_files:

            logger.info(f"{filename} has already been ingested, skipping.")
            continue
        
        if file_path.endswith(".pdf"):

            logger.info("Parsing PDF...")
            pages = parse_pdf(file_path)

        elif file_path.endswith(".xhtml") or file_path.endswith(".html"):

            logger.info("Parsing HTML...")
            pages = parse_xhtml(file_path)
            
        else:

            logger.error(f"Unsupported file type: {file_path}")
            raise ValueError(f"Unsupported file type: {file_path}")
        
        logger.info(f"Parsed {len(pages)} pages.")
        logger.info("Chunking pages...")
        # chunk_pages: character-based with overlap — robust for table-heavy financial documents
        # semantic_chunk_pages: meaning-boundary splitting — better for prose-heavy documents
        chunks = chunk_pages(pages)
        all_chunks.extend(chunks)
        logger.info(f"Chunked into {len(chunks)} chunks.")
        logger.info("Updating manifest...")
        manifest["ingested_files"].append({
            "filename": filename,
            "ingested_at": datetime.now().isoformat(),
        })
        logger.info("Saving manifest...")

    if all_chunks:

        logger.info("Generating embeddings for chunks...")
        embeddings = get_embeddings([chunk["text"] for chunk in all_chunks])
        logger.info(f"Generated embeddings with shape: {embeddings.shape}")
        logger.info("Adding chunks and embeddings to vector store...")
        add_chunks(all_chunks, embeddings)
        logger.info("Chunks added to vector store.")
        logger.info("Saving manifest...")
        save_manifest(manifest)
        logger.info("Manifest saved.")
        logger.info("Ingestion complete!")

    else:

        logger.info("No new files to ingest.")
        

if __name__ == "__main__": 

    from amf_rag_agent.config import setup_logging

    setup_logging()

    raw_dir = Path("data/raw")
    file_paths = sorted([str(f) for f in raw_dir.iterdir() if f.suffix in [".pdf", ".xhtml", ".html"]])
    run_ingestion_pipeline(file_paths)