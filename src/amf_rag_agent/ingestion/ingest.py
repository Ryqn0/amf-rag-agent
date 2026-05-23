from amf_rag_agent.ingestion.parser import parse_pdf
from amf_rag_agent.ingestion.chunker import chunk_pages
from amf_rag_agent.retrieval.embedder import get_embeddings
from amf_rag_agent.retrieval.store import add_chunks


def run_ingestion_pipeline(file_path: str):
    """
    Run the ingestion pipeline for a given PDF file.
    Args:
        file_path (str): The path to the PDF file to ingest.
    """

    print(f"Starting ingestion for {file_path}...")
    
    print("Parsing PDF...")

    pages = parse_pdf(file_path)

    print(f"Parsed {len(pages)} pages.")

    print("Chunking pages...")

    chunks = chunk_pages(pages)

    print(f"Chunked into {len(chunks)} chunks.")

    texts = [chunk["text"] for chunk in chunks]

    print("Generating embeddings for chunks...")

    embeddings = get_embeddings(texts)

    print(f"Generated embeddings with shape: {embeddings.shape}")

    print("Adding chunks and embeddings to vector store...")

    add_chunks(chunks, embeddings)

    print("Ingestion complete!")

if __name__ == "__main__": 
    run_ingestion_pipeline("data/raw/totalenergies_universal-registration-document-2025_2026_en.pdf")