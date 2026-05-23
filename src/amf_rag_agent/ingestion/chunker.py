

def chunk_pages(pages: list[dict], chunk_size: int = 800, overlap: int = 200) -> list[dict]:
    """Split parsed pages into smaller chunks with overlap."""

    chunks = []
    chunk_index = 0

    for page in pages:
        text = page["text"]

        paragraphs = text.split("\n")

        current_chunk = ""

        for para in paragraphs:

            para = para.strip()
            
            if len(current_chunk) + len(para) > chunk_size and current_chunk :
                # Save the current chunk
                chunks.append({
                    "text": current_chunk.strip(),
                    "page_number": page["page_number"],
                    "source": page["source"],
                    "chunk_index": chunk_index,
                })
                chunk_index += 1

                current_chunk = current_chunk[-overlap:] + "\n" + para
            else:

                if current_chunk:
                    current_chunk += "\n" + para
                else:
                    current_chunk = para

        if current_chunk:
            chunks.append({
                "text": current_chunk.strip(),
                "page_number": page["page_number"],
                "source": page["source"],
                "chunk_index": chunk_index,
            })
            chunk_index += 1

    return chunks

if __name__ == "__main__":
    from amf_rag_agent.ingestion.parser import parse_pdf
    pages = parse_pdf("data/raw/totalenergies_universal-registration-document-2025_2026_en.pdf")
    chunks = chunk_pages(pages)
    print(f"Total chunks: {len(chunks)}")
    print(f"\n--- Sample chunk ---")
    print(f"Page: {chunks[405]['page_number']}")
    print(f"Text: {chunks[405]['text'][:400]}")