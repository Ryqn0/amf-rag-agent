import numpy as np
from amf_rag_agent.retrieval.embedder import get_embeddings
from logging import getLogger

logger = getLogger(__name__)


def chunk_pages(pages: list[dict], chunk_size: int = 800, overlap: int = 200) -> list[dict]:
    """Split parsed pages into smaller chunks with overlap.
    Args:
        pages (list[dict]): A list of parsed pages, where each page is a dict with keys "text", "page_number", and "source".
        chunk_size (int): The maximum number of characters in each chunk.
        overlap (int): The number of overlapping characters between consecutive chunks.
    Returns:
        list[dict]: A list of chunks, where each chunk is a dict with keys "text", "page_number", "source", and "chunk_index".
    """

    chunks = []
    chunk_index = 0

    logger.info(f"Starting chunking of pages with chunk_size={chunk_size} and overlap={overlap}. Total pages: {len(pages)}")


    for page in pages:

        logger.info(f"Chunking page {page['page_number']} from source {page['source']} with text length {len(page['text'])} characters.")
        text = page["text"]
        logger.info(f"Splitting page text into paragraphs.")
        paragraphs = text.split("\n")
        logger.info(f"Total paragraphs found: {len(paragraphs)}")
        current_chunk = ""

        logger.info(f"Processing paragraphs for page {page['page_number']}.")
        for para in paragraphs:

            logger.info(f"Processing paragraph with length {len(para)} characters.")
            para = para.strip()
            
            logger.info(f"Current chunk length: {len(current_chunk)} characters. Paragraph length: {len(para)} characters.")
            if len(current_chunk) + len(para) > chunk_size and current_chunk :
                # Save the current chunk
                logger.info(f"Chunk size exceeded. Saving chunk with length {len(current_chunk)} characters for page {page['page_number']}.")
                chunks.append({
                    "text": current_chunk.strip(),
                    "page_number": page["page_number"],
                    "source": page["source"],
                    "chunk_index": chunk_index,
                })
                logger.info(f"Chunk saved. Moving to next chunk with overlap of {overlap} characters.")
                chunk_index += 1
                logger.info(f"Applying overlap. Keeping last {overlap} characters for next chunk.")
                current_chunk = current_chunk[-overlap:] + "\n" + para
            else:

                logger.info(f"Adding paragraph to current chunk. New chunk length will be {len(current_chunk) + len(para)} characters.")
                if current_chunk:
                    logger.info(f"Current chunk is not empty. Adding newline before paragraph.")
                    current_chunk += "\n" + para

                else:
                    logger.info(f"Current chunk is empty. Starting new chunk with paragraph.")
                    current_chunk = para

        logger.info(f"Finished processing paragraphs for page {page['page_number']}. Checking if there is a remaining chunk to save.")
        if current_chunk:

            logger.info(f"Saving remaining chunk with length {len(current_chunk)} characters for page {page['page_number']}.")
            chunks.append({
                "text": current_chunk.strip(),
                "page_number": page["page_number"],
                "source": page["source"],
                "chunk_index": chunk_index,
            })
            chunk_index += 1

    logger.info(f"Finished chunking all pages. Total chunks created: {len(chunks)}")
    
    return chunks



def semantic_chunk_pages(pages: list[dict], threshold: float = 0.6, min_chunk_size: int = 100, max_chunk_size: int = 1500) -> list[dict]:
    """
    Split parsed pages into smaller chunks based on semantic similarity of sentences.
    Args:        pages (list[dict]): A list of parsed pages, where each page is a dict with keys "text", "page_number", and "source".
        threshold (float): The cosine similarity threshold for determining whether sentences belong in the same chunk. Higher means more similar.
    Returns:
        list[dict]: A list of chunks, where each chunk is a dict with keys "text", "page_number", "source", and "chunk_index".
    """

    def cosine_similarity(vec1, vec2):
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
    
    semantic_chunks = []
    chunk_index = 0

    logger.info(f"Starting semantic chunking for {len(pages)} pages.")
    for page in pages:

        logger.info(f"Processing page {page['page_number']} from source {page['source']} with text length {len(page['text'])} characters.")
        sentences = [s.strip() for s in page["text"].split(". ") if s.strip()]

        if not sentences:
            continue

        
        logger.info(f"Extracted {len(sentences)} sentences from page {page['page_number']}. Getting embeddings for sentences.")
        sentence_embeddings = get_embeddings(sentences)
        logger.info(f"Got embeddings for sentences on page {page['page_number']}. Starting to create semantic chunks based on cosine similarity with threshold {threshold}.")
        current_chunk = sentences[0]

        logger.info(f"Processing sentences for semantic chunking. Starting with first sentence: '{sentences[0][:50]}...'")
        for i in range(len(sentences) - 1):

            if len(current_chunk) < min_chunk_size:

                logger.debug(f"Current chunk length {len(current_chunk)} is below minimum chunk size {min_chunk_size}. Adding sentence {i+1} to current chunk without checking similarity.")
                current_chunk += ". " + sentences[i+1]
                continue

            if len(current_chunk) > max_chunk_size:

                logger.debug(f"Current chunk length {len(current_chunk)} exceeds maximum chunk size {max_chunk_size}. Saving current chunk and starting new chunk with sentence {i+1}.")
                semantic_chunks.append({
                    "text": current_chunk.strip(),
                    "page_number": page["page_number"],
                    "source": page["source"],
                    "chunk_index": chunk_index,
                })
                logger.debug(f"Chunk saved. Moving to next chunk with sentence {i+1}.")
                chunk_index += 1
                current_chunk = sentences[i+1]
                continue

            logger.info(f"Comparing sentence {i} with sentence {i+1} for semantic similarity.")
            sim = cosine_similarity(sentence_embeddings[i], sentence_embeddings[i+1])

            logger.info(f"Cosine similarity between sentence {i} and sentence {i+1}: {sim:.4f}")
            if sim < threshold:

                logger.info(f"Similarity below threshold. Saving current chunk and starting new chunk with sentence {i+1}.")
                semantic_chunks.append({
                    "text": current_chunk.strip(),
                    "page_number": page["page_number"],
                    "source": page["source"],
                    "chunk_index": chunk_index,
                })

                logger.info(f"Chunk saved. Moving to next chunk with sentence {i+1}.")
                chunk_index += 1
                current_chunk = sentences[i+1]

            else:

                logger.info(f"Similarity above threshold. Adding sentence {i+1} to current chunk.")
                current_chunk += ". " + sentences[i+1]

        logger.info(f"Finished processing sentences for page {page['page_number']}. Saving last chunk.")
        semantic_chunks.append({
            "text": current_chunk.strip(),
            "page_number": page["page_number"],
            "source": page["source"],
            "chunk_index": chunk_index,
        })
        chunk_index += 1

    logger.info(f"Finished semantic chunking all pages. Total semantic chunks created: {len(semantic_chunks)}")

    return semantic_chunks






if __name__ == "__main__":
    from amf_rag_agent.ingestion.parser import parse_pdf
    pages = parse_pdf("data/raw/totalenergie-deu-2025-en.pdf")
    # chunks = chunk_pages(pages)
    chunks = semantic_chunk_pages(pages, threshold=0.6)
    # print(f"Total chunks: {len(chunks)}")
    # print(f"\n--- Sample chunk ---")
    # print(f"Page: {chunks[405]['page_number']}")
    # print(f"Text: {chunks[405]['text'][:400]}")
    print(f"Total semantic chunks: {len(chunks)}")
    for c in chunks[:5]:
        print(f"\n--- Sample semantic chunk ---")
        print(f"Page: {c['page_number']}")
        print(f"Text: {c['text'][:400]}")
        print(f"\n[{len(c['text'])} characters] {c['text'][:400]}...")
    
    sizes = [len(c["text"]) for c in chunks]
    print(f"Total semantic chunks: {len(chunks)}")
    print(f"Min size: {min(sizes)}")
    print(f"Max size: {max(sizes)}")
    print(f"Avg size: {sum(sizes) // len(sizes)}")

    chunks_old = chunk_pages(pages)
    sizes_old = [len(c["text"]) for c in chunks_old]
    print(f"Original: {len(chunks_old)} chunks, min={min(sizes_old)}, max={max(sizes_old)}, avg={sum(sizes_old)//len(sizes_old)}")