import pytest
from amf_rag_agent.ingestion.chunker import chunk_pages

@pytest.fixture
def sample_pages():
    """Provides a sample list of parsed pages for testing the chunking function."""

    return [
        {
            "text": "Present in about 120 countries, the Company works with a network of\
                over 100,000 suppliers of goods and services.",
            "page_number": 412,
            "source": "totalenergies_universal-registration-document-2025_2026_en.pdf"
        },
        {
            "text": "suppliers of goods and services. In 2025, the Company’s\
                purchases of goods and services (excluding pe",
            "page_number": 412,
            "source": "totalenergies_universal-registration-document-2025_2026_en.pdf"
        }
    ]

def test_shape_list_dict(sample_pages):
    """Helper function to verify that all chunks have the correct shape."""

    chunks = chunk_pages(sample_pages, chunk_size=50, overlap=20)

    for chunk in chunks:

        assert "text" in chunk, f"Chunk {chunk['chunk_index']} is missing 'text' key."
        assert "page_number" in chunk, f"Chunk {chunk['chunk_index']} is missing 'page_number' key."
        assert "source" in chunk, f"Chunk {chunk['chunk_index']} is missing 'source' key."
        assert "chunk_index" in chunk, f"Chunk {chunk['chunk_index']} is missing 'chunk_index' key."


def test_metadata_consistency(sample_pages):
    """Helper function to verify that the metadata in the chunks is consistent with the original pages."""

    chunks = chunk_pages(sample_pages, chunk_size=100, overlap=20)
    valid_sources = {p["source"] for p in sample_pages}
    valid_page_numbers = {p["page_number"] for p in sample_pages}

    for chunk in chunks:
        assert chunk["source"] in valid_sources
        assert chunk["page_number"] in valid_page_numbers


def test_overlap(sample_pages):
    """Helper function to verify that the specified overlap is maintained between consecutive chunks."""

    chunks = chunk_pages(sample_pages, chunk_size=50, overlap=20)

    for chunk in chunks:

        if chunk["chunk_index"] > 0:

            previous_chunk = chunks[chunk["chunk_index"] - 1]
            overlap_text = chunk["text"][:20]
            assert overlap_text in previous_chunk["text"], f"Chunk {chunk['chunk_index']} does not maintain the specified overlap with the previous chunk."


def test_no_empty_text(sample_pages):
    """Helper function to verify that no chunk has empty text."""

    chunks = chunk_pages(sample_pages, chunk_size=50, overlap=20)

    for chunk in chunks:

        assert chunk["text"].strip() != "", f"Chunk {chunk['chunk_index']} has empty text."


def test_chunk_index_increment(sample_pages):
    """Helper function to verify that the chunk_index is incremented correctly for each chunk."""

    chunks = chunk_pages(sample_pages, chunk_size=50, overlap=20)

    for i, chunk in enumerate(chunks):
        
        assert chunk["chunk_index"] == i, f"Chunk {chunk['chunk_index']} has incorrect chunk index."