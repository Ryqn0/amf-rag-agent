import pytest
from amf_rag_agent.ingestion.chunker import chunk_pages


@pytest.fixture
def sample_pages():
    """Provides a sample list of parsed pages for testing the chunking function."""

    return [
        {
            "text": (
                "TotalEnergies is a global integrated energy company.\n"
                "It produces and markets oil, natural gas, and electricity.\n"
                "The company operates in more than 130 countries worldwide.\n"
                "Its strategy focuses on profitable growth and the energy transition.\n"
                "Renewable energy investments reached 4 billion dollars in 2025."
            ),
            "page_number": 6,
            "source": "test_document.pdf"
        },
        {
            "text": (
                "Risk factors represent the main uncertainties facing TotalEnergies.\n"
                "Market risks include oil price volatility and currency fluctuations.\n"
                "Climate risks encompass both physical and transition risks.\n"
                "Geopolitical risks arise from operations in unstable regions.\n"
                "Cybersecurity threats continue to grow in sophistication.\n"
            ),
            "page_number": 130,
            "source": "test_document.pdf"
        }
    ]


@pytest.mark.parametrize("chunk_size, overlap", [
    (100, 20),
    (200, 40),
    (800, 160)
])
def test_shape_list_dict(sample_pages, chunk_size, overlap):
    """Helper function to verify that all chunks have the correct shape."""

    chunks = chunk_pages(sample_pages, chunk_size=chunk_size, overlap=overlap)

    for chunk in chunks:

        assert "text" in chunk, f"Chunk {chunk['chunk_index']} is missing 'text' key."
        assert "page_number" in chunk, f"Chunk {chunk['chunk_index']} is missing 'page_number' key."
        assert "source" in chunk, f"Chunk {chunk['chunk_index']} is missing 'source' key."
        assert "chunk_index" in chunk, f"Chunk {chunk['chunk_index']} is missing 'chunk_index' key."


@pytest.mark.parametrize("chunk_size, overlap", [
    (100, 20),
    (200, 40),
    (800, 160)
])
def test_metadata_consistency(sample_pages, chunk_size, overlap):
    """Helper function to verify that the metadata in the chunks is consistent with the original pages."""

    chunks = chunk_pages(sample_pages, chunk_size=chunk_size, overlap=overlap)
    valid_sources = {p["source"] for p in sample_pages}
    valid_page_numbers = {p["page_number"] for p in sample_pages}

    for chunk in chunks:
        assert chunk["source"] in valid_sources
        assert chunk["page_number"] in valid_page_numbers


@pytest.mark.parametrize("chunk_size, overlap", [
    (100, 20),
    (200, 40),
    (800, 160)
])
def test_overlap(sample_pages, chunk_size, overlap):
    """Helper function to verify that the specified overlap is maintained between consecutive chunks."""

    chunks = chunk_pages(sample_pages, chunk_size=chunk_size, overlap=overlap)

    for chunk in chunks:

        if chunk["chunk_index"] > 0:

            previous_chunk = chunks[chunk["chunk_index"] - 1]

            if chunk["page_number"] == previous_chunk["page_number"]:

                overlap_text = chunk["text"][:20]
                assert overlap_text in previous_chunk["text"], f"Chunk {chunk['chunk_index']} does not maintain the specified overlap with the previous chunk."


@pytest.mark.parametrize("chunk_size, overlap", [
    (100, 20),
    (200, 40),
    (800, 160)
])
def test_no_empty_text(sample_pages, chunk_size, overlap):
    """Helper function to verify that no chunk has empty text."""

    chunks = chunk_pages(sample_pages, chunk_size=chunk_size, overlap=overlap)

    for chunk in chunks:

        assert chunk["text"].strip() != "", f"Chunk {chunk['chunk_index']} has empty text."


@pytest.mark.parametrize("chunk_size, overlap", [
    (100, 20),
    (200, 50),
    (800, 160)
])
def test_chunk_index_increment(sample_pages, chunk_size, overlap):
    """Helper function to verify that the chunk_index is incremented correctly for each chunk."""

    chunks = chunk_pages(sample_pages, chunk_size=chunk_size, overlap=overlap)

    for i, chunk in enumerate(chunks):

        assert chunk["chunk_index"] == i, f"Chunk {chunk['chunk_index']} has incorrect chunk index."