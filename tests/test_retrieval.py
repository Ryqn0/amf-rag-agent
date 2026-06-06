from unittest.mock import patch, MagicMock
from amf_rag_agent.agent.loop import search_documents
import numpy as np


def test_search_documents():
    """
    Test the search_documents function to ensure it returns results in the expected format.
    """

    with patch("amf_rag_agent.agent.loop.get_embeddings") as mock_embed:
        mock_embed.return_value = np.zeros((1, 384))
        # inside here, get_embeddings returns zeros instantly

        with patch("amf_rag_agent.agent.loop.search") as mock_search:
            mock_search.return_value = [
                {"text": "Document 1", "source": "doc1.pdf", "page_number": 1, "distance": 0.1},
                {"text": "Document 2", "source": "doc2.pdf", "page_number": 5, "distance": 0.2}
            ]

            with patch("amf_rag_agent.agent.loop.search_es") as mock_search_es:
                mock_search_es.return_value = []

                results = search_documents("What are the main risk factors?")
                assert isinstance(results, list)
                assert len(results) == 2
                assert all(isinstance(result, dict) for result in results)
                assert all("text" in result and "source" in result and "page_number" in result for result in results)
                mock_embed.assert_called_once_with(["What are the main risk factors?"])
                mock_search.assert_called_once()
