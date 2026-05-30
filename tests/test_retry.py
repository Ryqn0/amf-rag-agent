import pytest
from unittest.mock import patch, AsyncMock
from langchain_core.messages import AIMessage
from amf_rag_agent.agent.graph_v2 import invoke_model, run_agent, search_documents

from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI


@pytest.mark.asyncio
async def test_retry_recovers_after_transient_failures():
    """Test that the retry mechanism correctly retries after transient failures and eventually succeeds."""

    mock_model = AsyncMock()
    mock_model.ainvoke = AsyncMock(side_effect=[
        Exception("503 Service Unavailable"),
        Exception("503 Service Unavailable"),
        AIMessage(content="success"),
    ])

    with patch("asyncio.sleep", new=AsyncMock()):

        with patch("amf_rag_agent.agent.graph_v2.model", mock_model):

            result = await invoke_model(["test message"])

            assert result.content == "success"
            assert mock_model.ainvoke.call_count == 3


@pytest.mark.asyncio
async def test_retry_fails_after_max_attempts():
    """Test that the retry mechanism raises an exception after exceeding the maximum number of attempts."""

    mock_model = AsyncMock()
    mock_model.ainvoke = AsyncMock(side_effect=Exception("503 Service Unavailable"))

    with patch("asyncio.sleep", new=AsyncMock()):

        with patch("amf_rag_agent.agent.graph_v2.model", mock_model):

            with pytest.raises(Exception):

                await invoke_model(["test message"])

            assert mock_model.ainvoke.call_count == 3