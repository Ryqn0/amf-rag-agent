import os
import logging
import pydantic
import fastapi
from fastapi import Security, HTTPException, status, Depends, Request
from fastapi.security import APIKeyHeader
from anthropic import RateLimitError, APIError
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from contextlib import asynccontextmanager
import uuid

from amf_rag_agent import config
# from amf_rag_agent.agent.graph import run_agent
# from amf_rag_agent.agent.loop import run_agent, tools
from amf_rag_agent.agent.graph_v2 import run_agent
from amf_rag_agent.session_store import get_history, append_messages, dicts_to_message
# from amf_rag_agent.retrieval.bm25_store import build_bm25_index
# from amf_rag_agent.retrieval.store import load_all_chunks

# os.environ["LANGCHAIN_TRACING"] = "true"
# os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY", "")
# os.environ["LANGCHAIN_PROJECT"] = "amf-rag-agent"
# os.environ["LANGSMITH_ENDPOINT"] = "https://eu.api.smith.langchain.com"


logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: fastapi.FastAPI):

    config.setup_logging()
    # chunks = load_all_chunks()
    # build_bm25_index(chunks)
    
    yield
    # shutdown code here


api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)
VALID_API_KEYS = set(os.getenv("API_KEYS", "").split(","))


def verify_api_key(api_key: str = Security(api_key_header)) -> str:
    """
    Verify the provided API key against a list of valid keys.
    Args:
        api_key (str): The API key provided in the request header.
    Returns:
        str: The verified API key.
    Raises:
        HTTPException: If the API key is missing or invalid."""
    
    if api_key not in VALID_API_KEYS:

        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API key")
    
    return api_key



def get_api_key_or_ip(request: Request) -> str:
    """
    Get the API key from the request header or fall back to the client's IP address for rate limiting.
    Args:
        request (Request): The incoming HTTP request.
    Returns:
        str: The API key if provided, otherwise the client's IP address.
    """

    return request.headers.get("X-API-Key") or get_remote_address(request)


limiter = Limiter(key_func=get_api_key_or_ip)


app = fastapi.FastAPI(lifespan=lifespan)

app.state.limiter = limiter

app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

class QuestionRequest(pydantic.BaseModel):

    question: str
    session_id: str | None = None


class AnswerResponse(pydantic.BaseModel):

    answer: str
    sources: list[dict]
    session_id: str


@app.post("/ask", response_model=AnswerResponse)
@limiter.limit("10/minute")
async def ask(request: Request, question_request: QuestionRequest, api_key: str = Depends(verify_api_key)) -> AnswerResponse:

    try:

        logger.info(f"Received question: {question_request.question} with session_id: {question_request.session_id}")
        session_id = question_request.session_id or str(uuid.uuid4())

        logger.info(f"Using session_id: {session_id} for conversation history.")
        history_dicts = get_history(session_id)
        logger.info(f"Retrieved history for session_id {session_id}: {history_dicts}")
        history = dicts_to_message(history_dicts)

        logger.info(f"Converted history dicts to messages for session_id {session_id}: {history}")
        response = await run_agent(question_request.question, history=history)

        logger.info(f"Agent response for session_id {session_id}: {response}")
        append_messages(session_id, [{"role": "user", "content": question_request.question}, {"role": "assistant", "content": response["answer"]},])

        logger.info(f"Appended user question and agent answer to history for session_id {session_id}.")
        return AnswerResponse(
            answer=response['answer'],
            sources=response['sources'],
            session_id=session_id,
        )
    
    except RateLimitError as e:

        raise fastapi.HTTPException(status_code=429, detail=str("Rate limit exceeded. Please try again later."))
    
    except APIError as e:

        raise fastapi.HTTPException(status_code=500, detail=str("An error occurred while processing your request."))
    
    except Exception as e:
    
        logger.error(f"Error processing request: {e}")

        raise fastapi.HTTPException(status_code=500, detail=str("An unexpected error occurred. Please try again later."))