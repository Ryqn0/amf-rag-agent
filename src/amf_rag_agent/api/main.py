import os

import pydantic
import fastapi
from contextlib import asynccontextmanager

from amf_rag_agent import config
from amf_rag_agent.agent.graph import run_agent
# from amf_rag_agent.agent.loop import run_agent, tools
from amf_rag_agent.retrieval.bm25_store import build_bm25_index
from amf_rag_agent.retrieval.store import load_all_chunks

# os.environ["LANGCHAIN_TRACING"] = "true"
# os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY", "")
# os.environ["LANGCHAIN_PROJECT"] = "amf-rag-agent"
# os.environ["LANGSMITH_ENDPOINT"] = "https://eu.api.smith.langchain.com"

@asynccontextmanager
async def lifespan(app: fastapi.FastAPI):

    config.setup_logging()
    chunks = load_all_chunks()
    build_bm25_index(chunks)
    
    yield
    # shutdown code here


app = fastapi.FastAPI(lifespan=lifespan)

class QuestionRequest(pydantic.BaseModel):

    question: str


class AnswerResponse(pydantic.BaseModel):

    answer: str
    sources: list[dict]


@app.post("/ask", response_model=AnswerResponse)
async def ask(request: QuestionRequest) -> AnswerResponse:

    response = await run_agent(request.question)

    return AnswerResponse(
        answer=response['answer'],
        sources=response['sources']
    )