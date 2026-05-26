import pydantic
import fastapi
from contextlib import asynccontextmanager

from amf_rag_agent import config
from amf_rag_agent.agent.loop import run_agent, tools

@asynccontextmanager
async def lifespan(app: fastapi.FastAPI):

    config.setup_logging()
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

    response = await run_agent(request.question, tools)

    return AnswerResponse(
        answer=response['answer'],
        sources=response['sources']
    )