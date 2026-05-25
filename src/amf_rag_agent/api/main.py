import pydantic
import fastapi

from amf_rag_agent.agent.loop import run_agent, tools

app = fastapi.FastAPI()

class QuestionRequest(pydantic.BaseModel):

    question: str


class AnswerResponse(pydantic.BaseModel):

    answer: str
    sources: list[dict]


@app.post("/ask", response_model=AnswerResponse)
def ask(request: QuestionRequest) -> AnswerResponse:

    response = run_agent(request.question, tools)

    return AnswerResponse(
        answer=response['answer'],
        sources=response['sources']
    )