from amf_rag_agent import config
import asyncio
from langchain_anthropic import ChatAnthropic
from langsmith import evaluate
import json
from amf_rag_agent.agent.graph_v2 import run_agent
import logging
import re




logger = logging.getLogger(__name__)

judge = ChatAnthropic(model="claude-haiku-4-5-20251001")


def run_agent_target(inputs: dict) -> dict:
    """Run the agent with the given inputs and return the results.
    Args:
        inputs: A dictionary containing the initial inputs for the agent.
    Returns:
        A dictionary containing the final state of the agent after execution.
    """

    logger.info(f"Running agent with inputs: {inputs}")
    results = asyncio.run(run_agent(inputs["question"]))

    logger.info(f"Agent results: {results}")
    return {"answer": results["answer"],
            "sources": results["sources"]}


def faithfulness(run, example) -> dict:
    """Evaluate the faithfulness of the agent's answer to the question based on the provided sources.
    Args:
        run: The output from the agent execution, containing the answer and sources.
        example: The original example containing the question and expected answer.
    Returns:
        A dictionary containing the faithfulness score and reasoning.
    """

    logger.info("Evaluating faithfulness of the answer to the question.")
    answer = run.outputs["answer"]
    sources = run.outputs["sources"]
    question = example.inputs["question"]

    logger.info(f"Question: {question}")
    logger.info(f"Answer: {answer}")
    logger.info(f"Sources: {sources}")
    sources_text = "\n\n".join(s["text"] for s in sources)

    logger.info("Constructing prompt for faithfulness evaluation.")
    prompt = f"""You are evaluating whether an answer is faithful to its sources.
An answer is FAITHFUL if every claim it makes is supported by the source documents.
It is UNFAITHFUL if it states facts not found in the sources (hallucination).

Question: {question}

Answer: {answer}

Source documents:
{sources_text}

Score 1 if every claim in the answer is supported by the sources, 0 if any claim is unsupported.
Respond with ONLY a JSON object: {{"score": 0 or 1, "reasoning": "brief explanation"}}"""

    logger.info(f"Prompt for faithfulness evaluation: {prompt}")
    response = judge.invoke(prompt)
    logger.info(f"Faithfulness evaluation response: {response}")
    raw = response.content
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        logger.error(f"Failed to parse JSON from judge response: {raw}")
        return {"key": "faithfulness", "score": None, "comment": f"Failed to parse judge response: {raw[:200]}..."}
    else:
        result = json.loads(match.group(0))

    logger.info(f"Parsed faithfulness evaluation result: {result}")
    return {"key": "faithfulness", "score": result["score"], "comment": result["reasoning"]}

logger.info("Starting evaluation of agent performance on faithfulness.")
results = evaluate(
    run_agent_target,
    data="amf-rag-eval",
    evaluators=[faithfulness],
    experiment_prefix="faithfulness-baseline",
)