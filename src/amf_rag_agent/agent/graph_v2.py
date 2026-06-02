from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage, SystemMessage
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END
from typing import Literal, TypedDict, Annotated
from operator import add
from tenacity import retry, stop_after_attempt, wait_exponential
import asyncio

import logging

from langsmith import traceable

logger = logging.getLogger(__name__)

class AgentState(TypedDict):

    messages: Annotated[list[BaseMessage], add]    # Conversation history, including user messages, assistant responses, and tool results.
    sources: Annotated[list[dict], add]             # Retrieved documents or sources relevant to the query.


SYSTEM_PROMPT = """You are a financial research assistant answering questions about AMF regulatory filings.

Answer ONLY using information from the retrieved source documents. If the sources do not contain enough information to fully answer, explicitly state what you could not find rather than filling in from general knowledge. Never state facts about a company that are not present in the retrieved sources.

Always cite which company and document your information comes from."""


@tool
async def search_documents(query: str) -> list[dict]:
    """Search AMF financial filings for relevant document chunks.

    Args:
        query: The search query.
    Returns:
        A list of document chunks, each with 'text', 'source', and 'page_number'.
    """

    from amf_rag_agent.agent.loop import search_documents as search_func
    return await asyncio.to_thread(search_func, query)


primary_model = ChatAnthropic(model="claude-haiku-4-5-20251001", max_tokens=1024).bind_tools([search_documents])

fallback_model = ChatOpenAI(model="gpt-4o-mini").bind_tools([search_documents])

model = primary_model.with_fallbacks([fallback_model])

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
async def invoke_model(messages):
    """Invoke the primary model with the given messages, falling back to the secondary model if the primary fails."""

    return await model.ainvoke(messages)



async def call_llm(state: AgentState) -> AgentState:
    """Call the LLM with the current conversation history and tools.

    Args:
        state: The current state of the agent, including messages and sources.
    Returns:
        Updated state with the LLM's response added to the messages.
    """

    logger.info("Calling LLM with current messages and tools.")
    response = await invoke_model(state["messages"])
    logger.info(f"LLM response received: {response.content}")

    return {"messages": [response]}

tools = [search_documents]
tools_by_name = {tool.name: tool for tool in tools}


async def execute_tools(state: AgentState) -> AgentState:
    """Execute any tools that the LLM has indicated it wants to use.

    Args:
        state: The current state of the agent, including messages and sources.
    Returns:
        Updated state with tool results added to messages and sources.
    """

    logger.info("Checking for tool calls in LLM response.")
    last_message = state["messages"][-1]

    search_results = await asyncio.gather(*[search_documents.ainvoke(call['args']) for call in last_message.tool_calls])


    messages = []
    all_sources = []

    for call, results in zip(last_message.tool_calls, search_results):

        logger.info(f"Tool results for call with args '{call['args']}': {results}")
        tool_content = "\n".join([result["text"] for result in results])
        logger.info(f"Tool content for call with args '{call['args']}': {tool_content}")
        all_sources.extend(results)
        logger.info(f"Adding tool result for call with args '{call['args']}' to messages.")
        messages.append(ToolMessage(content=tool_content, tool_call_id=call["id"]))
    
    return {
        "messages": messages,
        "sources": all_sources
    }


def should_continue(state: AgentState) -> str:
    """Decide if we should continue the loop or stop based upon whether the LLM made a tool call"""

    logger.info("Deciding whether to continue loop based on LLM response.")
    messages = state["messages"]
    last_message = messages[-1]

    # If the LLM makes a tool call, then perform an action
    if last_message.tool_calls:

        logger.info("LLM made a tool call, continuing to execute tools.")
        return "execute_tools"

    # Otherwise, we stop (reply to the user)
    logger.info("LLM did not make a tool call, ending loop.")
    return END


graph = StateGraph(AgentState)
graph.add_node("call_llm", call_llm)
graph.add_node("execute_tools", execute_tools)
graph.set_entry_point("call_llm")
graph.add_conditional_edges("call_llm", should_continue)
graph.add_edge("execute_tools", "call_llm")

app = graph.compile()

@traceable
async def run_agent(query: str, history: list[BaseMessage] | None = None) -> AgentState:
    """Run the agent with the given initial state."""

    history = history or []

    logger.info("Running agent with initial state.")
    initial_state = {
        "messages": [SystemMessage(content=SYSTEM_PROMPT)] + history + [HumanMessage(content=query)],
        "sources": []
    }

    logger.info(f"Initial state prepared: {initial_state}")
    final_state = await app.ainvoke(initial_state)

    logger.info(f"Agent execution completed. Final state: {final_state}")
    answer = final_state["messages"][-1].content
    sources = final_state["sources"]

    logger.info(f"Final answer: {answer}")
    logger.info(f"Sources used: {sources}")
    return {"answer": answer,
        "sources": sources
    }