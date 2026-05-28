from typing import TypedDict, Annotated
from anthropic.types import MessageParam
from langgraph.graph import StateGraph, END
from operator import add
from amf_rag_agent.agent.loop import client, tools
from logging import getLogger
from amf_rag_agent.agent.loop import search_documents


logger = getLogger(__name__)


class AgentState(TypedDict):

    messages: Annotated[list[MessageParam], add]    # Conversation history, including user messages, assistant responses, and tool results.
    sources: Annotated[list[dict], add]             # Retrieved documents or sources relevant to the query.


async def call_llm(state: AgentState):
    """
    Make a call to the LLM with the current conversation history and tools.
    Args:
    state (AgentState): The current state of the agent, including messages and sources.
    Returns:
    AgentState: Updated state with the LLM's response added to the messages.
    """

    logger.info("Calling LLM with current messages and tools.")
    response = await client.messages.create(
    model="claude-haiku-4-5-20251001",
    tools=tools,
    max_tokens=1024,
    messages=state["messages"]
    )

    logger.info(f"LLM response received): {response.content}")
    return {
    "messages": [{"role": "assistant", "content": response.content}]
    }


async def execute_tools(state: AgentState) -> AgentState:
    """
    Execute any tools that the LLM has indicated it wants to use.
    Args:
    state (AgentState): The current state of the agent, including messages and sources.
    Returns:
    AgentState: Updated state with tool results added to messages and sources.
    """

    logger.info("Executing tools.")
    last_message = state["messages"][-1]
    logger.info(f"Last message content: {last_message['content']}")
    logger.info("Checking for tool use blocks in the last message.")
    tool_blocks = [block for block in last_message["content"] if block.type == "tool_use"]
    tool_results = []
    all_sources = []

    for block in tool_blocks:

        logger.info(f"Processing tool use block: {block.name} with input: {block.input}")

        try:

            logger.info(f"Executing tool: {block.name} with query: {block.input['query']}")
            results = search_documents(block.input["query"])
            logger.info(f"Tool execution results: {results}")
            logger.info(f"Formatting tool results for message content.")
            tool_content = "\n".join(r["text"] for r in results)
            logger.info(f"Tool content formatted: {tool_content}")
            logger.info(f"Adding tool results to sources.")
            all_sources.extend(results)

        except Exception as e:

            logger.error(f"Error executing tool: {block.name} with query: {block.input['query']}. Error: {str(e)}")
            tool_content = f"Search failed: {str(e)}"
        
        logger.info(f"Appending tool result to tool_results list.")
        tool_results.append({
            "type": "tool_result",
            "tool_use_id": block.id,
            "content": tool_content
        })


    logger.info(f"Finalizing tool execution with results: {tool_results}")
    return {
        "messages": [{"role": "user", "content": tool_results}],
        "sources": all_sources
    }


def should_continue(state: AgentState) -> str:

    logger.info("Determining if agent should continue or end.")
    last_message = state["messages"][-1]

    if last_message["content"] and any(block.type == "tool_use" for block in last_message["content"]):

        logger.info("LLM indicated tool use, continuing to execute tools.")
        return "execute_tools"
    
    logger.info("No tool use indicated, ending agent execution.")
    return END


graph = StateGraph(AgentState)
graph.add_node("call_llm", call_llm)
graph.add_node("execute_tools", execute_tools)
graph.set_entry_point("call_llm")
graph.add_conditional_edges("call_llm", should_continue)
graph.add_edge("execute_tools", "call_llm")

app = graph.compile()


async def run_agent(query: str) -> dict:
    """
    Run the agent with a given query.
    Args:
    query (str): The input query for the agent.
    Returns:
    dict: The final state of the agent after execution, including messages and sources.
    """

    logger.info(f"Running agent with query: {query}")
    initial_state = {
        "messages": [{"role": "user", "content": query}],
        "sources": []
    }

    logger.info("Starting agent execution.")
    final_state = await app.ainvoke(initial_state)
    logger.info("Agent execution completed.")

    last_message = final_state["messages"][-1]
    logger.info(f"Final message content: {last_message['content']}")
    answer = last_message["content"][0].text

    return {
        "answer": answer,
        "sources": final_state["sources"]
    }