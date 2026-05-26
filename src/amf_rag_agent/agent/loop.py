from amf_rag_agent.retrieval.embedder import get_embeddings
from amf_rag_agent.retrieval.store import search
from amf_rag_agent.retrieval.reranker import rerank_chunks
import asyncio
from amf_rag_agent import config

from anthropic import AsyncAnthropic
import logging
logger = logging.getLogger(__name__)


api_key = config.ANTHROPIC_API_KEY
client = AsyncAnthropic(api_key=api_key)

def search_documents(query: str) -> list[dict]:
    """
    Search for relevant documents based on a query.
    Args:
        query (str): The search query.
    Returns:
        list[str]: A list of relevant document texts.
    """

    embedding = get_embeddings([query])[0]
    results = search(embedding, k=20)
    results = rerank_chunks(query, results, top_k=5)

    return [{"text": result["text"], 
             "source": result["source"],
             "page_number": result["page_number"]} for result in results]
    # return [chunk["text"] for chunk in result]

tools = [
    {
        "name": "search_documents",
        "description": "Search for relevant documents based on a query.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "The search query."}
            },
            "required": ["query"]
        }
    }
]


async def run_agent(query: str, tools: list[dict]):
    """Run the RAG agent with the given query and tools.
    Args:
        query (str): The user's query.
        tools (list[dict]): A list of tools that the agent can use.
        
    Returns:
        str: The agent's final response.
    """

    sources = []
    history = [
        {
            "role": "user",
            "content": query,
        }
    ]
    done = False

    while not done:

        logger.info("Calling model...")
        response = await client.messages.create(model="claude-haiku-4-5-20251001",
                                          tools=tools,
                                          max_tokens=1024, 
                                          messages=history)
    
        if response.stop_reason == "tool_use":
            
            tool_blocks = [block for block in response.content if block.type == "tool_use"]
            history.append(
                {
                    "role": "assistant",
                    "content": response.content
                }
            )
            tool_results = []

            for tool_block in tool_blocks:
                logger.info(f"Model wants to use tool: {tool_block.name}")

                try :

                    result = search_documents(tool_block.input["query"])
                    tool_content = "\n".join(r["text"] for r in result)
                    sources.extend(result)

                except Exception as e:

                    tool_content = f"Search failed: {str(e)}"


                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": tool_block.id,
                    "content": tool_content
                })

            history.append({
                "role": "user",
                "content": tool_results
            })
            
            logger.info(f"Tool result added, looping back...")

        elif response.stop_reason == "end_turn":
            # model is done
            logger.info("Model finished.")
            done = True
    
    return {"answer": response.content[0].text, 
            "sources": sources}

if __name__ == "__main__":

    from amf_rag_agent.config import setup_logging
    setup_logging()
    answer = asyncio.run(run_agent("What are the main risk factors for TotalEnergies?", tools))["answer"]
    print(answer)