from amf_rag_agent.retrieval.embedder import get_embeddings
from amf_rag_agent.retrieval.store import search
from amf_rag_agent.retrieval.bm25_store import search_bm25
from amf_rag_agent.retrieval.reranker import rerank_chunks
# from amf_rag_agent.retrieval.query_expander import needs_expansion, expand_query
import asyncio
from amf_rag_agent import config
from langsmith import traceable

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
        list[dict]: A list of relevant document texts.
    """

    logger.info(f"Received query: {query}")
    
    '''
    It seems that Claude Haiku already do query decomposition so it is better to let it handle that instead of doing it ourselves.
    logger.info("Checking if query needs expansion...")

    if needs_expansion(query):

        logger.info("Query needs expansion, generating alternatives...")
        queries = await expand_query(query)
        logger.info(f"Generated {len(queries)} expanded queries.")

    else:

        logger.info("Query does not need expansion.")
        '''
    
    queries = [query]


    seen = set()
    combined_results = []

    logger.info("Combining and deduplicating results from semantic search and BM25...")

    for q in queries:

        logger.info(f"Processing query: {q}")
        logger.info("Getting embedding for query...")
        embedding = get_embeddings([q])[0]

        for result in search(embedding, k=20) + search_bm25(q, k=20):

            logger.info(f"Processing result: {result['text'][:50]}...")
            
            if result["text"] not in seen:

                logger.info(f"Adding new result to combined results: {result['text'][:50]}...")
                seen.add(result["text"])
                combined_results.append(result)

    return rerank_chunks(queries[0], combined_results, top_k=5)


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

@traceable
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

    logger.info(f"Starting agent loop with query: {query}")
    logger.info(f"Query length: {len(query.split())} words.")

    while not done:

        logger.info("Calling model...")
        response = await client.messages.create(model="claude-haiku-4-5-20251001",
                                          tools=tools,
                                          max_tokens=1024, 
                                          messages=history)
    
        if response.stop_reason == "tool_use":
            
            logger.info("Model requested tool use.")
            tool_blocks = [block for block in response.content if block.type == "tool_use"]
            logger.info(f"Model requested {len(tool_blocks)} tool uses.")
            history.append(
                {
                    "role": "assistant",
                    "content": response.content
                }
            )
            logger.info("Executing tool calls in parallel...")
            search_results = await asyncio.gather(*[asyncio.to_thread(search_documents, block.input["query"]) for block in tool_blocks])
            tool_results = []

            logger.info("Processing tool results...")
            for block, results in zip(tool_blocks, search_results):

                logger.info(f"Processing results for tool use with query: {block.input['query']}")
                tool_content = "\n".join(r["text"] for r in results)
                logger.info(f"Tool returned {len(results)} results, adding to sources and tool results.")
                sources.extend(results)
                logger.info(f"Adding tool results to tool results list.")
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
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