from logging import getLogger
from anthropic import AsyncAnthropic
from amf_rag_agent import config

api_key = config.ANTHROPIC_API_KEY

logging = getLogger(__name__)

_client = None


def get_client() -> AsyncAnthropic:
    global _client
    if _client is None:
        logging.info("Initializing new AsyncAnthropic client.")
        _client = AsyncAnthropic(api_key=api_key)
    return _client


def needs_expansion(query: str) -> bool:
    """
    Determines if a query needs expansion.
    Args:        query (str): The input query string.
    Returns:        bool: True if the query needs expansion, False otherwise.
    """

    logging.info(f"Evaluating if query needs expansion: {query}")
    logging.info(f"Query length: {len(query.split())} words.")
    
    if len(query.split()) < 6:
        
        logging.info("Query is short, does not need expansion.")
        return False
    
    comparison_signals = ["compare", "difference", "versus", "vs", "contrast", "similarity", "difference between", "accross", "between"]
    logging.info(f"Checking for comparison signals in query: {query}")

    if any(signal in query.lower() for signal in comparison_signals):

        logging.info("Query contains comparison signals, need expansion.")
        return True
    
    if "?" in query and len(query) > 100:

        logging.info("Query is a long question, need expansion.")
        return True
    
    logging.info("Query does not need expansion.")
    return False


PROMPT = """
Given the following query, generate 3 alternative phrasings that capture the same intent but use different wording. The alternatives should be concise and varied in structure. It will return 3 alternatives one per line separated by a newline character. The original query is: {query}
"""

async def expand_query(query: str) -> list[str]:
    """
    Use the LLM to generate alternative phrasings of the query.
    Returns a list containing the original query plus expansions.
    """

    response = await get_client().messages.create(model="claude-haiku-4-5-20251001",
                                                  max_tokens=1024, 
                                                  messages=[
                                                      {
                                                          "role": "user",
                                                          "content": PROMPT.format(query=query)
                                                      }
                                                  ])
    
    expansions = [query] + response.content[0].text.split("\n")

    return expansions