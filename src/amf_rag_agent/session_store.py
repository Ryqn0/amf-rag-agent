import os
import json
import logging
import redis
from langchain_core.messages import HumanMessage, AIMessage

logger = logging.getLogger(__name__)

_client = None

def get_redis_client():

    global _client

    if _client is None:

        redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
        _client = redis.from_url(redis_url, decode_responses=True)

    return _client


SESSION_TTL_SECONDS = 3600  # 1 hour


def get_history(session_id: str) -> list[dict]:
    """Retrieve the conversation history for a given session ID from Redis.
    Args:
        session_id (str): The unique identifier for the conversation session.
    Returns:
        list[dict]: The conversation history as a list of message dictionaries.
    """
    
    logger.debug(f"Retrieving history for session_id: {session_id}")
    client = get_redis_client()
    logger.debug(f"Redis client obtained: {client}")
    history_json = client.get(f"session:{session_id}")

    if history_json is None:
        
        logger.debug(f"No history found for session_id: {session_id}")
        return []
    
    else:
    
        logger.debug(f"History found for session_id: {session_id}")
        return json.loads(history_json)
    

def append_messages(session_id: str, messages: list[dict]) -> None:
    """Append new messages to the conversation history for a given session ID in Redis.
    Args:
        session_id (str): The unique identifier for the conversation session.
        messages (list[dict]): A list of message dictionaries to append to the history.
    """
    
    logger.debug(f"Appending messages to session_id: {session_id}, messages: {messages}")
    client = get_redis_client()
    existing_history = get_history(session_id)
    existing_history.extend(messages)
    client.set(f"session:{session_id}", json.dumps(existing_history), ex=SESSION_TTL_SECONDS)


def dicts_to_message(history: list[dict]) -> list:
    """Convert a list of message dictionaries to a list of HumanMessage and AIMessage objects.
    Args:
        history (list[dict]): A list of message dictionaries, each with 'role' and 'content' keys.
    Returns:
        list: A list of HumanMessage and AIMessage objects representing the conversation history.
    """
    
    logger.debug(f"Converting history dicts to messages, history: {history}")
    messages = []

    for entry in history:

        if entry["role"] == "user":

            logger.debug(f"Adding HumanMessage for entry: {entry}")
            messages.append(HumanMessage(content=entry["content"]))

        else:

            logger.debug(f"Adding AIMessage for entry: {entry}")
            messages.append(AIMessage(content=entry["content"]))
    
    logger.debug(f"Converted messages: {messages}")
    return messages