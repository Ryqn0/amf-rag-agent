from dotenv import load_dotenv
import logging
import os
import sys

load_dotenv()

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY", "")
os.environ["LANGCHAIN_PROJECT"] = "amf-rag-agent"
os.environ["LANGSMITH_ENDPOINT"] = "https://eu.api.smith.langchain.com"


# VALIDATION
def validate_config():
    if not ANTHROPIC_API_KEY:
        raise ValueError("ANTHROPIC_API_KEY is not set")
    
validate_config()

def setup_logging(level = logging.INFO):
    """
    Set up logging configuration.
    """

    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)]
    )