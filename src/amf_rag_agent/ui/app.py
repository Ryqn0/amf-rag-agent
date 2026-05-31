import logging

import streamlit as st
# from amf_rag_agent.agent.loop import run_agent, tools
# from amf_rag_agent.agent.graph import run_agent
from amf_rag_agent.agent.graph_v2 import run_agent
# from amf_rag_agent.retrieval.store import load_all_chunks
# from amf_rag_agent.retrieval.bm25_store import build_bm25_index
import asyncio
from amf_rag_agent.config import setup_logging

from anthropic import RateLimitError, APIError

# os.environ["LANGCHAIN_TRACING"] = "true"
# os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY", "")
# os.environ["LANGCHAIN_PROJECT"] = "amf-rag-agent"
# os.environ["LANGSMITH_ENDPOINT"] = "https://eu.api.smith.langchain.com"

logger = logging.getLogger(__name__)

setup_logging()
# chunks = load_all_chunks()
# build_bm25_index(chunks)


st.title("Bilingual (FR/EN) AMF RAG AGENT Chat Interface")
st.subheader("Ask questions about the ingested documents and get answers with sources")
st.write("Current source documents: TotalEnergies, BNP Paribas, LVMH, Airbus - all from 2025 Universal Registration Documents (URD).")

if "messages" not in st.session_state:

    st.session_state.messages = []

for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])

if prompt := st.chat_input("What is up?"):

    # st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):

        st.markdown(prompt)

    with st.chat_message("assistant"):

        with st.spinner("Searching documents..."):

            try:

                
                data = asyncio.run(run_agent(prompt))
                st.session_state.messages.append({"role": "user", "content": prompt})
                st.markdown(data['answer'])
                
                with st.expander("Sources"):

                    for source in data['sources']:

                        st.write(f"Page {source['page_number']} - {source['source']}")
                
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": data['answer']
                })

            except RateLimitError as e:

                st.error("Rate limit exceeded. Please try again later.")

            except APIError as e:

                st.error("An error occurred while processing your request.")

            except Exception as e:

                logger.error(f"Error processing request: {e}")
                st.error("Something went wrong - please try again")
