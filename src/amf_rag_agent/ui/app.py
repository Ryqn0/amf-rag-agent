import logging
import os
import uuid
import httpx
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

API_URL = os.getenv("API_URL", "http://api:8000")

st.title("Bilingual (FR/EN) AMF RAG AGENT Chat Interface")
st.subheader("Ask questions about the ingested documents and get answers with sources")
st.write("Current source documents: TotalEnergies, BNP Paribas, LVMH, Airbus - all from 2025 Universal Registration Documents (URD).")


if "api_key" not in st.session_state:

    st.session_state.api_key = ""

if not st.session_state.api_key:

    st.subheader("Please enter your API key to get started")
    key_input = st.text_input("API Key", type="password")

    if st.button("Login"):

        if key_input:

            st.session_state.api_key = key_input
            st.session_state.session_id = str(uuid.uuid4())
            st.rerun()
    
    st.stop()



if "messages" not in st.session_state:

    st.session_state.messages = []

for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])

if prompt := st.chat_input("Ask about the documents..."):

    # st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):

        st.markdown(prompt)
    
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):

        with st.spinner("Searching documents..."):

            try:

                response = httpx.post(
                    f"{API_URL}/ask",
                    headers={"X-API-Key": st.session_state.api_key},
                    json={"question": prompt, "session_id": st.session_state.session_id},
                    timeout=600.0
                )
                # data = asyncio.run(run_agent(prompt))
                # st.session_state.messages.append({"role": "user", "content": prompt})
                # st.markdown(data['answer'])
                
                # with st.expander("Sources"):

                #     for source in data['sources']:

                #         st.write(f"Page {source['page_number']} - {source['source']}")
                
                # st.session_state.messages.append({
                #     "role": "assistant",
                #     "content": data['answer']
                # })

                if response.status_code == 401:

                    st.error("Invalid API key. Please log out and try again.")

                elif response.status_code == 429:

                    st.error("Rate limit exceeded. Please try again later.")

                elif response.status_code != 200:

                    st.error("An error occurred while processing your request. Please try again.")

                else:

                    data = response.json()
                    st.markdown(data['answer'])
                    
                    with st.expander("Sources"):

                        for source in data['sources']:

                            st.write(f"Page {source['page_number']} - {source['source']}")
                    
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": data['answer']
                    })

            # except RateLimitError as e:

            #     st.error("Rate limit exceeded. Please try again later.")

            # except APIError as e:

            #     st.error("An error occurred while processing your request.")

            except httpx.RequestError as e:

                logger.error(f"HTTP request error: {e}")

                st.error("Could not connect to the service. Please try again later.")

            
            except httpx.TimeoutException as e:

                logger.error(f"HTTP timeout error: {e}")

                st.error("Request timed out. Please try again.")

            except Exception as e:

                logger.error(f"Error processing request: {e}")

                st.error("Something went wrong and service couldn't be reached. Please try again.")
