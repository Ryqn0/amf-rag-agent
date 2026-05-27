import streamlit as st
from amf_rag_agent.agent.loop import run_agent, tools
from amf_rag_agent.retrieval.store import load_all_chunks
from amf_rag_agent.retrieval.bm25_store import build_bm25_index
import asyncio
from amf_rag_agent.config import setup_logging


setup_logging()
chunks = load_all_chunks()
build_bm25_index(chunks)


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

                
                data = asyncio.run(run_agent(prompt, tools))
                st.session_state.messages.append({"role": "user", "content": prompt})
                st.markdown(data['answer'])
                
                with st.expander("Sources"):

                    for source in data['sources']:

                        st.write(f"Page {source['page_number']} - {source['source']}")
                
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": data['answer']
                })

            except Exception as e:

                st.error("Something went wrong - please try again")
