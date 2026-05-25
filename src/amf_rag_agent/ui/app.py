import streamlit as st
import requests
import os

API_URL = os.getenv("API_URL", "http://localhost:8000")


st.title("AMF RAG AGENT Chat Interface")

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

                
                data = requests.post(f"{API_URL}/ask", json={"question": prompt}).json()
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
