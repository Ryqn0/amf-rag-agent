import streamlit as st
import requests


st.title("AMF RAG AGENT Chat Interface")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("What is up?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Searching documents..."):

            response = requests.post("http://localhost:8000/ask", json={"question": prompt})
            # print(response.status_code)  # debug
            # print(response.text)         # debug
            data = response.json()
            st.markdown(data['answer'])
            
            with st.expander("Sources"):
                for source in data['sources']:
                    st.write(f"Page {source['page_number']} - {source['source']}")
            
            st.session_state.messages.append({
                "role": "assistant",
                "content": data['answer']
            })
