"""
app.py
------
The Streamlit UI. Run this with:  streamlit run app.py

Streamlit re-runs this whole script top-to-bottom every time the user interacts
with the page. That's why we store things like chat history and the chatbot
object in `st.session_state` -- it's the one thing that survives between reruns.
"""

import os
import tempfile

import streamlit as st
from dotenv import load_dotenv

from chatbot import RAGChatbot

load_dotenv()  # reads GROQ_API_KEY from a local .env file, if present

st.set_page_config(page_title="RAG Chatbot", page_icon="💬")
st.title("💬 RAG Chatbot")
st.caption("Chat normally, or upload a document and ask questions about it.")

# ----------------------------------------------------------------------
# Session state setup (runs once per browser session)
# ----------------------------------------------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []  # chat history shown on screen + sent to the API

if "chatbot" not in st.session_state:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        st.error("No GROQ_API_KEY found. Add it to a .env file (see .env.example).")
        st.stop()
    st.session_state.chatbot = RAGChatbot(api_key=api_key)

# ----------------------------------------------------------------------
# Sidebar: file upload for RAG
# ----------------------------------------------------------------------

with st.sidebar:
    st.header("📄 Document (optional)")
    uploaded_file = st.file_uploader("Upload a .txt or .pdf", type=["txt", "pdf"])

    if uploaded_file is not None:
        # Chunking/reading logic expects a real file path, so we save the
        # upload to a temporary file first.
        suffix = os.path.splitext(uploaded_file.name)[1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name

        if st.button("Ingest document"):
            with st.spinner("Reading and indexing document..."):
                num_chunks = st.session_state.chatbot.ingest_file(tmp_path)
            st.success(f"Indexed {num_chunks} chunks from {uploaded_file.name}")

    if st.session_state.chatbot.has_document:
        st.info("Answering using your uploaded document.")

    if st.button("Clear chat"):
        st.session_state.messages = []
        st.rerun()

# ----------------------------------------------------------------------
# Main chat area
# ----------------------------------------------------------------------

# Replay the existing conversation on every rerun
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

user_input = st.chat_input("Ask me anything...")

if user_input:
    # Show and store the user's message
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Generate and show the assistant's reply
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            reply = st.session_state.chatbot.generate_response(
                user_input,
                st.session_state.messages[:-1],  # history *before* this new message
            )
        st.markdown(reply)

    st.session_state.messages.append({"role": "assistant", "content": reply})
