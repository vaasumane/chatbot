# RAG Chatbot

A chatbot built with Python, [Streamlit](https://streamlit.io/), and the
free [Groq API](https://console.groq.com/) (running Llama 3.1 open-source
models). Chat normally, or upload a `.txt`/`.pdf` file and ask questions
about its contents — the bot retrieves the most relevant chunks of the
document and uses them to ground its answer (this pattern is called
**Retrieval-Augmented Generation**, or RAG).

## How it works

1. **Ingest**: an uploaded document is split into overlapping text chunks.
2. **Embed & store**: each chunk is embedded and saved in a local vector
   database ([Chroma](https://www.trychroma.com/)).
3. **Retrieve**: when you ask a question, the chunks most similar to your
   question are pulled from the database.
4. **Generate**: those chunks are inserted into the system prompt, and
   Llama (via Groq) generates an answer grounded in your document.

If no document has been uploaded, the bot just chats normally.

## Setup

```bash
# 1. Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Add your API key (free — no credit card required)
cp .env.example .env
# then open .env and paste in your key from https://console.groq.com/keys

# 4. Run the app
streamlit run app.py
```

The app will open at `http://localhost:8501`.

## Deploying it live (free)

1. Push this project to a public GitHub repo.
2. Go to [share.streamlit.io](https://share.streamlit.io), sign in with GitHub,
   and deploy the repo (`app.py` as the entry point).
3. In the app's settings, add `GROQ_API_KEY` under "Secrets" —
   this replaces your local `.env` file in production.

## Project structure

```
├── app.py           # Streamlit UI
├── chatbot.py        # RAG logic: ingestion, retrieval, generation
├── requirements.txt   # Dependencies
├── .env.example        # Template for your API key
└── README.md
```

## What I learned building this

This was my first hands-on project in Python, coming from a PHP/Laravel and
React.js background — so a lot of the learning here was about how backend
concepts I already knew (API integration, request/response handling, state
management, separating logic from presentation) translate into a different
language and ecosystem.

- **Retrieval-Augmented Generation (RAG)**: how splitting a document into
  overlapping chunks, embedding them, and doing similarity search lets an LLM
  answer questions grounded in a specific document instead of only its
  training data.
- **Vector databases**: how Chroma stores embeddings and retrieves the most
  relevant chunks for a query — conceptually similar to indexing a MySQL
  table for fast lookups, just over meaning instead of exact values.
- **Chunk size trade-offs**: too small and you lose context; too large and
  irrelevant text dilutes the retrieved match. This is a tuning problem, not
  a one-time setting.
- **Streamlit's rerun model**: the entire script re-runs on every user
  interaction, so anything that needs to persist (chat history, the chatbot
  instance) has to live in `st.session_state` — a different mental model
  from PHP's request/response lifecycle I was used to.
- **Structuring a Python project**: separating UI (`app.py`) from core logic
  (`chatbot.py`) so the RAG pipeline could be tested or reused independently
  of the Streamlit interface — the same separation-of-concerns instinct I
  apply in Laravel with controllers vs. services.

## Possible next steps

- Swap the chunker for a sentence-aware splitter (e.g. via `nltk` or `spaCy`)
- Show which document chunks were used to answer each question (source citations)
- Support multiple documents at once
- Add streaming responses instead of waiting for the full reply
