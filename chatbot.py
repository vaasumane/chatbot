"""
chatbot.py
----------
This file holds all the "brains" of the chatbot, separate from the UI code in app.py.
Splitting code like this (UI vs. logic) is a Python best practice that also happens to
make your project read as more "engineered" to anyone reviewing your GitHub.

Concepts you'll see used here (good things to be able to explain in an interview):
- Classes and methods (bundling related state + behavior together)
- Type hints (List[str], Optional[str]) for readability
- List comprehensions
- Working with an external API (Groq)
- A vector database (Chroma) for retrieval-augmented generation (RAG)
"""

import os
from typing import List, Optional

from groq import Groq
import chromadb
from chromadb.utils import embedding_functions
from pypdf import PdfReader


class RAGChatbot:
    """
    A chatbot that can optionally answer questions using content from an uploaded
    document (RAG = Retrieval-Augmented Generation), or fall back to a normal
    chat if no document has been uploaded.
    """

    def __init__(self, api_key: Optional[str] = None):
        # If no key is passed in, the Groq client will look for the
        # GROQ_API_KEY environment variable automatically.
        self.client = Groq(api_key=api_key)
        # Groq's free tier hosts several open-source models. gpt-oss-20b is
        # fast and has generous free rate limits, making it a good fit for a
        # public demo. (Note: Groq periodically retires older model IDs —
        # check https://console.groq.com/docs/models for the current list
        # if this ever stops working.)
        self.model = "openai/gpt-oss-20b"

        # Chroma is a lightweight local vector database. We use its default
        # built-in embedding function so you don't need to download a separate
        # embedding model or manage embeddings by hand.
        self.chroma_client = chromadb.Client()
        self.embedding_fn = embedding_functions.DefaultEmbeddingFunction()
        self.collection = self.chroma_client.get_or_create_collection(
            name="documents",
            embedding_function=self.embedding_fn,
        )
        self.has_document = False

    # ------------------------------------------------------------------
    # Ingestion: turn an uploaded file into searchable chunks
    # ------------------------------------------------------------------

    def ingest_file(self, file_path: str) -> int:
        """
        Reads a .txt or .pdf file, splits it into chunks, and stores those
        chunks in the vector database. Returns the number of chunks created.
        """
        text = self._extract_text(file_path)
        chunks = self._chunk_text(text)

        # Clear out any previous document so searches don't mix documents.
        self.chroma_client.delete_collection("documents")
        self.collection = self.chroma_client.get_or_create_collection(
            name="documents",
            embedding_function=self.embedding_fn,
        )

        # Chroma needs a unique string ID for every chunk we add.
        ids = [f"chunk_{i}" for i in range(len(chunks))]
        self.collection.add(documents=chunks, ids=ids)

        self.has_document = True
        return len(chunks)

    def _extract_text(self, file_path: str) -> str:
        """Reads raw text out of a .txt or .pdf file."""
        if file_path.lower().endswith(".pdf"):
            reader = PdfReader(file_path)
            return "\n".join(page.extract_text() or "" for page in reader.pages)
        else:
            with open(file_path, "r", encoding="utf-8") as f:
                return f.read()

    def _chunk_text(self, text: str, chunk_size: int = 800, overlap: int = 100) -> List[str]:
        """
        Splits text into overlapping chunks. Overlap helps avoid losing context
        that falls right on a chunk boundary. This is a simple, readable chunker --
        good enough for a portfolio project; production systems often chunk by
        sentence or paragraph instead of raw character count.
        """
        chunks = []
        start = 0
        while start < len(text):
            end = start + chunk_size
            chunks.append(text[start:end])
            start += chunk_size - overlap
        return [c.strip() for c in chunks if c.strip()]

    # ------------------------------------------------------------------
    # Retrieval: find the most relevant chunks for a question
    # ------------------------------------------------------------------

    def retrieve(self, query: str, n_results: int = 3) -> List[str]:
        """Returns the most relevant chunks of the uploaded document for a query."""
        if not self.has_document:
            return []
        results = self.collection.query(
            query_texts=[query], n_results=n_results)
        return results["documents"][0] if results["documents"] else []

    # ------------------------------------------------------------------
    # Generation: ask Claude, optionally grounded in retrieved context
    # ------------------------------------------------------------------

    def generate_response(self, query: str, history: List[dict]) -> str:
        """
        Builds a prompt (with retrieved context if we have a document) and
        calls the Claude API. `history` is the running list of prior
        {"role": ..., "content": ...} messages, which gives the bot memory.
        """
        context_chunks = self.retrieve(query)

        if context_chunks:
            context_text = "\n\n---\n\n".join(context_chunks)
            system_prompt = (
                "You are a helpful assistant. Answer the user's question using "
                "the context below when it's relevant. If the context doesn't "
                "contain the answer, say so and answer from general knowledge.\n\n"
                f"Context:\n{context_text}"
            )
        else:
            system_prompt = "You are a helpful, friendly assistant."

        # Groq's API follows the OpenAI-style format: the system prompt is
        # just another message in the list, rather than a separate parameter.
        messages = (
            [{"role": "system", "content": system_prompt}]
            + history
            + [{"role": "user", "content": query}]
        )

        response = self.client.chat.completions.create(
            model=self.model,
            max_tokens=1024,
            messages=messages,
        )
        return response.choices[0].message.content
