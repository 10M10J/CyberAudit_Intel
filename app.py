# app.py
import os
import io
import json
import hashlib
from typing import List, Tuple

import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader

# Mistral SDK
from mistralai import Mistral

# FAISS for vector search
import faiss
import numpy as np


# =========================
# -------- Utilities -------
# =========================

def read_pdf_text(uploaded_pdf) -> str:
    """Extract text safely from a PDF file."""
    try:
        reader = PdfReader(uploaded_pdf)
        pages = [page.extract_text() or "" for page in reader.pages]
        text = "\n".join(pages)
        # light cleanup
        text = " \n".join(line.strip() for line in text.splitlines() if line.strip())
        return text
    except Exception as e:
        raise RuntimeError(f"Failed to read PDF: {e}")


def chunk_text(text: str, chunk_size: int = 1200, overlap: int = 200) -> List[str]:
    """
    Splits text into overlapping chunks of roughly 'chunk_size' words.
    """
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
    return chunks


def sanitize_text(s: str) -> str:
    """
    Basic defense against prompt injection by stripping a few common patterns.
    (You should expand this for production.)
    """
    patterns = [
        "ignore all previous instructions",
        "disregard previous",
        "system prompt:",
        "you are chatgpt",
        "developer instructions:"
    ]
    lowered = s.lower()
    for p in patterns:
        lowered = lowered.replace(p, "")
    return lowered


def words_cap(s: str, max_words: int) -> str:
    """Hard-cap the output to a maximum word count as a safeguard."""
    tokens = s.split()
    if len(tokens) <= max_words:
        return s
    return " ".join(tokens[:max_words])


def ensure_store_dir() -> str:
    store_dir = "vector_store"
    os.makedirs(store_dir, exist_ok=True)
    return store_dir


def content_hash(text: str) -> str:
    """Hash the entire document content to create a unique store key."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]


# =========================
# ---- Mistral Helpers ----
# =========================

def get_mistral_client() -> Mistral:
    load_dotenv()
    api_key = os.getenv("MISTRAL_API_KEY")
    if not api_key:
        raise RuntimeError("MISTRAL_API_KEY not found in environment.")
    return Mistral(api_key=api_key)


def embed_texts(_client: Mistral, texts: List[str], model: str = "mistral-embed", batch_size: int = 32) -> np.ndarray:
    """
    Embed a list of texts using Mistral embeddings. Returns a 2D numpy array.
    """
    embeddings: List[List[float]] = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        resp = _client.embeddings.create(model=model, inputs=batch)
        # SDK returns in the same order
        for d in resp.data:
            embeddings.append(d.embedding)
    return np.array(embeddings, dtype="float32")


def chat_complete(_client: Mistral, model: str, messages: List[dict], temperature: float = 0.2) -> str:
    """
    Call Mistral chat completion and return string content.
    """
    resp = _client.chat.complete(
        model=model,
        messages=messages,
        temperature=temperature
    )
    return resp.choices[0].message.content


# =========================
# ---- FAISS Utilities ----
# =========================

def build_faiss_index(vectors: np.ndarray) -> faiss.IndexFlatIP:
    """
    Build an Inner Product index; we will L2-normalize vectors so this becomes cosine similarity.
    """
    # Normalize to unit length for cosine similarity
    faiss.normalize_L2(vectors)
    dim = vectors.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(vectors)
    return index


def save_faiss_index(index: faiss.IndexFlatIP, path: str):
    faiss.write_index(index, path)


def load_faiss_index(path: str) -> faiss.IndexFlatIP:
    return faiss.read_index(path)


def retrieve(index: faiss.IndexFlatIP, query_vec: np.ndarray, top_k: int = 5) -> Tuple[np.ndarray, np.ndarray]:
    """
    query_vec: shape (1, dim) already normalized
    Returns (scores, indices)
    """
    faiss.normalize_L2(query_vec)
    scores, ids = index.search(query_vec, top_k)
    return scores, ids


# =========================
# ----- RAG Pipeline ------
# =========================

def build_or_load_rag_store(_client: Mistral, all_chunks: List[str], store_key: str) -> Tuple[faiss.IndexFlatIP, List[str]]:
    """
    Create or load a FAISS index and the associated chunk store.
    Uses content hash to persist per-document version.
    """
    store_dir = ensure_store_dir()
    doc_dir = os.path.join(store_dir, store_key)
    os.makedirs(doc_dir, exist_ok=True)

    index_path = os.path.join(doc_dir, "index.faiss")
    meta_path = os.path.join(doc_dir, "chunks.json")

    if os.path.exists(index_path) and os.path.exists(meta_path):
        # Load
        index = load_faiss_index(index_path)
        with open(meta_path, "r", encoding="utf-8") as f:
            chunks = json.load(f)
        return index, chunks

    # Build fresh
    # Sanitize chunks before embedding
    sanitized_chunks = [sanitize_text(c) for c in all_chunks]
    vectors = embed_texts(_client, sanitized_chunks, model="mistral-embed")
    index = build_faiss_index(vectors)
    save_faiss_index(index, index_path)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(sanitized_chunks, f, ensure_ascii=False)

    return index, sanitized_chunks


def retrieve_context(_client: Mistral, index: faiss.IndexFlatIP, chunks: List[str], question: str, top_k: int = 5) -> str:
    """Embed the question, retrieve top chunks, and return concatenated context."""
    q_vec = embed_texts(_client, [question], model="mistral-embed")
    scores, ids = retrieve(index, q_vec, top_k=top_k)
    # Flatten ids and collect chunks (filter -1 just in case)
    got = []
    for i in ids[0]:
        if 0 <= i < len(chunks):
            got.append(chunks[i])
    # Keep context to a safe size
    return "\n\n---\n\n".join(got)


# =========================
# ---- Prompt Builders ----
# =========================

def build_summary_prompt(context: str, language: str) -> List[dict]:
    lang_line = "English" if language == "English" else "Hindi"
    system = (
        f"You are a cybersecurity audit assistant. Summarize the provided content in {lang_line}. "
        "Be faithful to the source. Use plain language. "
        "STRICT LIMIT: maximum 250 words."
    )
    user = f"Document content:\n\n{context}\n\nSummarize as requested."
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def build_qa_prompt(question: str, context: str, language: str) -> List[dict]:
    lang_line = "English" if language == "English" else "Hindi"
    system = (
        f"You are a concise cybersecurity audit assistant. Answer only using the given context. "
        f"Reply in {lang_line}. If the answer is not in the context, say you don't have enough information. "
        "STRICT LIMIT: maximum 150 words."
    )
    user = (
        f"Context (from the audit report):\n{context}\n\n"
        f"Question: {question}\n\n"
        "Answer strictly from the context."
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


# =========================
# --------- UI ------------
# =========================

def main():
    st.set_page_config(page_title="CyberAudit Intel (RAG + Mistral)", page_icon="🛡️", layout="wide")

    # ---- Sidebar ----
    st.sidebar.title("CyberAudit Intel")
    st.sidebar.caption("RAG • FAISS • Mistral")

    # Model controls
    model_name = st.sidebar.selectbox(
        "Mistral chat model",
        ("mistral-small-latest", "mistral-small-2503"),
        index=0
    )
    temperature = 0.2  # fixed as requested
    language = st.sidebar.radio("Language", ("English", "Hindi"), horizontal=True, index=0)

    mode = st.sidebar.radio("Mode", ("Summary", "Chat"), index=0)

    # Clear buttons (affect session_state)
    if st.sidebar.button("Clear ALL"):
        for key in ("summary_text", "chat_history", "rag_store_key", "rag_index", "rag_chunks", "doc_hash", "full_text"):
            st.session_state.pop(key, None)
        st.sidebar.success("Cleared all state.")

    # ---- Body ----
    st.header("Chat with your Cybersecurity Audit Report")

    uploaded_pdf = st.file_uploader("Upload your Audit Report (PDF only)", type=["pdf"])

    # Exit early if no file
    if not uploaded_pdf:
        st.info("Upload a PDF to get started.")
        return

    # Load text
    try:
        full_text = read_pdf_text(uploaded_pdf)
    except Exception as e:
        st.error(str(e))
        return

    if not full_text.strip():
        st.error("No extractable text found in the PDF.")
        return

    # Compute content hash & build/load RAG store once per document content
    doc_key = content_hash(full_text)

    # Rebuild RAG store if new doc uploaded (by content)
    if st.session_state.get("doc_hash") != doc_key:
        # Reset state when new doc is detected
        for key in ("summary_text", "chat_history", "rag_store_key", "rag_index", "rag_chunks"):
            st.session_state.pop(key, None)
        st.session_state["doc_hash"] = doc_key
        st.session_state["full_text"] = full_text

    # Build chunks (kept small for better retrieval granularity)
    chunks = chunk_text(st.session_state["full_text"], chunk_size=2500, overlap=300)

    # Prepare (or load) FAISS index + chunks (persisted)
    # We keep these in session_state to avoid hashing issues with Streamlit caching.
    if "rag_index" not in st.session_state or "rag_chunks" not in st.session_state:
        with st.spinner("Preparing RAG store..."):
            try:
                client = get_mistral_client()
                index, stored_chunks = build_or_load_rag_store(client, chunks, store_key=doc_key)
                st.session_state["rag_index"] = index
                st.session_state["rag_chunks"] = stored_chunks
                st.session_state["rag_store_key"] = doc_key
            except Exception as e:
                st.error(f"Failed to prepare RAG store: {e}")
                return

    # UI per mode
    if mode == "Summary":
        st.subheader("Summary (≈150 words)")
        col1, col2 = st.columns([1, 1], gap="large")
        with col1:
            if st.button("Generate Summary"):
                with st.spinner("Summarizing..."):
                    try:
                        # For summary, we can summarize over the first N chunks concatenated (or all if short)
                        # Keep context size reasonable
                        context = "\n\n".join(st.session_state["rag_chunks"][:8])
                        client = get_mistral_client()
                        messages = build_summary_prompt(context=context, language=language)
                        out = chat_complete(client, model=model_name, messages=messages, temperature=temperature)
                        out = words_cap(out, 160)  # hard cap ~150-160 words
                        st.session_state["summary_text"] = out
                    except Exception as e:
                        st.error(f"Error during summarization: {e}")

        with col2:
            if st.button("Clear Summary"):
                st.session_state.pop("summary_text", None)
                st.success("Summary cleared.")

        if "summary_text" in st.session_state:
            st.write(st.session_state["summary_text"])

    else:  # Chat mode
        st.subheader("Chat (answers ≈100 words)")
        if "chat_history" not in st.session_state:
            st.session_state["chat_history"] = []

        # Render chat history
        for turn in st.session_state["chat_history"]:
            st.chat_message("user").write(turn["q"])
            st.chat_message("assistant").write(turn["a"])

        # Clear chat
        if st.button("Clear Chat History"):
            st.session_state["chat_history"] = []
            st.success("Chat history cleared.")

        # Chat input
        user_q = st.chat_input("Ask a question about the report")
        if user_q:
            with st.spinner("Retrieving and answering..."):
                try:
                    client = get_mistral_client()
                    # Retrieve top-k relevant chunks for the question
                    context = retrieve_context(
                        client,
                        st.session_state["rag_index"],
                        st.session_state["rag_chunks"],
                        user_q,
                        top_k=5
                    )
                    messages = build_qa_prompt(question=user_q, context=context, language=language)
                    answer = chat_complete(client, model=model_name, messages=messages, temperature=temperature)
                    answer = words_cap(answer, 110)  # hard-cap ~100-110 words
                    # Save & render
                    st.session_state["chat_history"].append({"q": user_q, "a": answer})
                    st.chat_message("user").write(user_q)
                    st.chat_message("assistant").write(answer)
                except Exception as e:
                    st.error(f"Error answering question: {e}")


if __name__ == "__main__":
    main()
