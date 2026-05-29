"""
Free AI Chatbot — General Chat + RAG (Document Q&A)
Uses Ollama (local AI, no API key needed) + Streamlit UI
"""

import os
import glob
import streamlit as st
import ollama
import chromadb
from chromadb.utils import embedding_functions

# ── Config ────────────────────────────────────────────────────────────────────

DOCS_DIR   = "sample_docs"
DB_DIR     = "chroma_db"
COLLECTION = "my_documents"
TOP_K      = 3
MODEL      = "llama3.2"

# ── Page Setup ────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Free AI Chatbot",
    page_icon="🤖",
    layout="centered"
)

st.title("🤖 Free AI Chatbot")
st.caption("General chat + Document Q&A — powered by Llama 3.2, runs 100% free")

# ── Vector DB ─────────────────────────────────────────────────────────────────

@st.cache_resource
def get_collection():
    client = chromadb.PersistentClient(path=DB_DIR)
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )
    return client.get_or_create_collection(name=COLLECTION, embedding_function=ef)


def ingest_documents():
    files = glob.glob(os.path.join(DOCS_DIR, "*.txt"))
    if not files:
        return 0
    collection = get_collection()
    existing = collection.get()
    if existing["ids"]:
        collection.delete(ids=existing["ids"])
    all_chunks, all_ids, all_metas = [], [], []
    for path in files:
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
        chunk_size, overlap = 500, 100
        start, i = 0, 0
        filename = os.path.basename(path)
        while start < len(text):
            chunk = text[start:start+chunk_size].strip()
            if len(chunk) > 50:
                all_chunks.append(chunk)
                all_ids.append(f"{filename}__chunk_{i}")
                all_metas.append({"source": filename})
                i += 1
            start += chunk_size - overlap
    collection.add(documents=all_chunks, ids=all_ids, metadatas=all_metas)
    return len(all_chunks)


def retrieve(query):
    collection = get_collection()
    if collection.count() == 0:
        return []
    results = collection.query(query_texts=[query], n_results=TOP_K)
    return [{"text": d, "source": m["source"]}
            for d, m in zip(results["documents"][0], results["metadatas"][0])]


# ── Ollama Chat ───────────────────────────────────────────────────────────────

def ask_ollama(question, context_chunks=None, history=None):
    if context_chunks:
        context_text = "\n\n---\n\n".join(
            f"[Source: {c['source']}]\n{c['text']}" for c in context_chunks
        )
        system = f"""You are a helpful assistant. Answer strictly based on the context below.
If the answer is not in the context, say: I don't have enough information in the documents to answer that.

Context:
{context_text}"""
    else:
        system = """You are a helpful, friendly AI assistant like ChatGPT.
Answer clearly and helpfully. You can help with coding, writing, math, general knowledge, and more."""

    messages = [{"role": "system", "content": system}]
    if history:
        for msg in history[-6:]:
            messages.append({"role": msg["role"], "content": msg["content"]})
    messages.append({"role": "user", "content": question})

    response = ollama.chat(model=MODEL, messages=messages)
    return response["message"]["content"]


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.header("⚙️ Settings")

    mode = st.radio(
        "Chat mode:",
        ["💬 General Chat", "📄 Document Q&A (RAG)"],
        index=0
    )

    st.divider()

    if "RAG" in mode:
        st.header("📂 Documents")
        collection = get_collection()
        count = collection.count()
        if count > 0:
            st.success(f"✅ {count} chunks indexed")
        else:
            st.warning("No documents indexed yet")

        if st.button("🔄 Load Documents", use_container_width=True):
            with st.spinner("Ingesting documents..."):
                n = ingest_documents()
            if n > 0:
                st.success(f"✅ {n} chunks loaded!")
                st.rerun()
            else:
                st.error("No .txt files found in sample_docs/")

        st.divider()
        st.markdown("**Sample questions:**")
        sample_questions = [
            "What is RAG and how does it work?",
            "What is Acme Corp's refund policy?",
            "What are embeddings?",
            "How does fine-tuning differ from RAG?",
        ]
        for q in sample_questions:
            if st.button(q, use_container_width=True, key=q):
                st.session_state.sample_q = q
    else:
        st.markdown("**Try asking:**")
        general_questions = [
            "Explain machine learning simply",
            "Write a Python function to sort a list",
            "What is the difference between AI and ML?",
            "How do I reverse a string in Python?",
        ]
        for q in general_questions:
            if st.button(q, use_container_width=True, key=q):
                st.session_state.sample_q = q

    st.divider()
    st.markdown("**Model info:**")
    st.info("🦙 Llama 3.2 (local)\n\n✅ Free forever\n✅ No API key\n✅ Works offline")

    if st.button("🗑️ Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# ── Chat History ──────────────────────────────────────────────────────────────

if "messages" not in st.session_state:
    st.session_state.messages = []

if "mode" not in st.session_state:
    st.session_state.mode = mode

if st.session_state.mode != mode:
    st.session_state.messages = []
    st.session_state.mode = mode

if not st.session_state.messages:
    if "RAG" in mode:
        st.info("📄 Document Q&A mode — Click 'Load Documents' in the sidebar first!")
    else:
        st.info("💬 General chat mode — Ask me anything!")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            st.caption(f"📄 Sources: {msg['sources']}")

# ── Chat Input ────────────────────────────────────────────────────────────────

prompt = st.chat_input("Type your message here...")

if "sample_q" in st.session_state and st.session_state.sample_q:
    prompt = st.session_state.sample_q
    st.session_state.sample_q = None

if prompt:
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            sources = ""
            if "RAG" in mode:
                chunks = retrieve(prompt)
                if not chunks:
                    answer = "⚠️ No documents loaded yet. Click **'Load Documents'** in the sidebar first!"
                else:
                    answer = ask_ollama(prompt, context_chunks=chunks, history=st.session_state.messages)
                    sources = ", ".join({c["source"] for c in chunks})
            else:
                answer = ask_ollama(prompt, history=st.session_state.messages)

        st.markdown(answer)
        if sources:
            st.caption(f"📄 Sources: {sources}")

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": sources
    })