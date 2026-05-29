import os
import glob
import streamlit as st
from groq import Groq

# ── Config ────────────────────────────────────────────────────────────────────

DOCS_DIR = "sample_docs"
MODEL    = "llama-3.2-3b-preview"
GROQ_KEY = os.environ.get("GROQ_API_KEY", "")

# ── Page Setup ────────────────────────────────────────────────────────────────

st.set_page_config(page_title="Free AI Chatbot", page_icon="🤖", layout="centered")
st.title("🤖 Free AI Chatbot")
st.caption("General chat + Document Q&A — powered by Groq + Llama 3.2, 100% free")

if not GROQ_KEY:
    st.error("GROQ_API_KEY not set. Add it in Streamlit Cloud Secrets.")
    st.stop()

client = Groq(api_key=GROQ_KEY)

# ── Simple RAG (no ChromaDB needed) ──────────────────────────────────────────

@st.cache_data
def load_documents():
    files = glob.glob(os.path.join(DOCS_DIR, "*.txt"))
    docs = []
    for path in files:
        with open(path, "r", encoding="utf-8") as f:
            docs.append({"filename": os.path.basename(path), "text": f.read()})
    return docs

def simple_retrieve(query, docs, top_k=2):
    query_words = set(query.lower().split())
    scored = []
    for doc in docs:
        chunks = [doc["text"][i:i+500] for i in range(0, len(doc["text"]), 400)]
        for chunk in chunks:
            chunk_words = set(chunk.lower().split())
            score = len(query_words & chunk_words)
            scored.append({"text": chunk, "source": doc["filename"], "score": score})
    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:top_k]

def ask_groq(question, context_chunks=None, history=None):
    if context_chunks:
        context_text = "\n\n---\n\n".join(
            f"[Source: {c['source']}]\n{c['text']}" for c in context_chunks
        )
        system = f"""You are a helpful assistant. Answer strictly based on the context below.
If the answer is not in the context, say: I don't have enough information in the documents to answer that.

Context:
{context_text}"""
    else:
        system = "You are a helpful, friendly AI assistant. Answer clearly and helpfully."

    messages = [{"role": "system", "content": system}]
    if history:
        for msg in history[-6:]:
            messages.append({"role": msg["role"], "content": msg["content"]})
    messages.append({"role": "user", "content": question})

    response = client.chat.completions.create(model=MODEL, messages=messages, max_tokens=1024)
    return response.choices[0].message.content

# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.header("⚙️ Settings")
    mode = st.radio("Chat mode:", ["💬 General Chat", "📄 Document Q&A (RAG)"], index=0)
    st.divider()

    if "RAG" in mode:
        docs = load_documents()
        if docs:
            st.success(f"✅ {len(docs)} document(s) loaded")
            for d in docs:
                st.caption(f"📄 {d['filename']}")
        else:
            st.warning("No .txt files found in sample_docs/")
        st.divider()
        st.markdown("**Sample questions:**")
        for q in ["What is RAG and how does it work?",
                   "What is Acme Corp's refund policy?",
                   "What are embeddings?"]:
            if st.button(q, use_container_width=True, key=q):
                st.session_state.sample_q = q
    else:
        st.markdown("**Try asking:**")
        for q in ["Explain machine learning simply",
                   "Write a Python hello world",
                   "What is the difference between AI and ML?"]:
            if st.button(q, use_container_width=True, key=q):
                st.session_state.sample_q = q

    st.divider()
    st.info("⚡ Groq + Llama 3.2\n\n✅ Free forever\n✅ Super fast")
    if st.button("🗑️ Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# ── Chat ──────────────────────────────────────────────────────────────────────

if "messages" not in st.session_state:
    st.session_state.messages = []
if "mode" not in st.session_state:
    st.session_state.mode = mode
if st.session_state.mode != mode:
    st.session_state.messages = []
    st.session_state.mode = mode

if not st.session_state.messages:
    if "RAG" in mode:
        st.info("📄 Document Q&A mode — ask anything about your documents!")
    else:
        st.info("💬 General chat mode — ask me anything!")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            st.caption(f"📄 Sources: {msg['sources']}")

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
                docs = load_documents()
                chunks = simple_retrieve(prompt, docs)
                answer = ask_groq(prompt, context_chunks=chunks, history=st.session_state.messages)
                sources = ", ".join({c["source"] for c in chunks})
            else:
                answer = ask_groq(prompt, history=st.session_state.messages)
        st.markdown(answer)
        if sources:
            st.caption(f"📄 Sources: {sources}")

    st.session_state.messages.append({"role": "assistant", "content": answer, "sources": sources})
