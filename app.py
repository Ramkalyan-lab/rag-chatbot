import os
import glob
import zipfile
import io
import streamlit as st
from groq import Groq
import base64
import tempfile

# ── Config ────────────────────────────────────────────────────────────────────

DOCS_DIR = "sample_docs"
MODEL    = "llama-3.3-70b-versatile"
GROQ_KEY = os.environ.get("GROQ_API_KEY", "")

# ── Page Setup ────────────────────────────────────────────────────────────────

st.set_page_config(page_title="Free AI Chatbot", page_icon="robot", layout="wide")

st.markdown("""
<style>
    .stApp { background: linear-gradient(135deg, #0e1117 0%, #1a1f2e 100%); }
    .chat-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem 2rem;
        border-radius: 16px;
        margin-bottom: 1.5rem;
        text-align: center;
    }
    .chat-header h1 { color: white; font-size: 2rem; margin: 0; }
    .chat-header p { color: rgba(255,255,255,0.8); margin: 0.25rem 0 0; font-size: 0.95rem; }
    .feature-card {
        background: rgba(255,255,255,0.05);
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 12px;
        padding: 0.75rem 1rem;
        margin-bottom: 0.5rem;
        color: white;
        font-size: 0.85rem;
    }
    .source-badge {
        background: rgba(102,126,234,0.2);
        border: 1px solid rgba(102,126,234,0.4);
        border-radius: 8px;
        padding: 4px 10px;
        font-size: 0.75rem;
        color: #a0aec0;
    }
    div[data-testid="stSidebar"] { background: #1a1f2e !important; }
</style>
""", unsafe_allow_html=True)

# ── API Key Check ─────────────────────────────────────────────────────────────

if not GROQ_KEY:
    st.error("GROQ_API_KEY not set. Add it in Streamlit Secrets.")
    st.stop()

client = Groq(api_key=GROQ_KEY)

# ── Header ────────────────────────────────────────────────────────────────────

st.markdown("""
<div class="chat-header">
    <h1>Free AI Chatbot</h1>
    <p>General Chat + Document Q&A + Image Analysis | Powered by Groq + Llama 3.3 | 100% Free</p>
</div>
""", unsafe_allow_html=True)

# ── Document Extraction ───────────────────────────────────────────────────────

def extract_text_from_pdf(file_bytes):
    try:
        import pypdf
        reader = pypdf.PdfReader(io.BytesIO(file_bytes))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    except Exception as e:
        return f"[PDF read error: {e}]"

def extract_text_from_docx(file_bytes):
    try:
        import docx
        doc = docx.Document(io.BytesIO(file_bytes))
        return "\n".join(p.text for p in doc.paragraphs if p.text.strip())
    except Exception as e:
        return f"[DOCX read error: {e}]"

def extract_from_zip(file_bytes):
    docs = []
    try:
        with zipfile.ZipFile(io.BytesIO(file_bytes)) as z:
            for name in z.namelist():
                ext = name.lower().split(".")[-1]
                with z.open(name) as f:
                    content = f.read()
                if ext == "txt":
                    docs.append({"filename": name, "text": content.decode("utf-8", errors="ignore")})
                elif ext == "pdf":
                    docs.append({"filename": name, "text": extract_text_from_pdf(content)})
                elif ext == "docx":
                    docs.append({"filename": name, "text": extract_text_from_docx(content)})
    except Exception as e:
        docs.append({"filename": "zip_error.txt", "text": f"ZIP read error: {e}"})
    return docs

def process_uploaded_files(uploaded_files):
    docs = []
    for f in uploaded_files:
        name = f.name.lower()
        content = f.read()
        if name.endswith(".txt"):
            docs.append({"filename": f.name, "text": content.decode("utf-8", errors="ignore")})
        elif name.endswith(".pdf"):
            docs.append({"filename": f.name, "text": extract_text_from_pdf(content)})
        elif name.endswith(".docx"):
            docs.append({"filename": f.name, "text": extract_text_from_docx(content)})
        elif name.endswith(".zip"):
            docs.extend(extract_from_zip(content))
    return docs

# ── RAG Helpers ───────────────────────────────────────────────────────────────

@st.cache_data
def load_default_documents():
    files = glob.glob(os.path.join(DOCS_DIR, "*.txt"))
    docs = []
    for path in files:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            docs.append({"filename": os.path.basename(path), "text": f.read()})
    return docs

def simple_retrieve(query, docs, top_k=3):
    query_words = set(query.lower().split())
    scored = []
    for doc in docs:
        chunks = [doc["text"][i:i+600] for i in range(0, len(doc["text"]), 450)]
        for chunk in chunks:
            chunk_words = set(chunk.lower().split())
            score = len(query_words & chunk_words)
            if score > 0:
                scored.append({"text": chunk, "source": doc["filename"], "score": score})
    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:top_k]

# ── Groq Chat ─────────────────────────────────────────────────────────────────

def ask_groq(question, context_chunks=None, history=None, image_b64=None, image_type=None):
    if context_chunks:
        context_text = "\n\n---\n\n".join(
            f"[Source: {c['source']}]\n{c['text']}" for c in context_chunks
        )
        system = f"""You are a helpful assistant. Answer based on the context below.
If the answer is not in the context, say so clearly.

Context:
{context_text}"""
    else:
        system = "You are a helpful, friendly AI assistant. Answer clearly and helpfully."

    if image_b64 and image_type:
        vision_messages = [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": f"data:{image_type};base64,{image_b64}"}},
            {"type": "text", "text": question or "Describe this image in detail."}
        ]}]
        response = client.chat.completions.create(
            model="llama-4-scout-17b-16e-instruct",
            messages=vision_messages,
            max_tokens=1024
        )
    else:
        messages = [{"role": "system", "content": system}]
        if history:
            for msg in history[-6:]:
                if isinstance(msg.get("content"), str):
                    messages.append({"role": msg["role"], "content": msg["content"]})
        messages.append({"role": "user", "content": question})
        response = client.chat.completions.create(model=MODEL, messages=messages, max_tokens=1024)

    return response.choices[0].message.content

def transcribe_audio(audio_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(audio_file.read())
        tmp_path = tmp.name
    with open(tmp_path, "rb") as f:
        transcription = client.audio.transcriptions.create(
            model="whisper-large-v3", file=f, response_format="text"
        )
    os.unlink(tmp_path)
    return transcription

# ── Session State ─────────────────────────────────────────────────────────────

if "messages" not in st.session_state:
    st.session_state.messages = []
if "mode" not in st.session_state:
    st.session_state.mode = "General Chat"
if "uploaded_docs" not in st.session_state:
    st.session_state.uploaded_docs = []

# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("### Settings")

    mode = st.radio(
        "Chat Mode",
        ["General Chat", "Document Q&A (RAG)", "Image Analysis"],
        index=["General Chat", "Document Q&A (RAG)", "Image Analysis"].index(
            st.session_state.mode) if st.session_state.mode in [
            "General Chat", "Document Q&A (RAG)", "Image Analysis"] else 0
    )

    if mode != st.session_state.mode:
        st.session_state.messages = []
        st.session_state.mode = mode
        st.rerun()

    st.divider()

    if mode == "Document Q&A (RAG)":
        st.markdown("### Upload Documents")
        st.caption("Supports TXT, PDF, DOCX, ZIP")
        uploaded_files = st.file_uploader(
            "Upload files",
            type=["txt", "pdf", "docx", "zip"],
            accept_multiple_files=True,
            help="Upload TXT, PDF, DOCX files or a ZIP containing multiple files"
        )
        if uploaded_files:
            with st.spinner("Reading files..."):
                st.session_state.uploaded_docs = process_uploaded_files(uploaded_files)
            st.success(f"{len(st.session_state.uploaded_docs)} document(s) loaded!")

        all_docs = load_default_documents() + st.session_state.uploaded_docs
        if all_docs:
            st.markdown("**Loaded documents:**")
            for d in all_docs:
                size = len(d["text"])
                st.markdown(
                    f"<div class='feature-card'>📄 {d['filename']}<br><small>{size:,} chars</small></div>",
                    unsafe_allow_html=True
                )

        st.divider()
        st.markdown("**Sample questions:**")
        for q in ["What is RAG and how does it work?",
                   "Summarise the main points of the document",
                   "What are the key findings?"]:
            if st.button(q, use_container_width=True, key=q):
                st.session_state.sample_q = q

    elif mode == "General Chat":
        st.markdown("**Try asking:**")
        for q in ["Explain machine learning simply",
                   "Write a Python hello world",
                   "What is the difference between AI and ML?",
                   "How do neural networks work?"]:
            if st.button(q, use_container_width=True, key=q):
                st.session_state.sample_q = q

    elif mode == "Image Analysis":
        st.markdown("**Try asking:**")
        for q in ["What is in this image?",
                   "Describe this image in detail",
                   "What text can you see?",
                   "What are the main colors?"]:
            if st.button(q, use_container_width=True, key=q):
                st.session_state.sample_q = q

    st.divider()

    st.markdown("### Voice Input")
    audio_file = st.file_uploader(
        "Upload audio (.wav, .mp3, .m4a)",
        type=["wav", "mp3", "m4a"],
        help="Upload a recorded voice message"
    )
    if audio_file:
        with st.spinner("Transcribing..."):
            try:
                transcribed = transcribe_audio(audio_file)
                st.success(f"Transcribed: {transcribed}")
                st.session_state.sample_q = transcribed
            except Exception as e:
                st.error(f"Transcription failed: {str(e)[:100]}")

    st.divider()
    st.markdown("""
    <div class="feature-card">
        Groq + Llama 3.3<br>
        Free forever | Super fast<br>
        PDF, DOCX, ZIP support
    </div>
    """, unsafe_allow_html=True)

    if st.button("Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# ── Main Area ─────────────────────────────────────────────────────────────────

uploaded_image = None
image_b64 = None
image_type = None

if mode == "Image Analysis":
    uploaded_image = st.file_uploader(
        "Upload an image",
        type=["jpg", "jpeg", "png", "gif", "webp"],
        key="image_upload"
    )
    if uploaded_image:
        st.image(uploaded_image, caption="Uploaded image", use_column_width=True)
        img_bytes = uploaded_image.read()
        image_b64 = base64.b64encode(img_bytes).decode("utf-8")
        image_type = uploaded_image.type

if not st.session_state.messages:
    if mode == "Document Q&A (RAG)":
        st.info("Document Q&A mode - Upload TXT, PDF, DOCX or ZIP files in the sidebar!")
    elif mode == "Image Analysis":
        st.info("Image Analysis mode - Upload an image above and ask anything about it!")
    else:
        st.info("General Chat mode - Ask me anything!")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            st.markdown(
                f"<span class='source-badge'>Sources: {msg['sources']}</span>",
                unsafe_allow_html=True
            )

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
            try:
                if mode == "Image Analysis":
                    if image_b64:
                        answer = ask_groq(prompt, image_b64=image_b64, image_type=image_type)
                    else:
                        answer = "Please upload an image first."

                elif mode == "Document Q&A (RAG)":
                    all_docs = load_default_documents() + st.session_state.uploaded_docs
                    if not all_docs:
                        answer = "No documents found. Upload files in the sidebar."
                    else:
                        chunks = simple_retrieve(prompt, all_docs)
                        if not chunks:
                            answer = ask_groq(prompt, history=st.session_state.messages)
                        else:
                            answer = ask_groq(prompt, context_chunks=chunks,
                                            history=st.session_state.messages)
                            sources = ", ".join({c["source"] for c in chunks})
                else:
                    answer = ask_groq(prompt, history=st.session_state.messages)

            except Exception as e:
                answer = f"Error: {str(e)[:200]}"

        st.markdown(answer)
        if sources:
            st.markdown(
                f"<span class='source-badge'>Sources: {sources}</span>",
                unsafe_allow_html=True
            )

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": sources
    })
