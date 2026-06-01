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
st.set_page_config(page_title="Free AI Chatbot", page_icon="🤖", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap');

* { font-family: 'Inter', sans-serif; }
.stApp { background: #0f0f0f; }
div[data-testid="stSidebar"] { background: #171717 !important; border-right: 1px solid #2a2a2a; }

/* Header */
.app-header {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
    padding: 1.25rem 2rem; border-radius: 16px;
    margin-bottom: 1rem; text-align: center;
    border: 1px solid rgba(99,102,241,0.3);
}
.app-header h1 { color: white; font-size: 1.75rem; margin: 0; font-weight: 600; }
.app-header p { color: rgba(255,255,255,0.6); margin: 0.2rem 0 0; font-size: 0.85rem; }

/* Mode pills */
.mode-pills { display: flex; gap: 8px; justify-content: center; margin-bottom: 1rem; flex-wrap: wrap; }
.mode-pill {
    padding: 6px 16px; border-radius: 100px; font-size: 13px; cursor: pointer;
    border: 1px solid #333; color: #aaa; background: #1a1a1a;
    transition: all 0.2s;
}
.mode-pill.active { background: #6366f1; color: white; border-color: #6366f1; }

/* Upload bar at bottom */
.bottom-bar {
    position: fixed; bottom: 0; left: 0; right: 0;
    background: #171717; border-top: 1px solid #2a2a2a;
    padding: 12px 20px; z-index: 100;
}

/* Feature cards */
.feat-card {
    background: #1e1e1e; border: 1px solid #2a2a2a;
    border-radius: 10px; padding: 10px 12px;
    margin-bottom: 8px; color: #ccc; font-size: 13px;
}
.feat-card strong { color: white; display: block; margin-bottom: 2px; }

/* Source badge */
.src-badge {
    background: rgba(99,102,241,0.15); border: 1px solid rgba(99,102,241,0.3);
    border-radius: 6px; padding: 3px 10px; font-size: 12px; color: #818cf8;
    display: inline-block; margin-top: 6px;
}

/* Chat messages */
.stChatMessage { border-radius: 12px !important; }

/* Hide streamlit branding */
#MainMenu, footer { visibility: hidden; }
header { visibility: hidden; }

/* Plus button style */
.upload-section {
    background: #1e1e1e; border: 1px solid #2a2a2a;
    border-radius: 12px; padding: 1rem; margin-bottom: 1rem;
}
.upload-section h4 { color: white; margin: 0 0 0.5rem; font-size: 14px; }

/* Voice recorder style */
.voice-box {
    background: linear-gradient(135deg, #1a1a2e, #0f3460);
    border: 1px solid rgba(99,102,241,0.4);
    border-radius: 12px; padding: 1rem;
    text-align: center; margin-bottom: 1rem;
}
.voice-box p { color: #a5b4fc; font-size: 13px; margin: 0.5rem 0 0; }

/* Sidebar nav items */
.nav-item {
    padding: 8px 12px; border-radius: 8px; color: #aaa;
    font-size: 13px; margin-bottom: 4px; cursor: pointer;
    border: 1px solid transparent;
}
.nav-item:hover { background: #222; border-color: #333; color: white; }
</style>
""", unsafe_allow_html=True)

# ── API Key ───────────────────────────────────────────────────────────────────
if not GROQ_KEY:
    st.error("GROQ_API_KEY not set. Add it in Streamlit Secrets.")
    st.stop()

client = Groq(api_key=GROQ_KEY)

# ── Document Extraction ───────────────────────────────────────────────────────
def extract_pdf(file_bytes):
    try:
        import pypdf
        reader = pypdf.PdfReader(io.BytesIO(file_bytes))
        text = "\n".join(page.extract_text() or "" for page in reader.pages)
        return text.strip() or "[No text found in PDF]"
    except Exception as e:
        return f"[PDF error: {e}]"

def extract_docx(file_bytes):
    try:
        import docx
        doc = docx.Document(io.BytesIO(file_bytes))
        return "\n".join(p.text for p in doc.paragraphs if p.text.strip())
    except Exception as e:
        return f"[DOCX error: {e}]"

def extract_zip(file_bytes):
    docs = []
    try:
        with zipfile.ZipFile(io.BytesIO(file_bytes)) as z:
            for name in z.namelist():
                if name.startswith("__") or name.endswith("/"): continue
                ext = name.lower().rsplit(".", 1)[-1]
                with z.open(name) as f:
                    content = f.read()
                if ext == "txt":
                    docs.append({"filename": name, "text": content.decode("utf-8", errors="ignore")})
                elif ext == "pdf":
                    docs.append({"filename": name, "text": extract_pdf(content)})
                elif ext == "docx":
                    docs.append({"filename": name, "text": extract_docx(content)})
    except Exception as e:
        docs.append({"filename": "error.txt", "text": f"ZIP error: {e}"})
    return docs

def process_files(uploaded_files):
    docs = []
    for f in uploaded_files:
        content = f.read()
        name = f.name.lower()
        if name.endswith(".txt"):
            docs.append({"filename": f.name, "text": content.decode("utf-8", errors="ignore")})
        elif name.endswith(".pdf"):
            docs.append({"filename": f.name, "text": extract_pdf(content)})
        elif name.endswith(".docx"):
            docs.append({"filename": f.name, "text": extract_docx(content)})
        elif name.endswith(".zip"):
            docs.extend(extract_zip(content))
    return docs

@st.cache_data
def load_default_docs():
    files = glob.glob(os.path.join(DOCS_DIR, "*.txt"))
    docs = []
    for path in files:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            docs.append({"filename": os.path.basename(path), "text": f.read()})
    return docs

def retrieve(query, docs, top_k=3):
    qwords = set(query.lower().split())
    scored = []
    for doc in docs:
        chunks = [doc["text"][i:i+800] for i in range(0, len(doc["text"]), 600)]
        for chunk in chunks:
            score = len(qwords & set(chunk.lower().split()))
            if score > 0:
                scored.append({"text": chunk, "source": doc["filename"], "score": score})
    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:top_k]

# ── Groq ──────────────────────────────────────────────────────────────────────
def ask_groq(question, context=None, history=None, image_b64=None, image_type=None):
    if image_b64 and image_type:
        response = client.chat.completions.create(
            model="llama-4-scout-17b-16e-instruct",
            messages=[{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": f"data:{image_type};base64,{image_b64}"}},
                {"type": "text", "text": question or "Describe this image in detail."}
            ]}],
            max_tokens=1024
        )
        return response.choices[0].message.content

    if context:
        system = f"""You are a helpful document assistant. 
You have been given the full content of uploaded documents below.
Read them carefully and answer the user's question based on this content.
If the answer is not in the documents, say so clearly.

DOCUMENT CONTENT:
{context}"""
    else:
        system = "You are a helpful, friendly AI assistant like Claude. Answer clearly, helpfully, and with examples where useful."

    messages = [{"role": "system", "content": system}]
    if history:
        for msg in history[-4:]:
            if isinstance(msg.get("content"), str):
                messages.append({"role": msg["role"], "content": msg["content"]})
    messages.append({"role": "user", "content": question})

    response = client.chat.completions.create(model=MODEL, messages=messages, max_tokens=1024)
    return response.choices[0].message.content

def transcribe(audio_bytes, ext=".wav"):
    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name
    with open(tmp_path, "rb") as f:
        result = client.audio.transcriptions.create(
            model="whisper-large-v3", file=f, response_format="text"
        )
    os.unlink(tmp_path)
    return result

# ── Session ───────────────────────────────────────────────────────────────────
for key, val in [("messages", []), ("mode", "General Chat"),
                  ("uploaded_docs", []), ("sample_q", ""),
                  ("uploaded_image_b64", None), ("uploaded_image_type", None)]:
    if key not in st.session_state:
        st.session_state[key] = val

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🤖 AI Chatbot")
    st.divider()

    st.markdown("**New Chat**")
    if st.button("+ New Conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.uploaded_docs = []
        st.session_state.uploaded_image_b64 = None
        st.rerun()

    st.divider()
    st.markdown("**Chat Mode**")
    for m in ["General Chat", "Document Q&A", "Image Analysis"]:
        icon = {"General Chat": "💬", "Document Q&A": "📄", "Image Analysis": "🖼️"}[m]
        is_active = st.session_state.mode == m
        if st.button(
            f"{icon} {m}",
            use_container_width=True,
            type="primary" if is_active else "secondary",
            key=f"mode_{m}"
        ):
            if st.session_state.mode != m:
                st.session_state.messages = []
                st.session_state.mode = m
                st.rerun()

    st.divider()

    # ── Plus Button Upload Section ─────────────────────────────────────────
    st.markdown("### ➕ Upload")

    with st.expander("📄 Documents (PDF, DOCX, TXT, ZIP)", expanded=False):
        uploaded_files = st.file_uploader(
            "Upload documents",
            type=["txt", "pdf", "docx", "zip"],
            accept_multiple_files=True,
            label_visibility="collapsed"
        )
        if uploaded_files:
            with st.spinner("Reading..."):
                st.session_state.uploaded_docs = process_files(uploaded_files)
            st.success(f"✅ {len(st.session_state.uploaded_docs)} file(s) loaded!")
            for d in st.session_state.uploaded_docs:
                st.caption(f"📄 {d['filename']} — {len(d['text']):,} chars")

    with st.expander("🖼️ Image", expanded=False):
        img_file = st.file_uploader(
            "Upload image",
            type=["jpg", "jpeg", "png", "webp"],
            label_visibility="collapsed",
            key="sidebar_image"
        )
        if img_file:
            st.image(img_file, use_column_width=True)
            img_bytes = img_file.read()
            st.session_state.uploaded_image_b64 = base64.b64encode(img_bytes).decode("utf-8")
            st.session_state.uploaded_image_type = img_file.type
            st.session_state.mode = "Image Analysis"
            st.success("✅ Image ready!")

    with st.expander("🎤 Voice Input", expanded=False):
        st.caption("Record your voice on your phone, then upload the audio file here.")
        audio_file = st.file_uploader(
            "Upload audio",
            type=["wav", "mp3", "m4a", "ogg"],
            label_visibility="collapsed",
            key="audio_upload"
        )
        if audio_file:
            st.audio(audio_file)
            if st.button("Transcribe & Send", use_container_width=True):
                with st.spinner("Transcribing your voice..."):
                    try:
                        ext = "." + audio_file.name.rsplit(".", 1)[-1]
                        text = transcribe(audio_file.read(), ext)
                        st.session_state.sample_q = text
                        st.success(f"Heard: {text}")
                    except Exception as e:
                        st.error(f"Error: {str(e)[:100]}")

    st.divider()

    # Default docs status
    default_docs = load_default_docs()
    all_docs = default_docs + st.session_state.uploaded_docs
    if all_docs:
        st.markdown(f"**{len(all_docs)} document(s) available**")
        for d in all_docs[:3]:
            st.caption(f"📄 {d['filename']}")
        if len(all_docs) > 3:
            st.caption(f"...and {len(all_docs)-3} more")

    st.divider()
    st.markdown("""
    <div class="feat-card">
        <strong>⚡ Powered by Groq</strong>
        Llama 3.3 70B · Free forever<br>
        PDF · DOCX · ZIP · Images · Voice
    </div>
    """, unsafe_allow_html=True)

# ── Main Chat ─────────────────────────────────────────────────────────────────

st.markdown("""
<div class="app-header">
    <h1>🤖 Free AI Chatbot</h1>
    <p>General Chat · Document Q&A · Image Analysis · Voice Input | 100% Free | Powered by Groq</p>
</div>
""", unsafe_allow_html=True)

# Mode indicator
mode = st.session_state.mode
mode_icons = {"General Chat": "💬", "Document Q&A": "📄", "Image Analysis": "🖼️"}
st.markdown(f"**{mode_icons.get(mode, '💬')} {mode}** mode")

# Image preview in main area
if st.session_state.uploaded_image_b64 and mode == "Image Analysis":
    img_data = base64.b64decode(st.session_state.uploaded_image_b64)
    st.image(img_data, caption="Uploaded image — ask anything about it", width=400)

# Welcome
if not st.session_state.messages:
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <div class="feat-card">
            <strong>💬 General Chat</strong>
            Ask anything — coding, math, writing, general knowledge
        </div>
        """, unsafe_allow_html=True)
        for q in ["Explain RAG in simple terms", "Write a Python function"]:
            if st.button(q, key=f"gen_{q}", use_container_width=True):
                st.session_state.sample_q = q
    with col2:
        st.markdown("""
        <div class="feat-card">
            <strong>📄 Document Q&A</strong>
            Upload PDF, DOCX, TXT or ZIP and ask questions
        </div>
        """, unsafe_allow_html=True)
        for q in ["Summarise this document", "What are the key points?"]:
            if st.button(q, key=f"doc_{q}", use_container_width=True):
                st.session_state.sample_q = q
                st.session_state.mode = "Document Q&A"
    with col3:
        st.markdown("""
        <div class="feat-card">
            <strong>🖼️ Image Analysis</strong>
            Upload any image and ask about it
        </div>
        """, unsafe_allow_html=True)
        for q in ["What is in this image?", "Describe in detail"]:
            if st.button(q, key=f"img_{q}", use_container_width=True):
                st.session_state.sample_q = q
                st.session_state.mode = "Image Analysis"

# Chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            st.markdown(f"<span class='src-badge'>📄 Sources: {msg['sources']}</span>", unsafe_allow_html=True)

# ── Chat Input with voice/upload hint ────────────────────────────────────────

col_input, col_voice = st.columns([6, 1])

with col_input:
    prompt = st.chat_input("Message AI Chatbot... (use ➕ in sidebar to upload files or voice)")

with col_voice:
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("🎤", help="Use sidebar voice upload to send voice messages", use_container_width=True):
        st.info("Use the 🎤 Voice Input section in the sidebar to upload audio!")

if st.session_state.sample_q:
    prompt = st.session_state.sample_q
    st.session_state.sample_q = ""

# ── Process Message ───────────────────────────────────────────────────────────
if prompt:
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            sources = ""
            try:
                if mode == "Image Analysis":
                    if st.session_state.uploaded_image_b64:
                        answer = ask_groq(
                            prompt,
                            image_b64=st.session_state.uploaded_image_b64,
                            image_type=st.session_state.uploaded_image_type
                        )
                    else:
                        answer = "Please upload an image first using the ➕ button in the sidebar."

                elif mode == "Document Q&A":
                    all_docs = load_default_docs() + st.session_state.uploaded_docs
                    if not all_docs:
                        answer = "No documents found. Please upload files using the ➕ button in the sidebar."
                    else:
                        chunks = retrieve(prompt, all_docs)
                        if not chunks:
                            context = " ".join(d["text"][:2000] for d in all_docs)
                            chunks = [{"text": context, "source": "uploaded documents"}]
                        context_text = "\n\n---\n\n".join(
                            f"[Source: {c['source']}]\n{c['text']}" for c in chunks
                        )
                        answer = ask_groq(prompt, context=context_text, history=st.session_state.messages)
                        sources = ", ".join({c["source"] for c in chunks})
                else:
                    answer = ask_groq(prompt, history=st.session_state.messages)

            except Exception as e:
                answer = f"Error: {str(e)[:300]}"

        st.markdown(answer)
        if sources:
            st.markdown(f"<span class='src-badge'>📄 Sources: {sources}</span>", unsafe_allow_html=True)

    st.session_state.messages.append({"role": "assistant", "content": answer, "sources": sources})
