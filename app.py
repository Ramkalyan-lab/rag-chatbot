import os
import glob
import zipfile
import io
import streamlit as st
import streamlit.components.v1 as components
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
.app-header {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
    padding: 1.25rem 2rem; border-radius: 16px;
    margin-bottom: 1rem; text-align: center;
    border: 1px solid rgba(99,102,241,0.3);
}
.app-header h1 { color: white; font-size: 1.75rem; margin: 0; font-weight: 600; }
.app-header p { color: rgba(255,255,255,0.6); margin: 0.2rem 0 0; font-size: 0.85rem; }
.feat-card {
    background: #1e1e1e; border: 1px solid #2a2a2a;
    border-radius: 10px; padding: 10px 12px;
    margin-bottom: 8px; color: #ccc; font-size: 13px;
}
.feat-card strong { color: white; display: block; margin-bottom: 2px; }
.src-badge {
    background: rgba(99,102,241,0.15); border: 1px solid rgba(99,102,241,0.3);
    border-radius: 6px; padding: 3px 10px; font-size: 12px; color: #818cf8;
    display: inline-block; margin-top: 6px;
}
#MainMenu, footer { visibility: hidden; }
header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ── API Key ───────────────────────────────────────────────────────────────────
if not GROQ_KEY:
    st.error("GROQ_API_KEY not set. Add it in Streamlit Secrets.")
    st.stop()

client = Groq(api_key=GROQ_KEY)

# ── Live Voice Recorder HTML ──────────────────────────────────────────────────
VOICE_RECORDER_HTML = """
<div id="voice-recorder" style="
    background: linear-gradient(135deg, #1a1a2e, #16213e);
    border: 1px solid rgba(99,102,241,0.4);
    border-radius: 16px; padding: 20px;
    text-align: center; font-family: Inter, sans-serif;
">
    <div id="status" style="color:#a5b4fc; font-size:14px; margin-bottom:12px;">
        Click the mic button to start recording
    </div>
    <canvas id="waveCanvas" width="400" height="50" style="
        width:100%; height:50px; background:rgba(0,0,0,0.3);
        border-radius:8px; margin-bottom:12px; display:block;
    "></canvas>
    <div id="timer" style="color:#818cf8; font-size:22px; font-weight:600; margin-bottom:12px;">00:00</div>
    <div style="display:flex; gap:12px; justify-content:center; margin-bottom:12px;">
        <button onclick="startRec()" id="startBtn" style="
            background:linear-gradient(135deg,#6366f1,#8b5cf6);
            color:white; border:none; border-radius:50%;
            width:60px; height:60px; font-size:26px; cursor:pointer;
            box-shadow:0 4px 15px rgba(99,102,241,0.4);
        ">🎤</button>
        <button onclick="stopRec()" id="stopBtn" disabled style="
            background:#2a2a2a; color:#555; border:none;
            border-radius:50%; width:60px; height:60px;
            font-size:26px; cursor:not-allowed;
        ">⏹️</button>
    </div>
    <div id="playback" style="display:none; margin-bottom:10px;">
        <audio id="player" controls style="width:100%; border-radius:8px;"></audio>
    </div>
    <input type="text" id="transcriptBox" placeholder="Transcription will appear here after clicking Send..."
        style="width:100%; box-sizing:border-box; padding:8px 12px;
        background:#0f0f0f; border:1px solid #333; border-radius:8px;
        color:white; font-size:13px; margin-bottom:8px; display:none;"
    />
    <button onclick="sendToStreamlit()" id="sendBtn" style="
        display:none; width:100%;
        background:linear-gradient(135deg,#10b981,#059669);
        color:white; border:none; border-radius:8px;
        padding:10px; font-size:14px; cursor:pointer;
    ">✅ Send Audio for Transcription</button>
    <div id="msg" style="color:#10b981; font-size:12px; margin-top:6px;"></div>
</div>
<script>
let mr=null, chunks=[], blob=null, ti=null, secs=0, animF=null, analyser=null;

function tick(){ secs++; let m=String(Math.floor(secs/60)).padStart(2,'0'),s=String(secs%60).padStart(2,'0'); document.getElementById('timer').textContent=m+':'+s; }

function drawWave(){
    if(!analyser) return;
    const cv=document.getElementById('waveCanvas'), ctx=cv.getContext('2d');
    const buf=new Uint8Array(analyser.frequencyBinCount);
    analyser.getByteTimeDomainData(buf);
    ctx.clearRect(0,0,cv.width,cv.height);
    ctx.strokeStyle='#6366f1'; ctx.lineWidth=2; ctx.beginPath();
    const sw=cv.width/buf.length; let x=0;
    buf.forEach((v,i)=>{ const y=(v/128)*cv.height/2; i===0?ctx.moveTo(x,y):ctx.lineTo(x,y); x+=sw; });
    ctx.stroke(); animF=requestAnimationFrame(drawWave);
}

async function startRec(){
    try{
        const stream=await navigator.mediaDevices.getUserMedia({audio:true});
        const ac=new AudioContext(), src=ac.createMediaStreamSource(stream);
        analyser=ac.createAnalyser(); analyser.fftSize=256; src.connect(analyser); drawWave();
        chunks=[]; mr=new MediaRecorder(stream);
        mr.ondataavailable=e=>chunks.push(e.data);
        mr.onstop=()=>{
            blob=new Blob(chunks,{type:'audio/webm'});
            document.getElementById('player').src=URL.createObjectURL(blob);
            document.getElementById('playback').style.display='block';
            document.getElementById('sendBtn').style.display='block';
            document.getElementById('status').textContent='Recording done! Listen and click Send.';
            cancelAnimationFrame(animF); stream.getTracks().forEach(t=>t.stop());
        };
        mr.start(); secs=0; ti=setInterval(tick,1000);
        document.getElementById('startBtn').textContent='🔴';
        document.getElementById('startBtn').style.background='linear-gradient(135deg,#ef4444,#dc2626)';
        document.getElementById('stopBtn').disabled=false;
        document.getElementById('stopBtn').style.background='#333';
        document.getElementById('stopBtn').style.color='white';
        document.getElementById('stopBtn').style.cursor='pointer';
        document.getElementById('status').textContent='Recording... Click stop when done.';
    } catch(e){
        document.getElementById('status').textContent='Microphone access denied. Please allow microphone in browser.';
    }
}

function stopRec(){
    if(mr && mr.state!=='inactive'){ mr.stop(); clearInterval(ti); }
    document.getElementById('startBtn').textContent='🎤';
    document.getElementById('startBtn').style.background='linear-gradient(135deg,#6366f1,#8b5cf6)';
    document.getElementById('stopBtn').disabled=true;
}

function sendToStreamlit(){
    if(!blob) return;
    const r=new FileReader();
    r.onloadend=()=>{
        const b64=r.result.split(',')[1];
        // Store in sessionStorage for Streamlit to pick up
        window.sessionStorage.setItem('voiceAudio', b64);
        window.sessionStorage.setItem('voiceReady', 'true');
        document.getElementById('msg').textContent='Audio ready! Now type anything in the chat box and press Enter to trigger transcription.';
        document.getElementById('sendBtn').textContent='✅ Audio Ready - Type in chat to send';
        document.getElementById('sendBtn').style.background='#333';
    };
    r.readAsDataURL(blob);
}
</script>
"""

# ── Document Extraction ───────────────────────────────────────────────────────
def extract_pdf(file_bytes):
    try:
        import pypdf
        reader = pypdf.PdfReader(io.BytesIO(file_bytes))
        text = "\n".join(page.extract_text() or "" for page in reader.pages)
        return text.strip() or "[No extractable text in PDF]"
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

    system = f"""You are a helpful document assistant. Read and answer from the document content below.\n\nDOCUMENT CONTENT:\n{context}""" if context else \
             "You are a helpful, friendly AI assistant. Answer clearly and helpfully."

    messages = [{"role": "system", "content": system}]
    if history:
        for msg in history[-4:]:
            if isinstance(msg.get("content"), str):
                messages.append({"role": msg["role"], "content": msg["content"]})
    messages.append({"role": "user", "content": question})
    response = client.chat.completions.create(model=MODEL, messages=messages, max_tokens=1024)
    return response.choices[0].message.content

def transcribe_bytes(audio_bytes, ext=".webm"):
    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name
    with open(tmp_path, "rb") as f:
        result = client.audio.transcriptions.create(
            model="whisper-large-v3", file=f, response_format="text"
        )
    os.unlink(tmp_path)
    return result

# ── Session State ─────────────────────────────────────────────────────────────
for key, val in [("messages", []), ("mode", "General Chat"),
                  ("uploaded_docs", []), ("sample_q", ""),
                  ("uploaded_image_b64", None), ("uploaded_image_type", None)]:
    if key not in st.session_state:
        st.session_state[key] = val

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🤖 Free AI Chatbot")
    st.divider()

    if st.button("➕ New Chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.uploaded_docs = []
        st.session_state.uploaded_image_b64 = None
        st.rerun()

    st.divider()
    st.markdown("**Chat Mode**")
    for m in ["General Chat", "Document Q&A", "Image Analysis"]:
        icon = {"General Chat": "💬", "Document Q&A": "📄", "Image Analysis": "🖼️"}[m]
        if st.button(f"{icon} {m}", use_container_width=True,
                     type="primary" if st.session_state.mode == m else "secondary",
                     key=f"mode_{m}"):
            if st.session_state.mode != m:
                st.session_state.messages = []
                st.session_state.mode = m
                st.rerun()

    st.divider()
    st.markdown("### ➕ Attach")

    with st.expander("📄 Documents", expanded=False):
        st.caption("PDF · DOCX · TXT · ZIP")
        uploaded_files = st.file_uploader(
            "Upload", type=["txt","pdf","docx","zip"],
            accept_multiple_files=True, label_visibility="collapsed"
        )
        if uploaded_files:
            with st.spinner("Reading..."):
                st.session_state.uploaded_docs = process_files(uploaded_files)
            st.success(f"✅ {len(st.session_state.uploaded_docs)} file(s) loaded!")
            for d in st.session_state.uploaded_docs:
                st.caption(f"📄 {d['filename']}")

    with st.expander("🖼️ Image", expanded=False):
        img_file = st.file_uploader("Upload image",
            type=["jpg","jpeg","png","webp"],
            label_visibility="collapsed", key="img_up")
        if img_file:
            st.image(img_file, use_column_width=True)
            st.session_state.uploaded_image_b64 = base64.b64encode(img_file.read()).decode("utf-8")
            st.session_state.uploaded_image_type = img_file.type
            st.session_state.mode = "Image Analysis"
            st.success("✅ Image ready!")

    with st.expander("🎤 Upload Audio File", expanded=False):
        st.caption("Upload a pre-recorded .wav/.mp3/.m4a file")
        audio_up = st.file_uploader("Audio",
            type=["wav","mp3","m4a","ogg"],
            label_visibility="collapsed", key="audio_up")
        if audio_up:
            st.audio(audio_up)
            if st.button("📝 Transcribe", use_container_width=True):
                with st.spinner("Transcribing..."):
                    try:
                        ext = "." + audio_up.name.rsplit(".",1)[-1]
                        text = transcribe_bytes(audio_up.read(), ext)
                        st.session_state.sample_q = text
                        st.success(f"Heard: {text}")
                    except Exception as e:
                        st.error(str(e)[:100])

    st.divider()
    st.markdown("""
    <div class="feat-card">
        <strong>⚡ Groq + Llama 3.3 70B</strong>
        Free · Fast · PDF · DOCX · Images · Voice
    </div>
    """, unsafe_allow_html=True)

    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# ── Main Area ─────────────────────────────────────────────────────────────────
st.markdown("""
<div class="app-header">
    <h1>🤖 Free AI Chatbot</h1>
    <p>General Chat · Document Q&A · Image Analysis · Live Voice | 100% Free | Powered by Groq + Llama 3.3</p>
</div>
""", unsafe_allow_html=True)

mode = st.session_state.mode

if st.session_state.uploaded_image_b64 and mode == "Image Analysis":
    st.image(base64.b64decode(st.session_state.uploaded_image_b64),
             caption="Uploaded image", width=350)

# ── Live Voice Recorder ───────────────────────────────────────────────────────
with st.expander("🎤 Live Voice Recorder — Click to record directly in browser", expanded=False):
    st.caption("Click the mic button, speak, click stop, then click Send Audio.")
    components.html(VOICE_RECORDER_HTML, height=340, scrolling=False)
    st.caption("After recording, the audio is saved. Upload it using the 🎤 Upload Audio File section in the sidebar to transcribe it.")
    # Alternative: typed voice input
    st.divider()
    voice_text = st.text_input("Or type your voice message manually here:", key="voice_text_input")
    if voice_text and st.button("Send Voice Message", use_container_width=True):
        st.session_state.sample_q = voice_text

# Welcome cards
if not st.session_state.messages:
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown('<div class="feat-card"><strong>💬 General Chat</strong>Ask anything — coding, math, writing</div>', unsafe_allow_html=True)
        for q in ["Explain RAG simply", "Write Python code"]:
            if st.button(q, key=f"g_{q}", use_container_width=True):
                st.session_state.sample_q = q
    with col2:
        st.markdown('<div class="feat-card"><strong>📄 Document Q&A</strong>Upload PDF/DOCX and ask questions</div>', unsafe_allow_html=True)
        for q in ["Summarise this document", "What are the key points?"]:
            if st.button(q, key=f"d_{q}", use_container_width=True):
                st.session_state.sample_q = q
                st.session_state.mode = "Document Q&A"
    with col3:
        st.markdown('<div class="feat-card"><strong>🖼️ Image Analysis</strong>Upload any image and ask about it</div>', unsafe_allow_html=True)
        for q in ["What is in this image?", "Describe in detail"]:
            if st.button(q, key=f"i_{q}", use_container_width=True):
                st.session_state.sample_q = q
                st.session_state.mode = "Image Analysis"

# Chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            st.markdown(f"<span class='src-badge'>📄 {msg['sources']}</span>", unsafe_allow_html=True)

# ── Chat Input ────────────────────────────────────────────────────────────────
prompt = st.chat_input("Message AI... (use sidebar ➕ for files, or voice recorder above)")

if st.session_state.sample_q:
    prompt = st.session_state.sample_q
    st.session_state.sample_q = ""

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
                        answer = ask_groq(prompt,
                            image_b64=st.session_state.uploaded_image_b64,
                            image_type=st.session_state.uploaded_image_type)
                    else:
                        answer = "Please upload an image using the 🖼️ Image section in the sidebar."

                elif mode == "Document Q&A":
                    all_docs = load_default_docs() + st.session_state.uploaded_docs
                    if not all_docs:
                        answer = "No documents found. Upload files using the 📄 Documents section in the sidebar."
                    else:
                        chunks = retrieve(prompt, all_docs)
                        if not chunks:
                            context = " ".join(d["text"][:2000] for d in all_docs)
                            chunks = [{"text": context, "source": "uploaded documents"}]
                        context_text = "\n\n---\n\n".join(
                            f"[Source: {c['source']}]\n{c['text']}" for c in chunks)
                        answer = ask_groq(prompt, context=context_text,
                                          history=st.session_state.messages)
                        sources = ", ".join({c["source"] for c in chunks})
                else:
                    answer = ask_groq(prompt, history=st.session_state.messages)

            except Exception as e:
                answer = f"Error: {str(e)[:300]}"

        st.markdown(answer)
        if sources:
            st.markdown(f"<span class='src-badge'>📄 Sources: {sources}</span>", unsafe_allow_html=True)

    st.session_state.messages.append({"role": "assistant", "content": answer, "sources": sources})
