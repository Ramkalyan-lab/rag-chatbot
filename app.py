import os
import glob
import zipfile
import io
import streamlit as st
import streamlit.components.v1 as components
from groq import Groq
import base64
import tempfile
import requests
import json
import urllib.parse

# ── Config ────────────────────────────────────────────────────────────────────
DOCS_DIR    = "sample_docs"
MODEL       = "llama-3.3-70b-versatile"
GROQ_KEY    = os.environ.get("GROQ_API_KEY", "")
TAVILY_KEY  = os.environ.get("TAVILY_API_KEY", "")

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
.web-badge {
    background: rgba(16,185,129,0.15); border: 1px solid rgba(16,185,129,0.3);
    border-radius: 6px; padding: 3px 10px; font-size: 12px; color: #6ee7b7;
    display: inline-block; margin-top: 6px;
}
.img-badge {
    background: rgba(245,158,11,0.15); border: 1px solid rgba(245,158,11,0.3);
    border-radius: 6px; padding: 3px 10px; font-size: 12px; color: #fcd34d;
    display: inline-block; margin-top: 6px;
}
.search-result {
    background: #1a1a1a; border: 1px solid #2a2a2a;
    border-radius: 8px; padding: 8px 12px; margin: 4px 0;
    font-size: 12px; color: #aaa;
}
.search-result a { color: #818cf8; text-decoration: none; }
.gen-image-box {
    border: 1px solid rgba(245,158,11,0.3);
    border-radius: 12px; padding: 4px;
    background: #1a1a1a; margin-top: 8px;
}
#MainMenu, footer { visibility: hidden; }
header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ── API Keys ──────────────────────────────────────────────────────────────────
if not GROQ_KEY:
    st.error("GROQ_API_KEY not set. Add it in Streamlit Secrets.")
    st.stop()

client = Groq(api_key=GROQ_KEY)

# ── Image Generation (Pollinations AI - Free, no key needed) ─────────────────
def generate_image(prompt):
    """Generate image using HuggingFace Inference API — free with token."""
    HF_TOKEN = os.environ.get("HF_TOKEN", "")
    if not HF_TOKEN:
        return None, "HF_TOKEN not set in Streamlit Secrets."
    try:
        r = requests.post(
            "https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-2-1",
            headers={"Authorization": f"Bearer {HF_TOKEN}"},
            json={"inputs": prompt},
            timeout=60
        )
        if r.status_code == 200 and r.headers.get("content-type","").startswith("image"):
            img_b64 = base64.b64encode(r.content).decode("utf-8")
            return f"data:image/jpeg;base64,{img_b64}", None
        elif r.status_code == 503:
            # Model loading — wait and retry once
            import time
            time.sleep(20)
            r2 = requests.post(
                "https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-2-1",
                headers={"Authorization": f"Bearer {HF_TOKEN}"},
                json={"inputs": prompt},
                timeout=60
            )
            if r2.status_code == 200:
                img_b64 = base64.b64encode(r2.content).decode("utf-8")
                return f"data:image/jpeg;base64,{img_b64}", None
        return None, f"HuggingFace error: {r.status_code} — {r.text[:100]}"
    except Exception as e:
        return None, str(e)[:150]

def is_image_request(text):
    """Detect if user wants to generate an image."""
    t = text.lower().strip()
    # Direct generation keywords
    gen_keywords = [
        "generate image", "create image", "make image", "draw",
        "generate a picture", "create a picture", "make a picture",
        "generate photo", "create photo", "show me an image",
        "generate an image", "create an illustration", "paint",
        "design an image", "make art", "generate art", "create art",
        "create a photo", "make a photo", "make an image",
        "generate a ", "create a scene", "show image of",
        "image of a", "picture of a", "photo of a",
    ]
    return any(k in t for k in gen_keywords)


def is_image_gen_mode_request(text):
    """Even simpler check when in Image Generator mode."""
    # In image generator mode, treat everything as an image prompt
    return True

def extract_image_prompt(text, ai_response=None):
    """Extract the actual image prompt from user message."""
    removes = [
        "generate image of", "create image of", "make image of",
        "generate a picture of", "create a picture of", "draw",
        "generate an image of", "create an illustration of",
        "show me an image of", "generate photo of", "create photo of",
        "make art of", "generate art of", "image of", "picture of",
        "photo of", "illustration of", "generate image", "create image",
        "make image", "generate a", "create a", "make a", "paint a",
        "paint an", "design an image of", "design a"
    ]
    prompt = text.lower()
    for r in removes:
        prompt = prompt.replace(r, "")
    return prompt.strip() or text

# ── Web Search ────────────────────────────────────────────────────────────────
def web_search(query, max_results=5):
    if not TAVILY_KEY:
        return None, []
    try:
        response = requests.post(
            "https://api.tavily.com/search",
            json={"api_key": TAVILY_KEY, "query": query,
                  "max_results": max_results, "search_depth": "basic",
                  "include_answer": True},
            timeout=10
        )
        data = response.json()
        return data.get("answer", ""), data.get("results", [])
    except:
        return None, []

def needs_web_search(question):
    keywords = ["2024","2025","2026","latest","recent","current","today",
                "now","new","news","price","weather","who is","trending",
                "live","right now","this year","last year","score","result",
                "election","launch","release","update"]
    return any(k in question.lower() for k in keywords)

# ── Live Voice Recorder ───────────────────────────────────────────────────────
VOICE_HTML = """
<div style="background:linear-gradient(135deg,#1a1a2e,#16213e);border:1px solid rgba(99,102,241,0.4);border-radius:16px;padding:20px;text-align:center;font-family:Inter,sans-serif;">
    <div id="status" style="color:#a5b4fc;font-size:14px;margin-bottom:12px;">Click mic to start recording</div>
    <canvas id="wv" width="400" height="50" style="width:100%;height:50px;background:rgba(0,0,0,0.3);border-radius:8px;margin-bottom:12px;display:block;"></canvas>
    <div id="timer" style="color:#818cf8;font-size:22px;font-weight:600;margin-bottom:12px;">00:00</div>
    <div style="display:flex;gap:12px;justify-content:center;margin-bottom:12px;">
        <button onclick="startR()" id="sb" style="background:linear-gradient(135deg,#6366f1,#8b5cf6);color:white;border:none;border-radius:50%;width:60px;height:60px;font-size:26px;cursor:pointer;">🎤</button>
        <button onclick="stopR()" id="pb" disabled style="background:#2a2a2a;color:#555;border:none;border-radius:50%;width:60px;height:60px;font-size:26px;cursor:not-allowed;">⏹️</button>
    </div>
    <div id="play" style="display:none;margin-bottom:10px;"><audio id="ap" controls style="width:100%;border-radius:8px;"></audio></div>
    <div id="msg" style="color:#10b981;font-size:12px;margin-top:8px;"></div>
</div>
<script>
let mr=null,chunks=[],blob=null,ti=null,secs=0,af=null,an=null;
function tick(){secs++;let m=String(Math.floor(secs/60)).padStart(2,'0'),s=String(secs%60).padStart(2,'0');document.getElementById('timer').textContent=m+':'+s;}
function draw(){if(!an)return;const cv=document.getElementById('wv'),ctx=cv.getContext('2d'),buf=new Uint8Array(an.frequencyBinCount);an.getByteTimeDomainData(buf);ctx.clearRect(0,0,cv.width,cv.height);ctx.strokeStyle='#6366f1';ctx.lineWidth=2;ctx.beginPath();const sw=cv.width/buf.length;let x=0;buf.forEach((v,i)=>{const y=(v/128)*cv.height/2;i===0?ctx.moveTo(x,y):ctx.lineTo(x,y);x+=sw;});ctx.stroke();af=requestAnimationFrame(draw);}
async function startR(){
    try{
        const stream=await navigator.mediaDevices.getUserMedia({audio:true});
        const ac=new AudioContext(),src=ac.createMediaStreamSource(stream);
        an=ac.createAnalyser();an.fftSize=256;src.connect(an);draw();
        chunks=[];mr=new MediaRecorder(stream);
        mr.ondataavailable=e=>chunks.push(e.data);
        mr.onstop=()=>{blob=new Blob(chunks,{type:'audio/webm'});document.getElementById('ap').src=URL.createObjectURL(blob);document.getElementById('play').style.display='block';document.getElementById('status').textContent='Done! Listen to your recording.';cancelAnimationFrame(af);stream.getTracks().forEach(t=>t.stop());};
        mr.start();secs=0;ti=setInterval(tick,1000);
        document.getElementById('sb').textContent='🔴';document.getElementById('sb').style.background='linear-gradient(135deg,#ef4444,#dc2626)';
        document.getElementById('pb').disabled=false;document.getElementById('pb').style.background='#333';document.getElementById('pb').style.color='white';document.getElementById('pb').style.cursor='pointer';
        document.getElementById('status').textContent='Recording... Click stop when done.';
    }catch(e){document.getElementById('status').textContent='Microphone denied. Please allow microphone access.';}
}
function stopR(){if(mr&&mr.state!=='inactive'){mr.stop();clearInterval(ti);}document.getElementById('sb').textContent='🎤';document.getElementById('sb').style.background='linear-gradient(135deg,#6366f1,#8b5cf6)';document.getElementById('pb').disabled=true;}
</script>
"""

# ── Document Extraction ───────────────────────────────────────────────────────
def extract_pdf(b):
    try:
        import pypdf
        r = pypdf.PdfReader(io.BytesIO(b))
        return "\n".join(p.extract_text() or "" for p in r.pages).strip() or "[No text in PDF]"
    except Exception as e: return f"[PDF error: {e}]"

def extract_docx(b):
    try:
        import docx
        d = docx.Document(io.BytesIO(b))
        return "\n".join(p.text for p in d.paragraphs if p.text.strip())
    except Exception as e: return f"[DOCX error: {e}]"

def extract_zip(b):
    docs = []
    try:
        with zipfile.ZipFile(io.BytesIO(b)) as z:
            for name in z.namelist():
                if name.startswith("__") or name.endswith("/"): continue
                ext = name.lower().rsplit(".",1)[-1]
                with z.open(name) as f: content = f.read()
                if ext=="txt": docs.append({"filename":name,"text":content.decode("utf-8",errors="ignore")})
                elif ext=="pdf": docs.append({"filename":name,"text":extract_pdf(content)})
                elif ext=="docx": docs.append({"filename":name,"text":extract_docx(content)})
    except Exception as e: docs.append({"filename":"error.txt","text":f"ZIP error: {e}"})
    return docs

def process_files(files):
    docs = []
    for f in files:
        c = f.read(); n = f.name.lower()
        if n.endswith(".txt"): docs.append({"filename":f.name,"text":c.decode("utf-8",errors="ignore")})
        elif n.endswith(".pdf"): docs.append({"filename":f.name,"text":extract_pdf(c)})
        elif n.endswith(".docx"): docs.append({"filename":f.name,"text":extract_docx(c)})
        elif n.endswith(".zip"): docs.extend(extract_zip(c))
    return docs

@st.cache_data
def load_default_docs():
    docs = []
    for path in glob.glob(os.path.join(DOCS_DIR, "*.txt")):
        with open(path,"r",encoding="utf-8",errors="ignore") as f:
            docs.append({"filename":os.path.basename(path),"text":f.read()})
    return docs

def retrieve(query, docs, top_k=3):
    qw = set(query.lower().split()); scored = []
    for doc in docs:
        for chunk in [doc["text"][i:i+800] for i in range(0,len(doc["text"]),600)]:
            score = len(qw & set(chunk.lower().split()))
            if score > 0: scored.append({"text":chunk,"source":doc["filename"],"score":score})
    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:top_k]

# ── Groq ──────────────────────────────────────────────────────────────────────
def ask_groq(question, context=None, web_context=None, history=None, image_b64=None, image_type=None):
    if image_b64 and image_type:
        response = client.chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            messages=[{"role":"user","content":[
                {"type":"image_url","image_url":{"url":f"data:{image_type};base64,{image_b64}"}},
                {"type":"text","text":question or "Describe this image in detail."}
            ]}], max_tokens=1024)
        return response.choices[0].message.content

    if web_context:
        system = f"""You are a helpful AI with access to current web search results.
Use the search results below to answer accurately with up-to-date information.

WEB SEARCH RESULTS:
{web_context}"""
    elif context:
        system = f"""You are a helpful document assistant.
Read the document content carefully and answer based on it.

DOCUMENT CONTENT:
{context}"""
    else:
        system = "You are a helpful, friendly AI assistant. Answer clearly and helpfully."

    messages = [{"role":"system","content":system}]
    if history:
        for msg in history[-4:]:
            if isinstance(msg.get("content"),str):
                messages.append({"role":msg["role"],"content":msg["content"]})
    messages.append({"role":"user","content":question})
    response = client.chat.completions.create(model=MODEL, messages=messages, max_tokens=1024)
    return response.choices[0].message.content

def transcribe_bytes(audio_bytes, ext=".webm"):
    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
        tmp.write(audio_bytes); tmp_path = tmp.name
    with open(tmp_path,"rb") as f:
        result = client.audio.transcriptions.create(
            model="whisper-large-v3", file=f, response_format="text")
    os.unlink(tmp_path)
    return result

# ── Session State ─────────────────────────────────────────────────────────────
for k,v in [("messages",[]),("mode","General Chat"),("uploaded_docs",[]),
            ("sample_q",""),("uploaded_image_b64",None),("uploaded_image_type",None),
            ("web_search_enabled",True)]:
    if k not in st.session_state: st.session_state[k] = v

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
    for m in ["General Chat","Document Q&A","Image Analysis","Web Search","Image Generator"]:
        icon = {"General Chat":"💬","Document Q&A":"📄","Image Analysis":"🖼️",
                "Web Search":"🌐","Image Generator":"🎨"}[m]
        if st.button(f"{icon} {m}", use_container_width=True,
                     type="primary" if st.session_state.mode==m else "secondary",
                     key=f"mode_{m}"):
            if st.session_state.mode != m:
                st.session_state.messages = []
                st.session_state.mode = m
                st.rerun()

    st.divider()
    st.markdown("### ➕ Attach")

    with st.expander("📄 Documents", expanded=False):
        st.caption("PDF · DOCX · TXT · ZIP")
        ufiles = st.file_uploader("Upload",type=["txt","pdf","docx","zip"],
            accept_multiple_files=True,label_visibility="collapsed")
        if ufiles:
            with st.spinner("Reading..."):
                st.session_state.uploaded_docs = process_files(ufiles)
            st.success(f"✅ {len(st.session_state.uploaded_docs)} file(s) loaded!")
            for d in st.session_state.uploaded_docs:
                st.caption(f"📄 {d['filename']}")

    with st.expander("🖼️ Image", expanded=False):
        img_f = st.file_uploader("Image",type=["jpg","jpeg","png","webp"],
            label_visibility="collapsed",key="img_up")
        if img_f:
            img_bytes = img_f.read()
            st.image(img_bytes, use_column_width=True)
            st.session_state.uploaded_image_b64 = base64.b64encode(img_bytes).decode("utf-8")
            st.session_state.uploaded_image_type = img_f.type
            st.session_state.mode = "Image Analysis"
            st.success(f"✅ Image ready! ({img_f.name})")
            st.caption("Now switch to Image Analysis mode and ask anything about it.")

    with st.expander("🎤 Upload Audio", expanded=False):
        aud_f = st.file_uploader("Audio",type=["wav","mp3","m4a","ogg"],
            label_visibility="collapsed",key="aud_up")
        if aud_f:
            st.audio(aud_f)
            if st.button("📝 Transcribe & Send", use_container_width=True):
                with st.spinner("Transcribing..."):
                    try:
                        ext = "."+aud_f.name.rsplit(".",1)[-1]
                        text = transcribe_bytes(aud_f.read(), ext)
                        st.session_state.sample_q = text
                        st.success(f"Heard: {text}")
                    except Exception as e:
                        st.error(str(e)[:100])

    st.divider()
    if TAVILY_KEY:
        st.session_state.web_search_enabled = st.toggle(
            "🌐 Auto Web Search", value=st.session_state.web_search_enabled)
    st.divider()
    st.markdown("""
    <div class="feat-card">
        <strong>⚡ Groq + Llama 3.3 70B</strong>
        Free · Fast · Web Search<br>
        PDF · DOCX · Images · Voice · AI Image Gen
    </div>
    """, unsafe_allow_html=True)

    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# ── Main Area ─────────────────────────────────────────────────────────────────
st.markdown("""
<div class="app-header">
    <h1>🤖 Free AI Chatbot</h1>
    <p>Chat · Documents · Image Analysis · Web Search · AI Image Generation · Voice | 100% Free</p>
</div>
""", unsafe_allow_html=True)

mode = st.session_state.mode

if st.session_state.uploaded_image_b64:
    st.image(base64.b64decode(st.session_state.uploaded_image_b64),
             caption="Uploaded image — ask about it in Image Analysis mode", width=300)

# Live Voice Recorder
with st.expander("🎤 Live Voice Recorder", expanded=False):
    components.html(VOICE_HTML, height=300, scrolling=False)
    st.divider()
    voice_text = st.text_input("Or type voice message:", key="vt")
    if voice_text and st.button("Send", key="vtsend"):
        st.session_state.sample_q = voice_text
        st.rerun()

# Image Generator tips
if mode == "Image Generator":
    st.markdown("""
    <div class="feat-card">
        <strong>🎨 AI Image Generator — Free, no limits!</strong>
        Type what you want to see. Examples below:
    </div>
    """, unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    examples = [
        "A sunset over mountains with golden sky",
        "A futuristic city at night with neon lights",
        "A cute robot sitting in a garden",
        "A wolf howling at the full moon",
        "A magical forest with glowing mushrooms",
        "An astronaut floating in colorful space",
    ]
    for i, ex in enumerate(examples):
        with [col1, col2, col3][i % 3]:
            if st.button(ex, key=f"ex_{i}", use_container_width=True):
                st.session_state.sample_q = f"Generate image of {ex}"

# Welcome cards for other modes
if not st.session_state.messages and mode != "Image Generator":
    col1, col2, col3, col4, col5 = st.columns(5)
    cards = [
        (col1, "💬", "General Chat", "Ask anything", ["Explain RAG", "Write Python code"]),
        (col2, "📄", "Document Q&A", "Upload & chat with docs", ["Summarise document", "Key points?"]),
        (col3, "🖼️", "Image Analysis", "Analyse any image", ["What is in image?", "Describe it"]),
        (col4, "🌐", "Web Search", "Search internet", ["AI news 2025", "Bitcoin price"]),
        (col5, "🎨", "Image Generator", "Create AI images", ["Sunset mountains", "Futuristic city"]),
    ]
    for col, icon, m, desc, questions in cards:
        with col:
            st.markdown(f'<div class="feat-card"><strong>{icon} {m}</strong>{desc}</div>', unsafe_allow_html=True)
            for q in questions:
                if st.button(q, key=f"wc_{m}_{q}", use_container_width=True):
                    st.session_state.sample_q = q
                    st.session_state.mode = m

# Chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg.get("generated_image"):
            st.markdown(msg["content"])
            img = msg["generated_image"]
            if isinstance(img, str) and img.startswith("data:"):
                img_data = base64.b64decode(img.split(",")[1])
                st.image(img_data, caption="🎨 AI Generated Image", use_column_width=True)
            elif img:
                st.image(img, caption="🎨 AI Generated Image", use_column_width=True)
            st.markdown(f"<span class='img-badge'>🎨 Generated with Pollinations AI</span>", unsafe_allow_html=True)
        else:
            st.markdown(msg["content"])
        if msg.get("sources"):
            st.markdown(f"<span class='src-badge'>📄 {msg['sources']}</span>", unsafe_allow_html=True)
        if msg.get("web_sources"):
            st.markdown("<span class='web-badge'>🌐 Web Results:</span>", unsafe_allow_html=True)
            for src in msg["web_sources"][:3]:
                st.markdown(f"<div class='search-result'><a href='{src.get('url','#')}' target='_blank'>{src.get('title','Source')}</a><br>{src.get('content','')[:150]}...</div>", unsafe_allow_html=True)

# ── Chat Input ────────────────────────────────────────────────────────────────
hint = {
    "General Chat": "Ask me anything...",
    "Document Q&A": "Ask about your documents...",
    "Image Analysis": "Ask about the uploaded image...",
    "Web Search": "Search the web for anything...",
    "Image Generator": "Describe the image you want to generate...",
}.get(mode, "Message AI...")

prompt = st.chat_input(hint)

if st.session_state.sample_q:
    prompt = st.session_state.sample_q
    st.session_state.sample_q = ""

if prompt:
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role":"user","content":prompt})

    with st.chat_message("assistant"):
        sources = ""; web_sources = []; generated_image = None

        with st.spinner("Thinking..."):
            try:
                # ── Image Generation ──────────────────────────────────────
                if mode == "Image Generator" or is_image_request(prompt):
                    if mode == "Image Generator":
                        img_prompt = prompt
                    else:
                        img_prompt = extract_image_prompt(prompt)
                    with st.spinner(f"🎨 Generating image... please wait 20-30 seconds..."):
                        try:
                            img_data, err = generate_image(img_prompt)
                            if img_data:
                                generated_image = img_data
                                answer = f"🎨 Here is your AI generated image!\n\n**Prompt:** {img_prompt}"
                            else:
                                generated_image = None
                                answer = f"Image generation failed: {err}. Please try again."
                        except Exception as img_err:
                            generated_image = None
                            answer = f"Image generation error: {str(img_err)[:100]}. Try again."

                # ── Image Analysis ────────────────────────────────────────
                elif mode == "Image Analysis":
                    if st.session_state.uploaded_image_b64:
                        answer = ask_groq(prompt,
                            image_b64=st.session_state.uploaded_image_b64,
                            image_type=st.session_state.uploaded_image_type)
                    else:
                        answer = "Please upload an image using the 🖼️ Image section in the sidebar."

                # ── Web Search ────────────────────────────────────────────
                elif mode == "Web Search" or (
                    st.session_state.web_search_enabled and needs_web_search(prompt)
                ):
                    with st.spinner("🌐 Searching the web..."):
                        tavily_answer, web_results = web_search(prompt)
                    if web_results:
                        web_sources = web_results
                        web_context = "\n\n".join(
                            f"Title: {r.get('title','')}\nURL: {r.get('url','')}\nContent: {r.get('content','')}"
                            for r in web_results[:4])
                        if tavily_answer:
                            web_context = f"Quick Answer: {tavily_answer}\n\n{web_context}"
                        answer = ask_groq(prompt, web_context=web_context,
                                          history=st.session_state.messages)
                    else:
                        answer = ask_groq(prompt, history=st.session_state.messages)

                # ── Document Q&A ──────────────────────────────────────────
                elif mode == "Document Q&A":
                    all_docs = load_default_docs() + st.session_state.uploaded_docs
                    if not all_docs:
                        answer = "No documents found. Upload files using the 📄 section in sidebar."
                    else:
                        chunks = retrieve(prompt, all_docs)
                        if not chunks:
                            context = " ".join(d["text"][:2000] for d in all_docs)
                            chunks = [{"text":context,"source":"uploaded documents"}]
                        context_text = "\n\n---\n\n".join(
                            f"[Source: {c['source']}]\n{c['text']}" for c in chunks)
                        answer = ask_groq(prompt, context=context_text,
                                          history=st.session_state.messages)
                        sources = ", ".join({c["source"] for c in chunks})

                # ── General Chat ──────────────────────────────────────────
                else:
                    answer = ask_groq(prompt, history=st.session_state.messages)

            except Exception as e:
                answer = f"Error: {str(e)[:300]}"

        st.markdown(answer)
        if generated_image:
            if generated_image.startswith("data:"):
                # base64 image - decode and display
                img_data = base64.b64decode(generated_image.split(",")[1])
                st.image(img_data, caption="🎨 AI Generated Image", use_column_width=True)
            else:
                st.image(generated_image, caption="🎨 AI Generated Image", use_column_width=True)
            st.markdown("<span class='img-badge'>🎨 Generated with Pollinations AI — Free</span>", unsafe_allow_html=True)
        if sources:
            st.markdown(f"<span class='src-badge'>📄 Sources: {sources}</span>", unsafe_allow_html=True)
        if web_sources:
            st.markdown("<span class='web-badge'>🌐 Web Results:</span>", unsafe_allow_html=True)
            for src in web_sources[:3]:
                st.markdown(f"<div class='search-result'><a href='{src.get('url','#')}' target='_blank'>{src.get('title','Source')}</a><br>{src.get('content','')[:150]}...</div>", unsafe_allow_html=True)

    st.session_state.messages.append({
        "role":"assistant","content":answer,
        "sources":sources,"web_sources":web_sources,
        "generated_image":generated_image
    })
