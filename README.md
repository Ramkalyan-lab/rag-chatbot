# 🤖 RAG Chatbot — Ask Questions About Your Own Documents

A command-line chatbot that lets you **chat with your own `.txt` files** using Retrieval-Augmented Generation (RAG). It finds the most relevant parts of your documents and passes them to Claude as context — so answers are grounded in *your* data, not hallucinated.

---

## How It Works

```
Your .txt files
      │
      ▼
  [1] Chunk text into overlapping segments
      │
      ▼
  [2] Embed each chunk → vector (all-MiniLM-L6-v2)
      │
      ▼
  [3] Store vectors in ChromaDB (local, on-disk)
      │
      ▼
  User asks a question
      │
      ▼
  [4] Embed the question → find Top-K similar chunks
      │
      ▼
  [5] Send chunks + question to Claude API
      │
      ▼
  Answer grounded in your documents ✅
```

---

## Tech Stack

| Layer | Tool |
|---|---|
| Embedding model | `all-MiniLM-L6-v2` (runs locally, free) |
| Vector database | ChromaDB (on-disk, no server needed) |
| LLM | Claude via Anthropic API |
| Language | Python 3.11+ |

---

## Setup

### 1. Clone the repo

```bash
git clone https://github.com/YOUR_USERNAME/rag-chatbot.git
cd rag-chatbot
```

### 2. Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate      # Mac/Linux
venv\Scripts\activate         # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

> First run downloads the `all-MiniLM-L6-v2` model (~80MB). This is a one-time download.

### 4. Set your API key

```bash
export ANTHROPIC_API_KEY=your_key_here
```

Get a free API key at [console.anthropic.com](https://console.anthropic.com).

---

## Usage

### Step 1 — Add your documents

Drop any `.txt` files into the `sample_docs/` folder. Two example files are already included:
- `company_faq.txt` — sample company FAQ
- `ai_concepts.txt` — AI/ML concept definitions

### Step 2 — Ingest (first time, or when you add new docs)

```bash
python chatbot.py --ingest
```

This chunks, embeds, and stores all your documents in ChromaDB.

### Step 3 — Start chatting

```bash
python chatbot.py
```

Or do both in one command:

```bash
python chatbot.py --ingest --chat
```

### Example session

```
🤖  RAG Chatbot ready! Ask anything about your documents.
    Type 'quit' or press Ctrl+C to exit.

    (47 chunks indexed from your docs)

You: What is RAG and why does it reduce hallucinations?

Bot: RAG (Retrieval-Augmented Generation) is a technique that improves
     LLM accuracy by supplying the model with relevant external documents
     before it generates an answer. Instead of relying only on its training
     data, the model grounds its response in retrieved chunks, which
     significantly reduces hallucinations because the model has factual
     source material to reference rather than having to "guess."
     📄 Sources: ai_concepts.txt

You: What is Acme Corp's refund policy?

Bot: Acme Corp offers a 30-day full refund on all products. Refunds are
     processed within 5-7 business days. After 30 days, only store credit
     is available. Contact support@acme.com with your order number to start.
     📄 Sources: company_faq.txt

You: quit
Goodbye! 👋
```

---

## Project Structure

```
rag-chatbot/
├── chatbot.py          ← Main application (ingest + chat)
├── requirements.txt    ← Python dependencies
├── .env.example        ← API key template
├── .gitignore
└── sample_docs/        ← Put your .txt files here
    ├── company_faq.txt
    └── ai_concepts.txt
```

---

## Key Concepts Demonstrated

- **Chunking with overlap** — text is split into 500-character segments with 100-character overlap to avoid losing context at boundaries
- **Local embeddings** — `all-MiniLM-L6-v2` runs on your machine; no paid embedding API needed
- **Vector similarity search** — ChromaDB retrieves the Top-3 most semantically similar chunks to each question
- **Context injection** — retrieved chunks are injected into the Claude prompt, keeping the model grounded
- **Hallucination control** — the system prompt explicitly instructs Claude not to answer outside the provided context

---

## Customisation

| Config variable | Default | What it does |
|---|---|---|
| `CHUNK_SIZE` | 500 | Characters per chunk |
| `CHUNK_OVERLAP` | 100 | Overlap between chunks |
| `TOP_K` | 3 | Chunks retrieved per query |
| `CLAUDE_MODEL` | `claude-sonnet-4-20250514` | Which Claude model to use |
| `DOCS_DIR` | `sample_docs` | Folder to read `.txt` files from |

---

## Extending This Project

Ideas to take this further:
- Add PDF support using `pypdf` or `pdfplumber`
- Build a web UI with Streamlit (`pip install streamlit`)
- Add conversation history (multi-turn chat memory)
- Support multiple collections / namespaces
- Add a confidence score based on ChromaDB distance values

---

## License

MIT — use freely, learn from it, build on it.
