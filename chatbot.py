import os, sys, glob, argparse
import google.generativeai as genai
import chromadb
from chromadb.utils import embedding_functions

DOCS_DIR = "sample_docs"
DB_DIR = "chroma_db"
COLLECTION = "my_documents"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100
TOP_K = 3
GOOGLE_KEY = os.environ.get("GOOGLE_API_KEY", "")

def check_api_key():
    if not GOOGLE_KEY:
        print("\n?  GOOGLE_API_KEY not set.")
        print("    Run: AIzaSyABOb90DX2n2BNFnrbFKMiBcIogDLwfqLs = 'your_key_here'\n")
        sys.exit(1)
    genai.configure(api_key=GOOGLE_KEY)

def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    chunks = []
    start = 0
    while start < len(text):
        chunks.append(text[start:start+chunk_size].strip())
        start += chunk_size - overlap
    return [c for c in chunks if len(c) > 50]

def load_documents(docs_dir):
    files = glob.glob(os.path.join(docs_dir, "*.txt"))
    if not files:
        print(f"No .txt files found in '{docs_dir}/'.")
        sys.exit(1)
    docs = []
    for path in files:
        with open(path, "r", encoding="utf-8") as f:
            docs.append({"filename": os.path.basename(path), "text": f.read()})
    return docs

def get_collection():
    client = chromadb.PersistentClient(path=DB_DIR)
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
    return client.get_or_create_collection(name=COLLECTION, embedding_function=ef)

def ingest_documents():
    print("\n??  Loading documents...")
    docs = load_documents(DOCS_DIR)
    collection = get_collection()
    existing = collection.get()
    if existing["ids"]:
        collection.delete(ids=existing["ids"])
    all_chunks, all_ids, all_metas = [], [], []
    for doc in docs:
        chunks = chunk_text(doc["text"])
        for i, chunk in enumerate(chunks):
            all_chunks.append(chunk)
            all_ids.append(f"{doc['filename']}__chunk_{i}")
            all_metas.append({"source": doc["filename"], "chunk_index": i})
    print(f"    Embedding {len(all_chunks)} chunks from {len(docs)} file(s)...")
    collection.add(documents=all_chunks, ids=all_ids, metadatas=all_metas)
    print(f"?  Done! {len(all_chunks)} chunks stored.\n")

def retrieve(query):
    collection = get_collection()
    results = collection.query(query_texts=[query], n_results=TOP_K)
    return [{"text": d, "source": m["source"]} for d, m in zip(results["documents"][0], results["metadatas"][0])]

def ask_gemini(question, context_chunks):
    context_text = "\n\n---\n\n".join(f"[Source: {c['source']}]\n{c['text']}" for c in context_chunks)
    prompt = f"""You are a helpful assistant. Answer strictly based on the context below.
If the answer is not in the context, say: I don't have enough information in the documents to answer that.

Context:
{context_text}

Question: {question}"""
    model = genai.GenerativeModel("gemini-2.0-flash-lite")
    response = model.generate_content(prompt)
    return response.text

def chat():
    print("\n??  RAG Chatbot ready! Type 'quit' to exit.\n")
    collection = get_collection()
    count = collection.count()
    if count == 0:
        print("??  No documents indexed. Run with --ingest first.\n")
        return
    print(f"    ({count} chunks indexed)\n")
    while True:
        try:
            question = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye! ??")
            break
        if not question:
            continue
        if question.lower() in ("quit", "exit", "q"):
            print("Goodbye! ??")
            break
        chunks = retrieve(question)
        sources = list({c["source"] for c in chunks})
        answer = ask_gemini(question, chunks)
        print(f"\nBot: {answer}")
        print(f"     ?? Sources: {', '.join(sources)}\n")

def main():
    check_api_key()
    parser = argparse.ArgumentParser()
    parser.add_argument("--ingest", action="store_true")
    parser.add_argument("--chat", action="store_true")
    args = parser.parse_args()
    if args.ingest:
        ingest_documents()
    if args.chat or not args.ingest:
        chat()

if __name__ == "__main__":
    main()
