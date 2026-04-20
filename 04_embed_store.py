"""
04_embed_store.py
-----------------
Purpose : Convert each text chunk into a vector (embedding) using a local
          sentence-transformers model, then store everything in ChromaDB.
          This is the LOAD step — the "L" in ETL.

DE Concept: After Extract (raw PDF text) and Transform (346 chunks), we now
            LOAD our transformed data into the target data store — ChromaDB.
            ChromaDB is our vector database. Instead of storing rows by ID
            or timestamp like a relational DB, it stores data by *meaning*,
            represented as vectors (lists of numbers called embeddings).

What is an Embedding?
---------------------
An embedding converts text into a list of 384 numbers (a vector) that
captures the *semantic meaning* of the text. Similar sentences produce
similar vectors — mathematically close in "vector space."

Example:
    "Apple's revenue declined"  →  [0.21, -0.54, 0.88, ...]
    "Apple saw a drop in sales" →  [0.22, -0.51, 0.85, ...]  ← very close!
    "The weather is sunny"      →  [-0.9,  0.12, -0.3, ...]  ← far away

Local vs Cloud Embeddings (Real DE Architecture Decision):
----------------------------------------------------------
Cloud embeddings (Google, OpenAI): Higher quality, costs money, needs internet
Local embeddings (sentence-transformers): Free, private, runs on your machine
→ We use local here. In production, the choice depends on cost, data privacy,
  and quality requirements — the same tradeoff exists for any data processing.

Usage:
    python 04_embed_store.py

NOTE: First run downloads the model (~90MB). After that it's instant.
      The ChromaDB store is saved to disk — run this once, query forever.
"""

import os
import fitz  # PyMuPDF
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

# ---------------------------------------------------------------------------
# SETUP
# ---------------------------------------------------------------------------
load_dotenv()

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
PDF_PATH        = os.path.join("data", "apl10-K-2025-As-Filed.pdf")
CHROMA_DB_PATH  = "chroma_db"
COLLECTION_NAME = "apple_10k_2025"
CHUNK_SIZE      = 1000
CHUNK_OVERLAP   = 200

# all-MiniLM-L6-v2: A compact, fast, production-grade embedding model.
# Used widely in enterprise search pipelines. 384-dimensional vectors.
EMBEDDING_MODEL = "all-MiniLM-L6-v2"


# ---------------------------------------------------------------------------
# STEP 1: EXTRACT — Pull raw text from PDF
# ---------------------------------------------------------------------------
def extract_full_text(pdf_path: str) -> str:
    print(f"\n📂 Extracting text from: {pdf_path}")
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"[ERROR] PDF not found: {pdf_path}")
    full_text = ""
    with fitz.open(pdf_path) as doc:
        for page in doc:
            full_text += page.get_text()
    print(f"   Characters extracted: {len(full_text):,}")
    return full_text


# ---------------------------------------------------------------------------
# STEP 2: TRANSFORM — Chunk the text
# ---------------------------------------------------------------------------
def chunk_text(full_text: str) -> list:
    print("\n✂️  Chunking text...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
    )
    chunks = splitter.create_documents([full_text])
    print(f"   Chunks created: {len(chunks)}")
    return chunks


# ---------------------------------------------------------------------------
# STEP 3: LOAD — Embed and store in ChromaDB
# ---------------------------------------------------------------------------
def embed_and_store(chunks: list) -> chromadb.Collection:
    """
    Loads all chunks into ChromaDB with local embeddings.

    ChromaDB stores three things per chunk:
        ids        → unique identifier  (like a primary key)
        documents  → the raw chunk text
        embeddings → the vector from sentence-transformers

    DE Concept: This is a one-time pipeline load. ChromaDB persists to disk
                at chroma_db/. Future queries just READ — no re-embedding.
                Build once, query many. Same principle as a data mart load.
    """
    print(f"\n🗄️  Connecting to ChromaDB at: {CHROMA_DB_PATH}/")

    # Local embedding function — no API key needed, runs on your CPU
    embedding_fn = SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL
    )

    # PersistentClient saves data to disk — survives restarts
    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)

    # Idempotent design — delete and recreate for a clean load every time
    existing = [c.name for c in client.list_collections()]
    if COLLECTION_NAME in existing:
        print(f"   ⚠️  Existing collection found — deleting for fresh load.")
        client.delete_collection(COLLECTION_NAME)

    collection = client.create_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_fn,
        metadata={"hnsw:space": "cosine"}  # cosine similarity — best for text
    )

    print(f"   Collection '{COLLECTION_NAME}' created.")
    print(f"\n🔢 Embedding {len(chunks)} chunks locally via {EMBEDDING_MODEL}...")
    print(f"   (First run downloads the model ~90MB — subsequent runs are instant)\n")

    # Prepare data
    documents = [chunk.page_content for chunk in chunks]
    ids       = [f"chunk_{i}" for i in range(len(chunks))]
    metadatas = [{"source": "apple_10k_2025", "chunk_index": i}
                 for i in range(len(chunks))]

    # ChromaDB calls the embedding function automatically
    collection.add(
        documents=documents,
        ids=ids,
        metadatas=metadatas,
    )

    return collection


# ---------------------------------------------------------------------------
# STEP 4: VALIDATE — Post-load sanity check
# ---------------------------------------------------------------------------
def validate_store(collection: chromadb.Collection) -> None:
    """
    Run a test similarity search to confirm the vector store is working.
    DE Concept: Post-load validation — same as a COUNT(*) check after
                an ETL job to confirm everything landed correctly.
    """
    print("\n🔍 Validation query: 'What is Apple total revenue?'")

    results = collection.query(
        query_texts=["What is Apple total revenue?"],
        n_results=2,
    )

    print("\n   Top 2 matching chunks:\n")
    for i, doc in enumerate(results["documents"][0]):
        print(f"   [{i+1}] {doc[:250].strip()}")
        print()


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # E — Extract
    full_text = extract_full_text(PDF_PATH)

    # T — Transform
    chunks = chunk_text(full_text)

    # L — Load
    collection = embed_and_store(chunks)

    # Validate
    validate_store(collection)

    print(f"\n✅ ETL Pipeline complete!")
    print(f"   {collection.count()} chunks stored in ChromaDB.")
    print(f"   Embedding model  : {EMBEDDING_MODEL} (local)")
    print(f"   Data persisted at: {CHROMA_DB_PATH}/")
    print(f"\n⏭️  Next step: 05_rag_query.py — Ask questions, get AI answers!\n")
