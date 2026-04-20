"""
05_rag_query.py
---------------
Purpose : The complete RAG (Retrieval-Augmented Generation) query engine.
          Ask any question about Apple's 10-K and get a grounded AI answer.

How RAG works (the full picture):
----------------------------------
WITHOUT RAG:
    You ask Gemini: "What was Apple's revenue in 2025?"
    Gemini guesses from training data → possibly wrong, possibly outdated.

WITH RAG:
    Step 1 — RETRIEVE: Your question is embedded into a vector.
                       ChromaDB finds the 5 most similar chunks from the 10-K.
    Step 2 — AUGMENT:  Those 5 chunks are injected into the prompt as context.
    Step 3 — GENERATE: Gemini reads the context and answers using REAL data.
    Result → Accurate, cited, grounded answer. No hallucination.

DE Concept: This is the READ side of our pipeline.
            ETL built the store (write once).
            RAG queries the store (read many times).
            Same separation of concerns as a data warehouse:
            ingestion pipelines write, BI tools/analysts read.

Usage:
    python 05_rag_query.py
"""

import os
from dotenv import load_dotenv
from google import genai
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

# ---------------------------------------------------------------------------
# SETUP
# ---------------------------------------------------------------------------
load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise EnvironmentError("[ERROR] GEMINI_API_KEY not found. Check your .env file.")

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
CHROMA_DB_PATH  = "chroma_db"
COLLECTION_NAME = "apple_10k_2025"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
GEMINI_MODEL    = "gemini-2.0-flash"   # Fast, free-tier friendly generation model
TOP_K           = 5                    # Number of chunks to retrieve per question


# ---------------------------------------------------------------------------
# CONNECT TO STORES
# ---------------------------------------------------------------------------
def load_collection() -> chromadb.Collection:
    """
    Opens the existing ChromaDB collection from disk.
    No re-embedding — just connecting to the store we already built.
    """
    embedding_fn = SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL
    )
    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    collection = client.get_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_fn
    )
    return collection


def load_gemini_client() -> genai.Client:
    """
    Initialises the Gemini client for answer generation.
    We use gemini-2.0-flash — fast, accurate, free-tier supported.
    """
    return genai.Client(api_key=GEMINI_API_KEY)


# ---------------------------------------------------------------------------
# THE RAG ENGINE
# ---------------------------------------------------------------------------
def retrieve_context(collection: chromadb.Collection, question: str) -> list[str]:
    """
    STEP 1 — RETRIEVE
    Converts the question into a vector and finds the TOP_K most
    semantically similar chunks in ChromaDB.

    DE Concept: This is a nearest-neighbour query against our vector index.
                In a traditional data warehouse this is like a SELECT with
                an ORDER BY similarity DESC LIMIT 5. Same idea, different math.
    """
    results = collection.query(
        query_texts=[question],
        n_results=TOP_K,
    )
    return results["documents"][0]   # list of TOP_K chunk strings


def build_prompt(question: str, context_chunks: list[str]) -> str:
    """
    STEP 2 — AUGMENT
    Injects the retrieved chunks into a structured prompt.
    The context IS the Apple 10-K data — Gemini reads it like a human
    would read notes before answering a question.

    The system instruction is critical — it tells Gemini to ONLY use
    the provided context, preventing hallucination.
    """
    context = "\n\n---\n\n".join(context_chunks)

    prompt = f"""You are a precise financial analyst assistant.
Answer the question below using ONLY the context provided from Apple's 2025 10-K filing.
If the answer is not found in the context, say: "This information is not available in the provided context."
Always be specific with numbers and figures when they appear in the context.

CONTEXT FROM APPLE 10-K:
{context}

QUESTION: {question}

ANSWER:"""

    return prompt


def generate_answer(client: genai.Client, prompt: str) -> str:
    """
    STEP 3 — GENERATE
    Sends the augmented prompt to Gemini and gets a grounded answer.
    """
    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt,
    )
    return response.text


def ask(collection: chromadb.Collection, client: genai.Client, question: str) -> None:
    """
    Full RAG pipeline for one question — retrieve, augment, generate, display.
    """
    print(f"\n{'='*60}")
    print(f"❓ QUESTION: {question}")
    print(f"{'='*60}")

    # Retrieve
    print(f"\n🔍 Retrieving top {TOP_K} relevant chunks from ChromaDB...")
    chunks = retrieve_context(collection, question)

    # Show sources (transparency — a hallmark of production RAG systems)
    print(f"   Retrieved {len(chunks)} chunks. Top match preview:")
    print(f"   → {chunks[0][:120].strip()}...")

    # Generate
    print(f"\n🤖 Generating answer with {GEMINI_MODEL}...")
    prompt  = build_prompt(question, chunks)
    answer  = generate_answer(client, prompt)

    print(f"\n📊 ANSWER:\n")
    print(answer)
    print()


# ---------------------------------------------------------------------------
# MAIN — Demo questions to prove the bot works end-to-end
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("\n🚀 Financial Analyst RAG Bot — Apple 10-K 2025")
    print("   Loading vector store and AI model...\n")

    collection = load_collection()
    client     = load_gemini_client()

    print(f"   ✅ ChromaDB loaded  : {collection.count()} chunks ready")
    print(f"   ✅ Gemini connected : {GEMINI_MODEL}")

    # --- Ask three demo questions that test different parts of the 10-K ---
    questions = [
        "What was Apple's total net sales revenue in 2025?",
        "What are the main risk factors Apple mentions related to AI?",
        "How much did Apple spend on research and development in 2025?",
    ]

    for question in questions:
        ask(collection, client, question)

    # --- Interactive mode — ask your own questions ---
    print("\n" + "="*60)
    print("💬 INTERACTIVE MODE — Ask your own questions!")
    print("   Type 'quit' to exit.")
    print("="*60)

    while True:
        user_question = input("\nYour question: ").strip()
        if user_question.lower() in ("quit", "exit", "q"):
            print("\n👋 Goodbye!\n")
            break
        if not user_question:
            continue
        ask(collection, client, user_question)
