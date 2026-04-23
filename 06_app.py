"""
06_app.py
---------
Purpose : Streamlit web UI for the Financial Analyst RAG Bot.
          Turns our pipeline into a real, shareable web application.

DE Concept: This is the PRESENTATION LAYER — the final layer in any
            data product. In a company this would be the BI dashboard,
            the internal tool, or the analyst-facing app that sits on
            top of your data warehouse. We built the warehouse (ChromaDB).
            This is the front door.

Usage:
    streamlit run 06_app.py
"""

import os
import requests
import streamlit as st
from dotenv import load_dotenv
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

# ---------------------------------------------------------------------------
# PAGE SETUP — MUST be the very first Streamlit call. Nothing st.* before this.
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Financial Analyst RAG Bot",
    page_icon="📊",
    layout="centered",
)

# ---------------------------------------------------------------------------
# CONFIG — Load API key from .env (local) or Streamlit secrets (cloud)
# ---------------------------------------------------------------------------
load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
if not GEMINI_API_KEY:
    try:
        GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY", "")
    except Exception:
        GEMINI_API_KEY = ""

if not GEMINI_API_KEY:
    st.error("❌ GEMINI_API_KEY is missing. Add it in Streamlit Cloud → Manage App → Secrets.")
    st.stop()

# Absolute path — works on any machine regardless of launch directory
BASE_DIR        = os.path.dirname(os.path.abspath(__file__))
CHROMA_DB_PATH  = os.path.join(BASE_DIR, "chroma_db")
COLLECTION_NAME = "apple_10k_2025"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
GEMINI_MODEL    = "gemini-2.0-flash"
TOP_K           = 5


# ---------------------------------------------------------------------------
# CACHED RESOURCE LOADERS
# @st.cache_resource loads once and reuses across all sessions —
# same principle as a connection pool in a backend service.
# ---------------------------------------------------------------------------
@st.cache_resource
def load_collection():
    embedding_fn = SentenceTransformerEmbeddingFunction(model_name=EMBEDDING_MODEL)
    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    return client.get_collection(name=COLLECTION_NAME, embedding_function=embedding_fn)


# ---------------------------------------------------------------------------
# RAG PIPELINE FUNCTIONS
# ---------------------------------------------------------------------------
def retrieve_chunks(collection, question: str) -> list[str]:
    results = collection.query(query_texts=[question], n_results=TOP_K)
    return results["documents"][0]


def build_prompt(question: str, chunks: list[str]) -> str:
    context = "\n\n---\n\n".join(chunks)
    return f"""You are a precise financial analyst assistant.
Answer the question below using ONLY the context provided from Apple's 2025 10-K filing.
If the answer is not found in the context, say: "This information is not available in the provided context."
Always be specific with numbers and figures when they appear in the context.

CONTEXT FROM APPLE 10-K:
{context}

QUESTION: {question}

ANSWER:"""


def get_answer(question: str, chunks: list[str]) -> str:
    """
    Calls Gemini via direct REST API — no SDK, no version conflicts.
    Returns the answer text, or an error string if the API call fails.
    """
    prompt  = build_prompt(question, chunks)
    url     = (
        f"https://generativelanguage.googleapis.com/v1beta/models/"
        f"{GEMINI_MODEL}:generateContent?key={GEMINI_API_KEY}"
    )
    payload = {"contents": [{"parts": [{"text": prompt}]}]}

    try:
        resp = requests.post(url, json=payload, timeout=30)
        if resp.status_code != 200:
            return f"❌ API Error {resp.status_code}: {resp.text}"
        return resp.json()["candidates"][0]["content"]["parts"][0]["text"]
    except Exception as e:
        return f"❌ Request failed: {str(e)}"


# ---------------------------------------------------------------------------
# UI — Header
# ---------------------------------------------------------------------------
st.title("📊 Financial Analyst RAG Bot")
st.caption("Powered by Apple 10-K 2025 · ChromaDB · Gemini 2.0 Flash · all-MiniLM-L6-v2")

st.markdown("""
Ask any question about Apple's 2025 annual report.
The bot retrieves real data from the 10-K filing before generating an answer.
""")

st.divider()

# ---------------------------------------------------------------------------
# LOAD RESOURCES
# ---------------------------------------------------------------------------
with st.spinner("Loading vector store..."):
    try:
        collection = load_collection()
    except Exception as e:
        st.error(f"❌ Failed to load ChromaDB: {str(e)}")
        st.stop()

st.success(f"✅ Ready — {collection.count()} chunks loaded from Apple 10-K 2025")

# ---------------------------------------------------------------------------
# CHAT HISTORY
# ---------------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant" and "sources" in message:
            with st.expander("📄 View source chunks retrieved from 10-K"):
                for i, chunk in enumerate(message["sources"], 1):
                    st.markdown(f"**Chunk {i}:**")
                    st.text(chunk[:400])
                    st.divider()

# ---------------------------------------------------------------------------
# CHAT INPUT
# ---------------------------------------------------------------------------
if prompt := st.chat_input("Ask a question about Apple's 10-K..."):

    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        with st.spinner("Retrieving relevant data and generating answer..."):
            chunks = retrieve_chunks(collection, prompt)
            answer = get_answer(prompt, chunks)

        st.markdown(answer)

        with st.expander("📄 View source chunks retrieved from 10-K"):
            for i, chunk in enumerate(chunks, 1):
                st.markdown(f"**Chunk {i}:**")
                st.text(chunk[:400])
                st.divider()

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": chunks,
    })

# ---------------------------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("💡 Try These Questions")
    st.markdown("""
- What was Apple's total revenue in 2025?
- How much did Apple spend on R&D?
- What AI risks does Apple mention?
- What is Apple's dividend policy?
- How many employees does Apple have?
- What are Apple's main product categories?
- What did Apple say about China revenue?
- What is Apple's gross margin for 2025?
    """)

    st.divider()
    st.header("🛠️ Pipeline Info")
    st.markdown(f"""
- **Embedding model:** `{EMBEDDING_MODEL}` (local)
- **LLM:** `{GEMINI_MODEL}` (REST API)
- **Vector DB:** ChromaDB (local)
- **Chunks:** {collection.count()}
- **Top-K retrieval:** {TOP_K} chunks per query
    """)

    st.divider()
    if st.button("🗑️ Clear chat history"):
        st.session_state.messages = []
        st.rerun()
