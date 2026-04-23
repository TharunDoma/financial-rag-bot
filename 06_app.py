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
# CONFIG
# ---------------------------------------------------------------------------
load_dotenv()

# Works locally (.env) and on Streamlit Cloud (st.secrets)
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "") or st.secrets.get("GEMINI_API_KEY", "")
if not GEMINI_API_KEY:
    st.error("❌ GEMINI_API_KEY is missing. Add it in Streamlit Cloud → Manage App → Secrets.")
    st.stop()

# Absolute path — works on any machine regardless of where the app is launched from
BASE_DIR        = os.path.dirname(os.path.abspath(__file__))
CHROMA_DB_PATH  = os.path.join(BASE_DIR, "chroma_db")
COLLECTION_NAME = "apple_10k_2025"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
GEMINI_MODEL    = "gemini-2.0-flash"
TOP_K           = 5

# ---------------------------------------------------------------------------
# PAGE SETUP — Must be first Streamlit call
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Financial Analyst RAG Bot",
    page_icon="📊",
    layout="centered",
)

# ---------------------------------------------------------------------------
# CACHED RESOURCE LOADERS
# DE Concept: @st.cache_resource loads once and reuses across all user
#             sessions — same principle as a connection pool in a backend
#             service. We don't reconnect to ChromaDB on every question.
# ---------------------------------------------------------------------------
@st.cache_resource
def load_collection():
    embedding_fn = SentenceTransformerEmbeddingFunction(model_name=EMBEDDING_MODEL)
    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    return client.get_collection(name=COLLECTION_NAME, embedding_function=embedding_fn)


def load_gemini():
    # No SDK — just return the API key, we call REST directly
    return GEMINI_API_KEY


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


def get_answer(api_key: str, question: str, chunks: list[str]) -> str:
    """
    Calls Gemini via direct REST API — no SDK.
    This gives us transparent error messages and zero SDK version conflicts.
    """
    prompt  = build_prompt(question, chunks)
    url     = (
        f"https://generativelanguage.googleapis.com/v1beta/models/"
        f"{GEMINI_MODEL}:generateContent?key={api_key}"
    )
    payload = {"contents": [{"parts": [{"text": prompt}]}]}

    resp = requests.post(url, json=payload, timeout=30)

    if resp.status_code != 200:
        # Show the raw API error so we can debug it
        return f"❌ API Error {resp.status_code}: {resp.text}"

    return resp.json()["candidates"][0]["content"]["parts"][0]["text"]


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
with st.spinner("Loading vector store and AI model..."):
    collection = load_collection()
    gemini     = load_gemini()

st.success(f"✅ Ready — {collection.count()} chunks loaded from Apple 10-K 2025")

# ---------------------------------------------------------------------------
# CHAT HISTORY — Stored in session state (survives reruns within a session)
# DE Concept: Session state is in-memory state management — the same pattern
#             as a stateful streaming job that tracks context between events.
# ---------------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

# Render all previous messages
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

    # Show user message
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Retrieve + Generate
    with st.chat_message("assistant"):
        with st.spinner("Retrieving relevant data and generating answer..."):
            chunks = retrieve_chunks(collection, prompt)
            try:
                answer = get_answer(gemini, prompt, chunks)
            except Exception as e:
                st.error(f"❌ Gemini API error: {str(e)}")
                st.stop()

        st.markdown(answer)

        with st.expander("📄 View source chunks retrieved from 10-K"):
            for i, chunk in enumerate(chunks, 1):
                st.markdown(f"**Chunk {i}:**")
                st.text(chunk[:400])
                st.divider()

    # Save assistant message with sources
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": chunks,
    })

# ---------------------------------------------------------------------------
# SIDEBAR — Suggested questions and info
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
- **LLM:** `{GEMINI_MODEL}`
- **Vector DB:** ChromaDB (local)
- **Chunks:** {collection.count()}
- **Top-K retrieval:** {TOP_K} chunks per query
    """)

    st.divider()
    if st.button("🗑️ Clear chat history"):
        st.session_state.messages = []
        st.rerun()
