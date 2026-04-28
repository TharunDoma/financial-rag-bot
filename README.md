# 📊 Financial Analyst RAG Bot

An end-to-end **Retrieval-Augmented Generation (RAG) pipeline** that lets you ask natural language questions about Apple's 2025 SEC 10-K filing and receive grounded, accurate answers — with source citations from the actual document.

> **Live Demo:** [your-app.streamlit.app](https://your-app.streamlit.app)

---

## 🎯 What This Project Demonstrates

This project was built to showcase **junior-to-mid level Data Engineering** skills across the full data product lifecycle — from raw file ingestion to a deployed, interactive AI application.

| Skill Area | Implementation |
|---|---|
| ETL Pipeline Design | Modular Extract → Transform → Load scripts |
| Data Transformation | Recursive chunking with sliding window overlap |
| Vector Database | ChromaDB with cosine similarity indexing |
| Embedding Strategy | Local `all-MiniLM-L6-v2` (no API cost, privacy-safe) |
| LLM Integration | Gemini REST API with direct HTTP calls |
| Secret Management | `.env` locally, Streamlit Secrets in production |
| Caching Layer | `st.cache_resource` + `st.cache_data` for query caching |
| Deployment | Streamlit Cloud with `runtime.txt` Python version pinning |
| Dependency Management | Pinned `requirements.txt` for reproducible environments |
| Production Patterns | Idempotent pipelines, absolute paths, fail-fast validation |

---

## 🏗️ Architecture

```
SEC 10-K PDF
     │
     ▼
┌─────────────────┐
│  EXTRACT        │  02_ingest_pdf.py
│  PyMuPDF        │  Raw text extraction, page-by-page
│  277,598 chars  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  TRANSFORM      │  03_chunk_text.py
│  LangChain      │  RecursiveCharacterTextSplitter
│  346 chunks     │  chunk_size=1000, overlap=200
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  LOAD           │  04_embed_store.py
│  ChromaDB       │  all-MiniLM-L6-v2 embeddings
│  Vector Store   │  Cosine similarity, persistent to disk
└────────┬────────┘
         │
     [Query Time]
         │
         ▼
┌─────────────────┐     ┌──────────────────┐
│  RETRIEVE       │────▶│  GENERATE        │
│  ChromaDB       │     │  Gemini REST API │
│  Top-5 chunks   │     │  Grounded answer │
└─────────────────┘     └──────────────────┘
         │
         ▼
┌─────────────────┐
│  PRESENT        │  06_app.py
│  Streamlit UI   │  Chat interface + source citations
└─────────────────┘
```

---

## 🛠️ Tech Stack

| Layer | Technology | Purpose |
|---|---|---|
| PDF Ingestion | `PyMuPDF (fitz)` | Extract text from SEC filings |
| Chunking | `LangChain RecursiveCharacterTextSplitter` | Sliding window text transformation |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` | Local semantic embedding (384-dim vectors) |
| Vector Store | `ChromaDB` | Persistent local vector database |
| LLM | `Google Gemini 2.0 Flash Lite` | Answer generation via REST API |
| Frontend | `Streamlit` | Interactive chat UI |
| Security | `python-dotenv` | Environment-based secret management |

---

## 🚀 Run Locally

### Prerequisites
- Python 3.11+
- A free [Google AI Studio](https://aistudio.google.com) API key

### Setup

```bash
# 1. Clone the repo
git clone https://github.com/YOUR_USERNAME/financial-rag-bot.git
cd financial-rag-bot

# 2. Create and activate virtual environment
python -m venv venv
source venv/bin/activate      # Mac/Linux
venv\Scripts\activate         # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure your API key
echo 'GEMINI_API_KEY="your_key_here"' > .env

# 5. Add your SEC 10-K PDF to the data/ folder
# Download from: https://www.sec.gov/cgi-bin/browse-edgar

# 6. Run the ETL pipeline (one-time setup)
python 02_ingest_pdf.py
python 03_chunk_text.py
python 04_embed_store.py

# 7. Launch the app
streamlit run 06_app.py
```

---

## 📁 Project Structure

```
financial_rag_bot/
│
├── data/                    ← Raw SEC 10-K PDFs (source data)
├── chroma_db/               ← Persistent vector store (built once)
│
├── 01_test_env.py           ← Environment validation (pre-flight check)
├── 02_ingest_pdf.py         ← E: PDF text extraction
├── 03_chunk_text.py         ← T: Recursive chunking transformation
├── 04_embed_store.py        ← L: Embedding + ChromaDB load
├── 05_rag_query.py          ← Terminal query interface
├── 06_app.py                ← Streamlit web application
│
├── requirements.txt         ← Pinned dependencies
├── runtime.txt              ← Python version lock (3.11)
└── .env                     ← Local secrets (never committed)
```

---

## 💡 Key Engineering Decisions

**Why local embeddings?** Using `all-MiniLM-L6-v2` runs entirely on CPU — no API costs, no data leaving the machine, and no quota limits. In production, this decision is made based on data sensitivity and cost-per-query trade-offs.

**Why ChromaDB?** Local persistent vector store that survives restarts without re-embedding. The same architectural pattern scales to managed services like Pinecone or Weaviate when team/traffic requirements grow.

**Why direct REST API instead of SDK?** SDK version conflicts across Python versions (especially 3.14 on Streamlit Cloud) make raw HTTP calls more portable and easier to debug. This is a production pattern used when SDK stability is uncertain.

**Why modular scripts instead of one file?** Each script has a single responsibility and can be run, tested, and debugged independently — the same principle as separate Airflow DAG tasks or dbt models in an enterprise pipeline.

---

## 🔮 Roadmap

- [ ] Multi-document support (compare multiple 10-K filings)
- [ ] Airflow DAG for scheduled re-indexing when new filings drop
- [ ] Source page citation in every answer
- [ ] Swap ChromaDB for Pinecone for multi-user scale
- [ ] Add dbt-style data quality tests on chunk output

---

## 👨‍💻 Author

**Tharun** — MS Computer Science  
Built as part of a structured Data Engineering learning path covering ETL pipelines, vector databases, and LLM integration.

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue)](https://linkedin.com/in/YOUR_PROFILE)
[![GitHub](https://img.shields.io/badge/GitHub-Follow-black)](https://github.com/YOUR_USERNAME)
