"""
03_chunk_text.py
----------------
Purpose : Take the raw extracted text from the PDF and split it into
          smaller, meaningful pieces called "chunks".
          This is the TRANSFORM step — the "T" in ETL.

DE Concept: In a traditional ETL pipeline, the Transform step cleans,
            reshapes, and prepares raw data for loading into the target store.
            Here our "target store" is ChromaDB (a vector database).
            ChromaDB doesn't store whole documents — it stores chunks.
            Chunking IS our transformation.

Why chunk at all?
-----------------
Two hard constraints force us to chunk:

  1. LLMs have a context window limit (e.g., Gemini 2.5 Flash: ~1M tokens,
     but embeddings models are much smaller — typically 512-8192 tokens).
     You cannot embed 277,000 characters as one unit.

  2. Precision: When a user asks "What are Apple's cybersecurity risks?",
     you want to retrieve the 3-4 paragraphs that answer that — NOT the
     entire 80-page document. Smaller chunks = more precise retrieval.

Think of it like this: instead of handing someone a 500-page book to find
one answer, you hand them the right page. That's what chunking enables.

Usage:
    python 03_chunk_text.py
"""

import fitz  # PyMuPDF
import os
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
PDF_PATH = os.path.join("data", "apl10-K-2025-As-Filed.pdf")

# CHUNK_SIZE: Maximum number of characters per chunk.
# Think of this as your "batch size" in a data pipeline.
# Too large → loses retrieval precision
# Too small → loses context (a sentence without its paragraph means nothing)
# 1000 chars ≈ ~200 words — a solid paragraph. Industry standard starting point.
CHUNK_SIZE = 1000

# CHUNK_OVERLAP: How many characters to repeat between consecutive chunks.
# DE Concept: This is like a sliding window — we overlap so that a sentence
# at the boundary of one chunk isn't "orphaned" from its context.
# If chunk 1 ends mid-sentence, chunk 2 starts 200 chars back to capture it.
CHUNK_OVERLAP = 200


# ---------------------------------------------------------------------------
# STEP 1: EXTRACT (reuse our extraction logic inline — keeping it simple)
# ---------------------------------------------------------------------------
def extract_full_text(pdf_path: str) -> str:
    """
    Extracts all text from the PDF and joins it into one continuous string.
    We combine all pages because the splitter will handle boundaries for us.
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"[ERROR] PDF not found: {pdf_path}")

    full_text = ""
    with fitz.open(pdf_path) as doc:
        for page in doc:
            full_text += page.get_text()

    return full_text


# ---------------------------------------------------------------------------
# STEP 2: TRANSFORM — Split into chunks
# ---------------------------------------------------------------------------
def chunk_text(full_text: str) -> list:
    """
    Splits raw text into overlapping chunks using LangChain's
    RecursiveCharacterTextSplitter.

    'Recursive' means it tries to split on natural boundaries in this order:
        1. Paragraph breaks (double newline)
        2. Single newlines
        3. Spaces
        4. Individual characters (last resort)

    This is smarter than splitting every 1000 chars blindly — it respects
    the natural structure of the document, just like a good ETL transform
    respects the schema of its source data.

    Returns a list of LangChain Document objects, each with:
        .page_content  → the chunk text
        .metadata      → dict (empty for now, we'll add source info later)
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,          # measure size in characters
    )

    chunks = splitter.create_documents([full_text])
    return chunks


# ---------------------------------------------------------------------------
# STEP 3: DATA QUALITY CHECK — Preview and validate the output
# ---------------------------------------------------------------------------
def preview_chunks(chunks: list, num_chunks: int = 3) -> None:
    """
    Print a preview of the first few chunks so we can visually confirm
    the splits look clean and meaningful.
    """
    print(f"\n--- PREVIEW: First {num_chunks} chunks ---\n")
    for i, chunk in enumerate(chunks[:num_chunks]):
        print(f"[Chunk {i + 1}] — {len(chunk.page_content)} chars")
        print(chunk.page_content[:300].strip())
        print("-" * 60)


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # E — Extract
    print("\n📂 Extracting text from PDF...")
    full_text = extract_full_text(PDF_PATH)
    print(f"   Total characters extracted: {len(full_text):,}")

    # T — Transform (chunk)
    print("\n✂️  Chunking text...")
    chunks = chunk_text(full_text)

    # Quality check
    preview_chunks(chunks, num_chunks=3)

    # Summary
    avg_chunk_size = sum(len(c.page_content) for c in chunks) // len(chunks)
    print(f"\n✅ Chunking complete!")
    print(f"   Total chunks created : {len(chunks)}")
    print(f"   Avg chunk size       : {avg_chunk_size} chars")
    print(f"   Chunk size setting   : {CHUNK_SIZE} chars")
    print(f"   Overlap setting      : {CHUNK_OVERLAP} chars")
    print(f"\n⏭️  Next step: Embeddings + ChromaDB (04_embed_store.py) — the 'L' in ETL.\n")
