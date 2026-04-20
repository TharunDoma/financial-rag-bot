"""
02_ingest_pdf.py
----------------
Purpose : Extract raw text from a SEC 10-K PDF, page by page.
          This is the EXTRACT step — the "E" in ETL.

DE Concept: In a real enterprise pipeline (think: Airflow DAG, AWS Glue job),
            the Extract step pulls raw data from a source system and lands it
            in a staging area — untouched, unmodified. We never transform
            during extraction. We extract first, validate second, transform third.

            Our "source system" here is the PDF file.
            Our "staging area" is the list of page dictionaries we build below.

Usage:
    python 02_ingest_pdf.py
"""

import fitz  # PyMuPDF — 'fitz' is the internal module name
import os

# ---------------------------------------------------------------------------
# CONFIG — One place to change the filename, nothing else needs to touch it.
#          In production this would come from a config file or pipeline param.
# ---------------------------------------------------------------------------
PDF_PATH = os.path.join("data", "apl10-K-2025-As-Filed.pdf")


def extract_text_from_pdf(pdf_path: str) -> list[dict]:
    """
    Opens a PDF and extracts text from every page.

    Returns a list of dicts — one per page — shaped like:
        [
            {"page": 1, "text": "...raw text from page 1..."},
            {"page": 2, "text": "...raw text from page 2..."},
            ...
        ]

    DE Concept: This structured list is our STAGING LAYER.
                Each dict is like a row in a staging table:
                    page_number  |  raw_text
                    -------------|-----------------------------
                         1       |  "Apple Inc. 10-K ..."
                         2       |  "Risk Factors ..."

                Before this project is done, these rows will flow into
                ChromaDB — our vector data store. Same concept, different medium.
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(
            f"\n[ERROR] PDF not found at: {pdf_path}\n"
            "  → Make sure you copied the file into the data/ folder."
        )

    print(f"\n📂 Opening: {pdf_path}")

    pages = []

    # fitz.open() is our "source connector" — equivalent to a database cursor
    # or an S3 file handle in a cloud pipeline.
    with fitz.open(pdf_path) as doc:
        total_pages = len(doc)
        print(f"   Total pages found: {total_pages}")

        for page_num, page in enumerate(doc, start=1):
            raw_text = page.get_text()  # Extract raw text from this page

            # Skip blank pages — no point storing empty rows (data quality!)
            if raw_text.strip():
                pages.append({
                    "page": page_num,
                    "text": raw_text
                })

    print(f"   Pages with content: {len(pages)} / {total_pages}")
    return pages


def preview_pages(pages: list[dict], num_pages: int = 3) -> None:
    """
    Print a short preview of the first few pages so you can visually
    confirm the extraction worked correctly.

    DE Concept: This is a lightweight DATA QUALITY CHECK — always validate
                your extract output before moving to the Transform step.
                In production, this would be a Great Expectations assertion
                or a dbt test. Here, we just eyeball it first.
    """
    print(f"\n--- PREVIEW: First {num_pages} pages of extracted text ---\n")
    for page_data in pages[:num_pages]:
        print(f"[Page {page_data['page']}]")
        # Show only the first 300 chars so the terminal doesn't flood
        print(page_data["text"][:300].strip())
        print("-" * 60)


# ---------------------------------------------------------------------------
# MAIN — Run the pipeline extract step
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Step 1: Extract
    pages = extract_text_from_pdf(PDF_PATH)

    # Step 2: Lightweight quality check
    preview_pages(pages, num_pages=3)

    # Step 3: Summary report
    total_chars = sum(len(p["text"]) for p in pages)
    print(f"\n✅ Extraction complete!")
    print(f"   Pages extracted : {len(pages)}")
    print(f"   Total characters: {total_chars:,}")
    print(f"\n⏭️  Next step: Chunking (02_chunk_text.py) — the 'T' in ETL.\n")
