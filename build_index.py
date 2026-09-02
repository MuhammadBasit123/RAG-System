"""
Build FAISS indices for both chunking strategies.

Run ONCE after populating the corpus/ directory (via download_corpus.py
or by manually adding your own PDFs).

What this script does:
  1. Extracts text + page numbers from every PDF in corpus/
  2. Chunks with Strategy A (fixed-size) → embeds → saves FAISS index
  3. Chunks with Strategy B (sentence-aware) → embeds → saves FAISS index

Output:
  data/chunks_a.json, data/chunks_b.json  — chunk metadata
  indices/strategy_a.index                 — FAISS index for Strategy A
  indices/strategy_b.index                 — FAISS index for Strategy B

Usage:
    python build_index.py
"""

from __future__ import annotations
import json
import os

import faiss
import numpy as np

from src.extractor import extract_corpus
from src.chunker import chunk_strategy_a, chunk_strategy_b
from src.embedder import embed_texts

CORPUS_DIR = "corpus"
DATA_DIR = "data"
INDEX_DIR = "indices"


# ── Helpers ───────────────────────────────────────────────────────────────────

def build_and_save(
    chunks: list[dict],
    label: str,
    index_path: str,
    chunks_path: str,
) -> None:
    print(f"\n" + "-"*55)
    print(f"  {label}  ({len(chunks)} chunks)")
    print("-"*55)

    if not chunks:
        print("  [WARNING] No chunks produced -- skipping.")
        return

    # Embed
    texts = [c["text"] for c in chunks]
    print("  Embedding ...")
    embeddings: np.ndarray = embed_texts(texts, show_progress=True)

    # Build FAISS index
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)        # inner product = cosine (vecs normalised)
    faiss.normalize_L2(embeddings)        # extra safety pass
    index.add(embeddings)

    # Persist
    faiss.write_index(index, index_path)
    print(f"  Saved index  -> {index_path}  ({index.ntotal} vectors)")

    with open(chunks_path, "w", encoding="utf-8") as fh:
        json.dump(chunks, fh, ensure_ascii=False, indent=2)
    print(f"  Saved chunks -> {chunks_path}  ({len(chunks)} entries)")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(INDEX_DIR, exist_ok=True)

    # ── Extract ────────────────────────────────────────────────────────────
    print(f"Extracting text from PDFs in '{CORPUS_DIR}/' ...")
    try:
        pages = extract_corpus(CORPUS_DIR)
    except FileNotFoundError as exc:
        print(f"\n[WARNING] {exc}")
        print("Run `python download_corpus.py` first, then retry.")
        return

    n_docs = len(set(p["source"] for p in pages))
    print(f"\nExtracted {len(pages)} pages from {n_docs} PDF(s)")

    if len(pages) < 50:
        print(
            f"\n[WARNING] only {len(pages)} pages found. "
            "Assignment requires 50+. Add more PDFs to corpus/."
        )

    # ── Strategy A ─────────────────────────────────────────────────────────
    print("\nChunking — Strategy A (fixed-size, no overlap) ...")
    chunks_a = chunk_strategy_a(pages)
    print(f"  -> {len(chunks_a)} chunks")
    build_and_save(
        chunks_a,
        "Strategy A",
        os.path.join(INDEX_DIR, "strategy_a.index"),
        os.path.join(DATA_DIR, "chunks_a.json"),
    )

    # ── Strategy B ─────────────────────────────────────────────────────────
    print("\nChunking — Strategy B (sentence-aware, with overlap) ...")
    chunks_b = chunk_strategy_b(pages)
    print(f"  -> {len(chunks_b)} chunks")
    build_and_save(
        chunks_b,
        "Strategy B",
        os.path.join(INDEX_DIR, "strategy_b.index"),
        os.path.join(DATA_DIR, "chunks_b.json"),
    )

    # ── Summary ────────────────────────────────────────────────────────────
    print("\n" + "="*55)
    print("  Index building complete!")
    print(f"  Strategy A  ->  {len(chunks_a):,} chunks")
    print(f"  Strategy B  ->  {len(chunks_b):,} chunks")
    print("="*55)
    print("\nNow run:  streamlit run app.py")


if __name__ == "__main__":
    main()
