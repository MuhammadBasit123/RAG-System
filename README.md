# RAG Question Answering System

A **Retrieval-Augmented Generation (RAG)** system that answers questions grounded in a document corpus, built for the assignment with all four required features:

| Requirement | Implementation |
|---|---|
| Two chunking strategies | Strategy A (fixed-size) vs Strategy B (sentence-aware + overlap) |
| Answer shows passages used | Source file + page number + score on every answer |
| "I don't know" fallback | Two-layer gate: similarity threshold + prompt instruction |
| 50+ page corpus | 7 Wikipedia AI articles auto-downloaded (~120 pages) |

---

## Quick Start (3 steps)

### Step 1 — Install dependencies
```bash
pip install -r requirements.txt
```
> **Note**: `sentence-transformers` will download a ~90 MB model on first run.

### Step 2 — Configure API key
```bash
copy .env.example .env
```
Edit `.env` and paste your **Groq API key** (free at [console.groq.com](https://console.groq.com)).

### Step 3 — Build the system
```bash
python download_corpus.py   # Download 7 Wikipedia articles as PDFs (~120 pages)
python build_index.py       # Chunk, embed, and index everything
streamlit run app.py        # Launch the web app
```

---

## How It Works

```
Question
   │
   ▼
[Embedder] — all-MiniLM-L6-v2 → 384-dim query vector
   │
   ▼
[FAISS Index] — cosine similarity search (top-k chunks)
   │
   ├─ top score < threshold (0.35)?
   │       └─► "I don't know" ← Layer 1 gate (hard block)
   │
   ▼
[Groq LLM] — llama3-8b-8192 with injected passages
   │
   │ prompt says: "if not in passages → I don't know"
   │       └─► Layer 2 gate (soft instruction)
   │
   ▼
Answer + Citations (source file, page number, similarity score)
```

---

## Chunking Strategies

| Feature | Strategy A | Strategy B |
|---|---|---|
| Method | Fixed character windows | Sentence tokenization (NLTK) |
| Chunk size | 800 chars (exact) | ~600 chars (target) |
| Overlap | **None** | **~150 chars** |
| Sentence-aware | No — cuts mid-sentence | Yes — respects boundaries |
| Typical use case | Uniform indexing | Better recall at boundaries |

---

## Project Structure

```
Ass1/
├── corpus/                  # PDFs (auto-downloaded)
├── data/                    # chunks_a.json, chunks_b.json
├── indices/                 # strategy_a.index, strategy_b.index
├── src/
│   ├── extractor.py         # PDF → (page_num, text) extraction
│   ├── chunker.py           # Strategy A + Strategy B implementations
│   ├── embedder.py          # sentence-transformers wrapper
│   ├── retriever.py         # FAISS search + threshold gate (Layer 1)
│   ├── generator.py         # Groq LLM call + prompt (Layer 2)
│   └── pipeline.py          # End-to-end orchestration
├── evaluation/
│   ├── test_questions.json  # 20 questions (15 in-corpus, 5 out-of-corpus)
│   ├── evaluator.py         # Runs all questions, prints comparison table
│   └── results.json         # Generated after running evaluator
├── app.py                   # Streamlit web UI
├── build_index.py           # One-time index builder
├── download_corpus.py       # Corpus downloader
└── requirements.txt
```

---

## Running the Evaluation

```bash
python evaluation/evaluator.py
```

This calls the Groq API for each of the 20 test questions × 2 strategies, then prints:
- A per-question results table (score, gated, correct?)
- A summary stats table (hit rate per strategy)
- Up to 3 divergence examples where the strategies gave different results

Results are saved to `evaluation/results.json` and displayed in the Streamlit app's **Evaluation Results** tab.

---

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `GROQ_API_KEY` | *(required)* | Your Groq API key |
| `LLM_BACKEND` | `groq` | `groq` or `ollama` |
| `OLLAMA_BASE_URL` | `http://localhost:11434/v1` | Only if using Ollama |
