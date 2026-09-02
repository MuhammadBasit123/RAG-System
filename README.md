# RAG-System
Retrieval-Augmented Generation (RAG) system built with LangChain, Chroma, and OpenAI embeddings.


## Table of Contents

1. [Why RAG Exists](#1-why-rag-exists)
2. [RAG Fundamentals](#2-rag-fundamentals)
3. [The Injection (Ingestion) Pipeline](#3-the-injection-ingestion-pipeline)
4. [The Retrieval Pipeline](#4-the-retrieval-pipeline)
5. [Cosine Similarity — The Math Behind Retrieval](#5-cosine-similarity--the-math-behind-retrieval)
6. [Answer Generation](#6-answer-generation)
7. [Conversational RAG (History-Aware Retrieval)](#7-conversational-rag-history-aware-retrieval)
8. [Chunking Strategies Deep Dive](#8-chunking-strategies-deep-dive)
9. [Multi-Query Retrieval](#9-multi-query-retrieval)
10. [Reciprocal Rank Fusion (RRF)](#10-reciprocal-rank-fusion-rrf)
11. [Hybrid Search](#11-hybrid-search)
12. [Reranking](#12-reranking)
13. [Multimodal RAG](#13-multimodal-rag)
14. [Production Best Practices](#14-production-best-practices)
15. [Tech Stack Cheat Sheet](#15-tech-stack-cheat-sheet)
16. [Glossary](#16-glossary)

---

## 1. Why RAG Exists

Large Language Models are trained on a fixed snapshot of data and have no built-in knowledge of your private documents, your company's internal policies, or anything created after their training cutoff. They also have a **context window** — a hard limit on how much text they can process in a single request.

| Model (approx.) | Context Window |
|---|---|
| Claude Sonnet-class models | ~200,000 tokens |
| GPT-5-class models | ~400,000 tokens |
| GPT-4.1-class models | ~1,000,000 tokens |

A **token** is roughly a word or word-fragment. Even a 1-million-token window sounds huge, but a mid-sized company's document store can easily reach hundreds of gigabytes to a terabyte of text — far more than any context window could hold. A single enterprise data center's worth of documents (~1 petabyte) works out to roughly a quadrillion tokens. You simply cannot stuff "all the documents" into a prompt.

**RAG solves this** by not sending everything to the model. Instead, it:
1. Stores your knowledge base in a searchable form ahead of time.
2. At question time, retrieves only the small number of passages that are actually relevant.
3. Hands those passages + the question to the LLM, which answers using that grounded context instead of guessing from memory.

This reduces hallucination, keeps costs down, and lets the system answer questions about data the model was never trained on.

---

## 2. RAG Fundamentals

A RAG system is best understood as **two independent pipelines**:

- **Injection Pipeline** — prepares and stores your knowledge base (done once, or whenever data changes).
- **Retrieval Pipeline** — runs every time a user asks a question.

### Vector Embeddings

An **embedding model** (different from an LLM) converts text into a list of numbers — a **vector** — that represents the text's meaning mathematically. For example, a toy 3-dimension embedding might represent:

- `cat` → `[34, 8, 7.5]`
- `kitten` → `[33, 8, 7.2]`
- `dog` → `[35, 8, 9]`
- `elephant` → `[95, 62, 45]`

Words with similar meaning end up with vectors that are numerically close together — cat and kitten are nearly identical, while elephant sits far away. In practice, embedding models don't use 3 dimensions; they use hundreds or thousands:

| Model | Typical Dimensions |
|---|---|
| OpenAI text-embedding-3-small | 1,536 (reducible) |
| OpenAI text-embedding-3-large | 3,072 (reducible) |
| Voyage-3-large | 1,024 |
| Cohere embed models | varies |

More dimensions capture more semantic nuance but cost more to compute and store.

**Critical rule:** you must use the *exact same* embedding model and dimension count for both your documents and your user queries. Mixing embedding models is like storing information in one language and searching in another — the vectors won't be comparable.

### Vector Databases

Once text is embedded, the vectors need somewhere to live that supports fast similarity search. Common choices:

- **Chroma DB** — simple, local, popular for tutorials/small projects.
- **Pinecone** — managed, cloud-native, scales easily.
- **Weaviate** — open-source, feature-rich.
- **FAISS** — Meta's high-performance similarity search library.
- **PostgreSQL (with pgvector)** — good if you already run Postgres and want vectors alongside relational data.

---

## 3. The Injection (Ingestion) Pipeline

This is the "build the knowledge base" half of RAG. Steps:

### Step 1 — Load Documents
Use a document loader appropriate to your file type (`TextLoader`, `DirectoryLoader`, PDF loaders, CSV loaders, web loaders, etc. in LangChain). Loading produces a list of "Document" objects, each holding `page_content` (the raw text) and `metadata` (like the source filename).

### Step 2 — Chunk the Documents
Large documents must be broken into smaller pieces ("chunks") before embedding, because:
- Embedding models perform better on focused passages than on huge blocks of text.
- Smaller chunks make retrieval more precise — you get back the specific paragraph that answers the question, not an entire 50-page document.

A common tool is LangChain's `RecursiveCharacterTextSplitter`, configured with a `chunk_size` (e.g., 800 characters) and optionally a `chunk_overlap` (some repeated text between consecutive chunks so context isn't lost at chunk boundaries).

**Example scale:** 5 Wikipedia-style articles totaling tens of thousands of characters might split into ~800 individual chunks.

### Step 3 — Embed and Store
Each chunk is passed through the embedding model and the resulting vector — along with the original text and metadata — is stored in the vector database. It's important to explicitly configure the database's similarity metric as **cosine similarity**, since that is the standard for RAG retrieval quality.

At the end of this pipeline you have a persistent vector store (e.g., a `chroma_db` folder) containing every chunk's text and its vector representation, ready to be searched.

---

## 4. The Retrieval Pipeline

This is the "answer a question" half of RAG, and it runs on every user query.

1. **Embed the query** — convert the user's natural-language question into a vector using the *same* embedding model used during injection.
2. **Compare against the vector store** — a **retriever** component computes a similarity score between the query vector and every stored chunk vector.
3. **Rank and select top-K** — the retriever returns the top `k` chunks (e.g., top 3 or top 5) with the highest similarity scores.
4. **(Optional) Score threshold** — instead of (or in addition to) a fixed `k`, you can set a minimum similarity score (e.g., 0.3) so weak matches are excluded entirely.

Important nuance: **no LLM is involved in retrieval.** It is pure vector math. The returned chunks are the *exact original text* from your source documents — nothing is generated or paraphrased at this stage. This is why retrieval quality can be validated directly: you can literally search the source file for the retrieved sentence and confirm it appears verbatim.

A good way to sanity-check a retriever while building a system is to manually inspect whether the retrieved chunks actually contain the answer to a set of test questions before ever involving the LLM in generation.

---

## 5. Cosine Similarity — The Math Behind Retrieval

Cosine similarity measures the **angle** between two vectors, not their length/magnitude. Its formula is:

```
cosine_similarity(A, B) = (A · B) / (|A| × |B|)
```

- `A · B` is the **dot product** (multiply corresponding dimensions, then sum).
- `|A|` and `|B|` are the **magnitudes** (lengths) of the vectors.

**Score range:** 0 (completely dissimilar) to 1 (identical direction / maximum similarity).

**Key simplification:** Most modern embedding models (OpenAI's included) output *normalized* vectors — meaning every vector already has a magnitude of exactly 1. When both magnitudes are 1, the formula collapses to:

```
cosine_similarity(A, B) = A · B   (just the dot product)
```

This is why, in many RAG tutorials, "dot product" and "cosine similarity" are used almost interchangeably.

### Worked Example
Query vector: `[0.6, 0.3, 0.2]`
Chunk vector: `[0.7, 0.4, 0.1]`

```
Dot product = (0.6×0.7) + (0.3×0.4) + (0.2×0.1)
            = 0.42 + 0.12 + 0.02
            = 0.56
```

With normalized vectors, similarity ≈ 0.56 — a decent, moderately strong match. Scores closer to 1 (e.g., 0.85–0.95) indicate a near-perfect topical match.

### Rough Score Interpretation

| Score | Meaning |
|---|---|
| 1.0 | Perfect match |
| 0.7 – 0.9 | Very good match |
| 0.5 – 0.7 | Good match |
| 0.3 – 0.5 | Moderate match |
| 0.0 – 0.3 | Poor match |

The retriever computes this score for *every* chunk in the database, ranks them, and returns the top K — modern vector databases do this over millions of chunks in milliseconds using approximate-nearest-neighbor indexing (e.g., HNSW).

---

## 6. Answer Generation

Once the top-K chunks are retrieved, the final step combines them with the user's question into a single prompt sent to the LLM. A typical prompt pattern:

```
Based on the following documents, please answer this question: {query}

Documents:
{chunk_1_text}
{chunk_2_text}
...

Please provide a clear, helpful answer using ONLY the information
from these documents. If you can't find the answer, say
"I don't have enough information to answer the question based on
the provided documents."
```

This instruction is what "grounds" the model — it explicitly forbids the LLM from falling back on its own pretrained knowledge and forces it to admit uncertainty rather than hallucinate. The LLM then reads through the supplied passages and synthesizes a natural-language answer, citing only what was actually retrieved.

That's the complete, minimal RAG loop: **retrieve → construct prompt → generate → respond.**

---

## 7. Conversational RAG (History-Aware Retrieval)

A single question-answer exchange is easy, but real conversations involve follow-up questions that rely on earlier context — pronouns like "it" or "their," or implicit references.

**The problem:** Vector search has no memory. If a user asks "Tell me about Nvidia's GPU architecture" and then follows up with "What is their revenue from it?", searching for the literal phrase "their revenue from it" retrieves nothing useful, because embeddings don't know what "their" or "it" refer to.

**The fix — Query Reformulation:** Before retrieval happens, an LLM call rewrites the latest question into a fully standalone, context-independent version using the chat history. For example:

- Raw follow-up: *"What is their revenue from it?"*
- Reformulated: *"What is Nvidia's revenue from its Hopper GPU architecture?"*

This reformulated query is what actually gets embedded and searched — not the user's literal wording.

### The Conversational RAG Flow
1. Maintain a running `chat_history` list (user turns + AI turns).
2. On a new user message: if history exists, ask the LLM to rewrite the question into a standalone query.
3. Retrieve chunks using the *reformulated* query.
4. Build the generation prompt using the *original* user question (for natural phrasing) plus the retrieved chunks plus the chat history for context.
5. Get the LLM's answer.
6. Append both the user's question and the AI's answer to `chat_history` for the next turn.

This adds one extra LLM call per turn (for reformulation) but is essential for any assistant meant to hold a real dialogue rather than answer isolated one-off questions.

---

## 8. Chunking Strategies Deep Dive

Chunking is repeatedly called out as **the single biggest factor in RAG quality** — bad chunking leads to fragmented, out-of-context retrieval no matter how good your embedding model or LLM is.

### Common Strategies

- **Fixed-size chunking** — split every N characters/tokens, optionally with overlap. Simple but can cut sentences or ideas in half.
- **Recursive character splitting** — tries to split on natural boundaries first (paragraphs, then sentences, then words) before falling back to a hard character limit. This is the most commonly used default (e.g., LangChain's `RecursiveCharacterTextSplitter`).
- **Chunk overlap** — repeating a portion of text between consecutive chunks (e.g., 100–200 characters) so an idea that spans a chunk boundary isn't lost entirely in either chunk.
- **Semantic chunking** — instead of a fixed size, split where the *meaning* shifts (detected via embedding similarity between adjacent sentences), producing chunks that are topically coherent regardless of length.
- **Document-structure-aware chunking** — for structured content (Markdown, HTML, code), split along headers, sections, or function boundaries so each chunk is a self-contained logical unit.
- **Contextual chunk headers** — prepend a short summary of the document/section to each chunk before embedding, so the chunk retains context about where it came from even in isolation.

### Practical Guidance
- Chunk size is a trade-off: too small and chunks lose context; too large and irrelevant text dilutes the embedding and wastes context window.
- A typical starting point is 500–1,000 characters (or a few hundred tokens) per chunk with modest overlap, then tune based on retrieval quality testing.
- Always test chunking choices against real queries — retrieve, inspect the returned chunks, and check whether they actually contain complete, usable answers.

---

## 9. Multi-Query Retrieval

A single user query might be phrased in a way that doesn't closely match how the answer is worded in your documents (vocabulary mismatch). **Multi-query retrieval** addresses this by having an LLM generate several *reworded variations* of the original question, then running retrieval separately for each variation.

**Example:**
Original: *"How do I reset my password?"*
Generated variants might include:
- "What is the process for changing a forgotten password?"
- "Steps to recover account access after losing a password"
- "How can a user update their login credentials?"

Each variant retrieves its own set of top-K chunks. The results across all variants are then combined (with duplicates removed) into a broader, more robust candidate pool — increasing the odds that the truly relevant passage is captured, even if it doesn't share vocabulary with the user's exact phrasing.

---

## 10. Reciprocal Rank Fusion (RRF)

When you have multiple ranked result lists (e.g., from multi-query retrieval, or from combining vector search with keyword search — see Hybrid Search below), you need a principled way to merge them into one final ranking. **Reciprocal Rank Fusion** is the standard technique.

### How It Works
For each document, RRF sums a score contribution from every list it appears in, based on its *rank position* in that list (not its raw similarity score):

```
RRF_score(doc) = Σ  1 / (k + rank(doc in list_i))
```

- `rank(doc in list_i)` is the document's position (1st, 2nd, 3rd...) in a given result list.
- `k` is a small constant (commonly 60) that dampens the influence of very high ranks and smooths the scoring.
- The sum is taken across all the result lists the document appears in.

**Why rank-based instead of score-based?** Different retrieval methods (vector similarity vs. keyword search) produce scores on totally different, non-comparable scales. Rank position, however, is directly comparable across methods — a document ranked #1 in two different lists is clearly a strong candidate, regardless of the raw score each method assigned. RRF elegantly rewards documents that consistently rank well across multiple retrieval strategies.

---

## 11. Hybrid Search

Pure vector (semantic) search is excellent at understanding meaning and paraphrasing, but it can underperform on queries involving exact terms — product codes, acronyms, names, or rare jargon — where a traditional keyword match is actually more reliable.

**Hybrid search** combines two retrieval methods:

1. **Dense retrieval (vector/semantic search)** — good at conceptual/meaning-based matches.
2. **Sparse retrieval (keyword search, typically BM25)** — good at exact term/lexical matches.

Both methods run against the same query, producing two ranked lists of candidate chunks. These lists are then merged — commonly using **Reciprocal Rank Fusion** (see above) — into a single, more robust final ranking that benefits from both semantic understanding and exact-match precision.

This is described as a technique that separates "amateur" RAG implementations from **enterprise-grade** systems, because production knowledge bases almost always contain a mix of conceptual content and precise structured terms (SKUs, legal clause numbers, technical identifiers) that pure vector search alone tends to miss.

---

## 12. Reranking

Retrieval (via cosine similarity or hybrid search) is fast but relatively coarse — it's optimized to search across potentially millions of chunks quickly, which requires some accuracy trade-offs. **Reranking** adds a second, more precise pass over a *smaller* candidate set.

### The Two-Stage Pattern
1. **Retrieve broadly** — pull a larger candidate pool than you actually need (e.g., top 20–50 chunks) using fast vector/hybrid search.
2. **Rerank precisely** — pass the user's query and each candidate chunk *together* through a **cross-encoder** model, which directly compares the pair and outputs a fine-grained relevance score.
3. **Keep the best few** — select the final top-K (e.g., top 3–5) from the reranked list to send to the LLM.

### Why Cross-Encoders Are More Accurate (But Slower)
- A standard embedding model (bi-encoder) encodes the query and each chunk *separately* into vectors, then compares them — fast, but the model never actually "sees" the query and chunk together.
- A **cross-encoder** feeds the query and chunk into the model *simultaneously*, letting it directly reason about how well they match — much more accurate, but too computationally expensive to run against an entire database. That's why it's only applied to the smaller shortlist from stage 1.

This retrieve-then-rerank pattern consistently improves answer quality in production RAG systems and is considered a core "advanced" technique.

---

## 13. Multimodal RAG

Real-world knowledge bases aren't pure text — they include tables, charts, scanned pages, and images (diagrams, screenshots, product photos). **Multimodal RAG** extends the standard pipeline to handle this.

### Common Approaches

- **Captioning approach** — run non-text elements (images, complex tables, slides) through a vision-capable model to generate a text description/caption. Store that caption's embedding alongside the regular text chunks, so a relevant image can be retrieved by matching its *description* to the query.
- **Table-aware parsing** — extract tables into a structured, LLM-readable text format (e.g., Markdown tables) rather than letting a table's rows collapse into unstructured prose during chunking, which preserves row/column relationships.
- **Direct visual embedding (e.g., ColPali-style approaches)** — instead of captioning, convert document pages directly into images and embed those images with a vision embedding model. At query time, the most relevant *page images* are retrieved and passed straight to a vision-capable LLM, skipping lossy text extraction entirely (useful for complex layouts, scanned documents, and image-heavy content).

Multimodal RAG is generally introduced as one of the more advanced, capstone-level topics, since it requires combining text embeddings, vision models, and sometimes multiple vector stores in one system.

---

## 14. Production Best Practices

Recurring lessons emphasized throughout the series for moving from a tutorial project to a real system:

- **Consistency is non-negotiable** — same embedding model, same dimensions, everywhere (ingestion and query time). Changing embedding models later requires re-embedding your *entire* knowledge base.
- **Always use cosine similarity** as your distance metric for text embeddings unless you have a specific reason not to.
- **Test retrieval independently from generation** — before trusting the LLM's final answer, verify the retriever is actually returning chunks that contain the answer. Feeding good chunks to a bad generation prompt is fixable; feeding bad chunks to a great LLM is not.
- **Chunking quality dominates everything downstream.** Invest time here before tuning anything else.
- **Score thresholds prevent irrelevant retrieval** — better to return zero chunks (and let the system say "I don't know") than to force in a low-similarity chunk that misleads the LLM.
- **Monitor and trace your pipeline** (e.g., with LangSmith or similar tooling) — track latency, cost per call, and what was actually retrieved for each query, especially once you move to open-source/local models where cost visibility disappears.
- **Layer techniques as needed, not by default** — multi-query, hybrid search, and reranking all add latency and cost. Start with a solid baseline pipeline, measure where it fails, and add the specific advanced technique that addresses that failure mode.

---

## 14b. Full Reference Implementation

A minimal, complete, runnable RAG system needs only two scripts.

### `ingest.py` — build the knowledge base (run once)

```python
import os
from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

load_dotenv()  # loads OPENAI_API_KEY from a .env file

def ingest(docs_path="docs", persist_dir="./chroma_db"):
    # 1. Load every .txt file in the docs folder
    loader = DirectoryLoader(docs_path, glob="*.txt", loader_cls=TextLoader)
    documents = loader.load()
    print(f"Loaded {len(documents)} documents")

    # 2. Chunk them — this is the step that matters most for quality
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,      # characters per chunk
        chunk_overlap=100,   # overlap so ideas aren't cut in half
    )
    chunks = splitter.split_documents(documents)
    print(f"Split into {len(chunks)} chunks")

    # 3. Embed + store — MUST use the same model at query time
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_dir,
        collection_metadata={"hnsw:space": "cosine"},
    )
    print(f"Vector store saved to {persist_dir}")

if __name__ == "__main__":
    ingest()
```

### `ask.py` — answer questions (run per query)

```python
import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain_core.messages import SystemMessage, HumanMessage

load_dotenv()

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")  # same model as ingest!
db = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)

retriever = db.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={"k": 5, "score_threshold": 0.3},  # top 5, but only if decent match
)

def ask(query: str) -> str:
    # Retrieval — pure vector math, no LLM involved yet
    results = retriever.invoke(query)
    if not results:
        return "I couldn't find anything relevant in the documents."

    context = "\n\n".join(doc.page_content for doc in results)

    # Generation — force the LLM to stick to retrieved context
    prompt = f"""Based only on the following documents, answer the question.
If the answer isn't in the documents, say "I don't have enough information."

Documents:
{context}

Question: {query}"""

    llm = ChatOpenAI(model="gpt-4o")
    response = llm.invoke([
        SystemMessage(content="You are a helpful assistant that answers strictly from the provided context."),
        HumanMessage(content=prompt),
    ])
    return response.content

if __name__ == "__main__":
    while True:
        q = input("\nYou: ")
        if q.lower() in ("quit", "exit"):
            break
        print("AI:", ask(q))
```

Run `python ingest.py` once, then `python ask.py` to chat. This ~60-line pair is a fully working, production-shaped RAG system. Every advanced technique in this document is an optimization layered on top of this skeleton — build this first, then add complexity only where testing shows you need it.

## 14c. Troubleshooting Guide — Symptom → Cause → Fix

| Symptom | Root Cause | Fix |
|---|---|---|
| Retriever returns nothing relevant | Embedding model/dimensions mismatch between ingest and query | Use the exact same model string both places; re-embed everything if you ever change it |
| Answers are confidently wrong (hallucination) | Prompt doesn't force grounding, or LLM ignores context | Add an explicit "answer ONLY from these documents, say 'I don't know' otherwise" instruction; lower temperature |
| Right document exists but never gets retrieved | Chunk is too big (diluted meaning) or too small (missing context) | Test chunk sizes 400–1200 chars; add ~10–15% overlap; try semantic chunking for tricky docs |
| Retrieval works for exact wording but not paraphrases | Pure vector search struggles with vocabulary mismatch | Add multi-query retrieval (LLM generates 3–5 phrasings, union the results) |
| Retrieval misses exact codes/names/IDs | Vector search is bad at exact-match lexical terms | Add hybrid search: BM25 keyword search + vector search, merge with Reciprocal Rank Fusion |
| Top-K results include a mix of great and mediocre chunks | Retrieval is coarse by design (optimized for speed, not precision) | Add a reranking step: retrieve top 20–30 broadly, then rerank with a cross-encoder, keep top 3–5 |
| Follow-up questions ("what about it?") retrieve garbage | Vector search doesn't understand pronouns/history | Add query reformulation: rewrite the follow-up into a standalone question using chat history before embedding it |
| System returns an answer even when nothing relevant exists | No similarity floor | Set a score_threshold (e.g., 0.3) so weak matches are dropped instead of forced through |
| Costs/latency too high | Re-embedding on every run, or embedding model too large, or reranking everything | Persist your vector store (don't re-ingest every time); use text-embedding-3-small unless quality demands -large; only rerank the top-K candidate pool, not the whole DB |
| Tables/images ignored or garbled | Standard text chunking mangles tabular/visual data | Extract tables as Markdown before chunking; caption images with a vision model and embed the caption |
| Great in dev, degrades as the doc set grows | No monitoring, chunking strategy doesn't scale, vector DB not indexed properly | Add tracing/monitoring (e.g., LangSmith); switch to a DB with proper ANN indexing (HNSW) at scale |

## 14d. Recommended Build Order

1. Get the basic pipeline above working end-to-end.
2. Manually test 10–15 real questions — check if the *retrieved chunks* (not the final answer) actually contain the answer. This isolates retrieval bugs from generation bugs.
3. Tune chunk size/overlap based on step 2.
4. Only add multi-query, hybrid search, or reranking once you've identified a *specific* retrieval failure mode from step 2 — don't add them preemptively.
5. Add conversational memory (query reformulation) once single-turn works reliably.
6. Add multimodal handling only if your actual documents contain tables/images that matter to answers.

---

## 15. Tech Stack Cheat Sheet

| Component | Common Tools Mentioned |
|---|---|
| Orchestration framework | LangChain (chains, LCEL, document loaders, splitters) |
| Observability / monitoring | LangSmith |
| API deployment | LangServe |
| Embedding models | OpenAI text-embedding-3-small/large, Voyage AI, Cohere, Mistral |
| Vector databases | Chroma, Pinecone, Weaviate, FAISS, PostgreSQL (pgvector) |
| LLMs (paid) | OpenAI GPT models, Anthropic Claude |
| LLMs (local/open-source) | Ollama running Llama 2, Gemma, Code Llama, etc. |
| Keyword search (hybrid) | BM25 |
| Rerankers | Cross-encoder models |
| App/demo layer | Streamlit |

---

## 16. Glossary

- **Chunk** — a small segment of a larger document, sized for embedding and retrieval.
- **Chunk overlap** — shared text between consecutive chunks to preserve boundary context.
- **Context window** — the maximum amount of text (in tokens) an LLM can process in one request.
- **Cosine similarity** — a measure of the angle between two vectors, used to score semantic similarity; ranges 0–1.
- **Cross-encoder** — a model that scores a query-document pair jointly (used for reranking), more accurate but slower than embedding-based comparison.
- **Dense retrieval** — retrieval based on vector/semantic similarity.
- **Dot product** — sum of element-wise multiplication of two vectors; equals cosine similarity when vectors are normalized.
- **Embedding / vector embedding** — a numerical (vector) representation of text capturing its meaning.
- **Hybrid search** — combining dense (vector) and sparse (keyword) retrieval.
- **Injection / ingestion pipeline** — the process of loading, chunking, embedding, and storing documents.
- **Query reformulation** — rewriting a context-dependent follow-up question into a standalone searchable question.
- **RAG (Retrieval-Augmented Generation)** — combining an LLM with an external retrieval system so answers are grounded in retrieved data rather than memory alone.
- **Reciprocal Rank Fusion (RRF)** — a method for merging multiple ranked retrieval lists using rank position rather than raw scores.
- **Reranking** — a second, more precise scoring pass over a retrieved candidate set, typically using a cross-encoder.
- **Retriever** — the component that searches the vector store and returns the top-K most relevant chunks for a query.
- **Sparse retrieval** — traditional keyword-based search (e.g., BM25).
- **Token** — the basic text unit an LLM processes; roughly one word or word-fragment.
- **Vector database** — a database optimized for storing and searching vector embeddings by similarity.

---

*These notes summarize and explain the concepts taught across the "Complete RAG Tutorial 2026" playlist (Harish Neel), covering fundamentals through advanced production techniques: injection/retrieval pipelines, cosine similarity, conversational RAG, chunking strategies, multi-query retrieval, reciprocal rank fusion, hybrid search, reranking, and multimodal RAG.*
