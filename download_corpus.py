"""
Download a 50+ page Deep Learning / Machine Learning corpus from Wikipedia.

Uses the Wikipedia REST API to export clean article text as .txt files
saved to corpus/. The extractor assigns virtual page numbers
(every LINES_PER_PAGE lines = 1 page) so retrieval still tracks page citations.

Topics: Core deep learning and machine learning concepts.

Usage:
    python download_corpus.py
"""

from __future__ import annotations
import os
import re
import time
import requests

CORPUS_DIR = "corpus"
LINES_PER_PAGE = 45   # how many text lines = 1 virtual page

HEADERS = {
    "User-Agent": (
        "RAGAssignment/2.0 (student educational project; deep learning corpus) "
        "python-requests/2.31"
    )
}

# Wikipedia articles to download — (article_title, filename_stem)
# All are freely available under CC-BY-SA
ARTICLES: list[tuple[str, str]] = [
    ("Deep learning",                           "deep_learning"),
    ("Backpropagation",                         "backpropagation"),
    ("Convolutional neural network",            "convolutional_neural_network"),
    ("Recurrent neural network",                "recurrent_neural_network"),
    ("Transformer (deep learning architecture)","transformer_architecture"),
    ("Gradient descent",                        "gradient_descent"),
    ("Overfitting",                             "overfitting"),
    ("Transfer learning",                       "transfer_learning"),
    ("Reinforcement learning",                  "reinforcement_learning"),
    ("Generative adversarial network",          "generative_adversarial_network"),
    ("Natural language processing",             "natural_language_processing"),
    ("Support vector machine",                  "support_vector_machine"),
    ("Attention (machine learning)",            "attention_mechanism"),
    ("Regularization (mathematics)",            "regularization"),
    ("Artificial neural network",               "artificial_neural_network"),
    ("Long short-term memory",                  "long_short_term_memory"),
    ("Autoencoder",                             "autoencoder"),
    ("Batch normalization",                     "batch_normalization"),
    ("Dropout (deep learning)",                 "dropout"),
    ("Stochastic gradient descent",             "stochastic_gradient_descent"),
]

WIKIPEDIA_API = "https://en.wikipedia.org/w/api.php"


def fetch_wikipedia_article(title: str) -> str:
    """
    Fetch a Wikipedia article's plain text using the MediaWiki API.
    Returns clean plain text (no wikitext markup).
    """
    params = {
        "action": "query",
        "titles": title,
        "prop": "extracts",
        "explaintext": True,      # plain text, no HTML
        "exsectionformat": "plain",
        "format": "json",
        "redirects": 1,
    }
    resp = requests.get(WIKIPEDIA_API, params=params, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    pages = data.get("query", {}).get("pages", {})
    for page_id, page in pages.items():
        if page_id == "-1":
            raise ValueError(f"Article not found: {title!r}")
        return page.get("extract", "")

    raise ValueError(f"Unexpected API response for: {title!r}")


def clean_text(text: str) -> str:
    """Clean up Wikipedia plain-text extracts."""
    # Remove excessive blank lines
    text = re.sub(r"\n{4,}", "\n\n\n", text)
    # Remove lines that are just whitespace
    lines = [line.rstrip() for line in text.splitlines()]
    text = "\n".join(lines)
    return text.strip()


def count_virtual_pages(txt_path: str) -> int:
    """Count how many virtual pages a text file would produce."""
    with open(txt_path, encoding="utf-8", errors="replace") as f:
        lines = f.readlines()
    non_empty = [l for l in lines if l.strip()]
    return max(1, len(non_empty) // LINES_PER_PAGE)


def main() -> None:
    os.makedirs(CORPUS_DIR, exist_ok=True)
    total_pages = 0

    print(f"Downloading {len(ARTICLES)} Wikipedia ML/DL articles -> {CORPUS_DIR}/\n")

    for article_title, stem in ARTICLES:
        out_path = os.path.join(CORPUS_DIR, f"{stem}.txt")

        if os.path.exists(out_path):
            vp = count_virtual_pages(out_path)
            total_pages += vp
            print(f"  [SKIP]  {article_title:<45} (~{vp:3d} pages, already downloaded)")
            continue

        print(f"  Fetching: {article_title:<45}", end="", flush=True)
        try:
            raw = fetch_wikipedia_article(article_title)
            if len(raw) < 500:
                print(f" [SKIP] Too short ({len(raw)} chars)")
                continue
            cleaned = clean_text(raw)
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(cleaned)
            vp = count_virtual_pages(out_path)
            total_pages += vp
            print(f" done  ({len(cleaned):,} chars, ~{vp} virtual pages) -> {out_path}")
            time.sleep(0.5)   # be polite to Wikipedia servers
        except Exception as exc:
            print(f" ERROR: {exc}")

    print(f"\nCorpus complete -- ~{total_pages} total virtual pages")
    if total_pages >= 50:
        print("[OK] Meets the 50-page requirement")
    else:
        print(
            f"[WARNING] Only ~{total_pages} pages. "
            "Add more articles to reach 50+."
        )
    print("\nNext step:  python build_index.py")


if __name__ == "__main__":
    main()
