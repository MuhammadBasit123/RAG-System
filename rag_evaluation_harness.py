"""
RAG Evaluation Harness
=======================
Implements the brief:
  - 30+ question/answer pairs
  - Retrieval and generation scored separately
  - At least one deliberately unanswerable question
  - Results as a table, with an interpretation

HOW TO USE THIS FOR REAL
-------------------------
This file runs standalone right now against a tiny toy corpus (an IT
helpdesk FAQ) with a keyword-overlap "retriever" and an echo "generator" —
just so you can see the harness actually work and produce real numbers.

To use it for your submission:
  1. Delete the DEMO section below and replace toy_retrieve/toy_generate
     with calls into your real RAG pipeline (the retrieval + LLM call
     you built for the "cites its sources" brief).
  2. Replace QA_PAIRS with 30+ real questions about your real corpus,
     including a handful of deliberately unanswerable ones.
  3. Run: python rag_evaluation_harness.py
"""

from dataclasses import dataclass, field
from typing import List, Optional, Callable


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class QAPair:
    qid: str
    question: str
    answerable: bool
    expected_keywords: List[str] = field(default_factory=list)  # auto-scores answerable Qs
    gold_source: Optional[str] = None  # doc/chunk id, if you track one — stricter than keywords


@dataclass
class RetrievedChunk:
    text: str
    source: Optional[str] = None
    score: Optional[float] = None


# ---------------------------------------------------------------------------
# Scoring
# Retrieval and generation are scored independently on purpose: a wrong
# final answer could mean retrieval failed, OR generation failed on
# perfectly good context — and you want to know which one it was.
# ---------------------------------------------------------------------------

def score_retrieval(qa: QAPair, retrieved: List[RetrievedChunk]) -> int:
    if qa.answerable:
        if not retrieved:
            return 0
        if qa.gold_source:
            return int(any(c.source == qa.gold_source for c in retrieved))
        combined = " ".join(c.text.lower() for c in retrieved)
        return int(any(kw.lower() in combined for kw in qa.expected_keywords))
    else:
        # correct behavior for an unanswerable question: retrieve nothing
        # the system is confident enough to pass along
        return int(len(retrieved) == 0)


ABSTAIN_PHRASES = [
    "i don't know", "i do not know", "cannot find", "can't find",
    "no relevant", "not in the", "unable to find", "not covered",
]

def score_generation(qa: QAPair, answer: str) -> Optional[int]:
    answer_l = answer.lower()
    if qa.answerable:
        if not qa.expected_keywords:
            return None  # needs manual / LLM-judge scoring instead
        return int(any(kw.lower() in answer_l for kw in qa.expected_keywords))
    else:
        return int(any(p in answer_l for p in ABSTAIN_PHRASES))


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_evaluation(qa_pairs: List[QAPair],
                    retrieve_fn: Callable[[str], List[RetrievedChunk]],
                    generate_fn: Callable[[str, List[RetrievedChunk]], str]) -> List[dict]:
    results = []
    for qa in qa_pairs:
        retrieved = retrieve_fn(qa.question)
        answer = generate_fn(qa.question, retrieved)
        results.append({
            "id": qa.qid,
            "question": qa.question,
            "answerable": qa.answerable,
            "retrieval_score": score_retrieval(qa, retrieved),
            "generation_score": score_generation(qa, answer),
            "answer": answer,
        })
    return results


def summarize(results: List[dict]) -> dict:
    ans = [r for r in results if r["answerable"]]
    unans = [r for r in results if not r["answerable"]]

    def avg(rows, key):
        vals = [r[key] for r in rows if r[key] is not None]
        return sum(vals) / len(vals) if vals else float("nan")

    return {
        "n_answerable": len(ans),
        "n_unanswerable": len(unans),
        "retrieval_hit_rate_answerable": avg(ans, "retrieval_score"),
        "generation_accuracy_answerable": avg(ans, "generation_score"),
        "retrieval_correct_rejection_unanswerable": avg(unans, "retrieval_score"),
        "generation_abstention_rate_unanswerable": avg(unans, "generation_score"),
    }


def print_console_table(results: List[dict]) -> None:
    print(f"{'id':<7} {'answerable':<11} {'retrieval':<10} {'generation':<11} question")
    print("-" * 95)
    for r in results:
        gen = "-" if r["generation_score"] is None else r["generation_score"]
        print(f"{r['id']:<7} {str(r['answerable']):<11} {r['retrieval_score']:<10} {str(gen):<11} {r['question'][:48]}")


def markdown_summary_table(summary: dict) -> str:
    rows = [
        ("Answerable questions", summary["n_answerable"], ""),
        ("Unanswerable questions", summary["n_unanswerable"], ""),
        ("Retrieval hit-rate", f"{summary['retrieval_hit_rate_answerable']:.0%}", "on answerable questions"),
        ("Generation accuracy", f"{summary['generation_accuracy_answerable']:.0%}", "on answerable questions"),
        ("Retrieval correct-rejection rate", f"{summary['retrieval_correct_rejection_unanswerable']:.0%}", "on unanswerable questions"),
        ("Generation abstention rate", f"{summary['generation_abstention_rate_unanswerable']:.0%}", "on unanswerable questions"),
    ]
    lines = ["| Metric | Value | Scope |", "|---|---|---|"]
    lines += [f"| {m} | {v} | {s} |" for m, v, s in rows]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# DEMO — replace everything below this line for your real submission
# ---------------------------------------------------------------------------

TOY_CORPUS = [
    ("faq-1", "Password resets can be self-served through the IT portal at itportal.example.com; resets take effect within five minutes."),
    ("faq-2", "New employees receive a laptop within their first three business days; loaner laptops are available from the IT desk on the 4th floor."),
    ("faq-3", "VPN access requires the GlobalProtect client, and multi-factor authentication is mandatory for all remote connections."),
    ("faq-4", "The helpdesk is staffed Monday to Friday, 8am to 6pm; urgent after-hours issues go through the on-call pager at extension 5115."),
    ("faq-5", "Printer issues should be reported through the ticketing system under category Hardware Printers; average resolution time is four hours."),
    ("faq-6", "Software installation requests over $50 require manager approval before IT will proceed."),
    ("faq-7", "Lost or stolen devices must be reported to IT security within one hour of discovery, and the device will be remotely wiped."),
    ("faq-8", "Guest wifi credentials are valid for 24 hours and can be generated at the front desk kiosk."),
]

def toy_retrieve(query: str, k: int = 2, min_overlap: int = 2) -> List[RetrievedChunk]:
    """Keyword-overlap stand-in for real embeddings + vector search."""
    q_words = set(query.lower().replace("?", "").split())
    scored = []
    for source, text in TOY_CORPUS:
        t_words = set(text.lower().replace(".", "").replace(",", "").split())
        overlap = len(q_words & t_words)
        if overlap >= min_overlap:
            scored.append((overlap, RetrievedChunk(text=text, source=source, score=overlap)))
    scored.sort(key=lambda x: -x[0])
    return [c for _, c in scored[:k]]


def toy_generate(query: str, retrieved: List[RetrievedChunk]) -> str:
    """Stand-in generator. The real version calls your LLM (e.g. the
    InferenceClient / ChatHuggingFace setup) with the retrieved chunks
    as context. This one just returns the top chunk, or abstains."""
    if not retrieved:
        return "I don't know — I couldn't find anything relevant in the documents."
    return retrieved[0].text


QA_PAIRS = [
    QAPair("q01", "How do I reset my password?", True, ["portal", "self-served"]),
    QAPair("q02", "How long does a password reset take to apply?", True, ["five minutes"]),
    QAPair("q03", "What website do I use to reset my password?", True, ["itportal"]),
    QAPair("q04", "Does IT need to manually reset my password for me?", True, ["self-served"]),
    QAPair("q05", "How soon do new employees get a laptop?", True, ["three business days"]),
    QAPair("q06", "Where can I get a loaner laptop?", True, ["4th floor", "it desk"]),
    QAPair("q07", "Are loaner laptops available while I wait for mine?", True, ["loaner"]),
    QAPair("q08", "What VPN client does the company use?", True, ["globalprotect"]),
    QAPair("q09", "Is multi-factor authentication required for VPN?", True, ["multi-factor", "mandatory"]),
    QAPair("q10", "Can I connect remotely without MFA?", True, ["mandatory", "multi-factor"]),
    QAPair("q11", "What are the helpdesk's operating hours?", True, ["8am", "6pm"]),
    QAPair("q12", "How do I reach IT after hours for an urgent issue?", True, ["5115", "pager"]),
    QAPair("q13", "Is the helpdesk open on weekends?", True, ["monday", "friday"]),
    QAPair("q14", "How do I report a broken printer?", True, ["ticketing", "printers"]),
    QAPair("q15", "How long does it take to fix a printer issue on average?", True, ["four hours"]),
    QAPair("q16", "What ticket category should printer problems go under?", True, ["hardware", "printers"]),
    QAPair("q17", "Do I need approval to install paid software?", True, ["manager approval"]),
    QAPair("q18", "What's the dollar threshold requiring manager sign-off for software?", True, ["$50"]),
    QAPair("q19", "Who approves software purchases over $50?", True, ["manager"]),
    QAPair("q20", "What should I do if I lose my company laptop?", True, ["one hour", "security"]),
    QAPair("q21", "What happens to a lost device after it's reported?", True, ["remotely wiped"]),
    QAPair("q22", "How quickly must a stolen device be reported?", True, ["one hour"]),
    QAPair("q23", "Will IT wipe a lost laptop remotely?", True, ["remotely wiped"]),
    QAPair("q24", "How long is a guest wifi code valid?", True, ["24 hours"]),
    QAPair("q25", "Where do I get a guest wifi code?", True, ["front desk", "kiosk"]),
    QAPair("q26", "Can I generate my own guest wifi credentials?", True, ["front desk", "kiosk"]),
    QAPair("q27", "Do guest wifi credentials expire?", True, ["24 hours"]),
    QAPair("q28", "What's the company's parental leave policy?", False),
    QAPair("q29", "Can I expense a new keyboard without approval?", False),
    QAPair("q30", "What's the wifi password for the office printers' network?", False),
    QAPair("q31", "Who approves annual performance bonuses?", False),
    QAPair("q32", "What's the maximum file size for email attachments?", False),
]


if __name__ == "__main__":
    results = run_evaluation(QA_PAIRS, toy_retrieve, toy_generate)
    summary = summarize(results)

    print(f"Ran {len(QA_PAIRS)} questions ({summary['n_answerable']} answerable, "
          f"{summary['n_unanswerable']} unanswerable)\n")
    print_console_table(results)

    md = markdown_summary_table(summary)
    print("\n" + md)

    with open("results_table.md", "w") as f:
        f.write(md + "\n")
    print("\nSaved results_table.md")
