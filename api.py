"""
RAG Question-Answering System — Flask REST API

Exposes the RAG pipeline as JSON endpoints so the standalone HTML/JS
front-end (rag-website/) can call it from any browser.

Run:
    python api.py

Then open  rag-website/index.html  in your browser.
"""

from __future__ import annotations
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
CORS(app)  # Allow the HTML file to call from any origin


# ── Lazy-load pipeline so startup is fast ────────────────────────────────────

def _run_rag(query: str, strategy: str, top_k: int, threshold: float) -> dict:
    from src.pipeline import run_rag
    return run_rag(query, strategy=strategy, top_k=top_k, threshold=threshold)


# ── Routes ───────────────────────────────────────────────────────────────────

@app.route("/api/ask", methods=["POST"])
def ask():
    """
    POST /api/ask
    Body (JSON): { "query": "...", "strategy": "A"|"B", "top_k": 3, "threshold": 0.05 }
    Returns:     { "answer": "...", "gated": bool, "top_score": float,
                   "strategy": "A"|"B", "sources": [...] }
    """
    data = request.get_json(force=True, silent=True) or {}
    query     = (data.get("query") or "").strip()
    strategy  = data.get("strategy", "A").upper()
    top_k     = int(data.get("top_k", 3))
    threshold = float(data.get("threshold", 0.05))

    if not query:
        return jsonify({"error": "query is required"}), 400
    if strategy not in ("A", "B"):
        return jsonify({"error": "strategy must be 'A' or 'B'"}), 400

    try:
        result = _run_rag(query, strategy, top_k, threshold)
    except FileNotFoundError as e:
        return jsonify({"error": f"Index not found: {e}. Run build_index.py first."}), 503
    except ValueError as e:
        return jsonify({"error": str(e)}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    # Serialize sources (chunk dicts are already JSON-safe)
    return jsonify(result)


@app.route("/api/compare", methods=["POST"])
def compare():
    """
    POST /api/compare
    Body (JSON): { "query": "...", "top_k": 3, "threshold": 0.35 }
    Returns:     { "A": <ask_result>, "B": <ask_result> }
    """
    data = request.get_json(force=True, silent=True) or {}
    query     = (data.get("query") or "").strip()
    top_k     = int(data.get("top_k", 3))
    threshold = float(data.get("threshold", 0.35))

    if not query:
        return jsonify({"error": "query is required"}), 400

    results = {}
    errors  = {}
    for strat in ("A", "B"):
        try:
            results[strat] = _run_rag(query, strat, top_k, threshold)
        except Exception as e:
            errors[strat] = str(e)

    return jsonify({"results": results, "errors": errors})


@app.route("/api/evaluation", methods=["GET"])
def evaluation():
    """
    GET /api/evaluation
    Returns the pre-run evaluation/results.json, or an empty list.
    """
    results_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "evaluation", "results.json"
    )
    if not os.path.exists(results_path):
        return jsonify({"results": [], "available": False})

    with open(results_path, encoding="utf-8") as fh:
        data = json.load(fh)
    return jsonify({"results": data, "available": True})


@app.route("/api/health", methods=["GET"])
def health():
    """Simple liveness probe."""
    indices_ok = all(
        os.path.exists(os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "indices", f"strategy_{s}.index"
        ))
        for s in ("a", "b")
    )
    return jsonify({
        "status": "ok",
        "indices_ready": indices_ok,
    })


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n[RAG]  API starting on  http://localhost:5000\n")
    print("   Open  rag-website/index.html  in your browser to use the UI.\n")
    app.run(host="0.0.0.0", port=5000, debug=False)
