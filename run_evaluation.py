import json
import os
import argparse
import pprint

def main():
    parser = argparse.ArgumentParser(description="Evaluate RAG system and pretty‑print results.")
    parser.add_argument("--path", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "evaluation", "results.json"), help="Path to results.json")
    args = parser.parse_args()
    if not os.path.exists(args.path):
        print(f"Results file not found: {args.path}\nRun `python evaluation/evaluator.py` first.")
        return
    with open(args.path, encoding="utf-8") as f:
        data = json.load(f)
    pp = pprint.PrettyPrinter(indent=2, width=120)
    pp.pprint(data)

if __name__ == "__main__":
    main()
