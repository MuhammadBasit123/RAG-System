import argparse
import subprocess
import sys

def run_command(command: list[str]):
    """Run a shell command and stream output to stdout."""
    result = subprocess.run(command, check=False)
    if result.returncode != 0:
        sys.exit(result.returncode)

def main() -> None:
    parser = argparse.ArgumentParser(description="One‑click script to set up and run the RAG system.")
    parser.add_argument("--skip-download", action="store_true", help="Skip the corpus download step.")
    parser.add_argument("--skip-build", action="store_true", help="Skip building FAISS indices.")
    parser.add_argument("--no-streamlit", action="store_true", help="Do not launch the Streamlit UI after setup.")
    args = parser.parse_args()

    if not args.skip_download:
        print("\n=== Downloading corpus ===")
        run_command([sys.executable, "download_corpus.py"])
    else:
        print("\n=== Skipping corpus download ===")

    if not args.skip_build:
        print("\n=== Building indices ===")
        run_command([sys.executable, "build_index.py"])
    else:
        print("\n=== Skipping index building ===")

    if not args.no_streamlit:
        print("\n=== Launching Streamlit UI ===")
        run_command(["streamlit", "run", "app.py"])
    else:
        print("\n=== Finished setup (no Streamlit) ===")

if __name__ == "__main__":
    main()
