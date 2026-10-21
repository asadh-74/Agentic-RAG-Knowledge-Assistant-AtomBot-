import argparse
from pathlib import Path

from rag import build_vectorstore


def main():
    parser = argparse.ArgumentParser(description="Build a FAISS vector store from a PDF.")
    parser.add_argument(
        "--pdf",
        default="data/feature_engineering.pdf",
        help="Path to the source PDF file.",
    )
    parser.add_argument(
        "--output",
        default="faiss_index",
        help="Directory where the FAISS index should be saved.",
    )
    args = parser.parse_args()

    pdf_path = Path(args.pdf)
    output_path = Path(args.output)
    build_vectorstore(pdf_path, output_path)
    print(f"Created FAISS index at {output_path} from {pdf_path}.")


if __name__ == "__main__":
    main()
