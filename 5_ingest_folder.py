import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

try:
    from dotenv import load_dotenv
except Exception:
    load_dotenv = None


def main() -> None:
    if load_dotenv is not None:
        env_path = Path(__file__).with_name(".env")
        if env_path.exists():
            load_dotenv(dotenv_path=env_path)

    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", required=True, help="Folder containing PDFs")
    parser.add_argument("--chunk-size", type=int, default=800)
    parser.add_argument("--chunking", choices=["char", "sentence", "paragraph"], default="paragraph")
    parser.add_argument("--qdrant-url", default=os.getenv("QDRANT_URL"))
    parser.add_argument("--qdrant-host", default=os.getenv("QDRANT_HOST", "localhost"))
    parser.add_argument("--qdrant-port", type=int, default=int(os.getenv("QDRANT_PORT", "6333")))
    parser.add_argument("--collection", default="pdf_chunks")
    parser.add_argument("--neo4j-uri", default="bolt://localhost:7687")
    parser.add_argument("--neo4j-user", default="neo4j")
    parser.add_argument("--neo4j-pass", default="password")
    parser.add_argument("--spacy-model", default="en_core_web_sm")
    parser.add_argument("--ruler-json", default=None, help="Path to EntityRuler JSON patterns")
    parser.add_argument("--gliner-model", default="urchade/gliner_mediumv2")
    parser.add_argument(
        "--gliner-labels",
        default="PERSON,ORG,PRODUCT,GPE,DATE,TECH,CRYPTO,STANDARD",
        help="Comma-separated labels or path to a text file for GLiNER",
    )
    parser.add_argument("--gliner-threshold", type=float, default=0.3)
    parser.add_argument("--embedding-model", default=None)
    parser.add_argument(
        "--entity-provider",
        choices=["spacy", "gliner", "azure", "gemini", "langextract"],
        default="gliner",
        help="Entity extraction provider",
    )
    args = parser.parse_args()

    folder = Path(args.folder)
    if not folder.exists():
        raise FileNotFoundError(folder)

    pdfs = sorted(folder.rglob("*.pdf"))
    if not pdfs:
        print("No PDF files found.")
        return

    total = len(pdfs)
    start_all = time.time()
    for idx, pdf in enumerate(pdfs, start=1):
        cmd = [
            sys.executable,
            "4_ingest_and_entities.py",
            "--pdf",
            str(pdf),
            "--chunk-size",
            str(args.chunk_size),
            "--chunking",
            args.chunking,
            "--collection",
            args.collection,
            "--neo4j-uri",
            args.neo4j_uri,
            "--neo4j-user",
            args.neo4j_user,
            "--neo4j-pass",
            args.neo4j_pass,
            "--spacy-model",
            args.spacy_model,
            "--ruler-json",
            args.ruler_json or "",
            "--entity-provider",
            args.entity_provider,
            "--gliner-model",
            args.gliner_model,
            "--gliner-labels",
            args.gliner_labels,
            "--gliner-threshold",
            str(args.gliner_threshold),
        ]
        if args.embedding_model:
            cmd.extend(["--embedding-model", args.embedding_model])
        if args.qdrant_url:
            cmd.extend(["--qdrant-url", args.qdrant_url])
        else:
            cmd.extend(["--qdrant-host", args.qdrant_host, "--qdrant-port", str(args.qdrant_port)])
        print(f"[{idx}/{total}] Processing: {pdf}")
        start = time.time()
        subprocess.run(cmd, check=True)
        elapsed = time.time() - start
        print(f"[{idx}/{total}] Done in {elapsed:.1f}s")
    total_elapsed = time.time() - start_all
    print(f"All done: {total} file(s) in {total_elapsed:.1f}s")


if __name__ == "__main__":
    main()
