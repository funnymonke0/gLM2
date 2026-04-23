import argparse
from pathlib import Path

from gaia_local.constants import (
    DEFAULT_COLLECTION_NAME,
    DEFAULT_MAX_SEQ_LENGTH,
    DEFAULT_METRICS_DIR,
    DEFAULT_MODEL_NAME,
    DEFAULT_QDRANT_URL,
)


def parse_args() -> argparse.Namespace: #so it goes seqhub_local.py command --args ...
    parser = argparse.ArgumentParser(description="Local indexing and querying with gLM2 + Qdrant.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    index_parser = subparsers.add_parser("index", help="Embed FASTA records and store them in local Qdrant.")
    add_shared_index_args(index_parser)

    query_parser = subparsers.add_parser("query", help="Query a local Qdrant collection with one or more FASTA proteins.")
    add_shared_runtime_args(query_parser)
    query_parser.add_argument("--query-fasta", type=Path, nargs="+", required=True, help="One or more FASTA files containing query proteins.")
    query_parser.add_argument("--collection-name", default=DEFAULT_COLLECTION_NAME)
    query_parser.add_argument("--qdrant-url", default=DEFAULT_QDRANT_URL, help="Docker-hosted Qdrant URL, typically http://localhost:6333.")
    query_parser.add_argument("--top-k", type=int, default=10)
    query_parser.add_argument("--expected-fasta", type=Path, help="Reference FASTA used to compare returned ids against SeqHub results.")
    query_parser.add_argument("--output-json", type=Path, help="Optional explicit output path for the query report JSON.")

    benchmark_parser = subparsers.add_parser("benchmark", help="Index a corpus, validate deterministic controls, then query benchmark proteins.")
    add_shared_index_args(benchmark_parser)
    benchmark_parser.add_argument("--query-fasta", type=Path, nargs="+", required=True, help="One or more FASTA files containing query proteins.")
    benchmark_parser.add_argument("--expected-fasta", type=Path, required=True, help="SeqHub reference FASTA to compare against.")
    benchmark_parser.add_argument("--top-k", type=int, default=10)

    return parser.parse_args()


def add_shared_runtime_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-seq-length", type=int, default=DEFAULT_MAX_SEQ_LENGTH)
    parser.add_argument("--metrics-dir", type=Path, default=DEFAULT_METRICS_DIR)


def add_shared_index_args(parser: argparse.ArgumentParser) -> None:
    add_shared_runtime_args(parser)
    parser.add_argument("--corpus-source", choices=["fasta", "omg"], default="omg", help="Source used for indexing records.")
    parser.add_argument("--corpus", type=Path, nargs="*", help="One or more FASTA files or directories to index when --corpus-source=fasta.")
    parser.add_argument("--omg-split", default="train", help="OMG split to load when --corpus-source=omg.")
    parser.add_argument("--omg-limit", type=int, default=200, help="Maximum OMG rows to load (use 0 or negative for no limit).")
    parser.add_argument("--omg-streaming", action="store_true", default=True, help="Stream OMG rows instead of full download (default: true).")
    parser.add_argument("--no-omg-streaming", dest="omg_streaming", action="store_false", help="Disable OMG streaming mode.")
    parser.add_argument("--collection-name", default=DEFAULT_COLLECTION_NAME)
    parser.add_argument("--qdrant-url", default=DEFAULT_QDRANT_URL, help="Docker-hosted Qdrant URL, typically http://localhost:6333.")
    parser.add_argument("--exclude-header", action="append", default=["Query"], help="FASTA record ids to exclude during indexing.")
    parser.add_argument("--recreate", action="store_true", help="Delete and recreate the collection before indexing.")
    parser.add_argument("--output-json", type=Path, help="Optional explicit output path for the index report JSON.")