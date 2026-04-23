from pathlib import Path

DEFAULT_MODEL_NAME = "tattabio/gLM2_650M_embed"
DEFAULT_COLLECTION_NAME = "seqhub_local"
DEFAULT_QDRANT_URL = "http://localhost:6333"
DEFAULT_METRICS_DIR = Path("artifacts") / "metrics"
DEFAULT_INDEX_BATCH_LOG = "index_batches.jsonl"
DEFAULT_INDEX_SUMMARY = "index_summary.json"
DEFAULT_QUERY_SUMMARY = "query_summary.json"
DEFAULT_BENCHMARK_SUMMARY = "benchmark_summary.json"
DEFAULT_MAX_SEQ_LENGTH = 4096
CONTROL_COLLECTION_NAME = "seqhub_validation_controls"
FASTA_SUFFIXES = {".fa", ".faa", ".fasta", ".fna"}
AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"