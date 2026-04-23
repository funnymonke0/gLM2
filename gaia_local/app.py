from gaia_local.cli import parse_args
from gaia_local.pipeline import index_corpus, query_collection, run_benchmark
from gaia_local.reporting import print_benchmark_summary, print_index_summary, print_query_summary


def main() -> None:
    args = parse_args()
    if args.command == "index":
        report = index_corpus(
            corpus_paths=args.corpus,
            corpus_source=args.corpus_source,
            omg_split=args.omg_split,
            omg_limit=args.omg_limit,
            omg_streaming=args.omg_streaming,
            model_name=args.model_name,
            collection_name=args.collection_name,
            qdrant_url=args.qdrant_url,
            batch_size=args.batch_size,
            max_seq_length=args.max_seq_length,
            metrics_dir=args.metrics_dir,
            recreate=args.recreate,
            excluded_ids=set(args.exclude_header),
            output_json=args.output_json,
        )
        print_index_summary(report)
        return
    if args.command == "query":
        report = query_collection(
            query_paths=args.query_fasta,
            model_name=args.model_name,
            collection_name=args.collection_name,
            qdrant_url=args.qdrant_url,
            batch_size=args.batch_size,
            max_seq_length=args.max_seq_length,
            metrics_dir=args.metrics_dir,
            top_k=args.top_k,
            expected_fasta=args.expected_fasta,
            output_json=args.output_json,
        )
        print_query_summary(report)
        return
    if args.command == "benchmark":
        report = run_benchmark(args)
        print_benchmark_summary(report)
        return
    raise ValueError(f"Unsupported command: {args.command}")