def print_index_summary(report: dict) -> None:
    print(f"Indexed {report['record_count']} records into '{report['collection_name']}'.")
    print(f"Vectors: {report['vector_size']} dimensions")
    print(f"Metrics: {report['summary_path']}")


def print_query_summary(report: dict) -> None:
    print(f"Queried collection '{report['collection_name']}' with {len(report['results'])} sequence(s).")
    for result in report["results"]:
        top_hit = result["hits"][0] if result["hits"] else None
        if top_hit is None:
            print(f"- {result['query_id']}: no hits returned")
            continue
        print(
            f"- {result['query_id']}: top-1 {top_hit['seq_id']} "
            f"(score={top_hit['score']:.4f}, identity={top_hit['sequence_identity_to_query']:.4f})"
        )
    print(f"Metrics: {report['summary_path']}")


def print_benchmark_summary(report: dict) -> None:
    top_result = report["query"]["results"][0]
    print(f"Control validation ranked ids: {', '.join(report['controls']['ranked_ids'])}")
    print(f"Benchmark query: {top_result['query_id']}")
    if top_result["hits"]:
        top_hit = top_result["hits"][0]
        print(
            f"Top-1 local hit: {top_hit['seq_id']} "
            f"(score={top_hit['score']:.4f}, identity={top_hit['sequence_identity_to_query']:.4f})"
        )
    if "top_1_matches_reference" in top_result:
        print(f"Matches SeqHub reference top-1: {top_result['top_1_matches_reference']}")
        print(f"Overlap@5: {top_result['overlap_at_5']}, Overlap@10: {top_result['overlap_at_10']}")
    print(f"Metrics: {report['summary_path']}")