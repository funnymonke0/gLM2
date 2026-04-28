import argparse
from pathlib import Path
from typing import Sequence

from gaia_local.constants import (
    CONTROL_COLLECTION_NAME,
    DATASET,
    DEFAULT_BENCHMARK_SUMMARY,
    DEFAULT_INDEX_BATCH_LOG,
    DEFAULT_INDEX_SUMMARY,
    DEFAULT_QUERY_SUMMARY,
)
from gaia_local.controls import make_control_records, sequence_identity
from gaia_local.embedder import GLM2Embedder
from gaia_local.fasta_io import load_fasta_records, resolve_fasta_inputs
from gaia_local.metrics import stage_metrics_dict, timed_stage, write_json
from gaia_local.omg_io import iter_stream_record_batches, load_stream_records
from gaia_local.qdrant_store import create_qdrant_client, ensure_collection, recreate_collection, upsert_records
from gaia_local.types import SequenceRecord


def index_corpus(
    corpus_paths: Sequence[Path] | None,
    corpus_source: str,
    stream_split: str,
    stream_limit: int | None,
    streaming: bool,
    batch_log: bool,
    model_name: str,
    collection_name: str,
    qdrant_url: str,
    batch_size: int,
    auto_batch_size: bool,
    max_seq_length: int,
    metrics_dir: Path,
    recreate: bool,
    excluded_ids: set[str],
    output_json: Path | None,
) -> dict:
    normalized_limit = None if stream_limit is None or stream_limit <= 0 else stream_limit
    records: list[SequenceRecord] = []
    if corpus_source == "fasta":
        if not corpus_paths:
            raise ValueError("--corpus is required when --corpus-source=fasta")
        records = load_fasta_records(corpus_paths, excluded_ids=excluded_ids)
        corpus_inputs = [str(path.resolve()) for path in resolve_fasta_inputs(corpus_paths)]
    elif corpus_source == "stream":
        corpus_inputs = [f"hf://{DATASET}/{stream_split}"]
    else:
        raise ValueError(f"Unsupported corpus source: {corpus_source}")

    embedder = GLM2Embedder(
        model_name=model_name,
        batch_size=batch_size,
        max_seq_length=max_seq_length,
        auto_batch_size=auto_batch_size,
    )
    client = create_qdrant_client(qdrant_url)
    batch_log_path = (metrics_dir / DEFAULT_INDEX_BATCH_LOG) if batch_log else None
    if batch_log_path is not None and batch_log_path.exists():
        batch_log_path.unlink()

    if corpus_source == "stream" and streaming:
        embed_metrics = []
        upsert_metrics = []
        record_count = 0
        next_point_id = 0
        vector_size: int | None = None

        for records_batch in iter_stream_record_batches(
            split=stream_split,
            limit=normalized_limit,
            streaming=streaming,
            batch_size=batch_size,
        ):
            batch_embeddings, batch_metrics = embedder.embed_records(records_batch, batch_log_path=batch_log_path)
            embed_metrics.extend(batch_metrics)

            if vector_size is None:
                vector_size = int(batch_embeddings.shape[1])
                ensure_collection(client, collection_name, vector_size, recreate=recreate)

            upsert_metrics.extend(
                upsert_records(
                    client,
                    collection_name,
                    records_batch,
                    batch_embeddings,
                    batch_size=len(records_batch),
                    point_id_offset=next_point_id,
                )
            )
            next_point_id += len(records_batch)
            record_count += len(records_batch)

        if record_count == 0 or vector_size is None:
            raise ValueError(f"No stream records were loaded for split '{stream_split}'.")
    else:
        if corpus_source == "stream":
            records = load_stream_records(split=stream_split, limit=normalized_limit, streaming=streaming)
        embeddings, embed_metrics = embedder.embed_records(records, batch_log_path=batch_log_path)
        ensure_collection(client, collection_name, embeddings.shape[1], recreate=recreate)
        upsert_metrics = upsert_records(client, collection_name, records, embeddings, batch_size=batch_size)
        record_count = len(records)
        vector_size = int(embeddings.shape[1])

    report = {
        "mode": "index",
        "model_name": model_name,
        "device": str(embedder.device),
        "dtype": str(embedder.dtype),
        "qdrant_url": qdrant_url,
        "corpus_source": corpus_source,
        "collection_name": collection_name,
        "record_count": record_count,
        "vector_size": vector_size,
        "corpus_inputs": corpus_inputs,
        "excluded_ids": sorted(excluded_ids),
        "load_metrics": stage_metrics_dict(embedder.load_metrics),
        "embed_metrics": [stage_metrics_dict(metric) for metric in embed_metrics],
        "upsert_metrics": [stage_metrics_dict(metric) for metric in upsert_metrics],
        "index_batch_log": str(batch_log_path.resolve()) if batch_log_path is not None else None,
    }
    summary_path = output_json or (metrics_dir / DEFAULT_INDEX_SUMMARY)
    write_json(summary_path, report)
    report["summary_path"] = str(summary_path.resolve())
    return report


def query_collection(
    query_paths: Sequence[Path],
    model_name: str,
    collection_name: str,
    qdrant_url: str,
    batch_size: int,
    auto_batch_size: bool,
    max_seq_length: int,
    metrics_dir: Path,
    top_k: int,
    expected_fasta: Path | None,
    output_json: Path | None,
) -> dict:
    query_records = load_fasta_records(query_paths)
    embedder = GLM2Embedder(
        model_name=model_name,
        batch_size=batch_size,
        max_seq_length=max_seq_length,
        auto_batch_size=auto_batch_size,
    )
    client = create_qdrant_client(qdrant_url)
    query_embeddings, embed_metrics = embedder.embed_records(query_records)
    query_done = timed_stage("qdrant_query", item_count=len(query_records))
    results = []
    for record, embedding in zip(query_records, query_embeddings):
        response = client.query_points(collection_name=collection_name, query=embedding.tolist(), limit=top_k)
        hits = []
        for point in getattr(response, "points", []):
            payload = point.payload or {}
            sequence = payload.get("sequence", "")
            hits.append(
                {
                    "seq_id": payload.get("seq_id"),
                    "description": payload.get("description"),
                    "score": point.score,
                    "sequence_identity_to_query": round(sequence_identity(record.sequence, sequence), 6),
                    "source_path": payload.get("source_path"),
                }
            )
        results.append(
            {
                "query_id": record.seq_id,
                "query_length": len(record.sequence),
                "hits": hits,
            }
        )
    query_metrics = query_done()

    report = {
        "mode": "query",
        "model_name": model_name,
        "device": str(embedder.device),
        "dtype": str(embedder.dtype),
        "qdrant_url": qdrant_url,
        "collection_name": collection_name,
        "query_inputs": [str(path.resolve()) for path in resolve_fasta_inputs(query_paths)],
        "top_k": top_k,
        "load_metrics": stage_metrics_dict(embedder.load_metrics),
        "embed_metrics": [stage_metrics_dict(metric) for metric in embed_metrics],
        "query_metrics": stage_metrics_dict(query_metrics),
        "results": results,
    }
    if expected_fasta is not None:
        expected_records = load_fasta_records([expected_fasta], excluded_ids={"Query"})
        expected_ids = [record.seq_id for record in expected_records]
        for result in report["results"]:
            actual_ids = [hit["seq_id"] for hit in result["hits"]]
            result["expected_top_1"] = expected_ids[0] if expected_ids else None
            result["actual_top_1"] = actual_ids[0] if actual_ids else None
            result["top_1_matches_reference"] = bool(expected_ids and actual_ids and actual_ids[0] == expected_ids[0])
            result["overlap_at_5"] = len(set(actual_ids[:5]) & set(expected_ids[:5]))
            result["overlap_at_10"] = len(set(actual_ids[:10]) & set(expected_ids[:10]))
            result["reference_rank_of_actual_top_1"] = expected_ids.index(actual_ids[0]) + 1 if actual_ids and actual_ids[0] in expected_ids else None
    summary_path = output_json or (metrics_dir / DEFAULT_QUERY_SUMMARY)
    write_json(summary_path, report)
    report["summary_path"] = str(summary_path.resolve())
    return report


def validate_controls(
    query_record: SequenceRecord,
    model_name: str,
    qdrant_url: str,
    batch_size: int,
    auto_batch_size: bool,
    max_seq_length: int,
) -> dict:
    controls = make_control_records(query_record)
    embedder = GLM2Embedder(
        model_name=model_name,
        batch_size=batch_size,
        max_seq_length=max_seq_length,
        auto_batch_size=auto_batch_size,
    )
    client = create_qdrant_client(qdrant_url)
    embeddings, embed_metrics = embedder.embed_records(controls)
    recreate_collection(client, CONTROL_COLLECTION_NAME, embeddings.shape[1])
    upsert_metrics = upsert_records(client, CONTROL_COLLECTION_NAME, controls, embeddings, batch_size=batch_size)
    query_embedding, _ = embedder.embed_records([query_record])
    response = client.query_points(collection_name=CONTROL_COLLECTION_NAME, query=query_embedding[0].tolist(), limit=4)
    ranked_ids = [point.payload.get("seq_id") for point in getattr(response, "points", [])]
    if not ranked_ids or ranked_ids[0] != "control_exact":
        raise RuntimeError(f"Deterministic control validation failed. Expected top-1 'control_exact', got {ranked_ids[:1]}.")
    return {
        "load_metrics": stage_metrics_dict(embedder.load_metrics),
        "embed_metrics": [stage_metrics_dict(metric) for metric in embed_metrics],
        "upsert_metrics": [stage_metrics_dict(metric) for metric in upsert_metrics],
        "ranked_ids": ranked_ids,
    }


def run_benchmark(args: argparse.Namespace) -> dict:
    index_report = index_corpus(
        corpus_paths=args.corpus,
        corpus_source=args.corpus_source,
        stream_split=args.stream_split,
        stream_limit=args.stream_limit,
        streaming=args.streaming,
        batch_log=args.batch_log,
        model_name=args.model_name,
        collection_name=args.collection_name,
        qdrant_url=args.qdrant_url,
        batch_size=args.batch_size,
        auto_batch_size=args.auto_batch_size,
        max_seq_length=args.max_seq_length,
        metrics_dir=args.metrics_dir,
        recreate=args.recreate,
        excluded_ids=set(args.exclude_header),
        output_json=args.metrics_dir / DEFAULT_INDEX_SUMMARY,
    )
    query_records = load_fasta_records(args.query_fasta)
    controls_report = validate_controls(
        query_record=query_records[0],
        model_name=args.model_name,
        qdrant_url=args.qdrant_url,
        batch_size=args.batch_size,
        auto_batch_size=args.auto_batch_size,
        max_seq_length=args.max_seq_length,
    )
    query_report = query_collection(
        query_paths=args.query_fasta,
        model_name=args.model_name,
        collection_name=args.collection_name,
        qdrant_url=args.qdrant_url,
        batch_size=args.batch_size,
        auto_batch_size=args.auto_batch_size,
        max_seq_length=args.max_seq_length,
        metrics_dir=args.metrics_dir,
        top_k=args.top_k,
        expected_fasta=args.expected_fasta,
        output_json=args.metrics_dir / DEFAULT_QUERY_SUMMARY,
    )
    report = {
        "mode": "benchmark",
        "index": index_report,
        "controls": controls_report,
        "query": query_report,
    }
    summary_path = args.output_json or (args.metrics_dir / DEFAULT_BENCHMARK_SUMMARY)
    write_json(summary_path, report)
    report["summary_path"] = str(summary_path.resolve())
    return report