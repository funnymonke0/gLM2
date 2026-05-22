#!/usr/bin/env python3
"""
GLM2-based sequence search using Qdrant vector database.
Index OG_prot90 corpus from HuggingFace, query with GLM2, validate against reference.
"""

import argparse
import json
import shutil
import subprocess
import tempfile
from pathlib import Path
from dataclasses import dataclass
from typing import Sequence, Optional
import hashlib
import time

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel, BitsAndBytesConfig
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from Bio import SeqIO
import datasets


@dataclass
class SequenceRecord:
    """Container for sequence metadata."""
    seq_id: str
    description: str
    sequence: str
    source: str


class GLM2Embedder:
    """GLM2 embedding model with automatic batch-size backoff on OOM."""

    def __init__(self, model_name: str = "tattabio/gLM2_650M_embed", batch_size: int = 32, max_length: int = 4096, use_4bit: bool = True):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = model_name
        self.batch_size = batch_size
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

        # 4-bit (bitsandbytes nf4) keeps weights at 4× less bandwidth than fp16
        # on this memory-bound GPU class (GDDR6), which outweighs dequant cost.
        if use_4bit and torch.cuda.is_available():
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
            self.model = AutoModel.from_pretrained(
                model_name,
                trust_remote_code=True,
                quantization_config=quantization_config,
                device_map="auto",
            )
        else:
            self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True).to(self.device)

        self.model.eval()

    @staticmethod
    def _safe_empty_cuda_cache() -> None:
        """Best-effort CUDA cache cleanup that tolerates allocator failures."""
        if not torch.cuda.is_available():
            return
        try:
            torch.cuda.empty_cache()
        except Exception:
            # If CUDA is in an error state after OOM, cache cleanup can fail too.
            pass

    @staticmethod
    def _is_oom_error(exc: BaseException) -> bool:
        """Detect CUDA OOM and related accelerator allocation failures."""
        text = str(exc).lower()
        return (
            "out of memory" in text
            or "cuda error" in text and "memory" in text
            or "cudaerrormemoryallocation" in text
        )

    def embed_batch(self, sequences: Sequence[str]) -> np.ndarray:
        """Embed a batch of sequences."""
        tokens = self.tokenizer(
            sequences,
            max_length=self.max_length,
            truncation=True,
            padding=True,
            return_tensors="pt",
        )
        tokens = {k: v.to(self.device) for k, v in tokens.items()}
        
        with torch.no_grad():
            outputs = self.model(**tokens)
            
            # Handle gLM2 returning tuple format
            if isinstance(outputs, tuple):
                embeddings = outputs[0]
            elif hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
                return outputs.pooler_output.cpu().float().numpy()
            else:
                embeddings = outputs.last_hidden_state
            
            attention_mask = tokens["attention_mask"].unsqueeze(-1).float()
            masked = (embeddings * attention_mask).sum(dim=1)
            counts = attention_mask.sum(dim=1).clamp_min(1.0)
            pooled = masked / counts
        
        return pooled.cpu().float().numpy()

    def embed_sequences(self, sequences: Sequence[str]) -> np.ndarray:
        """Embed sequences with automatic batch-size backoff on OOM."""
        embeddings = []
        effective_batch_size = self.batch_size
        position = 0
        
        while position < len(sequences):
            batch = sequences[position : position + effective_batch_size]
            try:
                batch_emb = self.embed_batch(batch)
                embeddings.append(batch_emb)
                position += len(batch)
            except (RuntimeError, torch.AcceleratorError) as e:
                if self._is_oom_error(e) and effective_batch_size > 1:
                    effective_batch_size = max(1, effective_batch_size // 2)
                    print(f"  OOM at batch {effective_batch_size * 2}. Retrying with {effective_batch_size}.")
                    self._safe_empty_cuda_cache()
                else:
                    raise
        
        return np.vstack(embeddings) if embeddings else np.zeros((0, 0), dtype=np.float32)


def load_fasta(path: Path, max_seq_length: int = 0) -> list[SequenceRecord]:
    """Load FASTA file into Sequence objects."""
    records = []
    for record in SeqIO.parse(str(path), "fasta"):
        seq = str(record.seq).replace("*", "").replace(" ", "").upper()
        if seq and (max_seq_length == 0 or len(seq) <= max_seq_length):
            records.append(SequenceRecord(
                seq_id=record.id.strip(),
                description=record.description.strip(),
                sequence=seq,
                source=str(path),
            ))
    if not records:
        raise ValueError(f"No valid sequences in {path}")
    return records


def load_hf_dataset(split: str = "train", limit: Optional[int] = None, streaming: bool = False) -> list[SequenceRecord]:
    """Load OG_prot90 dataset from HuggingFace."""
    print(f"Loading OG_prot90 dataset (split={split}, streaming={streaming}, limit={limit})...")
    ds = datasets.load_dataset("tattabio/OG_prot90", split=split, streaming=streaming)
    
    records = []
    for idx, row in enumerate(ds):
        if limit and len(records) >= limit:
            break
        
        # OG_prot90 has single sequence per row
        if row.get("sequence"):
            seq = str(row["sequence"]).strip().upper()
            if seq:
                records.append(SequenceRecord(
                    seq_id=str(row.get("id") or f"hf_og90_{idx}"),
                    description=str(row.get("id") or f"hf_og90_{idx}"),
                    sequence=seq,
                    source="hf://tattabio/OG_prot90/train",
                ))
    
    if not records:
        raise ValueError(f"No sequences loaded from OG_prot90")
    print(f"Loaded {len(records)} sequences from OG_prot90")
    return records


def iter_hf_dataset_chunks(
    split: str = "train",
    limit: Optional[int] = None,
    chunk_size: int = 2000,
    streaming: bool = False,
):
    """Yield OG_prot90 sequences in bounded chunks to avoid large-memory spikes."""
    print(
        f"Loading OG_prot90 dataset in chunks "
        f"(split={split}, streaming={streaming}, limit={limit}, chunk_size={chunk_size})..."
    )
    ds = datasets.load_dataset("tattabio/OG_prot90", split=split, streaming=streaming)

    chunk: list[SequenceRecord] = []
    total = 0
    for idx, row in enumerate(ds):
        if limit is not None and total >= limit:
            break

        sequence_value = row.get("sequence")
        if not sequence_value:
            continue

        seq = str(sequence_value).strip().upper()
        if not seq:
            continue

        chunk.append(
            SequenceRecord(
                seq_id=str(row.get("id") or f"hf_og90_{idx}"),
                description=str(row.get("id") or f"hf_og90_{idx}"),
                sequence=seq,
                source="hf://tattabio/OG_prot90/train",
            )
        )
        total += 1

        if len(chunk) >= chunk_size:
            yield chunk
            chunk = []

    if chunk:
        yield chunk


def sequence_fingerprint(sequence: str) -> str:
    """Compute sequence fingerprint for validation."""
    normalized = "".join(ch for ch in sequence.upper() if "A" <= ch <= "Z")
    if not normalized:
        return ""
    return hashlib.sha1(normalized.encode("utf-8")).hexdigest()


def _l2_normalize_rows(x: np.ndarray) -> np.ndarray:
    """L2-normalize each row vector for cosine similarity via dot-product."""
    if x.size == 0:
        return x
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-12, None)
    return x / norms


def is_timeout_error(exc: BaseException) -> bool:
    """Best-effort detection for HTTP timeout errors from Qdrant/httpx/httpcore."""
    keywords = ("timed out", "readtimeout", "timeout")
    current: Optional[BaseException] = exc
    for _ in range(6):
        if current is None:
            break
        text = str(current).lower()
        if any(k in text for k in keywords):
            return True
        current = current.__cause__ or current.__context__
    return False


def search_and_validate(
    query_fasta: Path,
    model_name: str = "tattabio/gLM2_650M_embed",
    batch_size: int = 32,
    max_length: int = 4096,
    top_k: int = 10,
    qdrant_url: str = "http://localhost:6333",
    qdrant_timeout: int = 300,
    qdrant_retries: int = 5,
    collection_name: str = "glm2_og90",
    dataset_limit: Optional[int] = None,
    corpus_chunk_size: int = 2000,
    upsert_batch_size: int = 100,
    streaming: bool = False,
    recreate: bool = True,
    expected_fasta: Optional[Path] = None,
) -> dict:
    """
    Index OG_prot90 corpus into Qdrant, query with GLM2 embeddings, optionally validate.
    """
    print(f"\n=== GLM2 Sequence Search ===")
    print(f"Model: {model_name}")
    print(f"Qdrant: {qdrant_url}")
    print(f"Qdrant timeout: {qdrant_timeout}s")
    print(f"Qdrant retries: {qdrant_retries}")
    print(f"Collection: {collection_name}")
    print(f"Max token length: {max_length}")
    print(f"HF loading mode: {'streaming' if streaming else 'download-first'}")
    
    # Load query sequences
    print(f"\nLoading queries from {query_fasta}...")
    query_seqs = load_fasta(query_fasta)
    query_text = [s.sequence for s in query_seqs]
    print(f"Loaded {len(query_seqs)} query sequences")
    
    # Initialize Qdrant
    print(f"\nConnecting to Qdrant at {qdrant_url}...")
    client = QdrantClient(url=qdrant_url, timeout=int(qdrant_timeout))

    def qdrant_call(operation_name: str, fn):
        """Retry Qdrant HTTP operations only on timeout failures."""
        for attempt in range(1, max(1, qdrant_retries) + 1):
            try:
                return fn()
            except Exception as exc:
                if not is_timeout_error(exc) or attempt >= max(1, qdrant_retries):
                    raise
                delay = min(30.0, 1.5 * attempt)
                print(
                    f"  Qdrant timeout during {operation_name} "
                    f"(attempt {attempt}/{qdrant_retries}); retrying in {delay:.1f}s..."
                )
                time.sleep(delay)
    
    # Delete and recreate collection if requested
    if recreate:
        try:
            qdrant_call("delete_collection", lambda: client.delete_collection(collection_name))
            print(f"Deleted existing collection '{collection_name}'")
        except Exception:
            pass
    
    # Initialize embedder
    print(f"\nInitializing GLM2 embedder...")
    embedder = GLM2Embedder(model_name=model_name, batch_size=batch_size, max_length=max_length)

    # Stream corpus in chunks so large trials do not allocate a full in-memory matrix.
    print("\nIndexing OG_prot90 corpus into Qdrant in chunks...")
    total_indexed = 0
    vector_size = -1
    chunk_times: list[float] = []
    import time as _time
    for chunk_idx, corpus_chunk in enumerate(
        iter_hf_dataset_chunks(limit=dataset_limit, chunk_size=corpus_chunk_size, streaming=streaming), start=1
    ):
        _chunk_t0 = _time.monotonic()
        chunk_text = [s.sequence for s in corpus_chunk]
        chunk_embeddings = embedder.embed_sequences(chunk_text)
        if chunk_embeddings.size == 0:
            continue

        if vector_size < 0:
            vector_size = chunk_embeddings.shape[1]
            if not qdrant_call("collection_exists", lambda: client.collection_exists(collection_name)):
                print(f"Creating collection '{collection_name}' with vector size {vector_size}...")
                qdrant_call(
                    "create_collection",
                    lambda: client.create_collection(
                        collection_name=collection_name,
                        vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
                    ),
                )

        points = []
        for i, (seq, embedding) in enumerate(zip(corpus_chunk, chunk_embeddings)):
            points.append(
                PointStruct(
                    id=total_indexed + i,
                    vector=embedding.tolist(),
                    payload={
                        "seq_id": seq.seq_id,
                        "description": seq.description,
                        "sequence": seq.sequence,
                        "source": seq.source,
                    },
                )
            )

        for i in range(0, len(points), upsert_batch_size):
            batch = points[i : i + upsert_batch_size]
            qdrant_call("upsert", lambda: client.upsert(collection_name=collection_name, points=batch))

        total_indexed += len(points)
        _chunk_elapsed = _time.monotonic() - _chunk_t0
        chunk_times.append(_chunk_elapsed)
        avg_sec = sum(chunk_times) / len(chunk_times)
        seqs_per_sec = corpus_chunk_size / avg_sec
        eta_str = ""
        if dataset_limit:
            remaining = dataset_limit - total_indexed
            if remaining > 0:
                eta_sec = remaining / seqs_per_sec
                eta_str = f"  ETA {eta_sec/60:.1f} min"
        print(
            f"  Chunk {chunk_idx}: indexed {len(points)} (total={total_indexed})"
            f"  {seqs_per_sec:.1f} seqs/s{eta_str}"
        )

        # Encourage early release of memory between large chunks.
        del chunk_embeddings
        del points
        embedder._safe_empty_cuda_cache()

    if total_indexed == 0:
        raise ValueError("No sequences indexed from OG_prot90")
    print(f"Indexed {total_indexed} sequences into Qdrant")
    
    # Embed queries
    print(f"\nEmbedding {len(query_text)} query sequences...")
    query_embeddings = embedder.embed_sequences(query_text)
    
    # Query Qdrant
    print(f"\nQuerying Qdrant for top-{top_k} matches per query...")
    results = []
    for query_idx, (query, embedding) in enumerate(zip(query_seqs, query_embeddings)):
        response = qdrant_call(
            "query_points",
            lambda: client.query_points(
                collection_name=collection_name,
                query=embedding.tolist(),
                limit=top_k,
            ),
        )
        points_iter = getattr(response, "points", None) or []

        matches = []
        for rank, point in enumerate(points_iter, start=1):
            payload = point.payload or {}
            matches.append(
                {
                    "rank": rank,
                    "corpus_id": payload.get("seq_id"),
                    "corpus_description": payload.get("description"),
                    "similarity_score": float(point.score),
                    "corpus_sequence": payload.get("sequence"),
                }
            )
        
        results.append({
            "query_id": query.seq_id,
            "query_description": query.description,
            "query_length": len(query.sequence),
            "matches": matches,
        })
    
    output = {
        "mode": "index_and_query",
        "model": model_name,
        "qdrant_url": qdrant_url,
        "collection_name": collection_name,
        "corpus_source": "hf://tattabio/OG_prot90/train",
        "corpus_count": total_indexed,
        "corpus_limit": dataset_limit,
        "query_file": str(query_fasta),
        "query_count": len(query_seqs),
        "top_k": top_k,
        "results": results,
    }
    
    # Validate against reference if provided
    if expected_fasta:
        print(f"\nValidating against reference {expected_fasta}...")
        expected_seqs = load_fasta(expected_fasta)
        expected_fps = {sequence_fingerprint(s.sequence): s.seq_id for s in expected_seqs}
        
        validation = []
        for result in results:
            actual_fps = [sequence_fingerprint(hit["corpus_sequence"]) for hit in result["matches"]]
            expected_hit_ids = [expected_fps.get(fp) for fp in actual_fps if fp in expected_fps]
            
            validation.append({
                "query_id": result["query_id"],
                "top_match_ids": [hit["corpus_id"] for hit in result["matches"][:3]],
                "expected_in_top_k": len(expected_hit_ids) > 0,
                "matched_ids": expected_hit_ids,
            })
        
        output["validation"] = {
            "reference_file": str(expected_fasta),
            "reference_count": len(expected_seqs),
            "results": validation,
        }
    
    return output


def search_local_fasta(
    query_fasta: Path,
    corpus_fasta: Path,
    model_name: str = "tattabio/gLM2_650M_embed",
    batch_size: int = 32,
    max_length: int = 4096,
    top_k: int = 10,
    expected_fasta: Optional[Path] = None,
) -> dict:
    """Minimal local search mode: embed corpus/query FASTA files and rank by cosine similarity."""
    print(f"\n=== GLM2 Local FASTA Search ===")
    print(f"Model: {model_name}")
    print(f"Corpus FASTA: {corpus_fasta}")
    print(f"Query FASTA: {query_fasta}")
    print(f"Max token length: {max_length}")

    print(f"\nLoading corpus from {corpus_fasta}...")
    corpus_records = load_fasta(corpus_fasta)
    print(f"Loaded {len(corpus_records)} corpus sequences")

    print(f"Loading queries from {query_fasta}...")
    query_records = load_fasta(query_fasta)
    print(f"Loaded {len(query_records)} query sequences")

    print(f"\nInitializing GLM2 embedder...")
    embedder = GLM2Embedder(model_name=model_name, batch_size=batch_size, max_length=max_length)

    print(f"Embedding corpus...")
    corpus_embeddings = embedder.embed_sequences([r.sequence for r in corpus_records]).astype(np.float32)
    corpus_embeddings = _l2_normalize_rows(corpus_embeddings)

    print(f"Embedding queries...")
    query_embeddings = embedder.embed_sequences([r.sequence for r in query_records]).astype(np.float32)
    query_embeddings = _l2_normalize_rows(query_embeddings)

    top_k = max(1, min(top_k, len(corpus_records)))
    results = []
    for q_idx, (query, q_emb) in enumerate(zip(query_records, query_embeddings)):
        scores = corpus_embeddings @ q_emb
        order = np.argsort(-scores)[:top_k]
        matches = []
        for rank, corpus_idx in enumerate(order, start=1):
            c = corpus_records[int(corpus_idx)]
            matches.append(
                {
                    "rank": rank,
                    "corpus_id": c.seq_id,
                    "corpus_description": c.description,
                    "similarity_score": float(scores[int(corpus_idx)]),
                    "corpus_sequence": c.sequence,
                }
            )
        results.append(
            {
                "query_id": query.seq_id,
                "query_description": query.description,
                "query_length": len(query.sequence),
                "matches": matches,
            }
        )

    output = {
        "mode": "local_fasta_query",
        "model": model_name,
        "corpus_source": str(corpus_fasta),
        "corpus_count": len(corpus_records),
        "query_file": str(query_fasta),
        "query_count": len(query_records),
        "top_k": top_k,
        "results": results,
    }

    if expected_fasta:
        print(f"\nValidating against reference {expected_fasta}...")
        expected_seqs = load_fasta(expected_fasta)
        expected_fps = {sequence_fingerprint(s.sequence): s.seq_id for s in expected_seqs}

        validation = []
        for result in results:
            actual_fps = [sequence_fingerprint(hit["corpus_sequence"]) for hit in result["matches"]]
            expected_hit_ids = [expected_fps.get(fp) for fp in actual_fps if fp in expected_fps]
            validation.append(
                {
                    "query_id": result["query_id"],
                    "top_match_ids": [hit["corpus_id"] for hit in result["matches"][:3]],
                    "expected_in_top_k": len(expected_hit_ids) > 0,
                    "matched_ids": expected_hit_ids,
                }
            )

        output["validation"] = {
            "reference_file": str(expected_fasta),
            "reference_count": len(expected_seqs),
            "results": validation,
        }

    return output


# ---------------------------------------------------------------------------
# DIAMOND pre-filter helpers
# ---------------------------------------------------------------------------

def _find_diamond_binary(diamond_bin: Optional[str] = None) -> str:
    """Locate the diamond executable: explicit path → bin/ subdir → PATH."""
    candidates: list[str] = []
    if diamond_bin:
        candidates.append(diamond_bin)
    script_dir = Path(__file__).parent
    candidates.append(str(script_dir / "bin" / "diamond.exe"))
    candidates.append(str(script_dir / "bin" / "diamond"))
    candidates.append("diamond")
    for c in candidates:
        if Path(c).exists() or shutil.which(c):
            return c
    raise FileNotFoundError(
        "diamond not found. Pass --diamond-bin or place diamond.exe in bin/."
    )


def _extract_sequences_from_db(
    db_path: Path,
    seq_ids: list[str],
    diamond_bin: str,
) -> list[SequenceRecord]:
    """Extract sequences from a DIAMOND database by ID using 'diamond getseq'."""
    out_fasta = Path(tempfile.mktemp(suffix="_getseq.fasta"))
    try:
        cmd = [diamond_bin, "getseq", "--db", str(db_path), "--out", str(out_fasta)]
        for sid in seq_ids:
            cmd.extend(["--seq", sid])
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if proc.stderr:
            for line in proc.stderr.splitlines():
                print(f"  [getseq] {line}")
        if proc.returncode != 0:
            raise RuntimeError(f"diamond getseq failed (exit {proc.returncode}):\n{proc.stdout or '(no output)'}")
        records: list[SequenceRecord] = []
        for record in SeqIO.parse(str(out_fasta), "fasta"):
            seq = str(record.seq).replace("*", "").replace(" ", "").upper()
            if seq:
                records.append(SequenceRecord(
                    seq_id=record.id,
                    description=record.description,
                    sequence=seq,
                    source=str(db_path),
                ))
        return records
    finally:
        out_fasta.unlink(missing_ok=True)


def export_corpus_to_fasta(
    output_path: Path,
    limit: Optional[int] = None,
    streaming: bool = False,
) -> int:
    """Write OG_prot90 sequences (from HF cache) to a FASTA file."""
    print(f"Exporting OG_prot90 corpus to {output_path} ...")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output_path.open("w") as fh:
        for chunk in iter_hf_dataset_chunks(limit=limit, chunk_size=5000, streaming=streaming):
            for rec in chunk:
                fh.write(f">{rec.seq_id}\n{rec.sequence}\n")
                count += 1
            print(f"  {count} sequences written...", end="\r")
    print(f"  Exported {count} sequences to {output_path}          ")
    return count


def _run_diamond_blastp(
    query_fasta: Path,
    corpus_fasta: Path,
    db_path: Path,
    output_path: Path,
    diamond_bin: str,
    top_n: int = 500,
    evalue: float = 0.001,
    sensitivity: str = "sensitive",
) -> list[tuple[str, str]]:
    """Build DIAMOND db (if absent) and run blastp. Returns ordered (seq_id, sequence) pairs."""
    db_file = Path(str(db_path) + ".dmnd")

    def _build_db() -> None:
        print(f"Building DIAMOND database from {corpus_fasta} ({corpus_fasta.stat().st_size / 1e6:.0f} MB) ...")
        proc = subprocess.run(
            [diamond_bin, "makedb", "--in", str(corpus_fasta), "--db", str(db_path)],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        )
        if proc.stderr:
            for line in proc.stderr.splitlines():
                print(f"  [makedb] {line}")
        if proc.returncode != 0:
            db_file.unlink(missing_ok=True)  # remove partial file
            raise RuntimeError(f"diamond makedb failed (exit {proc.returncode}):\n{proc.stdout or '(no output)'}")
        print(f"  Database built: {db_file} ({db_file.stat().st_size / 1e6:.0f} MB)")

    if not db_file.exists():
        _build_db()
    else:
        # Quick sanity-check: try opening the DB; rebuild if it's incomplete
        probe = subprocess.run(
            [diamond_bin, "dbinfo", "--db", str(db_path)],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        )
        combined = (probe.stdout + probe.stderr).lower()
        if probe.returncode != 0 or "incomplete" in combined or "error" in combined:
            print(f"DIAMOND database appears incomplete or corrupt — rebuilding ...")
            db_file.unlink(missing_ok=True)
            _build_db()
        else:
            print(f"Using existing DIAMOND database: {db_file} ({db_file.stat().st_size / 1e6:.0f} MB)")

    # Print sizes to help diagnose failures on large corpora
    if corpus_fasta.exists():
        print(f"  Corpus FASTA : {corpus_fasta.stat().st_size / 1e6:.0f} MB")
    if db_file.exists():
        print(f"  DIAMOND DB   : {db_file.stat().st_size / 1e6:.0f} MB")

    print(f"Running DIAMOND blastp (top {top_n} hits, e-value <= {evalue}, {sensitivity} mode) ...")
    cmd = [
        diamond_bin, "blastp",
        "--db", str(db_path),
        "--query", str(query_fasta),
        "--out", str(output_path),
        "--outfmt", "6", "sseqid", "sseq",
        "--max-target-seqs", str(top_n),
        "--evalue", str(evalue),
        f"--{sensitivity}",
    ]
    print(f"  Command: {' '.join(str(a) for a in cmd)}")
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    # DIAMOND writes progress to stderr — always print so failures are visible
    if proc.stderr:
        for line in proc.stderr.splitlines():
            print(f"  [diamond] {line}")
    if proc.returncode != 0:
        detail = (proc.stdout or "(no stdout)").strip()
        raise RuntimeError(f"diamond blastp failed (exit {proc.returncode}):\n{detail}")

    hits: list[tuple[str, str]] = []
    seen: set[str] = set()
    with output_path.open() as fh:
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 2:
                sid = parts[0].strip()
                seq = parts[1].strip().replace("-", "")  # strip alignment gaps
                if sid not in seen and seq:
                    seen.add(sid)
                    hits.append((sid, seq))
    print(f"  DIAMOND returned {len(hits)} unique hits")
    return hits


def _build_full_seqs_cache(hit_ids: set[str], output_path: Path) -> int:
    """
    Stream OG_prot90 from HuggingFace and save full sequences for the given hit IDs.

    Both DIAMOND hit IDs and OG_prot90 HF IDs use +/- for strand (e.g. |+| / |-|).
    No normalization is needed — IDs match directly.
    """
    remaining = set(hit_ids)

    print(f"Streaming OG_prot90 from HuggingFace to fetch {len(remaining)} full sequences ...")
    print("  (Requires internet; may take 1-4 hrs to scan all 85 M sequences; no full dataset cached)")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    found = 0
    total_scanned = 0

    tmp_path = output_path.with_suffix(".tmp")
    try:
        ds = datasets.load_dataset("tattabio/OG_prot90", split="train", streaming=True)
        with tmp_path.open("w") as fh:
            for row in ds:
                if not remaining:
                    break
                total_scanned += 1
                row_id = str(row.get("id", ""))
                if row_id in remaining:
                    seq = str(row.get("sequence", "")).strip().upper()
                    if seq:
                        fh.write(f">{row_id}\n{seq}\n")
                        found += 1
                    remaining.discard(row_id)
                if total_scanned % 1_000_000 == 0:
                    print(f"  Scanned {total_scanned / 1e6:.0f}M seqs, found {found}/{len(hit_ids)} ...", flush=True)
        tmp_path.replace(output_path)  # atomic rename — only exists if complete
    except Exception:
        tmp_path.unlink(missing_ok=True)  # don't leave a partial file
        raise

    print(f"  Done: {found}/{len(hit_ids)} sequences found after scanning {total_scanned / 1e6:.1f}M records")
    return found


def search_with_diamond_prefilter(
    query_fasta: Path,
    corpus_fasta: Optional[Path] = None,
    diamond_db: Optional[Path] = None,
    diamond_bin: Optional[str] = None,
    diamond_top_n: int = 500,
    diamond_evalue: float = 0.001,
    diamond_sensitivity: str = "sensitive",
    diamond_hits_file: Optional[Path] = None,
    full_seqs_cache: Optional[Path] = None,
    model_name: str = "tattabio/gLM2_650M_embed",
    batch_size: int = 8,
    max_length: int = 4096,
    top_k: int = 10,
    dataset_limit: Optional[int] = None,
    expected_fasta: Optional[Path] = None,
    streaming: bool = False,
) -> dict:
    """
    Two-stage search: DIAMOND blastp for fast recall, gLM2 cosine similarity for precision.

    If corpus_fasta does not exist it is created by exporting OG_prot90 from the
    HuggingFace cache (respecting dataset_limit).  The DIAMOND database is built
    automatically alongside the corpus file on the first run and reused thereafter.
    """
    print("\n=== GLM2 + DIAMOND Pre-filter Search ===")
    print(f"Model         : {model_name}")
    print(f"Query         : {query_fasta}")

    diamond_exe = _find_diamond_binary(diamond_bin)
    print(f"Diamond binary: {diamond_exe}")

    # ---- 1. Resolve paths; export FASTA only if DB doesn't already exist ----
    if corpus_fasta is None:
        corpus_fasta = Path("og90_corpus.fasta")

    # ---- 2. DIAMOND search ----
    if diamond_db is None:
        diamond_db = corpus_fasta.with_suffix("")  # strip .fasta → use as db prefix
    elif Path(diamond_db).suffix == ".dmnd":
        diamond_db = Path(diamond_db).with_suffix("")  # user passed path with extension — strip it

    db_file = Path(str(diamond_db) + ".dmnd")
    if db_file.exists():
        if corpus_fasta.exists():
            print(f"Corpus FASTA  : {corpus_fasta} (exists)")
        else:
            print(f"Corpus FASTA  : not present (DIAMOND DB already built, skipping export)")
    else:
        if not corpus_fasta.exists():
            export_corpus_to_fasta(corpus_fasta, limit=dataset_limit, streaming=streaming)
        else:
            print(f"Corpus FASTA  : {corpus_fasta} (exists, skipping export)")

    # ---- 2b. Load cached hits or run DIAMOND blastp ----
    if diamond_hits_file and Path(diamond_hits_file).exists():
        print(f"Loading cached DIAMOND hits from {diamond_hits_file} ...")
        hits: list[tuple[str, str]] = []
        seen: set[str] = set()
        with open(diamond_hits_file) as fh:
            for line in fh:
                parts = line.rstrip("\n").split("\t")
                if len(parts) >= 2:
                    sid, seq = parts[0].strip(), parts[1].strip().replace("-", "")
                    if sid not in seen and seq:
                        seen.add(sid)
                        hits.append((sid, seq))
        print(f"  Loaded {len(hits)} cached hits")
    else:
        # Use mktemp to get a path without creating a file — avoids Windows handle-locking issues
        diamond_out = Path(tempfile.mktemp(suffix="_diamond_out.tsv"))  # noqa: S306
        try:
            hits = _run_diamond_blastp(
                query_fasta=query_fasta,
                corpus_fasta=corpus_fasta,
                db_path=diamond_db,
                output_path=diamond_out,
                diamond_bin=diamond_exe,
                top_n=diamond_top_n,
                evalue=diamond_evalue,
                sensitivity=diamond_sensitivity,
            )
        finally:
            if diamond_hits_file and diamond_out.exists():
                diamond_out.rename(diamond_hits_file)
                print(f"  Saved DIAMOND hits to {diamond_hits_file}")
            else:
                diamond_out.unlink(missing_ok=True)

    if not hits:
        print("WARNING: DIAMOND found no hits. Try --diamond-evalue 1 or --diamond-sensitivity more-sensitive.")
        return {
            "mode": "diamond_prefilter",
            "diamond_hits": 0,
            "results": [],
        }

    # ---- 3. Build candidate records ----
    # Priority: full_seqs_cache > corpus_fasta > sseq (DIAMOND aligned region)
    # Using full sequences instead of aligned regions gives better gLM2 embeddings
    # for distant homologs where the alignment may cover only part of the protein.
    print(f"Loading {len(hits)} candidate sequences ...")
    candidate_records: list[SequenceRecord] = []
    hit_id_set = {sid for sid, _ in hits}
    hit_sseq: dict[str, str] = {sid: seq for sid, seq in hits}  # fallback sseq by id

    def _load_candidates_from_fasta(fasta_path: Path) -> list[SequenceRecord]:
        loaded = []
        for record in SeqIO.parse(str(fasta_path), "fasta"):
            if record.id in hit_id_set:
                seq = str(record.seq).replace("*", "").replace(" ", "").upper()
                if seq:
                    loaded.append(SequenceRecord(
                        seq_id=record.id,
                        description=record.description,
                        sequence=seq,
                        source=str(fasta_path),
                    ))
        return loaded

    def _fill_missing_with_sseq(records: list[SequenceRecord]) -> list[SequenceRecord]:
        """Append sseq fallback for any hit IDs not yet in records."""
        covered = {r.seq_id for r in records}
        missing = hit_id_set - covered
        if missing:
            print(f"  WARNING: {len(missing)} hits not in FASTA; falling back to aligned region (sseq)")
        for sid in missing:
            seq = hit_sseq.get(sid, "")
            if seq:
                records.append(SequenceRecord(seq_id=sid, description=sid, sequence=seq.upper(), source=str(diamond_db)))
        return records

    cache_path = Path(full_seqs_cache) if full_seqs_cache else None
    cache_ready = cache_path is not None and cache_path.exists() and cache_path.stat().st_size > 0

    if cache_ready and cache_path is not None:
        print(f"  Using full-sequence cache: {cache_path}")
        candidate_records = _load_candidates_from_fasta(cache_path)
        candidate_records = _fill_missing_with_sseq(candidate_records)
    elif corpus_fasta.exists():
        candidate_records = _load_candidates_from_fasta(corpus_fasta)
        candidate_records = _fill_missing_with_sseq(candidate_records)
    elif cache_path:
        # Build cache by streaming OG_prot90 from HF, then load
        _build_full_seqs_cache(hit_id_set, cache_path)
        candidate_records = _load_candidates_from_fasta(cache_path)
        candidate_records = _fill_missing_with_sseq(candidate_records)
    else:
        # Use aligned region (sseq) — lower quality for distant homologs
        print("  NOTE: using DIAMOND aligned regions (sseq). Pass --full-seqs-cache to fetch full sequences.")
        for sid, seq in hits:
            candidate_records.append(SequenceRecord(
                seq_id=sid, description=sid, sequence=seq.upper(), source=str(diamond_db),
            ))
    print(f"  Loaded {len(candidate_records)} candidates")

    query_records = load_fasta(query_fasta)

    # ---- 4. gLM2 embedding + cosine re-rank ----
    print(f"\nEmbedding {len(candidate_records)} candidates + {len(query_records)} queries with gLM2 ...")
    embedder = GLM2Embedder(model_name=model_name, batch_size=batch_size, max_length=max_length)

    cand_emb = _l2_normalize_rows(
        embedder.embed_sequences([r.sequence for r in candidate_records]).astype(np.float32)
    )
    q_emb = _l2_normalize_rows(
        embedder.embed_sequences([r.sequence for r in query_records]).astype(np.float32)
    )

    top_k_eff = max(1, min(top_k, len(candidate_records)))
    results = []
    for query, qe in zip(query_records, q_emb):
        scores = cand_emb @ qe
        order = np.argsort(-scores)[:top_k_eff]
        matches = []
        for rank, cidx in enumerate(order, start=1):
            c = candidate_records[int(cidx)]
            matches.append({
                "rank": rank,
                "corpus_id": c.seq_id,
                "corpus_description": c.description,
                "similarity_score": float(scores[int(cidx)]),
                "corpus_sequence": c.sequence,
            })
        results.append({
            "query_id": query.seq_id,
            "query_description": query.description,
            "query_length": len(query.sequence),
            "matches": matches,
        })

    output: dict = {
        "mode": "diamond_prefilter",
        "model": model_name,
        "corpus_source": str(corpus_fasta),
        "diamond_hits": len(hits),
        "candidates_embedded": len(candidate_records),
        "query_file": str(query_fasta),
        "query_count": len(query_records),
        "top_k": top_k,
        "results": results,
    }

    if expected_fasta:
        print(f"\nValidating against reference {expected_fasta} ...")
        expected_seqs = load_fasta(expected_fasta)
        # Build lookup by sequence fingerprint (works when sseq = full sequence)
        expected_fps = {sequence_fingerprint(s.sequence): s.seq_id for s in expected_seqs}
        # Build lookup by normalized ID: forward→+, reverse→- for robust ID matching
        def _normalize_strand(seq_id: str) -> str:
            return seq_id.replace("|forward|", "|+|").replace("|reverse|", "|-|")
        expected_norm_ids = {_normalize_strand(s.seq_id): s.seq_id for s in expected_seqs}
        validation = []
        for res in results:
            matched: dict[str, str] = {}  # normalized_corpus_id → reference_id
            for h in res["matches"]:
                cid = h["corpus_id"]
                norm_cid = _normalize_strand(cid)
                # ID-based match (primary — works even with truncated sseq)
                if norm_cid in expected_norm_ids:
                    matched[cid] = expected_norm_ids[norm_cid]
                # Fingerprint-based fallback (for cases where IDs differ but sequence matches)
                elif sequence_fingerprint(h["corpus_sequence"]) in expected_fps:
                    matched[cid] = expected_fps[sequence_fingerprint(h["corpus_sequence"])]
            matched_ref_ids = list(matched.values())
            validation.append({
                "query_id": res["query_id"],
                "top_match_ids": [h["corpus_id"] for h in res["matches"][:3]],
                "expected_in_top_k": len(matched_ref_ids) > 0,
                "matched_count": len(matched_ref_ids),
                "matched_ids": matched_ref_ids,
            })
        output["validation"] = {
            "reference_file": str(expected_fasta),
            "reference_count": len(expected_seqs),
            "results": validation,
        }

    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GLM2 sequence search using Qdrant and OG_prot90 corpus.")
    parser.add_argument("query_fasta", type=Path, help="Query FASTA file")
    parser.add_argument("--model", default="tattabio/gLM2_650M_embed", help="HF model name for embedding")
    parser.add_argument("--batch-size", type=int, default=8, help="Embedding batch size (default: 8, auto-backoff on OOM)")
    parser.add_argument("--max-length", type=int, default=4096, help="Tokenizer max length for truncation (default: 4096)")
    parser.add_argument("--top-k", type=int, default=10, help="Return top-k matches per query (default: 10)")
    parser.add_argument("--qdrant-url", default="http://localhost:6333", help="Qdrant server URL (default: http://localhost:6333)")
    parser.add_argument("--qdrant-timeout", type=int, default=300, help="Qdrant HTTP timeout in seconds (default: 300)")
    parser.add_argument("--qdrant-retries", type=int, default=5, help="Retries for Qdrant timeout errors (default: 5)")
    parser.add_argument("--collection-name", default="glm2_og90", help="Qdrant collection name (default: glm2_og90)")
    parser.add_argument("--corpus-fasta", type=Path, help="Local FASTA corpus to search (minimal mode, skips HF/Qdrant)")
    parser.add_argument("--dataset-limit", type=int, help="Limit corpus size for testing (default: no limit)")
    parser.add_argument(
        "--corpus-chunk-size",
        type=int,
        default=500,
        help="Corpus records processed per embed/upsert chunk (default: 500)",
    )
    parser.add_argument(
        "--upsert-batch-size",
        type=int,
        default=50,
        help="Qdrant upsert batch size within each chunk (default: 50)",
    )
    parser.add_argument("--no-recreate", action="store_true", help="Don't recreate collection if it exists (default: recreate)")
    parser.add_argument("--expected-fasta", type=Path, help="Reference FASTA for validation (default: none)")
    parser.add_argument(
        "--streaming",
        action="store_true",
        help="Use HuggingFace streaming instead of download-first cache mode (default: download-first)",
    )
    parser.add_argument("--output", type=Path, default="results.json", help="Output JSON file path (default: results.json)")

    # DIAMOND pre-filter options
    parser.add_argument(
        "--diamond-prefilter", action="store_true",
        help="Two-stage search: DIAMOND blastp for recall, gLM2 cosine for precision",
    )
    parser.add_argument(
        "--diamond-corpus", type=Path, default=None,
        help="Corpus FASTA for DIAMOND (created from OG_prot90 if absent; default: og90_corpus.fasta)",
    )
    parser.add_argument(
        "--diamond-db", type=Path, default=None,
        help="DIAMOND database prefix (built alongside --diamond-corpus if absent)",
    )
    parser.add_argument(
        "--diamond-bin", default=None,
        help="Path to diamond executable (default: auto-detect bin/diamond.exe then PATH)",
    )
    parser.add_argument(
        "--diamond-top-n", type=int, default=500,
        help="Max DIAMOND hits passed to gLM2 re-ranking (default: 500)",
    )
    parser.add_argument(
        "--diamond-evalue", type=float, default=0.001,
        help="DIAMOND e-value cutoff (default: 0.001; use 1 for very remote homologs)",
    )
    parser.add_argument(
        "--diamond-sensitivity", default="sensitive",
        choices=["fast", "mid-sensitive", "sensitive", "more-sensitive", "very-sensitive", "ultra-sensitive"],
        help="DIAMOND sensitivity mode (default: sensitive)",
    )
    parser.add_argument(
        "--diamond-hits-file", type=Path, default=None,
        help="Cache file for DIAMOND hits (TSV: sseqid\\tsseq). If it exists, skip blastp and load from cache.",
    )
    parser.add_argument(
        "--full-seqs-cache", type=Path, default=None,
        help="FASTA of full (non-truncated) sequences for DIAMOND hit IDs. "
             "If absent, streams OG_prot90 from HuggingFace to build it (~1-4 hrs). "
             "Greatly improves gLM2 ranking vs. DIAMOND aligned regions (sseq).",
    )

    args = parser.parse_args()

    if args.diamond_prefilter:
        result = search_with_diamond_prefilter(
            query_fasta=args.query_fasta,
            corpus_fasta=args.diamond_corpus,
            diamond_db=args.diamond_db,
            diamond_bin=args.diamond_bin,
            diamond_top_n=args.diamond_top_n,
            diamond_evalue=args.diamond_evalue,
            diamond_sensitivity=args.diamond_sensitivity,
            diamond_hits_file=args.diamond_hits_file,
            full_seqs_cache=args.full_seqs_cache,
            model_name=args.model,
            batch_size=args.batch_size,
            max_length=args.max_length,
            top_k=args.top_k,
            dataset_limit=args.dataset_limit,
            expected_fasta=args.expected_fasta,
            streaming=args.streaming,
        )
    elif args.corpus_fasta:
        result = search_local_fasta(
            query_fasta=args.query_fasta,
            corpus_fasta=args.corpus_fasta,
            model_name=args.model,
            batch_size=args.batch_size,
            max_length=args.max_length,
            top_k=args.top_k,
            expected_fasta=args.expected_fasta,
        )
    else:
        result = search_and_validate(
            query_fasta=args.query_fasta,
            model_name=args.model,
            batch_size=args.batch_size,
            max_length=args.max_length,
            top_k=args.top_k,
            qdrant_url=args.qdrant_url,
            qdrant_timeout=args.qdrant_timeout,
            qdrant_retries=args.qdrant_retries,
            collection_name=args.collection_name,
            dataset_limit=args.dataset_limit,
            corpus_chunk_size=args.corpus_chunk_size,
            upsert_batch_size=args.upsert_batch_size,
            streaming=args.streaming,
            recreate=not args.no_recreate,
            expected_fasta=args.expected_fasta,
        )
    
    output = json.dumps(result, indent=2)
    if args.output:
        args.output.write_text(output)
        print(f"\n✓ Results written to {args.output}")
    else:
        print(f"\n{output}")

