#!/usr/bin/env python3
"""
GLM2-based sequence search using Qdrant vector database.
Index OG_prot90 corpus from HuggingFace, query with GLM2, validate against reference.
"""

import argparse
import json
from pathlib import Path
from dataclasses import dataclass
from typing import Sequence, Optional
import hashlib

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
        
        # Load with 4-bit quantization to reduce memory usage
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
            except RuntimeError as e:
                if "out of memory" in str(e).lower() and effective_batch_size > 1:
                    effective_batch_size = max(1, effective_batch_size // 2)
                    print(f"  OOM at batch {effective_batch_size * 2}. Retrying with {effective_batch_size}.")
                    torch.cuda.empty_cache() if torch.cuda.is_available() else None
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


def load_hf_dataset(split: str = "train", limit: Optional[int] = None, streaming: bool = True) -> list[SequenceRecord]:
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


def sequence_fingerprint(sequence: str) -> str:
    """Compute sequence fingerprint for validation."""
    normalized = "".join(ch for ch in sequence.upper() if "A" <= ch <= "Z")
    if not normalized:
        return ""
    return hashlib.sha1(normalized.encode("utf-8")).hexdigest()


def search_and_validate(
    query_fasta: Path,
    model_name: str = "tattabio/gLM2_650M_embed",
    batch_size: int = 32,
    top_k: int = 10,
    qdrant_url: str = "http://localhost:6333",
    collection_name: str = "glm2_og90",
    dataset_limit: Optional[int] = None,
    recreate: bool = True,
    expected_fasta: Optional[Path] = None,
) -> dict:
    """
    Index OG_prot90 corpus into Qdrant, query with GLM2 embeddings, optionally validate.
    """
    print(f"\n=== GLM2 Sequence Search ===")
    print(f"Model: {model_name}")
    print(f"Qdrant: {qdrant_url}")
    print(f"Collection: {collection_name}")
    
    # Load query sequences
    print(f"\nLoading queries from {query_fasta}...")
    query_seqs = load_fasta(query_fasta)
    query_text = [s.sequence for s in query_seqs]
    print(f"Loaded {len(query_seqs)} query sequences")
    
    # Load corpus from HuggingFace
    corpus_seqs = load_hf_dataset(limit=dataset_limit)
    corpus_text = [s.sequence for s in corpus_seqs]
    
    # Initialize Qdrant
    print(f"\nConnecting to Qdrant at {qdrant_url}...")
    client = QdrantClient(url=qdrant_url)
    
    # Delete and recreate collection if requested
    if recreate:
        try:
            client.delete_collection(collection_name)
            print(f"Deleted existing collection '{collection_name}'")
        except Exception:
            pass
    
    # Initialize embedder
    print(f"\nInitializing GLM2 embedder...")
    embedder = GLM2Embedder(model_name=model_name, batch_size=batch_size)
    
    # Embed corpus in batches
    print(f"\nEmbedding {len(corpus_text)} corpus sequences...")
    corpus_embeddings = embedder.embed_sequences(corpus_text)
    print(f"Corpus embeddings shape: {corpus_embeddings.shape}")
    
    # Create collection
    if not client.collection_exists(collection_name):
        vector_size = corpus_embeddings.shape[1]
        print(f"Creating collection '{collection_name}' with vector size {vector_size}...")
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
        )
    
    # Upsert corpus into Qdrant
    print(f"Indexing corpus into Qdrant...")
    points = []
    for point_id, (seq, embedding) in enumerate(zip(corpus_seqs, corpus_embeddings)):
        points.append(
            PointStruct(
                id=point_id,
                vector=embedding.tolist(),
                payload={
                    "seq_id": seq.seq_id,
                    "description": seq.description,
                    "sequence": seq.sequence,
                    "source": seq.source,
                },
            )
        )
    
    # Batch upsert
    batch_upsert_size = 100
    for i in range(0, len(points), batch_upsert_size):
        batch = points[i : i + batch_upsert_size]
        client.upsert(collection_name=collection_name, points=batch)
    print(f"Indexed {len(points)} sequences into Qdrant")
    
    # Embed queries
    print(f"\nEmbedding {len(query_text)} query sequences...")
    query_embeddings = embedder.embed_sequences(query_text)
    
    # Query Qdrant
    print(f"\nQuerying Qdrant for top-{top_k} matches per query...")
    results = []
    for query_idx, (query, embedding) in enumerate(zip(query_seqs, query_embeddings)):
        response = client.query_points(
            collection_name=collection_name,
            query=embedding.tolist(),
            limit=top_k,
        )
        
        matches = []
        for rank, point in enumerate(response.points, start=1):
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
        "corpus_count": len(corpus_seqs),
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GLM2 sequence search using Qdrant and OG_prot90 corpus.")
    parser.add_argument("query_fasta", type=Path, help="Query FASTA file")
    parser.add_argument("--model", default="tattabio/gLM2_650M_embed", help="HF model name for embedding")
    parser.add_argument("--batch-size", type=int, default=32, help="Embedding batch size (default: 32, auto-backoff on OOM)")
    parser.add_argument("--top-k", type=int, default=10, help="Return top-k matches per query (default: 10)")
    parser.add_argument("--qdrant-url", default="http://localhost:6333", help="Qdrant server URL (default: http://localhost:6333)")
    parser.add_argument("--collection-name", default="glm2_og90", help="Qdrant collection name (default: glm2_og90)")
    parser.add_argument("--dataset-limit", type=int, help="Limit corpus size for testing (default: no limit)")
    parser.add_argument("--no-recreate", action="store_true", help="Don't recreate collection if it exists (default: recreate)")
    parser.add_argument("--expected-fasta", type=Path, help="Reference FASTA for validation (default: none)")
    parser.add_argument("--output", type=Path, default="results.json", help="Output JSON file path (default: results.json)")
    
    args = parser.parse_args()
    
    result = search_and_validate(
        query_fasta=args.query_fasta,
        model_name=args.model,
        batch_size=args.batch_size,
        top_k=args.top_k,
        qdrant_url=args.qdrant_url,
        collection_name=args.collection_name,
        dataset_limit=args.dataset_limit,
        recreate=not args.no_recreate,
        expected_fasta=args.expected_fasta,
    )
    
    output = json.dumps(result, indent=2)
    if args.output:
        args.output.write_text(output)
        print(f"\n✓ Results written to {args.output}")
    else:
        print(f"\n{output}")

