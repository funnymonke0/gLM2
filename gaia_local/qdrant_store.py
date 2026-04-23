import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from typing import Sequence

from gaia_local.metrics import timed_stage
from gaia_local.types import SequenceRecord, StageMetrics


def create_qdrant_client(qdrant_url: str) -> QdrantClient:
    return QdrantClient(url=qdrant_url)


def recreate_collection(client: QdrantClient, collection_name: str, vector_size: int) -> None:
    if client.collection_exists(collection_name=collection_name):
        client.delete_collection(collection_name=collection_name)
    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
    )


def ensure_collection(client: QdrantClient, collection_name: str, vector_size: int, recreate: bool) -> None:
    if recreate or not client.collection_exists(collection_name=collection_name):
        recreate_collection(client, collection_name, vector_size)


def upsert_records(
    client: QdrantClient,
    collection_name: str,
    records: Sequence[SequenceRecord],
    embeddings: np.ndarray,
    batch_size: int,
) -> list[StageMetrics]:
    metrics: list[StageMetrics] = []
    for start in range(0, len(records), batch_size):
        batch = records[start : start + batch_size]
        batch_embeddings = embeddings[start : start + batch_size]
        batch_done = timed_stage("qdrant_upsert_batch", item_count=len(batch))
        points = []
        for offset, (record, embedding) in enumerate(zip(batch, batch_embeddings), start=start):
            points.append(
                PointStruct(
                    id=offset,
                    vector=embedding.tolist(),
                    payload={
                        "seq_id": record.seq_id,
                        "description": record.description,
                        "sequence": record.sequence,
                        "source_path": record.source_path,
                    },
                )
            )
        client.upsert(collection_name=collection_name, points=points)
        metrics.append(batch_done())
    return metrics