
from typing import Iterator

import datasets

from gaia_local.constants import DATASET
from gaia_local.types import SequenceRecord


def _iter_omg_rows(split: str, streaming: bool) -> Iterator[dict]:
    ds = datasets.load_dataset(
        DATASET,
        split=split,
        streaming=streaming,
    )
    for row in ds:
        if isinstance(row, dict):
            yield row
        elif hasattr(row, "items"):
            yield dict(row)


def iter_stream_records(split: str = "train", limit: int | None = None, streaming: bool = True) -> Iterator[SequenceRecord]:
    produced = False
    yielded_records = 0
    for row_index, row in enumerate(_iter_omg_rows(split=split, streaming=streaming)):
        # OG_prot90-style rows already contain a single protein per example.
        if row.get("sequence"):
            seq_id = str(row.get("id") or f"{split}_row{row_index}")
            sequence = str(row["sequence"]).strip().upper()
            if sequence:
                produced = True
                yield SequenceRecord(
                    seq_id=seq_id,
                    description=seq_id,
                    sequence=sequence,
                    source_path=f"hf://{DATASET}/{split}",
                )
                yielded_records += 1
                if limit is not None and yielded_records >= limit:
                    break
            continue

        # Original OMG-style rows contain many CDS sequences per scaffold row.
        cds_seqs = row.get("CDS_seqs") or []
        cds_ids = row.get("CDS_ids") or []
        cds_orientations = row.get("CDS_orientations") or []

        for cds_index, seq in enumerate(cds_seqs):
            if not seq:
                continue
            orient = cds_orientations[cds_index] if cds_index < len(cds_orientations) else True
            orientation_token = "<+>" if orient else "<->"
            seq_id = cds_ids[cds_index] if cds_index < len(cds_ids) and cds_ids[cds_index] else f"{split}_row{row_index}_cds{cds_index}"
            sequence = f"{orientation_token}{str(seq).upper()}"
            produced = True
            yield SequenceRecord(
                seq_id=seq_id,
                description=str(seq_id),
                sequence=sequence,
                source_path=f"hf://{DATASET}/{split}",
            )
            yielded_records += 1

            if limit is not None and yielded_records >= limit:
                break

        if limit is not None and yielded_records >= limit:
            break

    if not produced:
        raise ValueError(f"No stream records were loaded for split '{split}'.")


def iter_stream_record_batches(
    split: str = "train",
    limit: int | None = None,
    streaming: bool = True,
    batch_size: int = 8,
) -> Iterator[list[SequenceRecord]]:
    if batch_size <= 0:
        raise ValueError("batch_size must be greater than zero.")

    batch: list[SequenceRecord] = []
    for record in iter_stream_records(split=split, limit=limit, streaming=streaming):
        batch.append(record)
        if len(batch) >= batch_size:
            yield batch
            batch = []

    if batch:
        yield batch


def load_stream_records(split: str = "train", limit: int | None = None, streaming: bool = True) -> list[SequenceRecord]:
    return list(iter_stream_records(split=split, limit=limit, streaming=streaming))


# Backward-compatible aliases
def iter_omg_records(split: str = "train", limit: int | None = None, streaming: bool = True) -> Iterator[SequenceRecord]:
    return iter_stream_records(split=split, limit=limit, streaming=streaming)


def iter_omg_record_batches(
    split: str = "train",
    limit: int | None = None,
    streaming: bool = True,
    batch_size: int = 8,
) -> Iterator[list[SequenceRecord]]:
    return iter_stream_record_batches(split=split, limit=limit, streaming=streaming, batch_size=batch_size)


def load_omg_records(split: str = "train", limit: int | None = None, streaming: bool = True) -> list[SequenceRecord]:
    return load_stream_records(split=split, limit=limit, streaming=streaming)