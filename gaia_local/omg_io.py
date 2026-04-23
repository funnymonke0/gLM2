from typing import Iterator

import datasets

from gaia_local.types import SequenceRecord


def _iter_omg_rows(split: str, streaming: bool) -> Iterator[dict]:
    ds = datasets.load_dataset("tattabio/OMG", split=split, streaming=streaming)
    for row in ds:
        yield row


def load_omg_records(split: str = "train", limit: int | None = None, streaming: bool = True) -> list[SequenceRecord]:
    records: list[SequenceRecord] = []
    for row_index, row in enumerate(_iter_omg_rows(split=split, streaming=streaming)):
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
            records.append(
                SequenceRecord(
                    seq_id=seq_id,
                    description=str(seq_id),
                    sequence=sequence,
                    source_path=f"hf://tattabio/OMG/{split}",
                )
            )

        if limit is not None and (row_index + 1) >= limit:
            break

    if not records:
        raise ValueError("No OMG records were loaded.")
    return records