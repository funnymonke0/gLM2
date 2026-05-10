from pathlib import Path
from typing import Sequence

from Bio import SeqIO

from gaia_local.constants import FASTA_SUFFIXES
from gaia_local.types import SequenceRecord


def resolve_fasta_inputs(paths: Sequence[Path]) -> list[Path]:
    resolved: list[Path] = []
    for raw_path in paths:
        path = raw_path.resolve()
        if path.is_dir():
            for child in sorted(path.rglob("*")):
                if child.is_file() and child.suffix.lower() in FASTA_SUFFIXES:
                    resolved.append(child)
            continue
        if path.is_file() and path.suffix.lower() in FASTA_SUFFIXES:
            resolved.append(path)
    unique_paths: list[Path] = []
    seen = set()
    for path in resolved:
        if path not in seen:
            unique_paths.append(path)
            seen.add(path)
    if not unique_paths:
        raise FileNotFoundError("No FASTA files were found in the provided corpus/query paths.")
    return unique_paths


def load_fasta_records(paths: Sequence[Path], excluded_ids: set[str] | None = None) -> list[SequenceRecord]:
    records: list[SequenceRecord] = []
    excluded = excluded_ids or set()
    for fasta_path in resolve_fasta_inputs(paths):
        for record in SeqIO.parse(str(fasta_path), "fasta"):
            seq_id = record.id.strip()
            if seq_id in excluded:
                continue
            sequence = str(record.seq).replace("*", "").replace(" ", "").replace("\n", "").upper()
            if not sequence:
                continue
            records.append(
                SequenceRecord(
                    seq_id=seq_id,
                    description=record.description.strip(),
                    sequence=sequence,
                    source_path=str(fasta_path),
                )
            )
    if not records:
        raise ValueError("No FASTA records were loaded after filtering.")
    return records