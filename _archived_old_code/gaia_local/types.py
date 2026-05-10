from dataclasses import dataclass


@dataclass
class SequenceRecord:
    seq_id: str
    description: str
    sequence: str
    source_path: str


@dataclass
class StageMetrics:
    stage: str
    seconds: float
    cuda_peak_allocated_bytes: int | None
    cuda_peak_reserved_bytes: int | None
    items: int | None = None