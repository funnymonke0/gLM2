import json
import time
from dataclasses import asdict
from pathlib import Path

import torch

from gaia_local.types import StageMetrics


def reset_cuda_peak() -> None:
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def cuda_peaks() -> tuple[int | None, int | None]:
    if not torch.cuda.is_available():
        return None, None
    return torch.cuda.max_memory_allocated(), torch.cuda.max_memory_reserved()


def timed_stage(stage: str, item_count: int | None = None):
    reset_cuda_peak()
    start = time.perf_counter()

    def _finish() -> StageMetrics:
        allocated, reserved = cuda_peaks()
        return StageMetrics(
            stage=stage,
            seconds=time.perf_counter() - start,
            cuda_peak_allocated_bytes=allocated,
            cuda_peak_reserved_bytes=reserved,
            items=item_count,
        )

    return _finish


def stage_metrics_dict(metrics: StageMetrics) -> dict:
    data = asdict(metrics)
    for key in ("cuda_peak_allocated_bytes", "cuda_peak_reserved_bytes"):
        value = data[key]
        data[key.replace("_bytes", "_mib")] = None if value is None else round(value / (1024 * 1024), 2)
    return data


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def append_jsonl(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload) + "\n")