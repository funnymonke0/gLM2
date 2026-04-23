from typing import Sequence

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

from gaia_local.auth import maybe_login_from_token_file
from gaia_local.metrics import append_jsonl, stage_metrics_dict, timed_stage
from gaia_local.types import SequenceRecord, StageMetrics


def format_for_glm2(sequence: str, strand_token: str = "<+>") -> str:
    cleaned = sequence.strip()
    if cleaned.startswith("<+>") or cleaned.startswith("<->"):
        return cleaned
    return f"{strand_token}{cleaned.upper()}"


def choose_device() -> tuple[torch.device, torch.dtype]:
    if torch.cuda.is_available():
        if torch.cuda.is_bf16_supported():
            return torch.device("cuda"), torch.bfloat16
        return torch.device("cuda"), torch.float16
    return torch.device("cpu"), torch.float32


class GLM2Embedder:
    def __init__(self, model_name: str, batch_size: int, max_seq_length: int):
        maybe_login_from_token_file()
        self.model_name = model_name
        self.batch_size = batch_size
        self.max_seq_length = max_seq_length
        self.device, self.dtype = choose_device()

        load_done = timed_stage("model_load")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(
            model_name,
            dtype=self.dtype,
            trust_remote_code=True,
        ).to(self.device)
        self.model.eval()
        self.load_metrics = load_done()

    def embed_records(self, records: Sequence[SequenceRecord], batch_log_path=None) -> tuple[np.ndarray, list[StageMetrics]]:
        embeddings: list[np.ndarray] = []
        metrics: list[StageMetrics] = []
        for batch_index, start in enumerate(range(0, len(records), self.batch_size), start=1):
            batch = records[start : start + self.batch_size]
            batch_done = timed_stage("embed_batch", item_count=len(batch))
            texts = [format_for_glm2(record.sequence) for record in batch]
            encodings = self.tokenizer(
                texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_seq_length,
            )
            encodings = {name: tensor.to(self.device) for name, tensor in encodings.items()}
            with torch.no_grad():
                outputs = self.model(**encodings)
                batch_embeddings = self.extract_embeddings(outputs, encodings["attention_mask"])
            embeddings.append(batch_embeddings.cpu().numpy())
            batch_metrics = batch_done()
            metrics.append(batch_metrics)
            if batch_log_path is not None:
                append_jsonl(
                    batch_log_path,
                    {
                        "batch_index": batch_index,
                        "record_ids": [record.seq_id for record in batch],
                        **stage_metrics_dict(batch_metrics),
                    },
                )
        stacked = np.concatenate(embeddings, axis=0) if embeddings else np.zeros((0, 0), dtype=np.float32)
        return stacked, metrics

    @staticmethod
    def extract_embeddings(outputs, attention_mask: torch.Tensor) -> torch.Tensor:
        if getattr(outputs, "pooler_output", None) is not None:
            return outputs.pooler_output.float()
        hidden = outputs.last_hidden_state.float()
        mask = attention_mask.unsqueeze(-1).float()
        summed = (hidden * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp_min(1.0)
        return summed / counts