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
        print("CUDA is available. Using GPU for embedding.")
        if torch.cuda.is_bf16_supported():
            print("Using bfloat16 precision.")
            return torch.device("cuda"), torch.bfloat16
        print("Using float16 precision.")
        return torch.device("cuda"), torch.float16
    print("CUDA is not available. Using CPU for embedding.")
    return torch.device("cpu"), torch.float32


class GLM2Embedder:
    def __init__(self, model_name: str, batch_size: int, max_seq_length: int, auto_batch_size: bool = True):
        maybe_login_from_token_file()
        self.model_name = model_name
        self.batch_size = batch_size
        self.max_seq_length = max_seq_length
        self.auto_batch_size = auto_batch_size
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

    @staticmethod
    def _is_oom_error(exc: BaseException) -> bool:
        message = str(exc).lower()
        return "out of memory" in message or "cuda error: out of memory" in message

    def _clear_cuda_cache(self) -> None:
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

    def embed_batch(
        self,
        records: Sequence[SequenceRecord],
        batch_log_path=None,
        batch_index: int | None = None,
    ) -> tuple[np.ndarray, StageMetrics]:
        batch_done = timed_stage("embed_batch", item_count=len(records))
        texts = [format_for_glm2(record.sequence) for record in records]
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

        batch_metrics = batch_done()
        if batch_log_path is not None:
            payload = {
                "record_ids": [record.seq_id for record in records],
                **stage_metrics_dict(batch_metrics),
            }
            if batch_index is not None:
                payload["batch_index"] = batch_index
            append_jsonl(batch_log_path, payload)
        return batch_embeddings.cpu().numpy(), batch_metrics

    def embed_records(self, records: Sequence[SequenceRecord], batch_log_path=None) -> tuple[np.ndarray, list[StageMetrics]]:
        embeddings: list[np.ndarray] = []
        metrics: list[StageMetrics] = []
        effective_batch_size = self.batch_size
        batch_index = 1
        position = 0
        while position < len(records):
            batch = records[position : position + effective_batch_size]
            try:
                batch_embeddings, batch_metrics = self.embed_batch(
                    batch,
                    batch_log_path=batch_log_path,
                    batch_index=batch_index,
                )
            except RuntimeError as exc:
                if not self.auto_batch_size or not self._is_oom_error(exc) or effective_batch_size <= 1:
                    raise
                effective_batch_size = max(1, effective_batch_size // 2)
                print(f"OOM at batch size {len(batch)}. Reducing to {effective_batch_size} and retrying.")
                self._clear_cuda_cache()
                continue
            embeddings.append(batch_embeddings)
            metrics.append(batch_metrics)
            position += len(batch)
            batch_index += 1

        if self.auto_batch_size and effective_batch_size < self.batch_size:
            self.batch_size = effective_batch_size
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