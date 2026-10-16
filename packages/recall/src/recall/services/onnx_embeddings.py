"""ONNX Runtime embedding backend for cross-platform CPU inference.

Uses ONNX Runtime with CPUExecutionProvider for generating text embeddings
from pre-exported ONNX models on HuggingFace Hub.
"""

from __future__ import annotations

import importlib.util
import json
import logging
import os
from pathlib import Path
from typing import Any

from recall.core.telemetry import Span, Stage, backend_loaded, embedding_batch, measured

logger = logging.getLogger("recall.onnx_embeddings")

_SUPPORTED_MODEL_TYPE = "bert"
_SUPPORTED_HIDDEN_SIZE = 384
# One inference run admits at most this many padded token cells: one input at
# the model's 512-token limit, or several shorter ones. Attention activations
# grow with rows x width^2, and on CPU, runs past this size cost more CPU per
# text, not less (measured on a 32-core Linux host, RECONCILIATION_QA.md).
_PADDED_TOKENS_PER_RUN_MAX = 512
# The daemon holds two ONNX sessions that can run at once: the embed phase and
# the query-model worker. Together they may take a quarter of the host's CPUs.
_CONCURRENT_SESSIONS = 2
_HOST_CPU_SHARE_DIVISOR = 4
_INTRA_OP_THREADS_MAX = 4


def _module_available(module_name: str) -> bool:
    try:
        return importlib.util.find_spec(module_name) is not None
    except ModuleNotFoundError:
        return False


def _available_cpus() -> int:
    if hasattr(os, "sched_getaffinity"):
        return len(os.sched_getaffinity(0))
    return os.cpu_count() or 1


def intra_op_threads(available_cpus: int) -> int:
    """Return the per-session intra-op thread count for a host with this many CPUs.

    ORT's default spawns one spinning thread per core in every session; on a
    shared host those threads contend with each other and with other work, so
    each embedded text costs several times the CPU of a small bounded pool.
    """
    if available_cpus <= 0:
        raise ValueError("available_cpus must be positive")
    share = available_cpus // (_HOST_CPU_SHARE_DIVISOR * _CONCURRENT_SESSIONS)
    return max(1, min(_INTRA_OP_THREADS_MAX, share))


def inference_runs(lengths: list[int], padded_tokens_max: int) -> list[list[int]]:
    """Group input positions into runs ordered by token length.

    Each run pads to its own longest input and holds at most
    ``padded_tokens_max`` padded cells, so short inputs never pad to the
    length of an unrelated long one.
    """
    if not lengths:
        return []
    if not 0 < max(lengths) <= padded_tokens_max:
        raise ValueError("embedding input exceeds the padded-token capacity")
    order = sorted(range(len(lengths)), key=lambda position: lengths[position])
    runs: list[list[int]] = []
    run: list[int] = []
    for position in order:
        # Lengths ascend, so this input sets the run's padded width.
        if run and (len(run) + 1) * lengths[position] > padded_tokens_max:
            runs.append(run)
            run = []
        run.append(position)
    runs.append(run)
    return runs


class ONNXBackend:
    """Embedding backend using ONNX Runtime for cross-platform CPU inference."""

    @classmethod
    def is_available(cls) -> bool:
        """Return whether this host can run the ONNX backend."""
        return (
            _module_available("onnxruntime")
            and _module_available("tokenizers")
            and _module_available("huggingface_hub")
        )

    def __init__(self, model_id: str = "BAAI/bge-small-en-v1.5", max_length: int = 512) -> None:
        if not self.is_available():
            raise RuntimeError(
                "ONNX Runtime is not installed. Install with: uv pip install 'recall[onnx]'"
            )
        if not 0 < max_length <= _PADDED_TOKENS_PER_RUN_MAX:
            raise ValueError(
                f"max_length must be between 1 and {_PADDED_TOKENS_PER_RUN_MAX} tokens"
            )
        self._model_id = model_id
        self._max_length = max_length
        self._session: Any = None
        self._tokenizer: Any = None
        self._config: dict[str, Any] = {}

    @property
    def dimensions(self) -> int:
        if self._config:
            return self._config.get("hidden_size", 384)
        return 384

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def query_prefix(self) -> str:
        if "bge" in self._model_id.lower():
            return "Represent this sentence for searching relevant passages: "
        return ""

    def _ensure_loaded(self) -> None:
        if self._session is not None:
            return
        self._load_model()

    @measured(Stage.MODEL_LOAD)
    def _load_model(self) -> None:
        import onnxruntime as ort
        from huggingface_hub import snapshot_download
        from tokenizers import Tokenizer

        try:
            model_path = Path(snapshot_download(self._model_id, local_files_only=True))
        except Exception:
            try:
                model_path = Path(snapshot_download(self._model_id))
            except Exception as download_err:
                raise RuntimeError(
                    f"failed to download model {self._model_id}. "
                    "Only public HuggingFace models are supported."
                ) from download_err

        config_path = model_path / "config.json"
        with open(config_path, encoding="utf-8") as f:
            self._config = json.load(f)

        model_type = str(self._config.get("model_type", "")).strip().lower()
        hidden_size = int(self._config.get("hidden_size", 0))
        if model_type != _SUPPORTED_MODEL_TYPE:
            raise RuntimeError(
                f"unsupported ONNX embedding model_type: {model_type or '<missing>'}. "
                f"Expected {_SUPPORTED_MODEL_TYPE}."
            )
        if hidden_size != _SUPPORTED_HIDDEN_SIZE:
            raise RuntimeError(
                f"unsupported embedding dimension: {hidden_size}. "
                f"Expected {_SUPPORTED_HIDDEN_SIZE} to match the current schema."
            )

        # Find ONNX model file: prefer onnx/model.onnx, fall back to model.onnx
        onnx_path = model_path / "onnx" / "model.onnx"
        if not onnx_path.exists():
            onnx_path = model_path / "model.onnx"
        if not onnx_path.exists():
            raise FileNotFoundError(
                f"No ONNX model found in {model_path}. Expected onnx/model.onnx or model.onnx."
            )

        options = ort.SessionOptions()
        options.intra_op_num_threads = intra_op_threads(_available_cpus())
        options.inter_op_num_threads = 1
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        # Spinning keeps idle pool threads busy-waiting between operators,
        # which is CPU taken from the rest of the host for no throughput.
        options.add_session_config_entry("session.intra_op.allow_spinning", "0")
        options.add_session_config_entry("session.inter_op.allow_spinning", "0")
        self._session = ort.InferenceSession(
            str(onnx_path),
            sess_options=options,
            providers=["CPUExecutionProvider"],
        )

        self._tokenizer = Tokenizer.from_pretrained(self._model_id)
        self._tokenizer.enable_truncation(max_length=self._max_length)
        self._tokenizer.no_padding()

        backend_loaded(
            backend="onnx",
            device=",".join(self._session.get_providers()),
            model=self._model_id,
            runtime=ort.__version__,
        )
        logger.info("loaded ONNX model %s (%d dims)", self._model_id, self.dimensions)

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        self._ensure_loaded()
        import numpy as np

        with Span(Stage.TOKENIZE):
            encodings = self._tokenizer.encode_batch(texts)
        lengths = [len(encoding.ids) for encoding in encodings]
        vectors = np.empty((len(texts), self.dimensions), dtype=np.float32)
        for run in inference_runs(lengths, _PADDED_TOKENS_PER_RUN_MAX):
            vectors[run] = self._embed_run([encodings[position] for position in run])
        return vectors.tolist()

    def _embed_run(self, encodings: list[Any]) -> Any:
        import numpy as np

        width = max(len(encoding.ids) for encoding in encodings)
        pad_id = self._config.get("pad_token_id", 0)
        input_ids = np.full((len(encodings), width), pad_id, dtype=np.int64)
        attention_mask = np.zeros((len(encodings), width), dtype=np.int64)
        for row, encoding in enumerate(encodings):
            input_ids[row, : len(encoding.ids)] = encoding.ids
            attention_mask[row, : len(encoding.ids)] = 1
        embedding_batch(attention_mask.tolist())
        token_type_ids = np.zeros_like(input_ids)

        with Span(Stage.INFERENCE):
            outputs = self._session.run(
                None,
                {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "token_type_ids": token_type_ids,
                },
            )

        # last_hidden_state is the first output; pool the CLS token and L2 normalize.
        cls_embeddings = outputs[0][:, 0, :]
        norms = np.maximum(np.linalg.norm(cls_embeddings, axis=-1, keepdims=True), 1e-12)
        return cls_embeddings / norms
