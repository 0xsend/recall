"""ONNX Runtime embedding backend for cross-platform CPU inference.

Uses ONNX Runtime with CPUExecutionProvider for generating text embeddings
from pre-exported ONNX models on HuggingFace Hub.
"""

from __future__ import annotations

import importlib.util
import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger("recall.onnx_embeddings")

_SUPPORTED_MODEL_TYPE = "bert"
_SUPPORTED_HIDDEN_SIZE = 384


def _module_available(module_name: str) -> bool:
    try:
        return importlib.util.find_spec(module_name) is not None
    except ModuleNotFoundError:
        return False


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

        self._session = ort.InferenceSession(
            str(onnx_path),
            providers=["CPUExecutionProvider"],
        )

        self._tokenizer = Tokenizer.from_pretrained(self._model_id)
        self._tokenizer.enable_truncation(max_length=self._max_length)
        self._tokenizer.enable_padding(
            pad_id=self._config.get("pad_token_id", 0),
            pad_token="[PAD]",
        )

        logger.info("loaded ONNX model %s (%d dims)", self._model_id, self.dimensions)

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        self._ensure_loaded()
        import numpy as np

        encodings = self._tokenizer.encode_batch(texts)
        input_ids = np.array([e.ids for e in encodings], dtype=np.int64)
        attention_mask = np.array([e.attention_mask for e in encodings], dtype=np.int64)
        token_type_ids = np.zeros_like(input_ids, dtype=np.int64)

        outputs = self._session.run(
            None,
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "token_type_ids": token_type_ids,
            },
        )

        # last_hidden_state is the first output
        last_hidden_state = outputs[0]

        # CLS token pooling (first token output)
        cls_embeddings = last_hidden_state[:, 0, :]

        # L2 normalize
        norms = np.linalg.norm(cls_embeddings, axis=-1, keepdims=True)
        norms = np.maximum(norms, 1e-12)
        normalized = cls_embeddings / norms

        return normalized.tolist()
