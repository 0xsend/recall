"""Tests for the ONNX embedding backend.

These tests verify registration, configuration, and critical contract paths
without requiring onnxruntime to be installed. Model loading validation
paths are tested by writing config files to temp dirs and monkey-patching
the snapshot_download import inside _load_model().
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import pytest
from recall.services.embeddings import _REGISTRY

if TYPE_CHECKING:
    from recall.services.onnx_embeddings import ONNXBackend


def test_onnx_is_registered() -> None:
    assert "onnx" in _REGISTRY
    assert _REGISTRY["onnx"].priority == 20


def test_onnx_backend_query_prefix_bge() -> None:
    from recall.services.onnx_embeddings import ONNXBackend

    backend = ONNXBackend.__new__(ONNXBackend)
    backend._model_id = "BAAI/bge-small-en-v1.5"
    backend._max_length = 512
    backend._session = None
    backend._tokenizer = None
    backend._config = {}
    assert backend.query_prefix == "Represent this sentence for searching relevant passages: "


def test_onnx_backend_query_prefix_non_bge() -> None:
    from recall.services.onnx_embeddings import ONNXBackend

    backend = ONNXBackend.__new__(ONNXBackend)
    backend._model_id = "sentence-transformers/all-MiniLM-L6-v2"
    backend._max_length = 512
    backend._session = None
    backend._tokenizer = None
    backend._config = {}
    assert backend.query_prefix == ""


def test_onnx_backend_dimensions_default() -> None:
    from recall.services.onnx_embeddings import ONNXBackend

    backend = ONNXBackend.__new__(ONNXBackend)
    backend._model_id = "BAAI/bge-small-en-v1.5"
    backend._max_length = 512
    backend._session = None
    backend._tokenizer = None
    backend._config = {}
    assert backend.dimensions == 384

    backend._config = {"hidden_size": 384}
    assert backend.dimensions == 384


def test_onnx_backend_embed_empty_returns_empty() -> None:
    from recall.services.onnx_embeddings import ONNXBackend

    backend = ONNXBackend.__new__(ONNXBackend)
    backend._model_id = "BAAI/bge-small-en-v1.5"
    backend._max_length = 512
    backend._session = None
    backend._tokenizer = None
    backend._config = {}
    assert backend.embed([]) == []


# ---- ISSUE-1: fail-fast on missing deps ----


def test_onnx_init_raises_when_deps_missing() -> None:
    from recall.services.onnx_embeddings import ONNXBackend

    with (
        patch.object(ONNXBackend, "is_available", return_value=False),
        pytest.raises(RuntimeError, match="ONNX Runtime is not installed"),
    ):
        ONNXBackend()


# ---- is_available() probe tests ----


@pytest.mark.parametrize(
    ("missing", "expected"),
    [(None, True), ("onnxruntime", False), ("tokenizers", False), ("huggingface_hub", False)],
)
def test_is_available_requires_every_runtime_dependency(
    missing: str | None, expected: bool
) -> None:
    from recall.services.onnx_embeddings import ONNXBackend

    with patch(
        "recall.services.onnx_embeddings._module_available",
        side_effect=lambda name: name != missing,
    ):
        assert ONNXBackend.is_available() is expected


# ---- _load_model() validation tests ----


def _make_backend() -> ONNXBackend:
    """Create an ONNXBackend instance bypassing __init__ for test isolation."""
    from recall.services.onnx_embeddings import ONNXBackend

    backend = ONNXBackend.__new__(ONNXBackend)
    backend._model_id = "test-model"
    backend._max_length = 512
    backend._session = None
    backend._tokenizer = None
    backend._config = {}
    return backend


def _write_config(model_dir: Path, config: dict) -> None:
    (model_dir / "config.json").write_text(json.dumps(config))


def test_load_model_rejects_non_bert_model_type(tmp_path: Path) -> None:
    """_load_model() must reject model_type != 'bert'."""
    backend = _make_backend()
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    _write_config(model_dir, {"model_type": "gpt2", "hidden_size": 384})

    # Mock the imports that happen inside _load_model()
    mock_hf = MagicMock()
    mock_hf.snapshot_download.return_value = str(model_dir)
    mock_ort = MagicMock()
    mock_tokenizers = MagicMock()

    modules = {"huggingface_hub": mock_hf, "onnxruntime": mock_ort, "tokenizers": mock_tokenizers}
    with (
        patch.dict(sys.modules, modules),
        pytest.raises(RuntimeError, match="unsupported ONNX embedding model_type: gpt2"),
    ):
        backend._load_model()


def test_load_model_rejects_wrong_hidden_size(tmp_path: Path) -> None:
    """_load_model() must reject hidden_size != 384."""
    backend = _make_backend()
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    _write_config(model_dir, {"model_type": "bert", "hidden_size": 768})

    mock_hf = MagicMock()
    mock_hf.snapshot_download.return_value = str(model_dir)
    mock_ort = MagicMock()
    mock_tokenizers = MagicMock()

    modules = {"huggingface_hub": mock_hf, "onnxruntime": mock_ort, "tokenizers": mock_tokenizers}
    with (
        patch.dict(sys.modules, modules),
        pytest.raises(RuntimeError, match="unsupported embedding dimension: 768"),
    ):
        backend._load_model()


def test_load_model_rejects_missing_onnx_file(tmp_path: Path) -> None:
    """_load_model() must raise FileNotFoundError when no .onnx file exists."""
    backend = _make_backend()
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    _write_config(model_dir, {"model_type": "bert", "hidden_size": 384})

    mock_hf = MagicMock()
    mock_hf.snapshot_download.return_value = str(model_dir)
    mock_ort = MagicMock()
    mock_tokenizers = MagicMock()

    modules = {"huggingface_hub": mock_hf, "onnxruntime": mock_ort, "tokenizers": mock_tokenizers}
    with (
        patch.dict(sys.modules, modules),
        pytest.raises(FileNotFoundError, match="No ONNX model found"),
    ):
        backend._load_model()


# ---- CLS pooling + L2 normalization test ----


def test_embed_applies_cls_pooling_and_l2_normalization() -> None:
    """embed() must select CLS token (index 0) and L2-normalize the output."""
    np = pytest.importorskip("numpy")
    from recall.services.onnx_embeddings import ONNXBackend

    backend = ONNXBackend.__new__(ONNXBackend)
    backend._model_id = "BAAI/bge-small-en-v1.5"
    backend._max_length = 512
    backend._config = {"hidden_size": 384}

    # Mock tokenizer
    mock_encoding = MagicMock()
    mock_encoding.ids = [101, 7592, 102]
    mock_encoding.attention_mask = [1, 1, 1]
    mock_tokenizer = MagicMock()
    mock_tokenizer.encode_batch.return_value = [mock_encoding]
    backend._tokenizer = mock_tokenizer

    # CLS vector = [3, 0, 0, ...], other tokens = [999, ...]
    cls_vector = np.array([3.0] + [0.0] * 383, dtype=np.float32)
    other_vector = np.array([999.0] * 384, dtype=np.float32)
    # Shape: (batch=1, seq_len=3, hidden=384)
    hidden_state = np.stack(
        [
            np.stack([cls_vector, other_vector, other_vector]),
        ]
    )

    mock_session = MagicMock()
    mock_session.run.return_value = [hidden_state]
    backend._session = mock_session

    result = backend.embed(["hello"])

    assert len(result) == 1
    embedding = result[0]
    assert len(embedding) == 384

    # CLS pooling: should use index 0 (cls_vector), not other tokens
    # L2 norm of [3, 0, 0, ...] = 3, so normalized = [1, 0, 0, ...]
    assert math.isclose(embedding[0], 1.0, rel_tol=1e-6)
    assert math.isclose(embedding[1], 0.0, abs_tol=1e-6)

    # Verify L2 norm of output is ~1.0
    norm = math.sqrt(sum(x * x for x in embedding))
    assert math.isclose(norm, 1.0, rel_tol=1e-6)
