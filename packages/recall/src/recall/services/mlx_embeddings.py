"""MLX embedding backend for Apple Silicon Macs.

Uses Metal GPU acceleration via Apple's MLX framework.
Implements BERT-family model forward pass for generating text embeddings.
"""

from __future__ import annotations

import importlib.util
import json
import logging
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any, cast

logger = logging.getLogger("recall.mlx_embeddings")

_mlx_available: bool | None = None
_SUPPORTED_MODEL_TYPE = "bert"


def _module_available(module_name: str) -> bool:
    try:
        return importlib.util.find_spec(module_name) is not None
    except ModuleNotFoundError:
        return False


def _probe_mlx_import() -> bool:
    """Import MLX in a subprocess so broken native wheels cannot abort this process."""
    try:
        result = subprocess.run(
            [sys.executable, "-c", "import mlx.core, mlx.nn"],
            capture_output=True,
            check=False,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        logger.debug("MLX subprocess probe failed", exc_info=True)
        return False
    if result.returncode != 0:
        logger.debug("MLX subprocess probe returned %s", result.returncode)
        return False
    return True


def _check_mlx_available() -> bool:
    global _mlx_available
    if _mlx_available is None:
        _mlx_available = _probe_mlx_import()
    return _mlx_available


def _clear_mlx_probe_cache() -> None:
    """Reset cached MLX probe state for tests."""
    global _mlx_available
    _mlx_available = None


# ---- BERT Model in MLX ----
#
# Weight name mapping from HuggingFace → MLX module structure:
#   bert.                          → (removed)
#   encoder.layer.N                → encoder.layers.N
#   attention.self.                → attention.self_attn.
#   LayerNorm                      → layer_norm
#   pooler.*, cls.*                → (skipped)


def _build_bert_model(config: dict[str, Any]) -> Any:
    """Build BERT model. All MLX imports are local to avoid requiring MLX at module level."""
    import mlx.core as mx
    import mlx.nn as nn

    class BertEmbeddings(nn.Module):
        def __init__(self, cfg: dict[str, Any]) -> None:
            super().__init__()
            self.word_embeddings = nn.Embedding(cfg["vocab_size"], cfg["hidden_size"])
            self.position_embeddings = nn.Embedding(
                cfg["max_position_embeddings"], cfg["hidden_size"]
            )
            self.token_type_embeddings = nn.Embedding(
                cfg.get("type_vocab_size", 2), cfg["hidden_size"]
            )
            self.layer_norm = nn.LayerNorm(cfg["hidden_size"], eps=cfg.get("layer_norm_eps", 1e-12))

        def __call__(self, input_ids: mx.array, token_type_ids: mx.array | None = None) -> mx.array:
            seq_len = input_ids.shape[1]
            position_ids = mx.arange(seq_len)
            if token_type_ids is None:
                token_type_ids = mx.zeros_like(input_ids)
            return self.layer_norm(
                self.word_embeddings(input_ids)
                + self.position_embeddings(position_ids)
                + self.token_type_embeddings(token_type_ids)
            )

    class BertSelfAttention(nn.Module):
        def __init__(self, cfg: dict[str, Any]) -> None:
            super().__init__()
            self.num_heads = cfg["num_attention_heads"]
            self.head_dim = cfg["hidden_size"] // self.num_heads
            self.query = nn.Linear(cfg["hidden_size"], cfg["hidden_size"])
            self.key = nn.Linear(cfg["hidden_size"], cfg["hidden_size"])
            self.value = nn.Linear(cfg["hidden_size"], cfg["hidden_size"])

        def __call__(self, x: mx.array, mask: mx.array | None = None) -> mx.array:
            B, L, _ = x.shape
            q = self.query(x).reshape(B, L, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
            k = self.key(x).reshape(B, L, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
            v = self.value(x).reshape(B, L, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)

            # Fused Metal kernel — eliminates intermediate allocations
            # from manual q @ k.T -> softmax -> @ v computation
            scale = self.head_dim**-0.5
            out = mx.fast.scaled_dot_product_attention(q, k, v, scale=scale, mask=mask)
            return out.transpose(0, 2, 1, 3).reshape(B, L, -1)

    class BertAttentionOutput(nn.Module):
        def __init__(self, cfg: dict[str, Any]) -> None:
            super().__init__()
            self.dense = nn.Linear(cfg["hidden_size"], cfg["hidden_size"])
            self.layer_norm = nn.LayerNorm(cfg["hidden_size"], eps=cfg.get("layer_norm_eps", 1e-12))

        def __call__(self, hidden: mx.array, residual: mx.array) -> mx.array:
            return self.layer_norm(self.dense(hidden) + residual)

    class BertAttention(nn.Module):
        def __init__(self, cfg: dict[str, Any]) -> None:
            super().__init__()
            self.self_attn = BertSelfAttention(cfg)
            self.output = BertAttentionOutput(cfg)

        def __call__(self, x: mx.array, mask: mx.array | None = None) -> mx.array:
            return self.output(self.self_attn(x, mask), x)

    class BertIntermediate(nn.Module):
        def __init__(self, cfg: dict[str, Any]) -> None:
            super().__init__()
            self.dense = nn.Linear(cfg["hidden_size"], cfg["intermediate_size"])

        def __call__(self, x: mx.array) -> mx.array:
            return nn.gelu(self.dense(x))

    class BertOutputLayer(nn.Module):
        def __init__(self, cfg: dict[str, Any]) -> None:
            super().__init__()
            self.dense = nn.Linear(cfg["intermediate_size"], cfg["hidden_size"])
            self.layer_norm = nn.LayerNorm(cfg["hidden_size"], eps=cfg.get("layer_norm_eps", 1e-12))

        def __call__(self, hidden: mx.array, residual: mx.array) -> mx.array:
            return self.layer_norm(self.dense(hidden) + residual)

    class BertLayer(nn.Module):
        def __init__(self, cfg: dict[str, Any]) -> None:
            super().__init__()
            self.attention = BertAttention(cfg)
            self.intermediate = BertIntermediate(cfg)
            self.output = BertOutputLayer(cfg)

        def __call__(self, x: mx.array, mask: mx.array | None = None) -> mx.array:
            attn_out = self.attention(x, mask)
            return self.output(self.intermediate(attn_out), attn_out)

    class BertEncoder(nn.Module):
        def __init__(self, cfg: dict[str, Any]) -> None:
            super().__init__()
            self.layers = [BertLayer(cfg) for _ in range(cfg["num_hidden_layers"])]

        def __call__(self, x: mx.array, mask: mx.array | None = None) -> mx.array:
            for layer in self.layers:
                x = layer(x, mask)
            return x

    class _BertModel(nn.Module):
        def __init__(self, cfg: dict[str, Any]) -> None:
            super().__init__()
            self.embeddings = BertEmbeddings(cfg)
            self.encoder = BertEncoder(cfg)

        def __call__(
            self,
            input_ids: mx.array,
            attention_mask: mx.array | None = None,
            token_type_ids: mx.array | None = None,
        ) -> mx.array:
            x = self.embeddings(input_ids, token_type_ids)
            mask = None
            if attention_mask is not None:
                # 1 = attend, 0 = ignore -> additive bias: 1 -> 0.0, 0 -> -1e9.
                # Mask dtype must match model weights (float16) to avoid
                # implicit promotion penalty in attention score computation.
                mask = (1.0 - attention_mask[:, None, None, :].astype(mx.float16)) * mx.array(
                    -1e4, dtype=mx.float16
                )
            return self.encoder(x, mask)

    return _BertModel(config)


def _map_weight_name(hf_name: str) -> str:
    """Map HuggingFace BERT weight name to MLX module structure."""
    name = hf_name
    if name.startswith("bert."):
        name = name[5:]
    if name.startswith("pooler.") or name.startswith("cls."):
        return ""
    # Skip non-parameter buffers
    if name.endswith("position_ids"):
        return ""
    name = name.replace("encoder.layer.", "encoder.layers.")
    name = name.replace("attention.self.", "attention.self_attn.")
    name = name.replace("LayerNorm", "layer_norm")
    return name


class MLXBackend:
    """Embedding backend using Apple MLX for Metal GPU acceleration on Apple Silicon."""

    @classmethod
    def is_available(cls) -> bool:
        """Return whether this host can run the MLX backend."""
        return (
            sys.platform == "darwin"
            and platform.machine().lower() in {"arm64", "aarch64"}
            and _check_mlx_available()
            and _module_available("tokenizers")
            and _module_available("huggingface_hub")
        )

    def __init__(self, model_id: str = "BAAI/bge-small-en-v1.5", max_length: int = 512) -> None:
        if not _check_mlx_available():
            raise RuntimeError("MLX is not installed. Install with: uv pip install 'recall[mlx]'")
        self._model_id = model_id
        self._max_length = max_length
        self._model: Any = None
        self._tokenizer: Any = None
        self._config: dict[str, Any] = {}

    @property
    def dimensions(self) -> int:
        if self._config:
            return int(self._config.get("hidden_size", 0))
        from recall.core.config import KNOWN_MODEL_DIMENSIONS

        resolved = KNOWN_MODEL_DIMENSIONS.get(self._model_id)
        if resolved is not None:
            return resolved
        self._ensure_loaded()
        return int(self._config.get("hidden_size", 0))

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def query_prefix(self) -> str:
        if "bge" in self._model_id.lower():
            return "Represent this sentence for searching relevant passages: "
        return ""

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        self._load_model()

    def _load_model(self) -> None:
        import mlx.core as mx
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
        if model_type != _SUPPORTED_MODEL_TYPE:
            raise RuntimeError(
                f"unsupported MLX embedding model_type: {model_type or '<missing>'}. "
                f"Expected {_SUPPORTED_MODEL_TYPE}."
            )

        self._model = _build_bert_model(self._config)

        # Load safetensors weights
        weights_path = model_path / "model.safetensors"
        if not weights_path.exists():
            raise FileNotFoundError(
                f"No model.safetensors found in {model_path}. Only safetensors format is supported."
            )

        raw_weights = cast("dict[str, Any]", mx.load(str(weights_path)))
        mapped_weights = []
        for hf_name, tensor in raw_weights.items():
            mlx_name = _map_weight_name(hf_name)
            if mlx_name:
                mapped_weights.append((mlx_name, tensor))

        self._model.load_weights(mapped_weights)
        mx.eval(self._model.parameters())

        # Cast to float16 — halves memory bandwidth, faster Metal inference.
        # Negligible quality loss for embedding similarity tasks.
        self._model.set_dtype(mx.float16)
        mx.eval(self._model.parameters())

        # Cap Metal cache to prevent unbounded GPU memory growth during
        # bulk indexing. Use 10% of system memory or 3x model active
        # memory, whichever is larger, with a 512MB floor.
        # Prefer top-level mx.get_active_memory / mx.set_cache_limit (MLX >=0.22),
        # fall back to mx.metal.* for older versions.
        _get_active = getattr(mx, "get_active_memory", None) or mx.metal.get_active_memory
        _set_cache = getattr(mx, "set_cache_limit", None) or mx.metal.set_cache_limit
        model_memory = _get_active()
        system_memory = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
        cache_limit = max(model_memory * 3, system_memory // 10, 512 * 1024 * 1024)
        _set_cache(cache_limit)
        logger.info(
            "MLX Metal cache limit set to %d MB",
            cache_limit // (1024 * 1024),
        )

        # Load tokenizer with padding and truncation
        self._tokenizer = Tokenizer.from_pretrained(self._model_id)
        self._tokenizer.enable_truncation(max_length=self._max_length)
        self._tokenizer.enable_padding(
            pad_id=self._config.get("pad_token_id", 0),
            pad_token="[PAD]",
        )

        # 4-bit quantization — faster matmul kernels on Metal with negligible
        # quality loss for embedding retrieval tasks. Must happen before
        # mx.compile since quantized layers have different ops.
        import mlx.nn as nn

        nn.quantize(self._model, bits=4, group_size=64)
        mx.eval(self._model.parameters())

        # JIT compile the forward pass — fuses element-wise ops across BERT
        # layers into optimized Metal kernels. First call per input shape is
        # slow (compilation); subsequent calls use the cached graph.
        self._compiled_forward = mx.compile(self._model)

        logger.info("loaded MLX model %s (%d dims)", self._model_id, self.dimensions)

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        self._ensure_loaded()
        import mlx.core as mx

        encodings = self._tokenizer.encode_batch(texts)
        input_ids = mx.array([e.ids for e in encodings])
        attention_mask = mx.array([e.attention_mask for e in encodings])

        hidden_states = self._compiled_forward(input_ids, attention_mask=attention_mask)

        # CLS token pooling (first token output)
        cls_embeddings = hidden_states[:, 0, :]

        # L2 normalize
        norms = mx.linalg.norm(cls_embeddings, axis=-1, keepdims=True)
        normalized = cls_embeddings / mx.maximum(norms, mx.array(1e-12))

        mx.eval(normalized)
        result = normalized.tolist()

        # Drop references so MLX LRU cache can reclaim Metal memory
        del input_ids, attention_mask, hidden_states, cls_embeddings, norms, normalized

        return result
