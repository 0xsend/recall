"""Embedding backend protocol and registry types."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@runtime_checkable
class EmbeddingBackend(Protocol):
    """Pluggable embedding backend for generating text embeddings.

    Backends handle model loading, tokenization, and inference.
    Each backend targets a specific runtime (MLX, ONNX, remote API).
    """

    @property
    def dimensions(self) -> int:
        """Number of dimensions in the embedding vectors."""
        ...

    @property
    def model_id(self) -> str:
        """Identifier of the embedding model."""
        ...

    @property
    def query_prefix(self) -> str:
        """Prefix to prepend to query text for retrieval tasks.

        Empty string for models that don't use query/document distinction.
        BGE models use: "Represent this sentence for searching relevant passages: "
        """
        ...

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for a batch of texts.

        Texts are treated as documents (no query prefix).
        Caller is responsible for prepending query_prefix when embedding queries.

        Args:
            texts: List of text strings to embed.

        Returns:
            List of embedding vectors (list of floats), one per input text.
            Each vector has length equal to self.dimensions.
        """
        ...


@dataclass(frozen=True)
class BackendDescriptor:
    """Describes an embedding backend without importing it."""

    name: str
    is_available: Callable[[], bool]
    factory: Callable[[str], EmbeddingBackend]
    priority: int


_backend_probe: Callable[[str], bool] | None = None


def set_backend_probe(probe: Callable[[str], bool]) -> None:
    """Install probe function from services layer for backend availability checks."""
    global _backend_probe
    _backend_probe = probe


def embedding_backend_available(backend: str) -> bool:
    """Return whether the current host can run the requested embedding backend."""
    return _backend_probe(backend) if _backend_probe is not None else False
