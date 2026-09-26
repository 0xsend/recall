from __future__ import annotations

from unittest.mock import MagicMock

from recall.services.mlx_embeddings import MLXBackend, _clear_mlx_probe_cache


def test_is_available_uses_subprocess_probe_on_supported_hosts(monkeypatch) -> None:
    """MLX availability must probe imports out-of-process to avoid native aborts."""
    _clear_mlx_probe_cache()
    probe = MagicMock(return_value=True)

    monkeypatch.setattr("recall.services.mlx_embeddings.sys.platform", "darwin")
    monkeypatch.setattr("recall.services.mlx_embeddings.platform.machine", lambda: "arm64")
    monkeypatch.setattr("recall.services.mlx_embeddings._probe_mlx_import", probe)
    monkeypatch.setattr("recall.services.mlx_embeddings._module_available", lambda _name: True)

    assert MLXBackend.is_available() is True
    probe.assert_called_once_with()


def test_is_available_returns_false_when_subprocess_probe_fails(monkeypatch) -> None:
    """A crashing MLX import must degrade to unavailable instead of aborting pytest."""
    _clear_mlx_probe_cache()

    monkeypatch.setattr("recall.services.mlx_embeddings.sys.platform", "darwin")
    monkeypatch.setattr("recall.services.mlx_embeddings.platform.machine", lambda: "arm64")
    monkeypatch.setattr("recall.services.mlx_embeddings._probe_mlx_import", lambda: False)
    monkeypatch.setattr("recall.services.mlx_embeddings._module_available", lambda _name: True)

    assert MLXBackend.is_available() is False
