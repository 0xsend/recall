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


def test_real_mlx_cache_budget_and_diagnostics(tmp_path) -> None:
    """A cached real model uses the process budget and emits allocator counters."""
    import json
    import os
    import subprocess
    import sys

    import pytest

    _clear_mlx_probe_cache()
    if sys.platform != "darwin" or not MLXBackend.is_available():
        pytest.skip("requires Apple Silicon with MLX")
    script = """
import json
import sys
from pathlib import Path

import mlx.core as mx
from huggingface_hub import snapshot_download
from huggingface_hub.errors import LocalEntryNotFoundError
from recall.services.diagnostics import Diagnostics
from recall.core.telemetry import current
from recall.services.mlx_embeddings import MLXBackend

try:
    snapshot_download('BAAI/bge-small-en-v1.5', local_files_only=True)
except LocalEntryNotFoundError:
    sys.exit(77)

diagnostics = Diagnostics(
    Path(sys.argv[1]), requested_backend='mlx', configured_model='BAAI/bge-small-en-v1.5'
)
token = current.set(diagnostics.recorder)
try:
    backend = MLXBackend()
    vectors = backend.embed(['memory pressure in daemon', 'how to rebase a branch'])
    set_cache_limit = getattr(mx, 'set_cache_limit', None) or mx.metal.set_cache_limit
    previous_limit = set_cache_limit(0)
    set_cache_limit(previous_limit)
    diagnostics.write()
    print(json.dumps({'limit': previous_limit, 'vectors': vectors}))
finally:
    current.reset(token)
"""
    path = tmp_path / "diagnostics.jsonl"
    result = subprocess.run(
        [sys.executable, "-c", script, str(path)],
        env={**os.environ, "HF_HUB_OFFLINE": "1"},
        capture_output=True,
        text=True,
        timeout=60,
    )
    if result.returncode == 77:
        pytest.skip("requires cached BGE-small model")
    assert result.returncode == 0, result.stderr
    body = json.loads(result.stdout)
    # 4 GiB total less 2 GB DuckDB: one eighth for reusable Metal buffers.
    assert body["limit"] == 286_870_912
    assert len(body["vectors"]) == 2
    for vector in body["vectors"]:
        assert len(vector) == 384
        assert sum(value * value for value in vector) == pytest.approx(1, abs=0.002)
    sample = json.loads(path.read_text())
    backend = sample["telemetry"]["backend"]
    assert backend["name"] == "mlx"
    assert backend["cache_limit_bytes"] == 286_870_912
    assert backend["active_bytes"] > 0
    assert backend["cache_bytes"] >= 0
    assert backend["peak_bytes"] >= backend["active_bytes"]
    assert backend["memory_sample_at"] > 0
    assert sample["rss_bytes"] > 0


def test_real_mlx_long_batches_fit_activation_budget(tmp_path) -> None:
    """BGE-small inference leaves process headroom even for fully padded batches."""
    import json
    import os
    import subprocess
    import sys

    import pytest

    _clear_mlx_probe_cache()
    if sys.platform != "darwin" or not MLXBackend.is_available():
        pytest.skip("requires Apple Silicon with MLX")
    script = """
import json
import sys
import mlx.core as mx
from huggingface_hub import snapshot_download
from huggingface_hub.errors import LocalEntryNotFoundError
from recall.services.mlx_embeddings import MLXBackend

try:
    snapshot_download('BAAI/bge-small-en-v1.5', local_files_only=True)
except LocalEntryNotFoundError:
    sys.exit(77)
backend = MLXBackend()
texts = [(f'database performance query {i} ' * 256) for i in range(33)]
vectors = backend.embed(texts)
peak = mx.get_peak_memory()
references = [backend.embed([texts[i]])[0] for i in (0, 32)]
print(json.dumps({'peak': peak, 'vectors': vectors, 'references': references}))
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        env={**os.environ, "HF_HUB_OFFLINE": "1"},
        capture_output=True,
        text=True,
        timeout=120,
    )
    if result.returncode == 77:
        pytest.skip("requires cached BGE-small model")
    assert result.returncode == 0, result.stderr
    body = json.loads(result.stdout)
    # One quarter of 4 GiB less the decimal 2 GB DuckDB pool is 547 MiB.
    assert 0 < body["peak"] <= 573_741_824
    assert len(body["vectors"]) == 33
    for vector in body["vectors"]:
        assert len(vector) == 384
        assert sum(value * value for value in vector) == pytest.approx(1, abs=0.002)
    for index, reference in zip((0, 32), body["references"], strict=True):
        assert body["vectors"][index] == pytest.approx(reference, abs=0.002)
