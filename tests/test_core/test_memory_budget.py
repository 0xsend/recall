from recall.core.memory_budget import MLX_CACHE_LIMIT_BYTES


def test_mlx_cache_allowance_from_process_budget() -> None:
    """REQ-SEM-018 reserves one eighth of 4 GiB less decimal 2 GB for cache."""
    assert MLX_CACHE_LIMIT_BYTES == 286_870_912
