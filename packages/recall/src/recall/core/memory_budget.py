"""Default process memory partitions, independent of physical host RAM.

The DuckDB buffer pool leaves room for native allocations and live inference.
The process budget uses binary GiB; DuckDB's GB allowance is decimal.
Only one eighth of that remainder is reusable Metal cache; the rest stays
available to live tensors, untracked engine memory, Python, FTS and libraries.
These allocator allowances do not bound the process's total resident memory.
"""

PROCESS_MEMORY_BUDGET_BYTES = 4 * 1024**3
DUCKDB_MEMORY_ALLOWANCE_GB = 2
DUCKDB_MEMORY_ALLOWANCE_BYTES = DUCKDB_MEMORY_ALLOWANCE_GB * 1000**3
DUCKDB_MEMORY_LIMIT = f"{DUCKDB_MEMORY_ALLOWANCE_GB}GB"
MLX_CACHE_LIMIT_BYTES = (PROCESS_MEMORY_BUDGET_BYTES - DUCKDB_MEMORY_ALLOWANCE_BYTES) // 8

MLX_INFERENCE_ALLOWANCE_BYTES = (PROCESS_MEMORY_BUDGET_BYTES - DUCKDB_MEMORY_ALLOWANCE_BYTES) // 4
