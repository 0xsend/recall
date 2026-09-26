"""Shared output post-processing for context backends.

Model-specific output hygiene lives here so every backend that drives a model
capable of emitting it can apply the same cleanup, without the shared
contextual-retrieval prompt (identical across backends) having to change.
"""

from __future__ import annotations

import re

# Thinking-mode models (Qwen3 "Thinking" variants, codex reasoning) wrap their
# chain-of-thought in <think>...</think>. That reasoning must never reach the
# stored context prefix — it is verbose internal monologue, not a situating
# summary. A no-op when absent, so it is safe to apply on every model path.
_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL)


def strip_think_blocks(text: str) -> str:
    """Remove `<think>...</think>` spans from thinking-mode model output.

    Qwen3 emits an empty `<think></think>` even with `/no_think`; larger reasoning
    models (DeepSeek-R1, etc.) can emit substantial reasoning before the final
    answer. Either way we keep only the post-thinking summary, since the think
    block pollutes embeddings and BM25 indexing without aiding retrieval.
    Universally safe: a no-op for models that never emit `<think>` blocks.
    """
    return _THINK_BLOCK_RE.sub("", text)
