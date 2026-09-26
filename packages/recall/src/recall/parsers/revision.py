"""Identity of the normalization an adapter performs.

A stored row, a stored byte offset and a stored normalization checkpoint are
only trustworthy while the code that produced them is unchanged. The revision
covers the adapter's own module and every first-party module its normalized
output depends on, so editing the identity generator, the capture reader or the
resume declaration invalidates resume for all five adapters at once.

This lives in ``parsers`` rather than ``services`` so an adapter can stamp its
own checkpoint without importing upwards. ``services.coordinator`` re-exports
it, which is the spelling the rest of the codebase already uses.
"""

from __future__ import annotations

import hashlib
import inspect
from functools import cache
from pathlib import Path


@cache
def _shared_module_paths() -> tuple[Path, ...]:
    """Shared code every adapter's normalized output depends on.

    This is the union of the adapters' first-party import closures, minus the
    adapters themselves and this module. `core.ids` decides every persisted
    identity, `core.bash` the stored bash command, `parsers.skills` the skill
    name, `parsers.js_object` Codex tool input, `parsers.protocol` which
    sidecars refresh a start time, and `core.models` / `core.types` the shapes
    all of them produce. An adapter that imports only some of these still
    hashes all of them; over-hashing costs one extra invalidation, under-
    hashing costs a session normalized by two builds that no reparse heals.

    Imported here rather than at module scope because the adapters import this
    module, and `recall.parsers.__init__` imports the adapters. Adding an entry
    invalidates every stored revision once, by design.
    """
    from recall.core import bash, ids, models, types
    from recall.parsers import capture, checkpoint, common, js_object, protocol, skills

    return tuple(
        Path(inspect.getfile(module))
        for module in (
            bash,
            ids,
            models,
            types,
            capture,
            checkpoint,
            common,
            js_object,
            protocol,
            skills,
        )
    )


@cache
def revision_inputs(parser_type: type) -> tuple[Path, ...]:
    """Every source file whose bytes this adapter's revision covers.

    Exposed so the dependency set is inspectable: a test compares it against
    the adapter's own first-party import closure, which is what keeps this
    list from rotting the way a hand-maintained one does.
    """
    return (Path(inspect.getfile(parser_type)), *_shared_module_paths())


@cache
def parser_revision(parser_type: type) -> str:
    """Hash the adapter's source together with the shared parsing modules."""
    digest = hashlib.sha256()
    for path in revision_inputs(parser_type):
        digest.update(path.read_bytes())
    return digest.hexdigest()
