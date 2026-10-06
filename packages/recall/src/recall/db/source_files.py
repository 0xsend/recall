"""Transactional durable source inventory used by reconciliation workers."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import cast

import duckdb

from recall.core.models import NORMALIZATION_CHECKPOINT_BYTES_MAX
from recall.db.queries import _bound_identifiers

# A bounded typed JSON vector avoids per-value Python/DuckDB parameter binding.
# BIGINT retains nanosecond signatures exactly, including above float precision.
_OBSERVATION_TYPES = (
    ("source_key", "VARCHAR"),
    ("source_path", "VARCHAR"),
    ("source", "VARCHAR"),
    ("root_path", "VARCHAR"),
    ("dev", "BIGINT"),
    ("inode", "BIGINT"),
    ("ctime_ns", "BIGINT"),
    ("mtime_ns", "BIGINT"),
    ("size", "BIGINT"),
    ("sidecar_mtime_ns", "BIGINT"),
    ("sidecar_signature", "VARCHAR"),
    ("desired_generation", "BIGINT"),
    ("first_pending_at", "DOUBLE"),
    ("observed_at", "DOUBLE"),
)
_OBSERVATION_COLUMNS = tuple(name for name, _ in _OBSERVATION_TYPES)
_OBSERVATION_SHAPE = json.dumps([dict(_OBSERVATION_TYPES)])

# One projection for every SourceFile read; `_file` positions follow this order.
_SOURCE_FILE_COLUMNS = (
    "source_path, source, root_path, desired_generation, committed_generation, "
    "first_pending_at, retry_count, next_retry_at, last_serviced_seq, missing, "
    "committed_offset, committed_prefix_sha256, content_epoch, last_error, "
    "CAST(diagnostics AS VARCHAR), dev, inode, ctime_ns, mtime_ns, size, "
    "sidecar_mtime_ns, sidecar_signature, normalization_checkpoint"
)


@dataclass(frozen=True)
class SourceSignature:
    dev: int | None
    inode: int | None
    ctime_ns: int
    mtime_ns: int
    size: int
    sidecar_mtime_ns: int = 0
    sidecar_signature: str = ""


@dataclass(frozen=True)
class SourceFile:
    source_path: str
    source: str
    root_path: str
    desired_generation: int
    committed_generation: int
    first_pending_at: float | None
    retry_count: int
    next_retry_at: float
    last_serviced_seq: int
    missing: bool
    committed_offset: int = 0
    committed_prefix_sha256: str | None = None
    # The encoded resume proof for `committed_offset`, or None when none exists.
    # `prepare_raw_sources` runs without a connection, so the resume decision can
    # only see what this carries (REQ-INDEX-025).
    normalization_checkpoint: str | None = None
    content_epoch: int = 0
    last_error: str | None = None
    diagnostics: str | None = None
    signature: SourceSignature | None = None

    @property
    def current(self) -> bool:
        return (
            not self.missing
            and self.last_error is None
            and self.desired_generation == self.committed_generation
            and self.signature is not None
            and self.committed_offset == self.signature.size
        )


@dataclass(frozen=True)
class PresentSource:
    """What one catalogued source under a walked root holds right now."""

    identity: tuple[object, ...]
    current: bool


@dataclass(frozen=True)
class CatalogPage:
    rows: tuple[SourceFile, ...]
    next_cursor: str | None


def prefix_sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def source_key(source: str, source_path: str) -> str:
    return f"{source}\x1f{source_path}"


def observation_identity(signature: SourceSignature) -> tuple[object, ...]:
    """The transcript and sidecar facts whose change means new content to read.

    The parser revision is deliberately absent: a parser build change re-parses
    a source only when its own content changes next, never the whole corpus.
    Rebuilding history is a versioned index migration (`REQ-RECON-010`).
    """
    return (
        signature.dev,
        signature.inode,
        signature.ctime_ns,
        signature.mtime_ns,
        signature.size,
        signature.sidecar_mtime_ns,
        signature.sidecar_signature,
    )


class SourceCatalog:
    """Catalog mutations; callers own the transaction around content + ack."""

    def __init__(self, conn: duckdb.DuckDBPyConnection, *, clock: Callable[[], float]) -> None:
        self._conn = conn
        self._clock = clock

    def observe(
        self, source: str, root_path: str, source_path: str, signature: SourceSignature
    ) -> int:
        self.observe_batch([(source, root_path, source_path, signature)])
        row = self._conn.execute(
            "SELECT desired_generation FROM source_files WHERE source_key = ?",
            [source_key(source, source_path)],
        ).fetchone()
        assert row is not None
        return int(row[0])

    def observe_batch(
        self, observations: Sequence[tuple[str, str, str, SourceSignature]]
    ) -> tuple[str, ...]:
        """Persist at most 256 observations; return existing paths whose content changed.

        The generation, retry and pending-age rules are shared by notifications
        and inventory. Only new, changed, returning or re-rooted sources are
        written: every root is walked on a timer, and re-observing an unchanged
        source must cost a read, not a row rewrite.
        """
        if not 1 <= len(observations) <= 256:
            raise ValueError("observation batch must contain 1..256 paths")
        keys = [source_key(source, path) for source, _, path, _ in observations]
        if len(set(keys)) != len(keys):
            # Aliases can resolve to one path. Preserve sequential observations
            # of that identity instead of letting an upsert discard a change.
            changed: set[str] = set()
            for observation in observations:
                changed.update(self.observe_batch([observation]))
            return tuple(sorted(changed))
        placeholders = ", ".join("?" for _ in keys)
        persisted = {
            str(row[0]): row[1:]
            for row in self._conn.execute(
                f"""SELECT source_key, root_path, missing, dev, inode,
                           ctime_ns, mtime_ns, size, sidecar_mtime_ns, sidecar_signature
                    FROM source_files WHERE source_key IN ({placeholders})""",
                keys,
            ).fetchall()
        }
        writes: list[tuple[str, str, str, SourceSignature]] = []
        changed_paths: list[str] = []
        for key, observation in zip(keys, observations, strict=True):
            _source, root, path, signature = observation
            row = persisted.get(key)
            if row is None:
                writes.append(observation)
                continue
            root_path, missing, *content = row
            content_changed = tuple(content) != observation_identity(signature)
            if content_changed:
                changed_paths.append(path)
            if content_changed or missing or root_path != root:
                writes.append(observation)
        if writes:
            self._merge_observations(writes)
        return tuple(sorted(changed_paths))

    def _merge_observations(
        self, observations: Sequence[tuple[str, str, str, SourceSignature]]
    ) -> None:
        now = self._clock()
        records: list[dict[str, object]] = []
        for source, root, path, signature in observations:
            values = (
                source_key(source, path),
                path,
                source,
                root,
                signature.dev,
                signature.inode,
                signature.ctime_ns,
                signature.mtime_ns,
                signature.size,
                signature.sidecar_mtime_ns,
                signature.sidecar_signature,
                1,
                now,
                now,
            )
            records.append(dict(zip(_OBSERVATION_COLUMNS, values, strict=True)))
        columns = ", ".join(_OBSERVATION_COLUMNS)
        selection = ", ".join(f"observation.{column}" for column in _OBSERVATION_COLUMNS)
        changed = " OR ".join(
            f"source_files.{column} IS DISTINCT FROM excluded.{column}"
            for column in (
                "dev",
                "inode",
                "ctime_ns",
                "mtime_ns",
                "size",
                "sidecar_mtime_ns",
                "sidecar_signature",
            )
        )
        # Naming an indexed column in UPDATE rewrites the row on DuckDB 1.5.5.
        # Separate MERGE arms keep a returning or re-rooted source off the indexes
        # while preserving one atomic statement for mixed existing/new sources.
        inserted = ", ".join(f"excluded.{column}" for column in _OBSERVATION_COLUMNS)
        self._conn.execute(
            f"""WITH incoming AS (
                SELECT unnest(json_transform_strict(?, '{_OBSERVATION_SHAPE}')) AS observation
            )
            MERGE INTO source_files USING (
                SELECT {selection},
                    (SELECT COALESCE(MAX(last_serviced_seq), 0) FROM source_files)
                        AS first_pending_seq
                FROM incoming
            ) AS excluded ON source_files.source_key = excluded.source_key
            WHEN MATCHED AND ({changed}) THEN UPDATE SET
                root_path = excluded.root_path,
                dev = excluded.dev, inode = excluded.inode, ctime_ns = excluded.ctime_ns,
                mtime_ns = excluded.mtime_ns, size = excluded.size,
                sidecar_mtime_ns = excluded.sidecar_mtime_ns,
                sidecar_signature = excluded.sidecar_signature,
                desired_generation = source_files.desired_generation + 1,
                first_pending_at = COALESCE(
                    source_files.first_pending_at, excluded.first_pending_at),
                first_pending_seq = CASE WHEN source_files.first_pending_at IS NULL
                    THEN excluded.first_pending_seq ELSE source_files.first_pending_seq END,
                next_retry_at = 0,
                missing = FALSE, observed_at = excluded.observed_at
            WHEN MATCHED THEN UPDATE SET
                root_path = excluded.root_path,
                missing = FALSE, observed_at = excluded.observed_at
            WHEN NOT MATCHED THEN INSERT ({columns}, first_pending_seq)
                VALUES ({inserted}, excluded.first_pending_seq)
            """,
            [json.dumps(records, separators=(",", ":"))],
        )

    def bind_session(self, source: str, source_path: str, session_id: str) -> None:
        """Attach the source to the session committed by an ingestion transaction."""
        self._conn.execute(
            "UPDATE source_files SET session_id = ? WHERE source_key = ?",
            [session_id, source_key(source, source_path)],
        )

    def get(self, source: str, source_path: str) -> SourceFile | None:
        row = self._conn.execute(
            f"""SELECT {_SOURCE_FILE_COLUMNS}
               FROM source_files WHERE source_key = ?""",
            [source_key(source, source_path)],
        ).fetchone()
        return self._file(row) if row is not None else None

    def acknowledge(
        self,
        source: str,
        source_path: str,
        generation: int,
        offset: int,
        prefix: str,
        *,
        semantic_rewrite: bool = False,
        checkpoint: str | None = None,
        full_parse: bool = True,
    ) -> bool:
        """Commit the prefix this generation reconciled, with its resume proof.

        `checkpoint` is the encoded envelope describing exactly `offset`, and it
        moves in this statement or not at all. Omitting it clears any stored
        proof: an adapter that declined to resume must not leave one pointing at
        an older offset beside the newer committed one (REQ-INDEX-025).

        A suffix may replace a proof only while the catalog still carries one.
        An explicit reparse clears it to revoke suffix authorization; a suffix
        already in flight must not silently restore it. A full parse needs no
        prior proof and may always stamp the boundary it just established.
        """
        if offset < 0 or len(prefix) != 64:
            raise ValueError("acknowledgment requires a complete non-negative offset and SHA-256")
        if checkpoint is not None and len(checkpoint) > NORMALIZATION_CHECKPOINT_BYTES_MAX:
            raise ValueError("acknowledgment checkpoint exceeds the durable size bound")
        result = self._conn.execute(
            """UPDATE source_files SET committed_generation = ?, committed_offset = ?,
                   committed_prefix_sha256 = ?,
                   normalization_checkpoint = CASE
                       WHEN ? OR normalization_checkpoint IS NOT NULL THEN ? ELSE NULL END,
                   retry_count = 0, next_retry_at = 0,
                   last_error = NULL, diagnostics = NULL,
                   content_epoch = content_epoch + CASE WHEN ? THEN 1 ELSE 0 END,
                   last_serviced_seq = (SELECT COALESCE(MAX(last_serviced_seq), 0) + 1
                                        FROM source_files),
                   first_pending_at = CASE WHEN desired_generation = ? AND size = ? THEN NULL
                                           ELSE COALESCE(first_pending_at, ?) END
               WHERE source_key = ? AND desired_generation >= ? AND committed_generation <= ?
                 AND (committed_generation < ? OR committed_offset < ? OR last_error IS NOT NULL)
               RETURNING source_path""",
            [
                generation,
                offset,
                prefix,
                full_parse,
                checkpoint,
                semantic_rewrite,
                generation,
                offset,
                self._clock(),
                source_key(source, source_path),
                generation,
                generation,
                generation,
                offset,
            ],
        )
        return result.fetchone() is not None

    def fail(
        self,
        source: str,
        source_path: str,
        error: str,
        diagnostics: dict[str, object] | None = None,
    ) -> None:
        row = self._conn.execute(
            "SELECT retry_count FROM source_files WHERE source_key = ?",
            [source_key(source, source_path)],
        ).fetchone()
        if row is None:
            raise KeyError(source_path)
        retries = min(int(row[0]) + 1, 8)
        delay = min(300.0, float(2**retries))
        self._conn.execute(
            """UPDATE source_files SET retry_count = ?, next_retry_at = ?,
                   last_error = ?, diagnostics = ?,
                   last_serviced_seq = (SELECT COALESCE(MAX(last_serviced_seq), 0) + 1
                                        FROM source_files)
               WHERE source_key = ?""",
            [
                retries,
                self._clock() + delay,
                error,
                json.dumps(diagnostics or {}),
                source_key(source, source_path),
            ],
        )

    def defer_unsupported(
        self, source: str, source_path: str, diagnostics: dict[str, object]
    ) -> None:
        """Keep unsupported content visible until its source/parser changes or an explicit retry."""
        self.defer_until_changed(source, source_path, "unsupported", diagnostics)

    def defer_until_changed(
        self, source: str, source_path: str, error: str, diagnostics: dict[str, object]
    ) -> None:
        """A stable incomplete input needs changed bytes/parser or an explicit retry."""
        result = self._conn.execute(
            """UPDATE source_files SET last_error = ?, diagnostics = ?,
                   first_pending_at = COALESCE(first_pending_at, ?), next_retry_at = 1e308,
                   last_serviced_seq = (SELECT COALESCE(MAX(last_serviced_seq), 0) + 1
                                        FROM source_files)
               WHERE source_key = ? RETURNING source_path""",
            [error, json.dumps(diagnostics), self._clock(), source_key(source, source_path)],
        )
        if result.fetchone() is None:
            raise KeyError(source_key(source, source_path))

    def force_reconcile(self, source: str, source_path: str) -> None:
        """Record an explicit full reparse as a new desired generation."""
        self.force_reconcile_batch([(source, source_path)])

    def force_reconcile_batch(self, requests: Sequence[tuple[str, str]]) -> None:
        """Record 1..256 distinct reparse requests in the caller's transaction.

        A batch shares one pending timestamp. Existing pending age and service
        order survive; missing or repeated identities reject before any write.

        The resume proof is cleared here, and this is the one statement besides
        the acknowledgement allowed to touch it (REQ-INDEX-025). A reparse
        raises the same desired generation an ordinary append raises, so
        without the clear a turn that no longer holds the request answers the
        rebuild with a suffix and acknowledges it. Clearing only ever costs a
        full parse the caller already asked for. The committed offset and
        digest stay put: a later turn still needs them to tell an append from
        a rewrite.
        """
        if not 1 <= len(requests) <= 256:
            raise ValueError("reparse batch must contain 1..256 paths")
        keys = [source_key(source, path) for source, path in requests]
        if len(set(keys)) != len(keys):
            raise ValueError("reparse batch must contain distinct source paths")
        with _bound_identifiers(self._conn, keys):
            existing = {
                row[0]
                for row in self._conn.execute(
                    "SELECT source_key FROM source_files "
                    "WHERE source_key IN (SELECT id FROM _recall_bound_ids)"
                ).fetchall()
            }
            missing = next((key for key in keys if key not in existing), None)
            if missing is not None:
                raise KeyError(missing)
            changed = self._conn.execute(
                """UPDATE source_files SET desired_generation = desired_generation + 1,
                       normalization_checkpoint = NULL,
                       first_pending_seq = CASE WHEN first_pending_at IS NULL
                           THEN (SELECT COALESCE(MAX(last_serviced_seq), 0) FROM source_files)
                           ELSE first_pending_seq END,
                       first_pending_at = COALESCE(first_pending_at, ?), next_retry_at = 0
                   WHERE source_key IN (SELECT id FROM _recall_bound_ids)
                   RETURNING source_key""",
                [self._clock()],
            ).fetchall()
            assert len(changed) == len(keys), "catalog changed outside its transaction owner"

    def retry_now(self, source: str, source_path: str) -> None:
        result = self._conn.execute(
            "UPDATE source_files SET next_retry_at = 0 WHERE source_key = ? RETURNING source_path",
            [source_key(source, source_path)],
        )
        if result.fetchone() is None:
            raise KeyError(source_key(source, source_path))

    def present_count(self, source: str, root_path: str) -> int:
        row = self._conn.execute(
            """SELECT COUNT(*) FROM source_files
               WHERE source = ? AND root_path = ? AND missing = FALSE""",
            [source, root_path],
        ).fetchone()
        assert row is not None
        return int(row[0])

    def present_states(self, source: str, root_path: str) -> dict[str, PresentSource]:
        """Map each present source under the root to its identity and currency.

        One read per root answers both questions a captured batch asks -- would
        observing this change anything, and does it still owe any work -- so an
        unchanged corpus is compared in memory instead of costing a lookup and
        a writer turn per file (`REQ-INDEX-023`).
        """
        rows = self._conn.execute(
            """SELECT source_path, dev, inode, ctime_ns, mtime_ns, size,
                      sidecar_mtime_ns, sidecar_signature,
                      (last_error IS NULL AND desired_generation = committed_generation
                       AND committed_offset = size) AS current
               FROM source_files
               WHERE source = ? AND root_path = ? AND missing = FALSE""",
            [source, root_path],
        ).fetchall()
        return {str(row[0]): PresentSource(tuple(row[1:8]), bool(row[8])) for row in rows}

    def present_page(
        self, source: str, root_path: str, *, after_key: str | None, limit: int = 256
    ) -> list[tuple[str, str]]:
        """Return `(source_key, source_path)` of present sources after `after_key`."""
        if not 1 <= limit <= 256:
            raise ValueError("limit must be between 1 and 256")
        rows = self._conn.execute(
            """SELECT source_key, source_path FROM source_files
               WHERE source = ? AND root_path = ? AND missing = FALSE
                 AND source_key > COALESCE(?, '')
               ORDER BY source_key LIMIT ?""",
            [source, root_path, after_key, limit],
        ).fetchall()
        return [(str(key), str(path)) for key, path in rows]

    def mark_missing(self, keys: Sequence[str]) -> None:
        if not 1 <= len(keys) <= 256:
            raise ValueError("missing batch must contain 1..256 sources")
        placeholders = ", ".join("?" for _ in keys)
        self._conn.execute(
            f"UPDATE source_files SET missing = TRUE WHERE source_key IN ({placeholders})",
            list(keys),
        )

    def start_scan(self, source: str, root_path: str, started_at: float) -> int:
        row = self._conn.execute(
            """INSERT INTO reconciliation_roots (
                   source, root_path, scan_started_at, scan_complete, scan_generation)
               VALUES (?, ?, ?, FALSE, 1)
               ON CONFLICT (source, root_path) DO UPDATE SET
                   scan_started_at = excluded.scan_started_at,
                   scan_complete = FALSE,
                   scan_generation = reconciliation_roots.scan_generation + 1
               RETURNING scan_generation""",
            [source, root_path, started_at],
        ).fetchone()
        assert row is not None
        return cast(int, row[0])

    def pending(self, *, limit: int = 256) -> list[SourceFile]:
        if not 1 <= limit <= 256:
            raise ValueError("limit must be between 1 and 256")
        return self._query_pending("", [], limit)

    def priority_pending(
        self,
        active_paths: set[str],
        *,
        limit: int = 256,
        eligible_roots: tuple[tuple[str, str], ...] | None = None,
        requested_paths: tuple[str, ...] = (),
        active_since_ns: int | None = None,
        excluded_keys: tuple[str, ...] = (),
    ) -> tuple[list[SourceFile], list[SourceFile], list[SourceFile]]:
        """Keep all three lanes inside one 256-row preparation roster."""
        if not 1 <= limit <= 256:
            raise ValueError("limit must be between 1 and 256")
        limit = min(limit, 256 // 3)
        scope = ""
        scope_args: list[str] = []
        if eligible_roots is not None:
            clauses = ["(source = ? AND root_path = ?)" for _ in eligible_roots]
            scope_args = [value for pair in eligible_roots for value in pair]
            if requested_paths:
                clauses.append("source_path IN (" + ", ".join("?" for _ in requested_paths) + ")")
                scope_args.extend(requested_paths)
            scope = " AND (" + (" OR ".join(clauses) or "FALSE") + ")"
        if excluded_keys:
            scope += " AND source_key NOT IN (" + ", ".join("?" for _ in excluded_keys) + ")"
            scope_args.extend(excluded_keys)
        paths = sorted(active_paths)
        placeholders = ", ".join("?" for _ in paths) or "''"
        activity = f"source_path IN ({placeholders})"
        activity_args: list[str | int] = list(paths)
        recent = "TRUE"
        recent_args: list[str | int] = []
        if active_since_ns is not None:
            recent = "GREATEST(mtime_ns, sidecar_mtime_ns) >= ?"
            recent_args.append(active_since_ns)
        # Admission is independent of service: newly observed work cannot jump
        # ahead of an existing pending interval, including after a restart.
        fair_order = (
            "GREATEST(last_serviced_seq, first_pending_seq), first_pending_at, source, source_path"
        )
        active = self._query_pending(
            f"AND {activity}" + scope,
            activity_args + scope_args,
            limit,
            fair_order,
        )
        recent_rows = self._query_pending(
            f"AND NOT ({activity}) AND ({recent})" + scope,
            activity_args + recent_args + scope_args,
            limit,
            "GREATEST(last_serviced_seq, first_pending_seq), "
            "GREATEST(mtime_ns, sidecar_mtime_ns) DESC, first_pending_at, source, source_path",
        )
        oldest = self._query_pending(scope, scope_args, limit, fair_order)
        return active, recent_rows, oldest

    def _query_pending(
        self,
        clause: str,
        args: Sequence[str | int],
        limit: int,
        order: str = "first_pending_at, last_serviced_seq, source_path",
    ) -> list[SourceFile]:
        rows = self._conn.execute(
            f"""SELECT {_SOURCE_FILE_COLUMNS}
                FROM source_files
                WHERE (desired_generation > committed_generation OR committed_offset < size
                       OR last_error IS NOT NULL) AND missing = FALSE
                  AND next_retry_at <= ? {clause} ORDER BY {order} LIMIT ?""",
            [self._clock(), *args, limit],
        ).fetchall()
        return [self._file(row) for row in rows]

    @staticmethod
    def _file(row: tuple[object, ...]) -> SourceFile:
        signature = SourceSignature(
            cast(int | None, row[15]),
            cast(int | None, row[16]),
            cast(int, row[17]),
            cast(int, row[18]),
            cast(int, row[19]),
            cast(int, row[20]),
            sidecar_signature=cast(str, row[21]),
        )
        return SourceFile(
            source_path=cast(str, row[0]),
            source=cast(str, row[1]),
            root_path=cast(str, row[2]),
            desired_generation=cast(int, row[3]),
            committed_generation=cast(int, row[4]),
            first_pending_at=cast(float | None, row[5]),
            retry_count=cast(int, row[6]),
            next_retry_at=cast(float, row[7]),
            last_serviced_seq=cast(int, row[8]),
            missing=cast(bool, row[9]),
            committed_offset=cast(int, row[10]),
            committed_prefix_sha256=cast(str | None, row[11]),
            content_epoch=cast(int, row[12]),
            last_error=cast(str | None, row[13]),
            diagnostics=cast(str | None, row[14]),
            normalization_checkpoint=cast(str | None, row[22]),
            signature=signature,
        )

    def status_page(self, *, limit: int = 100, cursor: str | None = None) -> CatalogPage:
        """Stable bounded catalog listing for a future RPC/status projection."""
        if not 1 <= limit <= 256:
            raise ValueError("limit must be between 1 and 256")
        rows = self._conn.execute(
            f"""SELECT {_SOURCE_FILE_COLUMNS}
               FROM source_files WHERE source_key > COALESCE(?, '')
               ORDER BY source_key LIMIT ?""",
            [cursor, limit + 1],
        ).fetchall()
        page = tuple(self._file(row) for row in rows[:limit])
        next_cursor = (
            source_key(page[-1].source, page[-1].source_path) if len(rows) > limit else None
        )
        return CatalogPage(rows=page, next_cursor=next_cursor)

    def record_scan(
        self,
        source: str,
        root_path: str,
        *,
        started_at: float,
        finished_at: float,
        discovered_count: int,
        failures: list[str],
        failure_count: int | None = None,
        complete: bool = True,
    ) -> None:
        self._conn.execute(
            """INSERT INTO reconciliation_roots (
                   source, root_path, scan_started_at, scan_finished_at,
                   discovered_count, failure_count, failures, scan_complete)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT (source, root_path) DO UPDATE SET
                   scan_started_at = excluded.scan_started_at,
                   scan_finished_at = excluded.scan_finished_at,
                   discovered_count = excluded.discovered_count,
                   failure_count = excluded.failure_count,
                   failures = excluded.failures, scan_complete = excluded.scan_complete""",
            [
                source,
                root_path,
                started_at,
                finished_at,
                discovered_count,
                len(failures) if failure_count is None else failure_count,
                json.dumps(failures),
                complete,
            ],
        )
