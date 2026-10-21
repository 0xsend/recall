"""Transactional durable source inventory used by reconciliation workers."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import cast

import duckdb

from recall.core.models import NORMALIZATION_CHECKPOINT_BYTES_MAX
from recall.core.types import Source
from recall.db.mutations import execute_count
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

# A source owed reconciliation now. Lane reads and claims share this text, so a
# source a lane offers is exactly a source `claimable` can return.
_PENDING_NOW = (
    "(desired_generation > committed_generation OR committed_offset < size"
    " OR last_error IS NOT NULL) AND missing = FALSE AND next_retry_at <= ?"
)

# Coverage and inventory share the committed-boundary definition. Unsupported
# records limit what the parser understood, not whether it published these bytes.
SOURCE_CURRENT = (
    "NOT missing AND committed_generation > 0"
    " AND desired_generation = committed_generation AND committed_offset = size"
    " AND (last_error IS NULL OR last_error = 'unsupported')"
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
            and self.last_error in (None, "unsupported")
            and self.committed_generation > 0
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
class PendingSource:
    """A pending source's place in a scheduling lane: its identity and active eligibility.

    Lanes are read by scanning the catalog, and a scan that projects a whole
    `SourceFile` reads every present row's resume proof (up to 64 KiB each)
    before the pending predicate discards it. On an aged catalog that made one
    selection cost hundreds of milliseconds of writer time while selecting
    nothing. A `SourceFile` comes only from a keyed read (`claimable`, `get`).
    """

    source: str
    source_path: str
    is_active: bool = False

    @property
    def key(self) -> str:
        return source_key(self.source, self.source_path)


@dataclass(frozen=True)
class CatalogCursor:
    """The position after one catalogued source, in `source_key` order.

    The token is the source key itself. A string becomes a position only
    through `SourceCatalog.issued_cursor`: an arbitrary string would otherwise
    be a valid keyset position and silently skip or end the listing.
    """

    source: Source
    source_path: str

    @classmethod
    def parse(cls, token: str) -> CatalogCursor:
        source, separator, source_path = token.partition("\x1f")
        if not separator or not source_path:
            raise ValueError(
                "malformed source page cursor; pass next_cursor exactly as it was returned"
            )
        try:
            return cls(Source(source), source_path)
        except ValueError as err:
            raise ValueError(f"source page cursor names an unknown source: {source!r}") from err

    @property
    def token(self) -> str:
        return source_key(self.source, self.source_path)


@dataclass(frozen=True)
class CatalogPage:
    rows: tuple[SourceFile, ...]
    next_cursor: CatalogCursor | None


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
        count = execute_count(
            self._conn,
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
            """,
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
        assert count <= 1
        return count == 1

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
        count = execute_count(
            self._conn,
            """UPDATE source_files SET last_error = ?, diagnostics = ?,
                   first_pending_at = COALESCE(first_pending_at, ?), next_retry_at = 1e308,
                   last_serviced_seq = (SELECT COALESCE(MAX(last_serviced_seq), 0) + 1
                                        FROM source_files)
               WHERE source_key = ?""",
            [error, json.dumps(diagnostics), self._clock(), source_key(source, source_path)],
        )
        if count == 0:
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
            count = execute_count(
                self._conn,
                """UPDATE source_files SET desired_generation = desired_generation + 1,
                       normalization_checkpoint = NULL,
                       first_pending_seq = CASE WHEN first_pending_at IS NULL
                           THEN (SELECT COALESCE(MAX(last_serviced_seq), 0) FROM source_files)
                           ELSE first_pending_seq END,
                       first_pending_at = COALESCE(first_pending_at, ?), next_retry_at = 0
                   WHERE source_key IN (SELECT id FROM _recall_bound_ids)""",
                [self._clock()],
            )
            assert count == len(keys), "catalog changed outside its transaction owner"

    def retry_now(self, source: str, source_path: str) -> None:
        count = execute_count(
            self._conn,
            "UPDATE source_files SET next_retry_at = 0 WHERE source_key = ?",
            [source_key(source, source_path)],
        )
        if count == 0:
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
            f"""SELECT source_path, dev, inode, ctime_ns, mtime_ns, size,
                      sidecar_mtime_ns, sidecar_signature,
                      ({SOURCE_CURRENT}) AS current
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
        """Own one transaction so the returned generation belongs to this scan."""
        self._conn.execute("BEGIN")
        try:
            self._conn.execute(
                """INSERT INTO reconciliation_roots (
                   source, root_path, scan_started_at, scan_complete, scan_generation)
               VALUES (?, ?, ?, FALSE, 1)
               ON CONFLICT (source, root_path) DO UPDATE SET
                   scan_started_at = excluded.scan_started_at,
                   scan_complete = FALSE,
                   scan_generation = reconciliation_roots.scan_generation + 1""",
                [source, root_path, started_at],
            )
            row = self._conn.execute(
                "SELECT scan_generation FROM reconciliation_roots "
                "WHERE source = ? AND root_path = ?",
                [source, root_path],
            ).fetchone()
            assert row is not None
            generation = cast(int, row[0])
            self._conn.execute("COMMIT")
        except Exception:
            self._conn.execute("ROLLBACK")
            raise
        return generation

    def pending(self, *, limit: int = 256) -> list[SourceFile]:
        if not 1 <= limit <= 256:
            raise ValueError("limit must be between 1 and 256")
        claimed = (
            self.claimable(item)
            for item in self._query_pending(
                "", [], limit, "first_pending_at, last_serviced_seq, source_path"
            )
        )
        return [item for item in claimed if item is not None]

    def claimable(self, pending: PendingSource) -> SourceFile | None:
        """The whole catalog row for a lane's source, or None once it is no longer pending."""
        row = self._conn.execute(
            f"""SELECT {_SOURCE_FILE_COLUMNS}
               FROM source_files WHERE source_key = ? AND {_PENDING_NOW}""",
            [pending.key, self._clock()],
        ).fetchone()
        return self._file(row) if row is not None else None

    def priority_pending(
        self,
        active_paths: set[str],
        *,
        limit: int = 256,
        eligible_roots: tuple[tuple[str, str], ...] | None = None,
        requested_paths: tuple[str, ...] = (),
        active_since_ns: int | None = None,
        excluded_keys: tuple[str, ...] = (),
    ) -> tuple[list[PendingSource], list[PendingSource], list[PendingSource]]:
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
        recent = "TRUE"
        recent_args: list[str | int] = []
        activity_evidence = f"source_path IN ({placeholders})"
        activity_args: list[str | int] = list(paths)
        if active_since_ns is not None:
            recent = "GREATEST(mtime_ns, sidecar_mtime_ns) >= ?"
            recent_args.append(active_since_ns)
            activity_evidence += f" OR ({recent})"
            activity_args.append(active_since_ns)
        activity = f"committed_generation > 0 AND ({activity_evidence})"
        # Durable capture, rather than notification origin, owns active eligibility.
        # Imports can carry the same modification events as existing appends.
        # A recent modification also supplies activity evidence without a watch
        # event; inventory observation time alone cannot make old history active.
        # Recent arrivals may overtake each other; the oldest lane retains durable
        # admission/service order and bounds waiting through its reserved slot.
        fair_order = (
            "GREATEST(last_serviced_seq, first_pending_seq), first_pending_at, source, source_path"
        )

        def query(clause: str, args: Sequence[str | int], order: str) -> list[PendingSource]:
            rows = self._pending_rows(
                clause,
                args,
                limit,
                order,
                activity=activity,
                activity_args=activity_args,
            )
            return [
                PendingSource(cast(str, row[0]), cast(str, row[1]), bool(row[-1])) for row in rows
            ]

        active = query(
            "AND is_active" + scope,
            scope_args,
            fair_order,
        )
        recent_rows = query(
            f"AND NOT is_active AND ({recent})" + scope,
            recent_args + scope_args,
            "GREATEST(mtime_ns, sidecar_mtime_ns) DESC, first_pending_at DESC, source, source_path",
        )
        oldest = query(scope, scope_args, fair_order)
        return active, recent_rows, oldest

    def _query_pending(
        self,
        clause: str,
        args: Sequence[str | int],
        limit: int,
        order: str = "first_pending_at, last_serviced_seq, source_path",
    ) -> list[PendingSource]:
        return [
            PendingSource(cast(str, row[0]), cast(str, row[1]))
            for row in self._pending_rows(clause, args, limit, order)
        ]

    def _pending_rows(
        self,
        clause: str,
        args: Sequence[str | int],
        limit: int,
        order: str,
        *,
        activity: str | None = None,
        activity_args: Sequence[str | int] = (),
    ) -> list[tuple[object, ...]]:
        columns = "source, source_path"
        if activity is not None:
            columns += f", ({activity}) AS is_active"
        return self._conn.execute(
            f"""SELECT {columns}
                FROM source_files
                WHERE {_PENDING_NOW} {clause} ORDER BY {order} LIMIT ?""",
            [*activity_args, self._clock(), *args, limit],
        ).fetchall()

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

    def issued_cursor(self, token: str) -> CatalogCursor:
        """The position a `next_cursor` token names, if this catalog could have issued it.

        A page's cursor names its last row, and catalog rows are never deleted
        (a vanished file is marked missing), so every issued cursor names a
        catalogued file. A well-formed token naming none was truncated, edited
        or forged; as a keyset position it would select another page.
        """
        cursor = CatalogCursor.parse(token)
        row = self._conn.execute(
            "SELECT 1 FROM source_files WHERE source_key = ?", [cursor.token]
        ).fetchone()
        if row is None:
            raise ValueError(
                "source page cursor names no catalogued file; "
                "pass next_cursor exactly as it was returned"
            )
        return cursor

    def status_page(self, *, limit: int = 100, cursor: CatalogCursor | None = None) -> CatalogPage:
        """Stable bounded catalog listing for a future RPC/status projection."""
        if not 1 <= limit <= 256:
            raise ValueError("limit must be between 1 and 256")
        # Choose the page by key first, then read the wide columns for those
        # rows alone: a top-N over the full projection reads every row's
        # checkpoint text before the limit applies. One statement keeps the
        # key choice and the hydration on one snapshot, so a concurrent delete
        # cannot shorten a page that still reports a next cursor.
        rows = self._conn.execute(
            f"""WITH page AS (
                   SELECT source_key FROM source_files
                   WHERE source_key > COALESCE(?, '') ORDER BY source_key LIMIT ?)
               SELECT {_SOURCE_FILE_COLUMNS}
               FROM page JOIN source_files USING (source_key)
               ORDER BY source_key""",
            [cursor.token if cursor is not None else None, limit + 1],
        ).fetchall()
        page = tuple(self._file(row) for row in rows[:limit])
        next_cursor = (
            CatalogCursor(Source(page[-1].source), page[-1].source_path)
            if len(rows) > limit
            else None
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
