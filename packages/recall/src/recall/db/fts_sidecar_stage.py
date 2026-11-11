"""Keyword-index changes of one writer turn, published after DuckDB COMMIT.

A turn stages the keyword rows its DuckDB transactions change and writes one
durable `upsert` intent per changed id inside each transaction. Staged rows
join the turn only once that transaction's COMMIT returns, so a rolled-back
change never reaches the sidecar. `KeywordTurn.publish` then applies the
joined rows in one SQLite transaction (one per `FTS_SIDECAR_BATCH_SIZE` rows),
waiting for sidecar locks at most `KEYWORD_PUBLISH_DEADLINE_S` in total, and
`Publication.retire` deletes the intents the committed batches applied
(REQ-FTS-SIDECAR-006, REQ-FTS-SIDECAR-018).
"""

from __future__ import annotations

import logging
import sqlite3
from collections import Counter
from collections.abc import Callable, Iterable, Iterator, Mapping
from contextlib import closing, contextmanager
from dataclasses import dataclass, field
from itertools import batched
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import duckdb

from recall.core import telemetry
from recall.core.failure_observation import FailureObserver
from recall.core.telemetry import Span, Stage
from recall.db import fts_sidecar
from recall.db.fts_sidecar import (
    FtsSidecarUnavailableError,
    KeywordIndexFailure,
    PublicationDeadline,
    open_sidecar_for_publication,
    sidecar_path,
)
from recall.db.lease import DbLease, StandaloneLifetime, TransactionCapability

if TYPE_CHECKING:
    from recall.core.config import AppConfig

logger = logging.getLogger(__name__)

type KeywordKind = Literal["message", "tool_call"]
type KeywordKey = tuple[KeywordKind, str]


@dataclass(frozen=True)
class _MessageRow:
    content: str
    thinking: str


@dataclass(frozen=True)
class _ToolCallRow:
    bash_command: str | None


# None stages a removal: the sidecar row is deleted.
type _StagedRow = _MessageRow | _ToolCallRow | None


class KeywordStage:
    """Keyword changes of one open DuckDB transaction.

    Each call records the rows in memory (last write wins per id) and inserts
    one `upsert` intent per id in the transaction, so the intent commits or
    rolls back with the change it covers.
    """

    def __init__(self, unit: TransactionCapability, *, writes_intents: bool) -> None:
        self._unit = unit
        self._writes_intents = writes_intents
        self.rows: dict[KeywordKey, _StagedRow] = {}
        self.intents: Counter[KeywordKey] = Counter()

    def messages(self, rows: Iterable[tuple[str, str, str]]) -> None:
        """Stage `(message_id, fts_content, fts_thinking)` rows."""
        self._stage(
            "message", {key: _MessageRow(content, thinking) for key, content, thinking in rows}
        )

    def tool_calls(self, rows: Iterable[tuple[str, str | None]]) -> None:
        """Stage `(tool_call_id, bash_command)` rows; a None command removes the row."""
        self._stage("tool_call", {key: _ToolCallRow(command) for key, command in rows})

    def removed(self, kind: KeywordKind, ids: Iterable[str]) -> None:
        """Stage removal of rows DuckDB deleted in this transaction."""
        self._stage(kind, dict.fromkeys(ids))

    def _stage(self, kind: KeywordKind, rows: Mapping[str, _StagedRow]) -> None:
        if not self._writes_intents or not rows:
            return
        ids = list(rows)
        with _registered(self._unit, "_recall_keyword_intents", kind=[kind] * len(ids), id=ids):
            self._unit.call(
                "keyword:stage_intents",
                lambda conn: conn.execute(
                    "INSERT INTO fts_sidecar_pending(kind, id, op) "
                    "SELECT kind, id, 'upsert' FROM _recall_keyword_intents"
                ).fetchall(),
            )
        for key, row in rows.items():
            self.rows[(kind, key)] = row
            self.intents[(kind, key)] += 1


@dataclass(frozen=True)
class Publication:
    """Outcome of one turn's publication and the intents it may retire."""

    outcome: Literal["published", "deferred", "skipped", "nothing"]
    # Intent count per id of every committed batch.
    applied: Mapping[KeywordKey, int] = field(default_factory=dict)
    commits: int = 0
    error: KeywordIndexFailure | None = None

    def retire(self, conn: duckdb.DuckDBPyConnection) -> bool:
        """Delete applied intents inside the caller's DuckDB transaction.

        An id is retired only when every pending row of it is one this turn
        wrote. Another row (a move's delete, an earlier deferred publication,
        an older binary's row) keeps all of them for repair to re-derive.
        Returns whether there was anything to retire.
        """
        with StandaloneLifetime().lease(conn) as lease:
            return self._retire(lease)

    def retire_on(self, transaction: TransactionCapability) -> bool:
        """Apply retirement inside the caller's active outer transaction."""
        return self._retire(transaction)

    def _retire(self, unit: DbLease | TransactionCapability) -> bool:
        if not self.applied:
            return False
        kinds, ids, counts = zip(
            *((kind, key, count) for (kind, key), count in self.applied.items()), strict=True
        )
        with _registered(
            unit, "_recall_keyword_retired", kind=list(kinds), id=list(ids), written=list(counts)
        ):
            unit.call(
                "keyword:retire_intents",
                lambda conn: conn.execute(
                    """
                DELETE FROM fts_sidecar_pending AS pending
                USING _recall_keyword_retired AS turn
                WHERE pending.kind = turn.kind AND pending.id = turn.id
                  AND turn.written = (
                    SELECT COUNT(*) FROM fts_sidecar_pending AS other
                    WHERE other.kind = turn.kind AND other.id = turn.id
                  )
                """
                ).fetchall(),
            )
        return True


class KeywordTurn:
    """Keyword changes of one writer turn, published once after its commits."""

    def __init__(self, path: Path | None, fields: tuple[str, ...] | None) -> None:
        # None: the DuckDB keyword backend, which owns no sidecar and no
        # intents (REQ-FTS-SIDECAR-015).
        self._path = path
        self._fields = fields
        self._rows: dict[KeywordKey, _StagedRow] = {}
        self._intents: Counter[KeywordKey] = Counter()
        self._transaction_open = False

    @classmethod
    def for_config(cls, config: AppConfig) -> KeywordTurn:
        if config.fts.backend == "sqlite_sidecar":
            return cls(sidecar_path(config.data_dir), config.fts.fields)
        return cls(None, config.fts.fields)

    @property
    def staged_rows(self) -> int:
        return len(self._rows)

    @contextmanager
    def transaction(
        self,
        conn: duckdb.DuckDBPyConnection,
        *,
        on_failure: Callable[[BaseException], None] | None = None,
    ) -> Iterator[KeywordStage]:
        """Own one DuckDB transaction; its staged rows join the turn after COMMIT.

        `on_failure` runs before the rollback, the last point where a failed
        commit's cause is certainly still in hand.
        """
        with (
            StandaloneLifetime().lease(conn) as lease,
            self.transaction_on(lease, on_failure=on_failure) as stage,
        ):
            yield stage

    @contextmanager
    def transaction_on(
        self,
        lease: DbLease,
        *,
        on_failure: Callable[[BaseException], None] | None = None,
    ) -> Iterator[KeywordStage]:
        assert not self._transaction_open, "a keyword turn runs one transaction at a time"
        lease.call("keyword:begin", lambda conn: conn.execute("BEGIN").fetchone())
        transaction = TransactionCapability(lease)
        stage = KeywordStage(transaction, writes_intents=self._path is not None)
        self._transaction_open = True
        try:
            began = lease.call("keyword:transaction_id", _transaction_id)
            try:
                yield stage
                if lease.call("keyword:transaction_id", _transaction_id) != began:
                    raise _TransactionReplacedError
                lease.call("keyword:commit", lambda conn: conn.execute("COMMIT").fetchone())
            except BaseException as primary:
                lease.observe(primary, "keyword:transaction")
                errors: list[BaseException] = []
                if on_failure is not None:
                    try:
                        on_failure(primary)
                    except BaseException as diagnostic:
                        lease.observe(diagnostic, "keyword:failure_diagnostic")
                        errors.append(diagnostic)
                try:
                    if isinstance(primary, _TransactionReplacedError):
                        _rollback_if_open_on(lease)
                    else:
                        lease.cleanup_call(
                            "keyword:rollback", lambda conn: conn.execute("ROLLBACK").fetchone()
                        )
                except BaseException as cleanup:
                    errors.append(cleanup)
                if isinstance(primary, _TransactionReplacedError):
                    primary = AssertionError(
                        "a keyword turn's body ended its transaction; its staged rows are discarded"
                    )
                if errors:
                    raise BaseExceptionGroup(
                        "keyword transaction and cleanup failed", [primary, *errors]
                    ) from None
                raise primary
        finally:
            transaction._active = False
            self._transaction_open = False
        self._rows.update(stage.rows)
        self._intents.update(stage.intents)

    def publish(self, *, attempt: bool) -> Publication:
        """Apply the joined rows after the turn's last COMMIT; never raises for SQLite.

        The turn is empty afterwards. A deferred or skipped publication leaves
        its intents to keyword repair.
        """
        return self.publish_on(StandaloneLifetime(), attempt=attempt)

    def publish_on(self, observer: FailureObserver, *, attempt: bool) -> Publication:
        assert not self._transaction_open, "publication follows the turn's last COMMIT"
        rows, intents = self._rows, self._intents
        self._rows, self._intents = {}, Counter()
        if self._path is None or not rows:
            return Publication("nothing")
        if not attempt:
            return Publication("skipped")
        batch_size = fts_sidecar.FTS_SIDECAR_BATCH_SIZE
        batches = list(batched(rows.items(), batch_size))
        telemetry.count(telemetry.Counter.KEYWORD_PUBLICATIONS, 1)
        telemetry.count(telemetry.Counter.KEYWORD_PUBLISH_SPLIT, int(len(batches) > 1))
        applied: dict[KeywordKey, int] = {}
        commits = 0
        with Span(Stage.KEYWORD_PUBLISH):
            try:
                with closing(open_sidecar_for_publication(self._path)) as conn:
                    deadline = PublicationDeadline()
                    for batch in batches:
                        if deadline.exhausted:
                            raise fts_sidecar.PublicationDeadlineExceededError(
                                "keyword publication lock wait exhausted before its next batch"
                            )
                        with fts_sidecar.publish_sidecar(conn, deadline=deadline) as publication:
                            _apply(publication, batch, self._fields)
                        commits += 1
                        telemetry.count(telemetry.Counter.KEYWORD_PUBLISH_COMMITS, 1)
                        applied.update((key, intents[key]) for key, _row in batch)
            except (sqlite3.Error, FtsSidecarUnavailableError) as err:
                if observer.observe(err, "keyword:publish"):
                    raise
                telemetry.count(telemetry.Counter.KEYWORD_PUBLISH_DEFERRALS, 1)
                logger.warning(
                    "keyword publication deferred rows=%d messages=%d tool_calls=%d "
                    "committed_batches=%d of %d error=%s: %s",
                    len(rows),
                    sum(kind == "message" for kind, _key in rows),
                    sum(kind == "tool_call" for kind, _key in rows),
                    commits,
                    len(batches),
                    type(err).__name__,
                    err,
                )
                return Publication(
                    "deferred",
                    applied,
                    commits,
                    KeywordIndexFailure.of("publication deferred", err),
                )
        return Publication("published", applied, commits)


@contextmanager
def _registered(
    unit: DbLease | TransactionCapability, name: str, **columns: list[str] | list[int]
) -> Iterator[None]:
    """Expose columns to one statement as a registered Arrow table.

    DuckDB 1.5.5 searches for an absent optional `pandas` for every bound
    non-NULL parameter value, list elements included; a registered table is
    read without per-value Python conversion (REQ-RECON-027).
    """
    import pyarrow as pa

    def register(conn: duckdb.DuckDBPyConnection) -> None:
        conn.register(name, pa.table(columns))

    def unregister(conn: duckdb.DuckDBPyConnection) -> None:
        conn.unregister(name)

    unit.call("keyword:register", register)
    try:
        yield
    except BaseException as primary:
        unit.observe(primary, "keyword:registered_statement")
        try:
            unit.cleanup_call("keyword:unregister", unregister)
        except BaseException as cleanup:
            raise BaseExceptionGroup(
                "statement and unregister failed", [primary, cleanup]
            ) from None
        raise
    else:
        unit.cleanup_call("keyword:unregister", unregister)


class _TransactionReplacedError(Exception):
    """The body committed or rolled back the manager's transaction itself."""


def _transaction_id(conn: duckdb.DuckDBPyConnection) -> int:
    row = conn.execute("SELECT txid_current()").fetchone()
    assert row is not None
    return int(row[0])


def _rollback_if_open_on(lease: DbLease) -> None:
    try:
        lease.cleanup_call("keyword:rollback", lambda conn: conn.execute("ROLLBACK").fetchone())
    except duckdb.TransactionException as error:
        if lease.observe(error, "keyword:rollback"):
            raise


def _apply(
    publication: fts_sidecar._Publication,
    batch: Iterable[tuple[KeywordKey, _StagedRow]],
    fields: tuple[str, ...] | None,
) -> None:
    removed: dict[KeywordKind, list[str]] = {"message": [], "tool_call": []}
    messages: list[tuple[str, str, str]] = []
    tool_calls: list[tuple[str, str | None]] = []
    for (kind, key), row in batch:
        match row:
            case None:
                removed[kind].append(key)
            case _MessageRow(content=content, thinking=thinking):
                messages.append((key, content, thinking))
            case _ToolCallRow(bash_command=command):
                tool_calls.append((key, command))
    for kind, ids in removed.items():
        if ids:
            publication.delete(ids, kind)
    if messages:
        publication.messages(messages, fields=fields)
    if tool_calls:
        publication.tools(tool_calls, fields=fields)


class OwnedKeywordTurn:
    """A keyword turn whose transactions require one active owner lease."""

    def __init__(self, lease: DbLease, turn: KeywordTurn) -> None:
        self._lease = lease
        self._turn = turn

    @classmethod
    def for_config(cls, lease: DbLease, config: AppConfig) -> OwnedKeywordTurn:
        return cls(lease, KeywordTurn.for_config(config))

    @property
    def staged_rows(self) -> int:
        return self._turn.staged_rows

    @contextmanager
    def transaction(
        self,
        conn: duckdb.DuckDBPyConnection,
        *,
        on_failure: Callable[[BaseException], None] | None = None,
    ) -> Iterator[KeywordStage]:
        assert self._lease.call("keyword:connection_identity", lambda active: active is conn), (
            "keyword transaction belongs to a different connection"
        )
        with self._turn.transaction_on(self._lease, on_failure=on_failure) as stage:
            yield stage

    def publish(self, *, attempt: bool) -> Publication:
        return self._turn.publish_on(self._lease, attempt=attempt)
