from __future__ import annotations

import asyncio
import importlib.metadata
import json
import logging
import os
import signal
import socket
import threading
import time
from collections import deque
from collections.abc import Awaitable, Callable, Coroutine, Iterator, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from contextlib import suppress
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import duckdb
from pydantic import ValidationError

from recall.core.config import AppConfig, create_private_dir
from recall.core.daemon_status import StorageMaintenanceStatus, WriteAdmissionStatus
from recall.core.read_cancellation import ReadCancelled, read_cancellation
from recall.core.rpc_params import (
    BashStatsParams,
    CheckIndexesParams,
    DaemonRunParams,
    FollowParams,
    IndexParams,
    ListBySessionParams,
    ListParams,
    LiveBySessionParams,
    LiveMarkParams,
    LiveParams,
    Params,
    ResponsiveReadParams,
    SchedulerParams,
    SearchParams,
    ShowParams,
    SkillStatsParams,
    StatusParams,
    StorageParams,
    UsageStatsParams,
)
from recall.core.rpc_types import (
    APP_CONFIRMATION_REQUIRED,
    APP_LOCKED,
    APP_NOT_FOUND,
    INTERNAL_ERROR,
    INVALID_PARAMS,
    INVALID_REQUEST,
    METHOD_NOT_FOUND,
    PARSE_ERROR,
    RpcError,
    SearchPhase,
    SearchReleased,
    SearchWaiting,
    config_fingerprint,
    serialize_rpc_value,
)
from recall.core.socket_path import validate_socket_path
from recall.core.source_path import SourcePath
from recall.core.telemetry import Counter, DiagnosticExecutor, Span, Stage, count, measured_async
from recall.core.types import (
    SearchMode,
    Source,
    default_session_host,
    parse_search_mode,
    parse_source,
)
from recall.db import FtsRebuildOutOfMemoryError, FtsSettingsRestoreError, RecallLockError
from recall.db.fatal import is_disk_full_error, is_fatal_db_invalidation
from recall.db.maintenance import IndexDivergenceReport
from recall.db.source_files import CatalogCursor, SourceCatalog, SourceProgress
from recall.read_budget import (
    RESPONSIVE_READ_METHODS,
    SERVER_READ_BUDGET,
    ReadBudget,
)
from recall.services.live_events import SessionIndexedNotifier
from recall.services.reconciler import Admission, RefusedSource
from recall.services.scheduling import demote_current_thread
from recall.services.search_request import SearchRequest, search_request
from recall.services.session_ids import UnknownSessionError
from recall.services.snapshot_cohorts import SnapshotCapacityError, SnapshotCohorts

logger = logging.getLogger("recall.rpc_server")

_serialize = serialize_rpc_value
_READ_WORKERS = 8
_QUERY_MODEL_WORKERS = 1
_WRITER_WORKERS = 4
_READ_REQUEST_TIMEOUT = 30.0
# Yield at most one normal writer turn, leaving useful time for the scan.
_SEARCH_ENRICHMENT_WAIT_SECONDS = 5.0
_MAX_CONNECTIONS = 64
# Contended populated snapshot workers measured up to 0.73 s. Reserve their
# completion and caller-local copying; raw work continues after this wait.
_FINAL_SNAPSHOT_RESERVE_SECONDS = 0.85
_ROSTER_CATCH_UP_SECONDS = 0.25
_ACTIVITY_HINT_LIMIT = 4096
# The complete walk is how old-mtime imports are discovered, so its rescan period
# stays inside the 45 s discovery floor (BRIEF) whatever `daemon.interval` says.
_INVENTORY_RESCAN_SECONDS_MAX = 30
_RAW_IDLE_RESCAN_SECONDS = 5.0
_WAL_BLOCK_ALL_BYTES = 192 * 1024 * 1024
_RAW_PREPARATION_BYTES = 512 * 1024 * 1024
_RAW_ACTIVE_PROTECTED_BYTES = 128 * 1024 * 1024
_RAW_MIN_RESERVATION_BYTES = 8 * 1024 * 1024
_RAW_CAPTURE_MULTIPLIER = 4
# How long a requested path waits on a scheduler that serves nobody at all
# before the request gives up and says so. Reset by every served turn, so a
# large backlog is patience rather than a stall.
_REQUESTED_SERVICE_STALL_SECONDS = 120.0
# The window the reported reconciliation drain rate covers, and the ring that
# holds its samples: a drain faster than this many sources a minute reports the
# ring's size rather than its true rate.
_DRAIN_RATE_WINDOW_SECONDS = 60.0
_DRAIN_RATE_SAMPLES = 1024
# How long one source may be served without the request saying so. A client's
# idle timeout bounds silence between frames, and one 300 MiB transcript takes
# longer to serve than a pass's per-file frames admit.
_PROGRESS_KEEPALIVE_SECONDS = 20.0
_BOUNDED_READ_METHODS = frozenset(
    {
        "recall.search",
        "recall.list",
        "recall.list_by_session",
        "recall.show",
        "recall.stats",
        "recall.stats_tools",
        "recall.stats_bash",
        "recall.stats_tokens",
        "recall.stats_usage",
        "recall.stats_skills",
        "recall.daemon_status",
        "recall.check_indexes",
        "recall.live_sessions",
        "recall.live",
        "recall.live_by_session",
    }
)
# Long writes: no deadline of their own, but a departed client still ends them.
_WATCHED_WRITE_METHODS = frozenset({"recall.index", "recall.daemon_run"})
_STREAMING_METHODS = _WATCHED_WRITE_METHODS | {"recall.show_follow"}
_CONTEXT_ENV_VARS = (
    "RECALL_CONTEXT_MODE",
    "RECALL_CONTEXT_FALLBACK",
    "RECALL_CONTEXT_MODEL",
    "RECALL_CONTEXT_CONCURRENCY",
)
_RECOVERABLE_SEARCH_MARKERS = (
    "FTS indexes",
    "Current transaction is aborted",
    "please ROLLBACK",
)

MethodHandler = Callable[[dict[str, Any], "ClientConnection | None"], Awaitable[Any]]

if TYPE_CHECKING:
    from recall.core.daemon_status import ReconciliationStatus
    from recall.core.types import RunKind
    from recall.db.source_files import SourceFile
    from recall.services.coordinator import IndexRequestScope, PreparedRawSource, RawIndexRequest
    from recall.services.embed_phase import EmbedPhaseState, PendingEmbeds, PreparedEmbedCycle
    from recall.services.indexer import IndexSummary, SessionContextStats
    from recall.services.live import LiveCoverage, LivePage
    from recall.services.usage_harvest import HarvestResult
    from recall.services.watcher import LiveWatchRuntime


def _departure_ends_write(method: str, params: dict[str, Any]) -> bool:
    """Whether a departed client's write may be cancelled (`REQ-RPC-018`).

    An incremental pass loses nothing: it commits per session and
    reconciliation owns whatever it had not reached. Work nothing else will
    redo is a different matter -- `--recreate` cancelled after its reset leaves
    an emptied database with no run recorded, a cancelled `--full` leaves the
    batches it never captured un-observed, and a cancelled context recompute
    abandons its snapshot -- so those finish even when nobody is listening.
    """
    if method != "recall.index":
        return True
    return not any(bool(params.get(name)) for name in ("recreate", "full", "recompute_context"))


def _is_recoverable_search_error(err: BaseException) -> bool:
    if isinstance(err, FtsSettingsRestoreError):
        return True
    err_msg = str(err)
    return any(marker in err_msg for marker in _RECOVERABLE_SEARCH_MARKERS)


def _is_search_fts_missing(err: BaseException) -> bool:
    return "FTS indexes" in str(err)


def _search_needs_shared_conn_recovery(err: BaseException) -> bool:
    if isinstance(err, FtsSettingsRestoreError):
        return True
    err_msg = str(err)
    return "Current transaction is aborted" in err_msg or "please ROLLBACK" in err_msg


def _is_oom_error(err: BaseException) -> bool:
    if isinstance(err, FtsRebuildOutOfMemoryError):
        return True
    if isinstance(err, duckdb.OutOfMemoryException):
        return True
    return str(err).startswith("Out of Memory Error")


def _resolve_probe_sample(value: object) -> int:
    from recall.db.maintenance import PROBE_SAMPLE_DEFAULT, PROBE_SAMPLE_MAX

    if value is None:
        return PROBE_SAMPLE_DEFAULT
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError("sample must be an integer")
    if not 1 <= value <= PROBE_SAMPLE_MAX:
        raise ValueError(f"sample must be between 1 and {PROBE_SAMPLE_MAX}")
    return value


def _is_shared_conn_failure(err: BaseException) -> bool:
    """Return True only when recovery should reset the shared DuckDB handle.

    Watch indexing handles many path-local parser failures. Those failures must
    be recorded and skipped without requeueing the rest of the ready batch.
    """
    if _is_oom_error(err):
        return True
    if isinstance(err, FtsRebuildOutOfMemoryError):
        return True
    if isinstance(err, duckdb.TransactionException):
        return True
    if isinstance(err, duckdb.ConnectionException):
        return True
    msg = str(err)
    if "Current transaction is aborted" in msg:
        return True
    return "please ROLLBACK" in msg


@dataclass(frozen=True)
class _RawReservation:
    bytes: int
    historical: bool

    @classmethod
    def for_source(cls, source: SourceFile, *, historical: bool) -> _RawReservation:
        assert source.signature is not None
        # A compressed file's physical size cannot bound its normalized rows.
        # Reserve the decoded ceiling before it enters either preparation slot.
        from recall.core.transcripts import MAX_COMPRESSED_CAPTURE_BYTES, transcript_file

        size = source.signature.size
        if source.source == "codex":
            try:
                compressed = transcript_file(Path(source.source_path)).suffix == ".zst"
            except OSError:
                # Preparation owns reporting an unreadable source; admission
                # conservatively reserves its largest supported capture.
                compressed = True
            if compressed:
                size = MAX_COMPRESSED_CAPTURE_BYTES
        return cls(
            max(_RAW_MIN_RESERVATION_BYTES, size * _RAW_CAPTURE_MULTIPLIER),
            historical,
        )

    @property
    def oversized(self) -> bool:
        limit = _RAW_PREPARATION_BYTES - (_RAW_ACTIVE_PROTECTED_BYTES if self.historical else 0)
        return self.bytes > limit


class _ReadAbandoned(Exception):
    """A cancelled request withdrew before its worker opened a cursor."""


@dataclass(frozen=True)
class _LiveProbe:
    page: LivePage

    def __deepcopy__(self, _memo: dict[int, Any]) -> _LiveProbe:
        return self


@dataclass(frozen=True)
class _LiveResponse(Mapping[str, Any]):
    page: LivePage
    coverage: LiveCoverage
    watching: bool
    encoded: bytes = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "encoded", json.dumps(_serialize(self), separators=(",", ":")).encode("utf-8")
        )

    def __getitem__(self, key: str) -> Any:
        match key:
            case "schema_version":
                return 2
            case "watching":
                return self.watching
            case "sessions":
                return self.page.sessions
            case "next_cursor":
                return self.page.next_cursor
            case "coverage":
                return self.coverage
            case _:
                raise KeyError(key)

    def __iter__(self) -> Iterator[str]:
        return iter(("schema_version", "watching", "sessions", "next_cursor", "coverage"))

    def __len__(self) -> int:
        return 5

    def __deepcopy__(self, _memo: dict[int, Any]) -> _LiveResponse:
        # The page, coverage and every nested value are frozen dataclasses,
        # tuples or scalars. Both the view and its wire bytes are immutable.
        return self


class _ReadExecution:
    """Own one cursor from worker admission through cancellation and close."""

    def __init__(self) -> None:
        self.started_at = time.monotonic()
        self._search_request = search_request.get()
        self._lock = threading.Lock()
        self._cursor: Any = None
        self._cancelled = threading.Event()

    def cancel(self) -> None:
        with self._lock:
            self._cancelled.set()
            if self._cursor is not None:
                self._cursor.interrupt()

    def run(self, server: RpcServer, operation: Callable[[Any], Any]) -> Any:
        server._conn_lifecycle_gate.acquire_read()
        cursor = None
        try:
            cursor = server._open_conn_unlocked().cursor()
            with self._lock:
                if self._cancelled.is_set():
                    raise _ReadAbandoned
                self._cursor = cursor
            if self._search_request is not None:
                self._search_request.cursor_opened()
            with read_cancellation(self._cancelled):
                return operation(cursor)
        finally:
            with self._lock:
                self._cursor = None
            if cursor is not None:
                cursor.close()
            server._conn_lifecycle_gate.release_read()


@dataclass(frozen=True)
class _DatabaseClaim:
    writer: bool
    scan: bool
    ready: asyncio.Future[None]


class _DatabaseAdmission:
    """Grant writer and scan capacity together, without partial reservations.

    FIFO among runnable claims lets a scan pass a writer waiting on a
    checkpoint, and a raw writer pass enrichment waiting on a scan. All state
    changes happen on the event loop, so granting both resources is atomic.
    """

    def __init__(self) -> None:
        self._waiting: deque[_DatabaseClaim] = deque()
        self._writer = False
        self._scan = False

    async def acquire(self, *, writer: bool, scan: bool) -> _DatabaseClaim:
        assert writer or scan
        claim = _DatabaseClaim(writer, scan, asyncio.get_running_loop().create_future())
        self._waiting.append(claim)
        self._wake()
        try:
            await claim.ready
        except BaseException:
            if claim.ready.cancelled():
                self._waiting.remove(claim)
                self._wake()
            else:
                self.release(claim)
            raise
        return claim

    def release(self, claim: _DatabaseClaim) -> None:
        assert claim.ready.done() and not claim.ready.cancelled()
        if claim.writer:
            assert self._writer
            self._writer = False
        if claim.scan:
            assert self._scan
            self._scan = False
        self._wake()

    def _wake(self) -> None:
        for claim in tuple(self._waiting):
            if claim.ready.cancelled():
                continue
            if (claim.writer and self._writer) or (claim.scan and self._scan):
                continue
            self._waiting.remove(claim)
            self._writer |= claim.writer
            self._scan |= claim.scan
            claim.ready.set_result(None)


class _WriterLock:
    """Serialize writers and invalidate empty enrichment observations by default.

    Inventory, source/active observation, raw scheduling, keyword repair,
    usage harvest, and embed snapshots may opt out: they preserve embedding
    inputs. Inventory move supersession invalidates explicitly before changing
    embedding companions.
    Direct lock users and unclassified operations always invalidate, including
    failed turns that may have committed some statements before raising.
    """

    def __init__(self, admission: _DatabaseAdmission) -> None:
        self._admission = admission
        self._claim: _DatabaseClaim | None = None
        self.enrichment_generation = object()

    async def acquire(
        self, *, affects_enrichment: bool = True, scan: bool = False
    ) -> Literal[True]:
        claim = await self._admission.acquire(writer=True, scan=scan)
        assert self._claim is None
        self._claim = claim
        if affects_enrichment:
            self.invalidate_enrichment()
        return True

    def locked(self) -> bool:
        return self._claim is not None

    def release(self) -> None:
        assert self._claim is not None
        claim, self._claim = self._claim, None
        self._admission.release(claim)

    async def __aenter__(self) -> None:
        await self.acquire()

    async def __aexit__(self, *exc: object) -> None:
        self.release()

    def invalidate_enrichment(self) -> None:
        assert self.locked(), "enrichment changes belong to the writer turn"
        self.enrichment_generation = object()


class _ConnectionLifecycleGate:
    """Allow concurrent readers while maintenance owns connection close/reopen."""

    def __init__(self) -> None:
        self._condition = threading.Condition()
        self._readers = 0
        self._writer_active = False
        self._writers_waiting = 0

    def acquire_read(self) -> None:
        with self._condition:
            while self._writer_active or self._writers_waiting > 0:
                self._condition.wait()
            self._readers += 1

    def release_read(self) -> None:
        with self._condition:
            self._readers -= 1
            if self._readers == 0:
                self._condition.notify_all()

    def acquire_write(self) -> None:
        with self._condition:
            self._writers_waiting += 1
            try:
                while self._writer_active or self._readers > 0:
                    self._condition.wait()
                self._writer_active = True
            finally:
                self._writers_waiting -= 1

    def release_write(self) -> None:
        with self._condition:
            self._writer_active = False
            self._condition.notify_all()


def next_embed_interval(drained: int, remaining: PendingEmbeds | None, config: AppConfig) -> float:
    """Seconds the embed loop waits after a cycle that drained `drained` rows.

    It re-arms at once only while the commit left eligible work: after a full
    drain the next cycle can find nothing, and its outcome would replace
    `drained` as the last one within milliseconds of the batch.
    """
    if drained < 0:
        return config.daemon.embed_backoff
    if drained > 0 and remaining is not None and remaining.session_ids:
        return 0.01
    return config.daemon.embed_interval


def _issued_catalog_cursor(conn: Any, value: object) -> CatalogCursor | None:
    """Only an absent cursor starts the source page from the beginning."""
    if value is None:
        return None
    if not isinstance(value, str):
        raise RpcError(code=INVALID_PARAMS, message="source page cursor must be a string")
    try:
        return SourceCatalog(conn, clock=time.time).issued_cursor(value)
    except ValueError as err:
        raise RpcError(code=INVALID_PARAMS, message=str(err)) from err


def _bind_owner_only_socket(path: Path) -> socket.socket:
    """Bind a Unix socket at *path* that only the owning user can connect to.

    The mode is set between bind() and listen(): a peer cannot connect to a
    socket that is not listening yet, so no connection lands before the mode
    is owner-only, whatever the umask or the directory's mode.
    """
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        sock.bind(str(path))
        os.chmod(path, 0o600)
    except BaseException:
        sock.close()
        raise
    return sock


def _package_version() -> str | None:
    try:
        return importlib.metadata.version("recall")
    except importlib.metadata.PackageNotFoundError:
        return None


def _config_with_context_mode(config: AppConfig, mode_value: object | None) -> AppConfig:
    if mode_value is None:
        return config
    context = config.embedding.context
    # Carry every ContextConfig field forward and override only `mode`. Using asdict
    # keeps this future-proof: any new field on ContextConfig (e.g. base_url, timeout)
    # flows through an RPC override without needing to touch this site.
    values = asdict(context)
    values["mode"] = str(mode_value)
    override = type(context).from_values(values)
    return replace(config, embedding=replace(config.embedding, context=override))


def _runtime_context_stats(conn: Any) -> dict[str, Any]:
    try:
        row = conn.execute(
            """
            SELECT
                COALESCE(last_index_total, 0),
                COALESCE(last_index_indexed, 0),
                COALESCE(last_index_skipped, 0),
                COALESCE(last_index_failed, 0),
                COALESCE(last_context_messages, 0),
                COALESCE(last_context_mode, 'off'),
                COALESCE(last_context_input_tokens, 0),
                COALESCE(last_context_output_tokens, 0),
                last_context_model
            FROM runtime_state
            WHERE singleton = TRUE
            """
        ).fetchone()
    except Exception:
        row = None
    if row is None:
        return {
            "last_index_total": 0,
            "last_index_indexed": 0,
            "last_index_skipped": 0,
            "last_index_failed": 0,
            "last_context_messages": 0,
            "last_context_mode": "off",
            "last_context_input_tokens": 0,
            "last_context_output_tokens": 0,
            "last_context_model": None,
        }
    return {
        "last_index_total": int(row[0] or 0),
        "last_index_indexed": int(row[1] or 0),
        "last_index_skipped": int(row[2] or 0),
        "last_index_failed": int(row[3] or 0),
        "last_context_messages": int(row[4] or 0),
        "last_context_mode": str(row[5] or "off"),
        "last_context_input_tokens": int(row[6] or 0),
        "last_context_output_tokens": int(row[7] or 0),
        "last_context_model": str(row[8]) if row[8] is not None else None,
    }


class ClientDisconnected(Exception):
    """The peer is gone; a producer writing for it should stop."""


class _PeerStreamReader(asyncio.StreamReader):
    """Expose transport departure without consuming pipelined request bytes."""

    def __init__(self) -> None:
        super().__init__()
        self.input_closed = asyncio.Event()
        self.disconnected = asyncio.Event()

    def feed_eof(self) -> None:
        super().feed_eof()
        # Half-close ends producers, but the peer can still receive replies
        # to ordinary requests already sent, including buffered frames.
        self.input_closed.set()

    def set_exception(self, exc: Exception) -> None:
        super().set_exception(exc)
        self.input_closed.set()
        self.disconnected.set()


class _PeerStreamProtocol(asyncio.StreamReaderProtocol):
    def __init__(self, reader: _PeerStreamReader, connected: Callable[..., Any]) -> None:
        super().__init__(reader, connected)
        self._peer_reader = reader

    def connection_lost(self, exc: Exception | None) -> None:
        self._peer_reader.disconnected.set()
        super().connection_lost(exc)


class ClientConnection:
    def __init__(self, writer: asyncio.StreamWriter) -> None:
        self._writer = writer

    async def send_notification(self, method: str, params: dict[str, Any]) -> None:
        """Emit one notification frame, or raise `ClientDisconnected`.

        Checked before the write as well as after, because a transport already
        closing accepts `write` silently and only fails at `drain` — or not at
        all, which would let a stream run its whole deadline for nobody.
        """
        if self._writer.is_closing():
            raise ClientDisconnected(method)
        msg: dict[str, Any] = {"jsonrpc": "2.0", "method": method, "params": params, "id": None}
        line = json.dumps(msg, separators=(",", ":")) + "\n"
        try:
            self._writer.write(line.encode("utf-8"))
            await self._writer.drain()
        except (ConnectionError, BrokenPipeError) as err:
            raise ClientDisconnected(method) from err

    async def send_progress(self, processed: int, total: int, status: str, **extra: Any) -> None:
        """Report progress, tolerating a client that stopped watching.

        Unlike a stream, progress is advisory: a departed client ends only the
        writes watch mode can resume, and the rest run on with nobody to report
        to (`REQ-RPC-018`).
        """
        params: dict[str, Any] = {
            "processed": processed,
            "total": total,
            "status": status,
        }
        params.update(extra)
        with suppress(ClientDisconnected):
            await self.send_notification("progress", params)


# How long `show --follow` streams before closing itself, when the caller
# names no deadline of its own.
DEFAULT_FOLLOW_TIMEOUT = 60.0


@dataclass(frozen=True)
class _FreshTarget:
    source: str
    source_path: str
    database_epoch: int
    generation: int
    serviced_seq: int


class NoParserForPath(ValueError):
    """No registered parser claims this transcript.

    A `ValueError` subclass so the RPC layer keeps mapping it to a validation
    error, but a distinct type so `--fresh` can skip exactly this and nothing
    else. A bare `except ValueError` there also swallowed a bad
    `[embedding.context]` (`AppConfig.load`), a `pydantic.ValidationError` from
    a torn last line, and the FTS sidecar's own `ValueError`s — every one of
    which then read as "that path has no parser".
    """


# The two conditions that refuse reconciliation work. They are distinct
# situations for the operator -- one is a state they chose and must undo, the
# other is transient and clears itself -- so they never share a message.
PAUSED_MESSAGE = "reconciliation is paused; resume it with `recall daemon resume`"
SHUTTING_DOWN_MESSAGE = "the daemon is shutting down; retry once it has restarted"


class IndexTurns:
    """In-flight operator index requests, so background enrichment can yield.

    The embed loop re-arms 0.01s after a draining cycle (`REQ-ADAPT-008`), and
    an index request takes the enrichment lock once per session it enriches.
    Without this counter the drain wins the lock back between every one of
    them and the request pays a whole embed cycle per changed session
    (`REQ-ADAPT-017`). A request counts from the moment it starts
    waiting for its turn, not from the moment it gets one, so a queued request
    is not made to wait out cycles the drain started meanwhile.
    """

    def __init__(self) -> None:
        self._in_flight = 0
        self._idle = asyncio.Event()
        self._idle.set()

    @property
    def in_flight(self) -> int:
        return self._in_flight

    def enter(self) -> None:
        self._in_flight += 1
        self._idle.clear()

    def leave(self) -> None:
        assert self._in_flight > 0, "index turn left without being entered"
        self._in_flight -= 1
        if self._in_flight == 0:
            self._idle.set()

    async def wait_idle(self, timeout: float) -> bool:
        """Wait out the in-flight requests, returning whether they all finished."""
        assert timeout > 0, "index turn wait needs a positive bound"
        try:
            await asyncio.wait_for(self._idle.wait(), timeout=timeout)
        except TimeoutError:
            return False
        return True

    def turn(self, lock: asyncio.Lock) -> IndexTurn:
        return IndexTurn(self, lock)


class IndexTurn:
    """One operator index request: counted while queued, serialised while served.

    Written as a context-manager object rather than an `@asynccontextmanager`
    generator because the generator form reassigns `__traceback__` on an
    exception passing through it, and `RpcError` is a frozen dataclass that
    refuses the assignment -- every paused-reconciliation refusal raised inside
    the turn would surface as `FrozenInstanceError`.
    """

    def __init__(self, turns: IndexTurns, lock: asyncio.Lock) -> None:
        self._turns = turns
        self._lock = lock

    async def __aenter__(self) -> None:
        self._turns.enter()
        try:
            await self._lock.acquire()
        except BaseException:
            self._turns.leave()
            raise

    async def __aexit__(self, *_exc_info: object) -> None:
        self._lock.release()
        self._turns.leave()


class RpcServer:
    """JSON-RPC 2.0 server over Unix domain socket.

    Threading model
    ===============
    The daemon process is the single owner of all DuckDB and embedding
    model access.  No CLI process opens DB connections.

    A single shared DuckDB connection (``_conn``) is opened at startup.
    All handlers create cursors from this connection for thread safety.
    Cursors share the parent connection's configuration, eliminating
    DuckDB's "different configuration" error between read-only and
    read-write connections.  Multiple concurrent read cursors are
    served in parallel (REQ-RPC-012).

    *Write* RPCs (index, embed) run in the executor under
    ``_write_lock``, which serialises writes.  Because writes run in
    the executor, they do not block the event loop — reads and
    progress streaming continue during long writes.

    The embedding backend is lazy-loaded on the first vector/hybrid
    search (REQ-RPC-009) with double-checked locking.
    """

    def __init__(self, config: AppConfig | None = None) -> None:
        self._config = config or AppConfig.load()
        self._daemon_version = _package_version()
        self._methods: dict[str, MethodHandler] = {}
        self._database_admission = _DatabaseAdmission()
        self._write_lock = _WriterLock(self._database_admission)
        self._raw_slots = asyncio.Semaphore(2)
        self._raw_turn_lock = asyncio.Lock()
        self._raw_claims: dict[str, _RawReservation] = {}
        self._raw_reserved_bytes = 0
        self._raw_historical_reserved_bytes = 0
        self._raw_oversized_reserved_bytes = 0
        self._raw_turns_served = 0
        self._raw_served_at: deque[float] = deque(maxlen=_DRAIN_RATE_SAMPLES)
        self._raw_wakeup = asyncio.Event()
        self._inventory_wakeup = asyncio.Event()
        self._enrichment_lock = asyncio.Lock()
        self._status_count_lock = asyncio.Lock()
        self._snapshot_cohorts = SnapshotCohorts(
            max_waiters=_MAX_CONNECTIONS,
            max_draining=_READ_WORKERS,
            on_error=lambda err: self._stop_on_fatal_db(err, "snapshot cohort"),
        )
        self._background_enrichment_idle = asyncio.Event()
        self._background_enrichment_idle.set()
        self._index_request_lock = asyncio.Lock()
        self._index_turns = IndexTurns()
        self._announced_index_yield = False
        from recall.services.reconciler import FairScheduler

        self._raw_scheduler = FairScheduler(clock=time.time)
        self._fresh_jobs: dict[SourcePath, asyncio.Future[None]] = {}
        self._fresh_batches: set[asyncio.Task[None]] = set()
        self._storage_maintenance_task: asyncio.Task[None] | None = None
        self._storage_maintenance_plan_id: str | None = None
        self._storage_maintenance_error: str | None = None
        self._storage_plan_snapshot: dict[str, Any] | None = None
        self._storage_status_snapshot: dict[str, Any] | None = None
        self._raw_requests: dict[SourcePath, RawIndexRequest] = {}
        self._activity_hints: dict[str, float] = {}
        self._activity_hints_saturated = False
        self._active_observation_error: str | None = None
        self._inventory_error: str | None = None
        self._inventory_complete = False
        # Wall clock, like the scan records it is compared with (REQ-DAEMON-079).
        self._process_started_at = time.time()
        self._database_epoch = 0
        self._recreate_backup: str | None = None
        self._fts_repair_cursor: tuple[datetime, int] | None = None
        self._fts_repair_error: str | None = None
        self._next_fts_repair_at = 0.0
        self._keyword_dirty = False
        self._runtime_watch = False
        self._server: asyncio.Server | None = None
        self._shutdown_event = asyncio.Event()
        # Resolved now so shutdown never imports. The documented upgrade flow
        # (`uv tool install --force …` then `recall daemon restart`) replaces
        # site-packages under the still-running daemon, so an import issued from
        # the signal handler raises ModuleNotFoundError and the codex
        # subprocesses this was meant to reap are orphaned instead.
        from recall.services.context_backends.codex_cli import terminate_active_codex_processes

        self._terminate_codex_processes = terminate_active_codex_processes
        self._checkpoint_wakeup = asyncio.Event()
        self._checkpoint_retry_seconds = 30.0
        self._checkpoint_retry_at = 0.0
        self._checkpoint_contention_retries = 0
        self._wal_pressure_blocked = False
        self._checkpoint_status: dict[str, Any] = {
            "attempts": 0,
            "successes": 0,
            "running": False,
            "last_started_at": None,
            "last_completed_at": None,
            "last_duration_seconds": None,
            "last_wal_bytes_before": None,
            "last_wal_bytes_after": None,
            "last_error": None,
        }
        self._socket_path = self._config.data_dir / "recall.sock"
        self._pid_path = self._config.data_dir / "recall.pid"
        self._idle_timeout: float | None = None
        self._last_request_time: float = 0.0
        self._active_connections: int = 0
        self._client_tasks: set[asyncio.Task[Any]] = set()
        self._owns_runtime_artifacts = False
        self._embed_backend: Any = None
        self._embed_lock = threading.Lock()
        self._config_fp = config_fingerprint(self._config)
        self._embed_state: Any = None
        self._fts_sidecar_startup: Any = None
        from recall.services.daemon import ServiceProbeCache

        self._service_probes = ServiceProbeCache(self._config)
        # REQ-LIVE-011: one event channel for --fresh and --follow, owned by the
        # server rather than the watch runtime so it survives a watch restart
        # and exists in poll mode too.
        self._session_indexed = SessionIndexedNotifier()
        # REQ-LIVE-010 counters, surfaced by `recall daemon status`. An operator
        # tuning `live.fresh_timeout` needs to see how often the budget is
        # actually being spent before raising it.
        self.live_fresh_requests = 0
        self.live_fresh_timeouts = 0
        self._watch_metrics: Any = None  # WatchIndexMetrics, set in watch mode
        self._watch_runtime: LiveWatchRuntime | None = None
        self._watch_drain_task: asyncio.Task[None] | None = None
        self._watch_discovery_task: asyncio.Task[None] | None = None
        self._conn: Any = None  # shared DuckDB connection, opened lazily
        self._conn_lifecycle_gate = _ConnectionLifecycleGate()
        self._conn_open_lock = threading.Lock()
        self._conn_closed_for_maintenance = False
        self.last_compact_check_at: float | None = None
        # Most recent bloat ratio, cached for `recall daemon status` / CLI health
        # notices. Populated at startup and refreshed on each auto-compact check.
        self._last_bloat_ratio: float | None = None
        # Most recent index/table divergence probe (startup, then any
        # `recall.check_indexes` call), exposed by `daemon status` (REQ-RESIL-018).
        self._last_index_probe: IndexDivergenceReport | None = None
        # Dedicated executor owned by the server so its lifecycle is decoupled
        # from asyncio's default-executor teardown. asyncio.run() shuts the
        # default executor down during loop close; using an explicit executor
        # avoids the "Executor shutdown has been called" error path that fires
        # when handlers race the loop's automatic teardown sequence.
        self._executor: ThreadPoolExecutor | None = None
        # Admission is derived from each pool's capacity, not host CPU count.
        # Ordinary reads and model preparation cannot queue ahead of a writer.
        self._read_executor: ThreadPoolExecutor | None = None
        self._read_slots = asyncio.BoundedSemaphore(_READ_WORKERS)
        self._read_executions: set[_ReadExecution] = set()
        self._reader_drained = asyncio.Event()
        self._reader_drained.set()
        self._recovery_waiters: set[float] = set()
        self._query_model_executor: ThreadPoolExecutor | None = None
        self._query_model_slots = asyncio.BoundedSemaphore(_QUERY_MODEL_WORKERS)
        # Enrichment model work (load, inference, unload) runs on its own
        # thread at utility QoS. It holds no database handle, so demoting it
        # cannot delay the writer; work under the writer lock stays on
        # `_executor` at default QoS.
        self._enrichment_executor: ThreadPoolExecutor | None = None
        self._register_methods()

    def _fail_on_watch_task_exit(self, label: str, task: asyncio.Task[None]) -> None:
        if task.cancelled():
            return
        if self._shutdown_event.is_set():
            return
        try:
            err = task.exception()
        except asyncio.CancelledError:
            return
        if err is None:
            logger.error("watch %s task exited unexpectedly; stopping daemon", label)
        elif self._stop_on_fatal_db(err, f"watch_{label}"):
            return
        else:
            logger.error(
                "watch %s task crashed; stopping daemon so scheduler can restart it",
                label,
                exc_info=(type(err), err, err.__traceback__),
            )
        self._shutdown_event.set()

    def _open_conn_unlocked(self) -> Any:
        """Return the shared DuckDB connection once lifecycle access is held.

        Uses lenient schema mode so the daemon can start even when the
        database has a stale schema version.  The ``recall.index --recreate``
        RPC handler drops and rebuilds the database before any reads occur,
        so strict validation here would create a chicken-and-egg deadlock:
        the daemon can't listen until the connection opens, and the recreate
        command can't reach the daemon until it's listening.
        """
        # Lifecycle readers may enter concurrently before the first handle
        # exists. Only connection/schema creation is exclusive; reads stay parallel.
        with self._conn_open_lock:
            if self._conn is None:
                from recall.db import connect

                self._conn = connect(self._config, lenient_schema=True)
            return self._conn

    async def _await_shared_conn_work(self, future: asyncio.Future[Any], label: str) -> Any:
        """Await executor work on the shared connection, even when cancelled.

        The write lock is what keeps the shared DuckDB connection single-user.
        A cancelled ``await`` would release that lock while the executor thread
        is still mid-statement, so the next lock holder -- or ``stop()`` closing
        the handle -- interleaves with it: the observed ``tuple index out of
        range`` / ``No open result set`` / ``Connection already closed`` trio
        (REQ-RESIL-021). Cancellation is honoured only once the work has
        finished.
        """
        try:
            return await asyncio.shield(future)
        except asyncio.CancelledError:
            logger.info("%s cancellation requested; waiting for active work", label)
            try:
                await asyncio.shield(future)
            except Exception:
                logger.exception("%s failed during shutdown", label)
            raise

    async def _recover_shared_conn_locked(self, err: BaseException, label: str) -> None:
        """Run `_recover_shared_conn` as shared-connection work.

        An event-loop handler sees its job's failure only after that job
        released the write lock; recovering inline would roll back or close the
        handle while another job may be mid-statement on it. So recovery takes
        the lock, runs in the executor, and is awaited to completion like every
        other statement on the shared connection (REQ-RESIL-021).
        """
        waiting_since = time.monotonic()
        self._recovery_waiters.add(waiting_since)
        try:
            async with self._write_lock:
                await self._await_shared_conn_work(
                    asyncio.get_running_loop().run_in_executor(
                        self._executor, self._recover_shared_conn, err, label
                    ),
                    f"{label} recovery",
                )
        finally:
            self._recovery_waiters.discard(waiting_since)

    def _get_conn(self) -> Any:
        """Return the shared DuckDB connection, opening it lazily on first use."""
        self._conn_lifecycle_gate.acquire_read()
        try:
            return self._open_conn_unlocked()
        finally:
            self._conn_lifecycle_gate.release_read()

    def _stop_on_fatal_db(self, err: BaseException, label: str) -> bool:
        """Signal shutdown when `err` means the DuckDB instance is dead.

        Returns True when the caller should abandon its work: a fatally
        invalidated instance is process-terminal, so the only correct response
        is to let the scheduler restart us. Callers that would otherwise
        log-and-continue must consult this first, or they spin forever
        (REQ-RESIL-011).

        Before signalling, spool the failure for the next start (REQ-RESIL-014):
        the invalidated instance cannot record it, which is how 1,679 restarts
        went by with `last_failure_message: null`. Neither the spool nor the log
        line may preempt the signal (REQ-RESIL-020) -- both fail on a full disk,
        the very condition that caused the divergence.
        """
        if not is_fatal_db_invalidation(err):
            return False
        # Observable: this instance cannot hold it -- on a full disk both the
        # marker write and the stderr handler fail, and there is nothing left to
        # report to. Permanent -- the terminal signal outranks diagnostics; the
        # next start's self-repair carries the diagnosis instead.
        with suppress(Exception):
            self._remember_fatal_failure(err, label)
        with suppress(Exception):
            logger.error(
                "fatal DuckDB invalidation in %s; stopping daemon so scheduler can restart it",
                label,
                exc_info=(type(err), err, err.__traceback__),
            )
        self._shutdown_event.set()
        return True

    def _remember_fatal_failure(self, err: BaseException, label: str) -> None:
        from recall.services import self_repair

        self_repair.remember_fatal_failure(self._config.data_dir, err, site=label)

    def _note_disk_full(self, err: BaseException, label: str) -> None:
        """Flag index verification for the next open after ENOSPC (REQ-RESIL-019).

        A checkpoint or WAL write that ran out of space can leave rows outside
        their ART indexes without invalidating the instance; the next start's
        probe decides whether a rebuild is due.
        """
        if not is_disk_full_error(err):
            return
        from recall.services import self_repair

        self_repair.remember_disk_full(self._config.data_dir, conn=self._conn)
        logger.warning(
            "disk full in %s; index verification scheduled for the next daemon start", label
        )

    def _note_oom(self, err: BaseException, label: str) -> None:
        """Surface a recoverable DuckDB memory-limit OOM as an actionable hint.

        A recoverable OOM closes and reopens the shared connection; without
        this the only visible signal is a later 'database has been invalidated'
        fatal, which hides the real fix.
        """
        if not _is_oom_error(err):
            return
        from recall.db.connection import resolve_memory_limit

        config = getattr(self, "_config", None)
        limit = resolve_memory_limit(config) if config is not None else "unknown"
        logger.error(
            "out of memory in %s: DuckDB memory_limit (%s) exceeded; "
            "raise [duckdb] memory_limit or set RECALL_DUCKDB_MEMORY_LIMIT; "
            "limits above the default allowance also require "
            "[duckdb] memory_limit_exceeds_floor = true (may exceed the 4 GiB RSS floor)",
            label,
            limit,
        )

    def _recover_shared_conn(self, err: BaseException, label: str) -> None:
        """Restore the shared write connection after an exception.

        Recovery proves the postcondition with a health check: after this method
        returns, `_conn` is either a connection that answers `SELECT 1` or it is
        cleared so the next caller opens a fresh handle.

        A fatally invalidated instance is the one case this cannot deliver:
        DuckDB caches the instance per path per process, so the fresh handle
        reattaches to the same dead database. Escalate instead of reporting a
        recovery that did not happen.
        """
        if self._stop_on_fatal_db(err, label):
            return
        self._note_disk_full(err, label)
        self._note_oom(err, label)
        path = "close_clear"
        conn = None
        self._conn_lifecycle_gate.acquire_write()
        try:
            conn = self._conn
            if conn is None:
                return
            if not _is_oom_error(err):
                try:
                    conn.execute("ROLLBACK")
                    row = conn.execute("SELECT 1").fetchone()
                    if row == (1,):
                        path = "rollback_verified"
                        return
                except duckdb.TransactionException as rollback_err:
                    if "no transaction is active" in str(rollback_err):
                        try:
                            row = conn.execute("SELECT 1").fetchone()
                            if row == (1,):
                                path = "no_active_txn"
                                return
                        except Exception:
                            pass
                    path = "close_clear"
                except Exception:
                    path = "close_clear"

            with suppress(Exception):
                conn.close()
            if self._conn is conn:
                self._conn = None
        finally:
            logger.warning(
                'shared connection recovery label="%s" origin="%s: %s" path="%s"',
                label,
                type(err).__name__,
                err,
                path,
            )
            self._conn_lifecycle_gate.release_write()

    def close_db_connection(self) -> None:
        """Drain readers and retain exclusive ownership until maintenance reopens."""
        assert not self._conn_closed_for_maintenance
        self._conn_lifecycle_gate.acquire_write()
        try:
            if self._conn is not None:
                self._conn.close()
                self._conn = None
            # Retain ownership even when there was no shared handle. A read RPC
            # must not lazily open the file while maintenance attaches it.
            self._conn_closed_for_maintenance = True
        except BaseException:
            self._conn_lifecycle_gate.release_write()
            raise

    def reopen_db_connection(self) -> None:
        """Reopen the shared handle before releasing exclusive maintenance ownership."""
        if not self._conn_closed_for_maintenance:
            self._get_conn()
            return

        try:
            self._open_conn_unlocked()
        finally:
            self._conn_closed_for_maintenance = False
            self._conn_lifecycle_gate.release_write()

    @property
    def socket_path(self) -> Path:
        return self._socket_path

    @property
    def pid_path(self) -> Path:
        return self._pid_path

    def _get_embed_backend(self) -> Any:
        if self._embed_backend is not None:
            return self._embed_backend
        with self._embed_lock:
            if self._embed_backend is None:
                from recall.services.embeddings import get_backend

                self._embed_backend = get_backend(self._config.embedding)
            return self._embed_backend

    def _enrichment_model_executor(self) -> ThreadPoolExecutor:
        if self._enrichment_executor is None:
            self._enrichment_executor = DiagnosticExecutor(
                max_workers=1,
                thread_name_prefix="recall-enrichment",
                initializer=demote_current_thread,
            )
        return self._enrichment_executor

    def _note_source_activity(self, source_path: str) -> None:
        if (
            source_path not in self._activity_hints
            and len(self._activity_hints) >= _ACTIVITY_HINT_LIMIT
        ):
            self._activity_hints_saturated = True
            self._inventory_wakeup.set()
            return
        self._activity_hints[source_path] = time.monotonic()
        self._raw_wakeup.set()

    def _active_source_paths(self, config: AppConfig) -> set[str]:
        cutoff = time.monotonic() - config.daemon.live_idle_threshold
        expired = [
            path for path, observed_at in self._activity_hints.items() if observed_at < cutoff
        ]
        for path in expired:
            del self._activity_hints[path]
        if len(self._activity_hints) < _ACTIVITY_HINT_LIMIT:
            self._activity_hints_saturated = False
        active = set(self._activity_hints)
        if self._watch_runtime is not None:
            active.update(
                str(member.path)
                for member in self._watch_runtime.live_set.members()
                if member.last_event_at is not None
            )
        return active

    def _raw_candidate_admitted(
        self,
        source: SourceFile,
        *,
        is_active: bool,
        priority: RefusedSource | None = None,
    ) -> Admission:
        """Admit within WAL and preparation capacity, reserving active headroom.

        Checkpoint contention defers non-active work so maintenance can drain;
        the scheduler retains a refused slot while trying admissible lanes.
        Total shared occupancy after a borrow must leave room for a memory-owed
        candidate. Incompatible active work can pause until pre-existing claims
        drain; new claims cannot prolong that pause. Checkpoint deferral reserves
        no preparation bytes.
        One oversized claim may coexist with ordinary claims (REQ-RECON-024).
        """
        if self._wal_pressure_blocked:
            return Admission("wal_pressure")
        if self._checkpoint_contention_retries > 0 and not is_active:
            return Admission("checkpoint")
        history_busy = self._raw_historical_reserved_bytes > 0
        reservation = _RawReservation.for_source(source, historical=not is_active)
        shared_reserved = self._raw_reserved_bytes - self._raw_oversized_reserved_bytes
        if priority is not None and priority.reason == "preparation":
            owed = _RawReservation.for_source(
                priority.scheduled.source, historical=not priority.scheduled.is_active
            )
            if owed.oversized:
                if reservation.oversized:
                    return Admission("preparation")
            elif shared_reserved + reservation.bytes > _RAW_PREPARATION_BYTES - owed.bytes:
                return Admission("preparation")
        if reservation.oversized:
            # The reverse order already permits ordinary work beside an
            # oversized claim. Requiring an idle budget in this order would let
            # a stream of ordinary claims keep the owed source out forever.
            if self._raw_oversized_reserved_bytes > 0 or (not is_active and history_busy):
                return Admission("preparation")
            return Admission()
        # Oversized preparation owns separate capacity. Ordinary claims still
        # fit the shared budget, including while an oversized source prepares.
        if shared_reserved + reservation.bytes > _RAW_PREPARATION_BYTES:
            return Admission("preparation")
        if not is_active and history_busy:
            return Admission("preparation")
        return Admission()

    def _reserve_raw_preparation(self, source: SourceFile, *, historical: bool) -> _RawReservation:
        reservation = _RawReservation.for_source(source, historical=historical)
        if reservation.historical:
            assert self._raw_historical_reserved_bytes == 0
        if reservation.oversized:
            assert self._raw_oversized_reserved_bytes == 0
        shared_reserved = self._raw_reserved_bytes - self._raw_oversized_reserved_bytes
        assert (
            reservation.oversized or shared_reserved + reservation.bytes <= _RAW_PREPARATION_BYTES
        )
        self._raw_reserved_bytes += reservation.bytes
        if historical:
            self._raw_historical_reserved_bytes += reservation.bytes
        if reservation.oversized:
            self._raw_oversized_reserved_bytes = reservation.bytes
        # A source too large for the budget is the only claim that may exceed
        # it; every other claim together still fits inside it.
        assert (
            self._raw_reserved_bytes - self._raw_oversized_reserved_bytes <= _RAW_PREPARATION_BYTES
        )
        return reservation

    def _release_raw_preparation(self, reservation: _RawReservation) -> None:
        assert self._raw_reserved_bytes >= reservation.bytes
        if reservation.historical:
            assert self._raw_historical_reserved_bytes == reservation.bytes
        if reservation.oversized:
            assert self._raw_oversized_reserved_bytes == reservation.bytes
        self._raw_reserved_bytes -= reservation.bytes
        if reservation.historical:
            self._raw_historical_reserved_bytes -= reservation.bytes
        if reservation.oversized:
            self._raw_oversized_reserved_bytes = 0
        assert self._raw_reserved_bytes >= 0
        assert self._raw_historical_reserved_bytes >= 0
        assert self._raw_oversized_reserved_bytes >= 0

    def _note_raw_turn_served(self) -> None:
        """Record one served turn: the scheduler's own progress signal.

        A requested path's patience and the backlog drain rate an index summary
        reports both read this, so both describe the scheduler rather than the
        caller that happened to be looking.
        """
        self._raw_turns_served += 1
        self._raw_served_at.append(time.monotonic())

    def _recent_drain_per_minute(self) -> float | None:
        """Sources served in the last minute, or None when none has been served."""
        if not self._raw_served_at:
            return None
        cutoff = time.monotonic() - _DRAIN_RATE_WINDOW_SECONDS
        return float(sum(1 for served_at in self._raw_served_at if served_at >= cutoff))

    def _load_runtime_config(self) -> AppConfig:
        """Refresh config fields that affect daemon-cycle indexing output.

        The daemon's socket, pid file, and database paths are startup-time
        concerns. Contextual retrieval is a write-time concern, so long-lived
        watch/embed loops must pick up `[embedding.context]` edits without
        moving the running daemon to a different database or socket.
        """
        loaded = AppConfig.load()
        context_env_present = any(name in os.environ for name in _CONTEXT_ENV_VARS)
        if not context_env_present and (
            not self._config.config_path.exists() or loaded.config_path != self._config.config_path
        ):
            cfg = self._config
        else:
            cfg = replace(
                self._config,
                embedding=replace(self._config.embedding, context=loaded.embedding.context),
            )
        fp = config_fingerprint(cfg)
        if fp != self._config_fp:
            logger.info("daemon runtime config changed; reloaded embedding context")
            self._config = cfg
            self._config_fp = fp
        return cfg

    async def _harvest_usage_once(
        self, path: Path | None, host: str | None, label: str
    ) -> HarvestResult | None:
        """Harvest one Grok usage batch (REQ-USAGE-010, REQ-USAGE-017).

        The log is read and parsed on a read cursor; only the parsed batch is
        committed in a writer turn. A harvest failure must not abort the watch
        tick or the index run that asked for it, so ordinary failures are
        logged and return None. A dead database stops the daemon.
        """
        from recall.services.usage_harvest import commit_grok_usage_batch, read_grok_usage_batch

        try:
            batch = await self._run_readonly(lambda conn: read_grok_usage_batch(conn, path))
        except asyncio.CancelledError:
            raise
        except Exception as err:
            self._fail_soft_usage_harvest(err)
            return None
        if batch is None:
            return None
        try:
            return await self._writer_call(
                lambda: commit_grok_usage_batch(self._get_conn(), batch, host=host), label
            )
        except asyncio.CancelledError:
            raise
        except duckdb.Error as err:
            # The writer turn already recovered the connection, or stopped the
            # daemon when the database is dead.
            if not self._shutdown_event.is_set():
                logger.warning("usage harvest failed: %s", err, exc_info=True)
            return None
        except Exception as err:
            self._fail_soft_usage_harvest(err)
            return None

    def _fail_soft_usage_harvest(self, err: Exception) -> None:
        if self._stop_on_fatal_db(err, "usage_harvest"):
            return
        self._note_disk_full(err, "usage_harvest")
        # Traceback included: `str(err)` alone hid which statement produced
        # `tuple index out of range` on two fleet hosts (REQ-RESIL-023).
        logger.warning("usage harvest failed: %s", err, exc_info=True)

    async def _harvest_usage_until_caught_up(
        self, path: Path | None, host: str | None, label: str
    ) -> None:
        """Harvest all log present at the start, one writer turn per batch.

        Other writers take turns between batches. Appends made during the
        drain may remain for the next watch tick or index run.
        """
        from recall.services.usage_harvest import harvest_calls_to_cover

        result = await self._harvest_usage_once(path, host, label)
        if result is None:
            return
        for _ in range(harvest_calls_to_cover(result.log_size_bytes) - 1):
            if result is None or result.caught_up:
                return
            result = await self._harvest_usage_once(path, host, label)

    def _register_method(self, name: str, schema: type[Params], handler: MethodHandler) -> None:
        async def validated(params: dict[str, Any], client: ClientConnection | None) -> Any:
            admitted_at = time.monotonic()
            try:
                parsed = schema.model_validate(params)
            except ValidationError as err:
                messages = []
                for error in err.errors(include_input=False, include_url=False):
                    parameter = ".".join(str(part) for part in error["loc"])
                    if error["type"] == "missing":
                        messages.append(f"{parameter} is required")
                    else:
                        messages.append(f"{parameter}: {error['msg']}")
                raise RpcError(code=INVALID_PARAMS, message="; ".join(messages)) from err
            token = None
            if name in RESPONSIVE_READ_METHODS:
                assert isinstance(parsed, ResponsiveReadParams)
                budget = ReadBudget(admitted_at + parsed.remaining_read_ms / 1000)
                token = SERVER_READ_BUDGET.set(budget)
            try:
                return await handler(parsed.model_dump(exclude_unset=True), client)
            except UnknownSessionError as err:
                raise RpcError(code=APP_NOT_FOUND, message=str(err)) from err
            finally:
                if token is not None:
                    SERVER_READ_BUDGET.reset(token)

        self._methods[name] = validated

    def _register_methods(self) -> None:
        self._register_method("recall.index", IndexParams, self._handle_index)
        self._register_method("recall.search", SearchParams, self._handle_search)
        self._register_method("recall.list", ListParams, self._handle_list)
        self._register_method("recall.list_by_session", ListBySessionParams, self._handle_list)
        self._register_method("recall.show", ShowParams, self._handle_show)
        self._register_method("recall.stats", Params, self._handle_stats)
        self._register_method("recall.stats_tools", Params, self._handle_stats_tools)
        self._register_method("recall.stats_bash", BashStatsParams, self._handle_stats_bash)
        self._register_method("recall.stats_tokens", Params, self._handle_stats_tokens)
        self._register_method("recall.stats_usage", UsageStatsParams, self._handle_stats_usage)
        self._register_method("recall.stats_skills", SkillStatsParams, self._handle_stats_skills)
        self._register_method("recall.daemon_status", StatusParams, self._handle_daemon_status)
        self._register_method("recall.daemon_pause", Params, self._handle_daemon_pause)
        self._register_method("recall.daemon_resume", Params, self._handle_daemon_resume)
        self._register_method("recall.migrate_storage", StorageParams, self._handle_migrate_storage)
        self._register_method(
            "recall._set_installed_scheduler", SchedulerParams, self._handle_set_installed_scheduler
        )
        self._register_method("recall.daemon_run", DaemonRunParams, self._handle_daemon_run)
        self._register_method(
            "recall.check_indexes", CheckIndexesParams, self._handle_check_indexes
        )
        self._register_method("recall.live_sessions", Params, self._handle_live_sessions)
        self._register_method("recall.live", LiveParams, self._handle_live)
        self._register_method("recall.live_by_session", LiveBySessionParams, self._handle_live)
        self._register_method("recall.show_follow", FollowParams, self._handle_show_follow)
        self._register_method("recall.live_mark", LiveMarkParams, self._handle_live_mark)

    # ---- Write handlers ----

    async def _handle_index(self, params: dict[str, Any], client: ClientConnection | None) -> Any:
        from recall.core.embeddings import embedding_backend_available
        from recall.core.time import parse_since
        from recall.services.coordinator import IndexRequestScope, config_with_home_root
        from recall.services.indexer import _resolve_worker_count

        runtime_config = self._load_runtime_config()
        source_value = params.get("source")
        source = parse_source(source_value) if source_value else None
        full = bool(params.get("full", False))
        recreate = bool(params.get("recreate", False))
        recompute_context = bool(params.get("recompute_context", False))
        since_value = params.get("since")
        since = parse_since(str(since_value)) if since_value else None
        only_mode_value = params.get("only_mode")
        only_mode = str(only_mode_value) if only_mode_value is not None else None
        project_value = params.get("project")
        project = str(project_value) if project_value is not None else None
        root_value = params.get("root")
        home_root = Path(str(root_value)).expanduser() if root_value else None
        host_value = params.get("host")
        host = str(host_value) if host_value is not None else None
        if home_root is not None and host is None:
            host = home_root.resolve().name or "remote"

        if recreate and not params.get("confirmed", False):
            raise RpcError(
                code=APP_CONFIRMATION_REQUIRED,
                message="recreate requires confirmed: true",
            )

        embed_value = params.get("embed")
        if embed_value is None:
            embed = embedding_backend_available(runtime_config.embedding.backend)
        else:
            embed = bool(embed_value)

        workers_raw = params.get("workers", "auto")
        try:
            workers: int | str = int(workers_raw)
        except (ValueError, TypeError):
            workers = str(workers_raw)

        _resolve_worker_count(workers, 1)
        cfg = _config_with_context_mode(runtime_config, params.get("context"))
        if recreate and (recompute_context or project):
            raise RpcError(
                code=INVALID_PARAMS,
                message="recreate cannot be combined with recompute_context or project",
            )
        if recompute_context:
            return await self._recompute_context_request(
                config=cfg, since=since, only_mode=only_mode, embed=embed, client=client
            )
        return await self._manual_index_request(
            config=config_with_home_root(cfg, home_root),
            source=source,
            client=client,
            scope=IndexRequestScope(
                full=full or recreate,
                recreate=recreate,
                since=since,
                project=project,
                home_root=home_root,
                host=host,
                embed=embed,
                context=params.get("context"),
            ),
        )

    def _reset_database(self, config: AppConfig) -> str:
        from recall.services.recreation import backup_before_recreate, replace_schema

        self._conn_lifecycle_gate.acquire_write()
        try:
            self._require_reconciliation(config)
            self._recreate_backup = None
            conn = self._open_conn_unlocked()
            backup = backup_before_recreate(conn, config)
            self._recreate_backup = str(backup)
            logger.warning("consistent pre-recreate backup retained at %s", backup)
            self._require_reconciliation(config)
            self._database_epoch += 1
            self._inventory_complete = False
            self._fts_repair_cursor = None
            try:
                replace_schema(conn, config)
            except Exception as err:
                logger.error("recreate failed; backup retained at %s: %s", backup, err)
                raise
            self._fts_repair_error = None
            self._keyword_dirty = False
            return str(backup)
        finally:
            self._conn_lifecycle_gate.release_write()

    async def _generate_requested_enrichment(
        self,
        config: AppConfig,
        prepare: Callable[[], PreparedEmbedCycle],
    ) -> PreparedEmbedCycle:
        """Own models across snapshot, detached generation and generation-checked publication."""
        from recall.services.embed_phase import (
            EmbedPhaseState,
            generate_prepared_embed_cycle,
            publish_prepared_embed_cycle,
        )
        from recall.services.enrichment_inputs import EnrichmentCapacityError

        async with self._enrichment_lock:
            self._require_reconciliation(config)
            if self._embed_state is None:
                self._embed_state = EmbedPhaseState(_config=config)
            state: EmbedPhaseState = self._embed_state
            # A requested cycle commits through the same phase state as the
            # timer loop, so it records the same trigger, outcome and batch.
            # Without this, the batch fields advance on every indexed session
            # while the loop fields keep reporting the timer's last verdict.
            # Its cycle count and stage stay its own: the timer is asleep in a
            # stage of its own, which this must neither age nor overwrite, and
            # one enriched session is not one timer iteration (REQ-ADAPT-012).
            state.begin_requested_cycle()
            try:
                state.enter_requested_stage("snapshot")
                prepared = await self._enrichment_writer_call(
                    prepare, "requested enrichment snapshot"
                )
                state.enter_requested_stage("configure")
                await self._await_shared_conn_work(
                    asyncio.get_running_loop().run_in_executor(
                        self._enrichment_model_executor(), state.configure, config
                    ),
                    "requested model configuration",
                )
                started = time.monotonic()
                state.enter_requested_stage("generate")
                prepared = await self._await_shared_conn_work(
                    asyncio.get_running_loop().run_in_executor(
                        self._enrichment_model_executor(),
                        lambda: generate_prepared_embed_cycle(
                            config, state, prepared, should_stop=self._shutdown_event.is_set
                        ),
                    ),
                    "requested enrichment generation",
                )
                state.last_error = prepared.error
                if prepared.error is not None:
                    state.record_requested_outcome("generation-error")
                    raise RuntimeError(prepared.error)
                try:
                    self._require_reconciliation(config)
                except RpcError:
                    state.record_requested_outcome("paused")
                    raise
                state.enter_requested_stage("commit")
                committed = await self._enrichment_writer_call(
                    lambda: publish_prepared_embed_cycle(prepared, config, conn=self._get_conn()),
                    "requested enrichment commit",
                )
                if prepared.searchable_changed and config.fts.backend == "duckdb":
                    self._keyword_dirty = True
                if prepared.committed_session_ids != prepared.generated_session_ids:
                    state.record_requested_outcome("stale-inputs")
                    raise RpcError(
                        code=APP_LOCKED,
                        message=(
                            "session inputs changed during context generation; retry the request"
                        ),
                    )
                state.record_batch(committed if prepared.embed else 0, time.monotonic() - started)
                state.record_requested_outcome("drained" if committed else "no-progress")
                return prepared
            except EnrichmentCapacityError as err:
                state.last_error = f"{type(err).__name__}: {err}"
                state.record_requested_outcome("capacity-error")
                raise
            finally:
                state.close_requested_cycle()

    async def _enrich_indexed_session(
        self,
        config: AppConfig,
        session_id: str,
        *,
        embed: bool,
    ) -> SessionContextStats:
        """Drain a frozen requested roster through admitted publication units."""
        from uuid import uuid4

        from recall.services.embed_phase import (
            PendingEmbeds,
            PreparedEmbedCycle,
            prepare_target_enrichment,
        )
        from recall.services.enrichment_inputs import (
            ITEM_BYTES_MAX,
            TARGET_ROWS_MAX,
            EnrichmentCapacityError,
            EnrichmentTargets,
            admit_targets,
            oversized_targets,
            selected_predicate,
        )
        from recall.services.indexer import SessionContextStats

        table = "_recall_enrich_" + uuid4().hex
        created = False
        context_messages = input_tokens = output_tokens = 0
        model = None
        try:

            def capture():
                conn = self._get_conn()
                conn.execute(
                    f"""CREATE TEMP TABLE {table} AS
                    SELECT 'message' AS kind, m.id, m.idx, false AS blocked FROM messages m
                    WHERE m.session_id = ? AND NOT EXISTS (
                        SELECT 1 FROM message_embeddings e WHERE e.message_id = m.id
                    ) AND (
                        ? OR ?
                    )
                    UNION ALL
                    SELECT 'tool', t.id, t.idx, false FROM tool_calls t
                    WHERE t.session_id = ? AND ? AND NOT EXISTS (
                        SELECT 1 FROM tool_call_embeddings e WHERE e.tool_call_id = t.id
                    )""",
                    [
                        session_id,
                        embed,
                        config.embedding.context.mode.startswith("llm-"),
                        session_id,
                        embed,
                    ],
                )
                return int(conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])

            created = True
            total = await self._writer_call(capture, "requested enrichment roster")
            completed = 0
            while completed < total:

                def prepare():
                    conn = self._get_conn()
                    rows = conn.execute(
                        f"SELECT kind, id FROM {table} WHERE NOT blocked "
                        "ORDER BY kind, idx, id LIMIT ?",
                        [TARGET_ROWS_MAX],
                    ).fetchall()
                    if not rows:
                        raise EnrichmentCapacityError(
                            f"session {session_id} contains inputs above the {ITEM_BYTES_MAX} byte "
                            "capacity; admissible work was processed "
                            "and oversized inputs remain pending"
                        )
                    candidates = EnrichmentTargets(
                        tuple(str(row[1]) for row in rows if row[0] == "message"),
                        tuple(str(row[1]) for row in rows if row[0] == "tool"),
                    )
                    blocked = oversized_targets(conn, candidates)
                    for kind, ids in (("message", blocked.message_ids), ("tool", blocked.tool_ids)):
                        predicate, params = selected_predicate("id", ids)
                        conn.execute(
                            f"UPDATE {table} SET blocked=true WHERE kind=? AND {predicate}",
                            [kind, *params],
                        )
                    admitted = EnrichmentTargets(
                        tuple(
                            identifier
                            for identifier in candidates.message_ids
                            if identifier not in blocked.message_ids
                        ),
                        tuple(
                            identifier
                            for identifier in candidates.tool_ids
                            if identifier not in blocked.tool_ids
                        ),
                    )
                    if admitted.count == 0:
                        return PreparedEmbedCycle(
                            pending=PendingEmbeds(0, 0, []),
                            sessions=[],
                            embed=embed,
                            selection_scanning=True,
                        )
                    selected = admit_targets(conn, admitted)
                    assert selected.count > 0
                    item = prepare_target_enrichment(config, session_id, selected, conn=conn)
                    return PreparedEmbedCycle(
                        pending=PendingEmbeds(
                            len(selected.message_ids), len(selected.tool_ids), [session_id]
                        ),
                        sessions=[item],
                        embed=embed,
                    )

                prepared = await self._generate_requested_enrichment(config, prepare)
                if not prepared.sessions:
                    continue
                targets = prepared.sessions[0].targets

                def acknowledge(targets=targets):
                    conn = self._get_conn()
                    for kind, ids in (("message", targets.message_ids), ("tool", targets.tool_ids)):
                        predicate, params = selected_predicate("id", ids)
                        conn.execute(
                            f"DELETE FROM {table} WHERE kind = ? AND {predicate}", [kind, *params]
                        )

                await self._writer_call(acknowledge, "requested enrichment completion")
                completed += targets.count
                context_messages += prepared.context_messages
                input_tokens += prepared.context_input_tokens
                output_tokens += prepared.context_output_tokens
                model = prepared.context_model or model
            return SessionContextStats(context_messages, input_tokens, output_tokens, model)
        finally:
            if created:
                await self._writer_call(
                    lambda: self._get_conn().execute(f"DROP TABLE IF EXISTS {table}"),
                    "requested enrichment roster release",
                )

    async def _recompute_context_request(
        self,
        *,
        config: AppConfig,
        since: datetime | None,
        only_mode: str | None,
        embed: bool,
        client: ClientConnection | None,
    ) -> IndexSummary:
        """Freeze target ids in DuckDB and publish admitted recomputation units."""
        from uuid import uuid4

        from recall.core.daemon_status import IndexRunCounts
        from recall.core.types import RunKind
        from recall.services.embed_phase import (
            PendingEmbeds,
            PreparedEmbedCycle,
            prepare_context_recompute,
        )
        from recall.services.enrichment_inputs import (
            EnrichmentCapacityError,
            EnrichmentTargets,
            oversized_targets,
            selected_predicate,
        )
        from recall.services.indexer import IndexSummary
        from recall.services.runtime_state import (
            record_run_attempt,
            record_run_failure,
            record_run_success,
        )

        if only_mode is not None and only_mode not in {
            "off",
            "template",
            "llm-local",
            "llm-remote",
            "llm-codex",
        }:
            raise RpcError(code=INVALID_PARAMS, message=f"unsupported context mode: {only_mode}")
        table = "_recall_recompute_" + uuid4().hex
        async with self._index_turn():
            self._require_reconciliation(config)
            attempted_at = await self._runtime_write(
                lambda: record_run_attempt(self._get_conn(), run_kind=RunKind.INDEX),
                "recompute_record_attempt",
            )
            started = time.monotonic()
            created = False
            try:

                def capture_targets():
                    conn = self._get_conn()
                    # The database owns the frozen roster. Only one session's ids
                    # and document enter Python at once, even for a full corpus.
                    conn.execute(
                        f"""CREATE TEMP TABLE {table} AS
                        SELECT m.session_id, m.id, m.idx, false AS blocked
                        FROM messages m JOIN message_state ms ON ms.message_id = m.id
                        JOIN session_state ss ON ss.session_id = m.session_id
                        WHERE (ms.content IS NOT NULL OR ms.thinking IS NOT NULL)
                          AND (? IS NULL OR
                               COALESCE(ss.ended_at, ss.started_at, ss.indexed_at) >= ?)
                          AND (? IS NULL OR COALESCE(ms.context_mode, 'off') = ?)
                    """,
                        [since, since, only_mode, only_mode],
                    )
                    row = conn.execute(
                        f"SELECT COUNT(DISTINCT session_id), COUNT(*) FROM {table}"
                    ).fetchone()
                    assert row is not None
                    return int(row[0]), int(row[1])

                created = True
                _, total = await self._enrichment_writer_call(
                    capture_targets, "recompute target selection"
                )
                completed = context_messages = input_tokens = output_tokens = 0
                model = None
                while completed < total:

                    def prepare():
                        conn = self._get_conn()
                        rows = conn.execute(
                            f"SELECT session_id, id FROM {table} WHERE NOT blocked "
                            "AND session_id = "
                            f"(SELECT session_id FROM {table} WHERE NOT blocked "
                            "ORDER BY session_id LIMIT 1) "
                            "ORDER BY idx, id LIMIT 128",
                        ).fetchall()
                        if not rows:
                            raise EnrichmentCapacityError(
                                "recompute targets exceed item capacity; "
                                "admissible targets were processed "
                                "and oversized inputs remain pending"
                            )
                        session_id, targets = str(rows[0][0]), tuple(str(row[1]) for row in rows)
                        blocked = oversized_targets(conn, EnrichmentTargets(targets, ()))
                        predicate, params = selected_predicate("id", blocked.message_ids)
                        conn.execute(f"UPDATE {table} SET blocked=true WHERE {predicate}", params)
                        admitted = tuple(
                            identifier
                            for identifier in targets
                            if identifier not in blocked.message_ids
                        )
                        if not admitted:
                            return PreparedEmbedCycle(
                                pending=PendingEmbeds(0, 0, []),
                                sessions=[],
                                embed=embed,
                                selection_scanning=True,
                            )
                        return prepare_context_recompute(
                            session_id,
                            admitted,
                            conn=conn,
                            embed=embed,
                            only_mode=only_mode,
                            config=config,
                        )

                    prepared = await self._generate_requested_enrichment(config, prepare)
                    if not prepared.sessions:
                        continue
                    selected_ids = prepared.sessions[0].targets.message_ids
                    completed += len(selected_ids)
                    context_messages += prepared.context_messages
                    input_tokens += prepared.context_input_tokens
                    output_tokens += prepared.context_output_tokens
                    model = prepared.context_model or model
                    await self._writer_call(
                        lambda selected_ids=selected_ids: self._get_conn().execute(
                            f"DELETE FROM {table} WHERE id IN (SELECT UNNEST(?::TEXT[]))",
                            [list(selected_ids)],
                        ),
                        "recompute target completion",
                    )
                    if client is not None:
                        await client.send_progress(
                            processed=completed, total=total, status="indexed"
                        )
                await self._init_fts(reason="context recompute")
                summary = IndexSummary(
                    total=total,
                    indexed=0,
                    skipped=0,
                    failed=0,
                    changed=completed,
                    fts_rebuilt=bool(config.fts.fields) and completed > 0,
                    total_seconds=time.monotonic() - started,
                    context_messages=context_messages,
                    context_mode=config.embedding.context.mode,
                    context_input_tokens=input_tokens,
                    context_output_tokens=output_tokens,
                    context_model=model,
                )
                await self._runtime_write(
                    lambda: record_run_success(
                        self._get_conn(),
                        run_kind=RunKind.INDEX,
                        attempted_at=attempted_at,
                        index_summary=IndexRunCounts(
                            total=summary.total,
                            indexed=0,
                            skipped=0,
                            failed=0,
                            changed=summary.changed,
                            total_seconds=summary.total_seconds,
                        ),
                        last_context_messages=summary.context_messages,
                        last_context_mode=summary.context_mode,
                        last_context_input_tokens=summary.context_input_tokens,
                        last_context_output_tokens=summary.context_output_tokens,
                        last_context_model=summary.context_model,
                    ),
                    "recompute_record_success",
                )
                return summary
            except Exception as err:
                try:
                    await self._runtime_write(
                        lambda err=err: record_run_failure(
                            self._get_conn(),
                            run_kind=RunKind.INDEX,
                            attempted_at=attempted_at,
                            message=str(err),
                        ),
                        "recompute_record_failure",
                    )
                except Exception:
                    logger.exception("context recompute could not persist failure metadata")
                raise
            finally:
                if created:
                    await self._writer_call(
                        lambda: self._get_conn().execute(f"DROP TABLE IF EXISTS {table}"),
                        "recompute target cleanup",
                    )

    def _require_reconciliation(self, config: AppConfig) -> None:
        """Refuse with the condition that actually holds, never with both.

        "paused or shutting down" named a state the operator could not act on:
        a shutdown-induced refusal read as a pause they had to undo, and
        `reconciliation.paused` said false the whole time.
        """
        from recall.services.coordinator import is_paused

        if is_paused(config):
            raise RpcError(code=APP_LOCKED, message=PAUSED_MESSAGE)
        if self._shutdown_event.is_set():
            raise RpcError(code=APP_LOCKED, message=SHUTTING_DOWN_MESSAGE)

    def _index_turn(self) -> IndexTurn:
        """Serialise operator index requests and make the drain yield to them."""
        return self._index_turns.turn(self._index_request_lock)

    async def _runtime_write(self, operation: Callable[[], Any], label: str) -> Any:
        """Retry one idempotent runtime-metadata write after writer recovery."""
        try:
            return await self._writer_call(operation, label)
        except duckdb.Error:
            return await self._writer_call(operation, label)

    async def _record_departed_run(self, run_kind: RunKind, attempted_at: datetime) -> None:
        """Close out a run cancelled because its client left.

        The departure answers nobody (`REQ-RPC-018`), so runtime state is the
        only place it can be read afterwards -- and an attempt with no outcome
        leaves `last_attempted_at` newer than every completion with no message
        to explain it, which reads as a run that started and never ended.
        """
        from recall.services.runtime_state import record_run_failure

        try:
            await self._runtime_write(
                lambda: record_run_failure(
                    self._get_conn(),
                    run_kind=run_kind,
                    attempted_at=attempted_at,
                    message="cancelled: the client that asked for it departed",
                ),
                "departed run failure",
            )
        except Exception:
            logger.exception("could not record the departure of a cancelled run")

    async def _manual_index_request(
        self,
        *,
        config: AppConfig,
        source: Source | None,
        client: ClientConnection | None,
        scope: IndexRequestScope,
    ) -> IndexSummary:
        from recall.core.daemon_status import IndexRunCounts
        from recall.core.types import RunKind
        from recall.services.runtime_state import (
            record_run_attempt,
            record_run_failure,
            record_run_success,
        )

        async with self._index_turn():
            self._require_reconciliation(config)
            backup = None
            if scope.recreate:
                async with self._raw_turn_lock, self._enrichment_lock:
                    try:
                        backup = await self._writer_call(
                            lambda: self._reset_database(config), "index recreate"
                        )
                    except Exception as err:
                        if self._recreate_backup is None:
                            raise
                        raise RpcError(
                            code=err.code if isinstance(err, RpcError) else INTERNAL_ERROR,
                            message=f"recreate failed; backup: {self._recreate_backup}; {err}",
                            data={"backup_path": self._recreate_backup},
                        ) from err
            attempted = await self._runtime_write(
                lambda: record_run_attempt(self._get_conn(), run_kind=RunKind.INDEX),
                "manual index attempt",
            )
            try:
                summary = await self._reconcile_index_request(
                    config=config,
                    source=source,
                    client=client,
                    scope=scope,
                )
            except asyncio.CancelledError:
                await self._record_departed_run(RunKind.INDEX, attempted)
                raise
            except Exception as err:
                failure = f"{err}; pre-recreate backup: {backup}" if backup else str(err)
                await self._runtime_write(
                    lambda message=failure: record_run_failure(
                        self._get_conn(),
                        run_kind=RunKind.INDEX,
                        attempted_at=attempted,
                        message=message,
                    ),
                    "manual index failure",
                )
                if backup:
                    raise RpcError(
                        code=err.code if isinstance(err, RpcError) else INTERNAL_ERROR,
                        message=failure,
                        data={"backup_path": backup},
                    ) from err
                raise
            summary = replace(summary, backup_path=backup)
            await self._runtime_write(
                lambda: record_run_success(
                    self._get_conn(),
                    run_kind=RunKind.INDEX,
                    attempted_at=attempted,
                    index_summary=IndexRunCounts(
                        total=summary.total,
                        indexed=summary.indexed,
                        skipped=summary.skipped,
                        failed=summary.failed,
                        changed=summary.changed,
                        total_seconds=summary.total_seconds,
                    ),
                    last_context_messages=summary.context_messages,
                    last_context_mode=summary.context_mode,
                    last_context_input_tokens=summary.context_input_tokens,
                    last_context_output_tokens=summary.context_output_tokens,
                    last_context_model=summary.context_model,
                ),
                "manual index success",
            )
            return summary

    async def _serve_with_keepalive(
        self,
        work: Coroutine[Any, Any, None],
        keepalive: Callable[[], Coroutine[Any, Any, None]],
    ) -> None:
        """Serve one source, telling an attached client it is still being served.

        A client's timeout bounds *silence* between frames, and progress is
        coalesced per captured batch: a single 143-301 MiB codex rollout takes
        longer to serve than that bound, so the operator's `recall index`
        failed on a daemon that was working (`REQ-RPC-018`).
        """
        served = asyncio.ensure_future(work)
        try:
            while True:
                done, _pending = await asyncio.wait({served}, timeout=_PROGRESS_KEEPALIVE_SECONDS)
                if done:
                    break
                await keepalive()
        except asyncio.CancelledError:
            served.cancel()
            raise
        await served

    async def _reconcile_index_request(
        self,
        *,
        config: AppConfig,
        source: Source | None,
        client: ClientConnection | None,
        scope: IndexRequestScope | None = None,
    ) -> IndexSummary:
        """Inventory bounded batches and serve each path through the shared queue."""
        from recall.db.source_files import PresentSource, SourceCatalog
        from recall.services.coordinator import (
            RawIndexRequest,
            persist_prepared_inventory,
            prepare_raw_cycle,
            reconciliation_status,
        )
        from recall.services.indexer import IndexSummary
        from recall.services.reconciler import InventoryBatch, is_observed, unobserved
        from recall.services.watcher import _lightweight_watch_context

        started = time.monotonic()
        total = indexed = skipped = failed = 0
        context_messages = context_input_tokens = context_output_tokens = 0
        context_reused = 0
        context_model = None
        events = prepare_raw_cycle(config, source=source)
        generation = None
        partial = scope is not None and (scope.since is not None or scope.project is not None)
        # A plain incremental pass waits for what its own observation made
        # pending and hands the pre-existing backlog to the shared drain: on a
        # 6,515-source backlog it otherwise blocked for the whole drain to
        # report work the daemon was already doing (REQ-RECON-025).
        hands_off_backlog = scope is not None and not scope.owns_every_observed_source()
        # `--full`/`--recreate` reconcile a source whether or not anything about
        # it changed, so those are the only requests that must still reach the
        # writer for an unchanged path (`REQ-INDEX-023`).
        forces_work = scope is not None and scope.full
        # A request carrying per-path options must observe each path together
        # with them, or background work races ahead with a different host.
        # Every other request commits one bounded batch at a time instead.
        batched = scope is None or not scope.owns_every_observed_source()
        # The walked root's catalog, read once per root. Comparing each captured
        # batch against it in memory is what keeps an unchanged corpus off the
        # writer entirely, and it is also the source's state *before* this
        # request touched it, which is what decides the backlog handoff
        # (`REQ-RECON-025`).
        present: dict[str, PresentSource] = {}
        last_frame = time.monotonic()

        async def send_frame(
            status: str, *, path: str | None = None, coalesce: bool = False
        ) -> None:
            """Report progress, never once per unchanged file (`REQ-INDEX-024`).

            Phase transitions, one update per captured inventory batch and the
            terminal update are sent as they happen. Everything a served file
            would say is coalesced to the same bound that times a client out,
            so a long run of served files still refreshes the socket without
            turning progress back into a per-file stream.
            """
            nonlocal last_frame
            if client is None:
                return
            now = time.monotonic()
            if coalesce and now - last_frame < _PROGRESS_KEEPALIVE_SECONDS:
                return
            last_frame = now
            extra = {"path": path} if path is not None else {}
            await client.send_progress(
                processed=total,
                total=total,
                status=status,
                indexed=indexed,
                skipped=skipped,
                failed=failed,
                inventory_complete=status == "done",
                **extra,
            )

        loop = asyncio.get_running_loop()
        while True:
            self._require_reconciliation(config)
            event = await self._await_shared_conn_work(
                loop.run_in_executor(self._executor, next, events, None),
                "requested inventory capture",
            )
            if event is None:
                break
            # A scoped request can filter out every file in every batch: a
            # `--since` run over a large corpus otherwise stayed silent past the
            # client's idle timeout while the daemon was working.
            await send_frame("scanning")
            # A partial request cannot establish absence elsewhere in the root,
            # so it persists neither its scan nor its batches.
            if not partial and (batched or not isinstance(event.event, InventoryBatch)):
                commit = event
                if isinstance(event.event, InventoryBatch):
                    # Only what the root's catalog does not already hold reaches
                    # the writer, so an unchanged batch costs no writer turn.
                    commit = replace(event, event=unobserved(event.event, present))
                if not isinstance(commit.event, InventoryBatch) or commit.event.files:
                    generation, _ = await self._writer_call(
                        lambda commit=commit, generation=generation: persist_prepared_inventory(
                            commit,
                            generation,
                            conn=self._get_conn(),
                            on_enrichment_change=self._invalidate_moved_history,
                        ),
                        "requested inventory commit",
                        affects_enrichment=False,
                    )
            if event.event is None:
                # One read per root, taken before any batch of it is compared.
                present = await self._run_readonly(
                    lambda conn, source=event.parser.source.value, root=str(event.root): (
                        SourceCatalog(conn, clock=time.time).present_states(source, root)
                    )
                )
            if not isinstance(event.event, InventoryBatch):
                continue
            files = event.event.files
            if scope is not None and scope.since is not None:
                since_ns = int(scope.since.timestamp() * 1_000_000_000)
                files = tuple(item for item in files if item.signature.mtime_ns >= since_ns)
            if scope is not None and scope.project is not None and files:
                paths = [item.source_path for item in files]
                project = scope.project
                matches = await self._run_readonly(
                    lambda conn, paths=paths, project=project, source=event.parser.source.value: (
                        conn.execute(
                            "SELECT s.source_path FROM sessions s JOIN session_state ss "
                            "ON ss.session_id = s.id WHERE s.source_path IN (SELECT UNNEST(?)) "
                            "AND s.source = ? AND ss.git_repo ILIKE ?",
                            [paths, source, f"%{project}%"],
                        ).fetchall()
                    )
                )
                matching_paths = {str(row[0]) for row in matches}
                files = tuple(item for item in files if item.source_path in matching_paths)
            for captured in files:
                total += 1
                known = present.get(captured.source_path)
                if (
                    not forces_work
                    and known is not None
                    and known.current
                    and is_observed(captured, present)
                ):
                    # The whole corpus, on a pass where nothing changed. One
                    # in-memory comparison against the root's single catalog
                    # read replaces a writer turn and two lookups per file
                    # (`REQ-INDEX-023`).
                    skipped += 1
                    continue
                raw_request = None
                # Pending before this request observed it, so its service is
                # reconciliation's rather than this pass's (`REQ-RECON-025`).
                backlogged = hands_off_backlog and known is not None and not known.current
                if scope is not None:
                    raw_request = RawIndexRequest(
                        config=config,
                        parser=event.parser,
                        full=scope.full,
                        host=scope.host,
                    )
                    await self._serve_with_keepalive(
                        self.index_session_now(
                            Path(captured.source_path),
                            request=raw_request,
                            await_service=not backlogged,
                        ),
                        lambda path=captured.source_path: send_frame("serving", path=path),
                    )
                    changed = raw_request.changed
                else:
                    changed = True
                    await self._serve_with_keepalive(
                        self.index_session_now(Path(captured.source_path), source=captured.source),
                        lambda path=captured.source_path: send_frame("serving", path=path),
                    )
                self._require_reconciliation(config)
                current = await self._run_readonly(
                    lambda conn, captured=captured: SourceCatalog(conn, clock=time.time).get(
                        captured.source, captured.source_path
                    )
                )
                if (
                    current is None
                    or current.missing
                    or (not backlogged and current.last_error not in (None, "unsupported"))
                ):
                    failed += 1
                    status = "failed"
                elif not current.current:
                    # Settling the captured generation does not settle later
                    # bytes. The shared drain owns those and pre-existing
                    # backlog; neither is this pass's failure (REQ-RECON-025).
                    status = "pending"
                elif not changed:
                    skipped += 1
                    status = "skipped"
                else:
                    indexed += 1
                    status = "indexed"
                    if raw_request is not None:
                        context_reused += raw_request.context.reused
                        context_messages += (
                            raw_request.context.reused
                            if config.embedding.context.mode.startswith("llm-")
                            else raw_request.context.messages
                        )
                    if scope is not None and (
                        scope.embed or config.embedding.context.mode.startswith("llm-")
                    ):
                        row = await self._run_readonly(
                            lambda conn, captured=captured: conn.execute(
                                "SELECT session_id FROM source_files "
                                "WHERE source = ? AND source_path = ?",
                                [captured.source, captured.source_path],
                            ).fetchone()
                        )
                        assert row is not None and row[0] is not None
                        session_id = str(row[0])
                        counts = await self._enrich_indexed_session(
                            config, session_id, embed=scope.embed
                        )
                        context_messages += counts.messages
                        context_input_tokens += counts.input_tokens
                        context_output_tokens += counts.output_tokens
                        context_model = counts.model or context_model
                await send_frame(status, path=captured.source_path, coalesce=True)
        if scope is not None and (scope.home_root is not None or scope.host is not None):
            await self._harvest_usage_until_caught_up(
                scope.home_root / ".grok/logs/unified.jsonl" if scope.home_root else None,
                scope.host,
                "requested scoped usage harvest",
            )
        else:
            await self._harvest_usage_until_caught_up(None, None, "requested usage harvest")
        fts_rebuilt = self._keyword_dirty and bool(config.fts.fields)
        if self._keyword_dirty:
            await self._init_fts(reason="index request")
        await send_frame("done")
        backlog_pending = await self._run_readonly(
            lambda conn: reconciliation_status(config, conn=conn, limit=1)["pending"]
        )
        return IndexSummary(
            total=total,
            indexed=indexed,
            skipped=skipped,
            failed=failed,
            changed=indexed + failed,
            fts_rebuilt=fts_rebuilt,
            total_seconds=time.monotonic() - started,
            backlog_pending=backlog_pending,
            backlog_drain_per_minute=self._recent_drain_per_minute(),
            context_mode=(
                config.embedding.context.mode
                if scope
                else _lightweight_watch_context(config.embedding.context).mode
            ),
            context_messages=context_messages,
            context_reused=context_reused,
            context_input_tokens=context_input_tokens,
            context_output_tokens=context_output_tokens,
            context_model=context_model,
        )

    async def _handle_daemon_run(
        self, params: dict[str, Any], client: ClientConnection | None
    ) -> Any:
        """Run an operator-requested cycle through fair raw turns and detached models."""
        from recall.core.daemon_status import IndexRunCounts
        from recall.core.types import RunKind
        from recall.services.daemon import DaemonCycleSummary, _maybe_run_auto_compact
        from recall.services.embed_phase import EmbedPhaseState
        from recall.services.runtime_state import (
            record_run_attempt,
            record_run_failure,
            record_run_success,
        )

        if not params.get("once", False):
            raise RpcError(code=INVALID_PARAMS, message="daemon_run via RPC requires once=true")
        config = self._load_runtime_config()
        self._require_reconciliation(config)
        await self._maybe_begin_index_migration(config)
        source = parse_source(params["source"]) if params.get("source") else config.daemon.source
        embed = config.daemon.embed if params.get("embed") is None else bool(params["embed"])
        batch_size = params.get("batch_size")
        if batch_size is not None:
            batch_size = int(batch_size)
            if batch_size <= 0:
                raise RpcError(code=INVALID_PARAMS, message="batch_size must be positive")
        # `verbose` is accepted and ignored: a client's flag must not re-level a
        # daemon shared by every other session. The daemon's own `[daemon]
        # log_level` decides what it writes to its log.

        async with self._index_turn():
            self._require_reconciliation(config)
            attempted_at = await self._runtime_write(
                lambda: record_run_attempt(self._get_conn(), run_kind=RunKind.DAEMON_ONCE),
                "daemon_run_record_attempt",
            )
            try:
                summary = await self._reconcile_index_request(
                    config=config, source=source, client=client
                )
            except asyncio.CancelledError:
                await self._record_departed_run(RunKind.DAEMON_ONCE, attempted_at)
                raise
            except Exception as err:
                try:
                    await self._runtime_write(
                        lambda err=err: record_run_failure(
                            self._get_conn(),
                            run_kind=RunKind.DAEMON_ONCE,
                            message=str(err),
                            attempted_at=attempted_at,
                        ),
                        "daemon_run_record_failure",
                    )
                except Exception:
                    logger.exception("daemon_run could not persist failure metadata")
                raise

            persisted = True
            try:
                await self._runtime_write(
                    lambda: record_run_success(
                        self._get_conn(),
                        run_kind=RunKind.DAEMON_ONCE,
                        attempted_at=attempted_at,
                        index_summary=IndexRunCounts(
                            total=summary.total,
                            indexed=summary.indexed,
                            skipped=summary.skipped,
                            failed=summary.failed,
                            changed=summary.changed,
                            total_seconds=summary.total_seconds,
                        ),
                        last_context_messages=summary.context_messages,
                        last_context_mode=summary.context_mode,
                        last_context_input_tokens=summary.context_input_tokens,
                        last_context_output_tokens=summary.context_output_tokens,
                        last_context_model=summary.context_model,
                    ),
                    "daemon_run_record_success",
                )
            except Exception:
                persisted = False
                logger.exception("daemon_run could not persist success metadata after recovery")

            embed_summary = None
            if embed:
                embed_config = (
                    config
                    if batch_size is None
                    else replace(config, embedding=replace(config.embedding, batch_size=batch_size))
                )
                if self._embed_state is None:
                    self._embed_state = EmbedPhaseState(_config=embed_config)
                try:
                    drained = await self._run_embed_batch(embed_config, self._embed_state)
                    if drained > 0:
                        embed_summary = {"embedded": drained}
                except Exception as err:
                    self._embed_state.last_error = f"{type(err).__name__}: {err}"
                    logger.exception("daemon_run enrichment failed")
                    if self._shutdown_event.is_set():
                        raise
            await self._writer_call(
                lambda: _maybe_run_auto_compact(self, config), "daemon_run_auto_compact"
            )
            return DaemonCycleSummary(
                index_summary=summary,
                embed_summary=embed_summary,
                swapped=False,
                record_status_persisted=persisted,
            )

    # ---- Read handlers ----

    async def _run_duckdb_scan(self, operation: Callable[[], Awaitable[Any]]) -> Any:
        """Serialize memory-heavy reads before they own a cursor.

        FIFO admission lets enrichment progress under continuous searches.
        The operation retains the permit through executor cancellation cleanup;
        a waiting RPC still owns its ordinary 30-second request deadline.
        """
        claim = await self._database_admission.acquire(writer=False, scan=True)
        try:
            return await operation()
        finally:
            self._database_admission.release(claim)

    async def _prepare_query_embedding(self, operation: Callable[[], list[float]]) -> list[float]:
        """Model work owns bounded capacity, never a database cursor."""
        await self._query_model_slots.acquire()
        if self._query_model_executor is None:
            self._query_model_executor = DiagnosticExecutor(
                max_workers=_QUERY_MODEL_WORKERS, thread_name_prefix="recall-query-model"
            )
        try:
            future = asyncio.get_running_loop().run_in_executor(
                self._query_model_executor, operation
            )
        except BaseException:
            self._query_model_slots.release()
            raise

        def finished(completed: asyncio.Future[Any]) -> None:
            self._query_model_slots.release()
            if not completed.cancelled():
                completed.exception()

        future.add_done_callback(finished)
        return await asyncio.shield(future)

    @measured_async(Stage.READ)
    async def _run_readonly(self, fn: Callable[..., Any]) -> Any:
        """Run bounded, independently cancellable reads outside the writer pool."""
        await self._read_slots.acquire()
        if self._read_executor is None:
            self._read_executor = DiagnosticExecutor(
                max_workers=_READ_WORKERS, thread_name_prefix="recall-read"
            )
        execution = _ReadExecution()
        try:
            future = asyncio.get_running_loop().run_in_executor(
                self._read_executor, execution.run, self, fn
            )
        except BaseException:
            self._read_slots.release()
            raise
        self._reader_drained.clear()
        self._read_executions.add(execution)

        def finished(_completed: asyncio.Future[Any]) -> None:
            self._read_executions.discard(execution)
            if not self._read_executions:
                self._reader_drained.set()
            self._read_slots.release()

        future.add_done_callback(finished)
        try:
            return await asyncio.shield(future)
        except asyncio.CancelledError:
            execution.cancel()
            try:
                await asyncio.shield(future)
            except (_ReadAbandoned, ReadCancelled, duckdb.InterruptException):
                pass
            except Exception:
                logger.exception("read cursor failed while handling cancellation")
            raise

    def _run_with_read_cursor_sync(self, fn: Callable[..., Any]) -> Any:
        """Run synchronous read work with a per-task cursor and lifecycle access held."""
        self._conn_lifecycle_gate.acquire_read()
        cursor = None
        try:
            cursor = self._open_conn_unlocked().cursor()
            return fn(cursor)
        finally:
            if cursor is not None:
                cursor.close()
            self._conn_lifecycle_gate.release_read()

    async def _run_readonly_snapshot(self, fn: Callable[..., Any]) -> Any:
        """Keep a multi-query read on one generation of an owned read cursor."""

        def read(conn: Any) -> Any:
            conn.execute("BEGIN")
            try:
                result = fn(conn)
                conn.execute("COMMIT")
                return result
            except BaseException:
                conn.execute("ROLLBACK")
                raise

        return await self._run_readonly(read)

    async def _responsive_read(
        self, params: dict[str, Any], operation: Callable[[], Awaitable[Any]]
    ) -> Any:
        budget = SERVER_READ_BUDGET.get() or ReadBudget.start(
            params.get("remaining_read_ms", 2000) / 1000
        )
        token = SERVER_READ_BUDGET.set(budget)
        try:
            async with asyncio.timeout_at(budget.deadline):
                return await operation()
        except TimeoutError as err:
            raise RpcError(code=INTERNAL_ERROR, message="responsive read timed out") from err
        finally:
            SERVER_READ_BUDGET.reset(token)

    async def _cohort_snapshot(
        self,
        key: tuple[object, ...],
        epoch: int,
        fn: Callable[[Any], Any],
        *,
        status: bool = False,
    ) -> Any:
        def check_epoch() -> None:
            if epoch != self._database_epoch:
                raise RpcError(
                    code=INTERNAL_ERROR, message="database replaced during responsive read"
                )

        async def execute(start: Callable[[], None]) -> Any:
            def read(conn: Any) -> Any:
                start()
                check_epoch()
                conn.execute("BEGIN")
                try:
                    result = fn(conn)
                    conn.execute("COMMIT")
                    return result
                except BaseException:
                    conn.execute("ROLLBACK")
                    raise

            if status:
                async with self._status_count_lock:
                    return await self._run_readonly(read)
            return await self._run_readonly(read)

        try:
            result = await self._snapshot_cohorts.read((epoch, *key), execute)
        except SnapshotCapacityError as err:
            raise RpcError(code=INTERNAL_ERROR, message=str(err)) from err
        check_epoch()
        return result

    async def _recover_search_shared_conn(self, err: BaseException) -> None:
        await self._recover_shared_conn_locked(err, "search_aborted_txn")

    def _defer_search_fts_repair(self, err: BaseException) -> None:
        """Make missing keyword coverage visible and leave repair to the writer loop."""
        self._keyword_dirty = True
        logger.warning("search found unavailable FTS coverage; background repair queued: %s", err)

    @measured_async(Stage.SEARCH)
    async def _handle_search(self, params: dict[str, Any], client: ClientConnection | None) -> Any:
        from recall.services.search import resolve_search_mode, search

        query = params.get("query")
        if not query:
            raise RpcError(code=INVALID_PARAMS, message="query is required")
        source_value = params.get("source")
        source = parse_source(source_value) if source_value else None
        tool = params.get("tool")
        session = params.get("session")
        limit = int(params.get("limit", 20))
        mode_value = params.get("mode")
        mode = parse_search_mode(mode_value) if mode_value else SearchMode.AUTO
        effective_mode = (
            SearchMode.KEYWORD
            if tool
            else await self._run_readonly(lambda conn: resolve_search_mode(conn, mode))
        )
        timing = search_request.get() or SearchRequest(None, _READ_REQUEST_TIMEOUT)
        query_embedding = None
        if effective_mode in (SearchMode.VECTOR, SearchMode.HYBRID):

            def embed_query() -> list[float]:
                backend = self._get_embed_backend()
                return backend.embed([backend.query_prefix + query])[0]

            timing.enter(SearchPhase.QUERY_EMBEDDING)
            try:
                query_embedding = await self._prepare_query_embedding(embed_query)
            except (ValueError, ImportError, OSError, RuntimeError) as err:
                if mode is not SearchMode.AUTO:
                    raise RuntimeError(f"embedding backend unavailable: {err}") from err
                logger.warning("automatic search using keyword: %s", err)
                effective_mode = SearchMode.KEYWORD

        def _do_search(conn: Any) -> Any:
            return search(
                query=query,
                source=source,
                tool=tool,
                session=session,
                limit=limit,
                mode=effective_mode,
                config=self._config,
                conn=conn,
                query_embedding=query_embedding,
            )

        async def notify(payload: SearchWaiting | SearchReleased, until: float) -> None:
            if client is None:
                return
            # Advisory I/O spends the existing window, never a new query budget.
            remaining = until - time.monotonic()
            if remaining <= 0:
                return
            with suppress(TimeoutError):
                async with asyncio.timeout(min(0.05, remaining)):
                    await client.send_notification("search_progress", dict(payload))

        async def read_search(attempt: int) -> Any:
            if effective_mode in (SearchMode.VECTOR, SearchMode.HYBRID):
                if not self._background_enrichment_idle.is_set():
                    timing.enter(SearchPhase.ENRICHMENT_WAIT)
                    began = time.monotonic()
                    preference_end = began + _SEARCH_ENRICHMENT_WAIT_SECONDS
                    request_end = (
                        timing.started_at + timing.deadline
                        if timing.deadline is not None
                        else float("inf")
                    )
                    await notify(
                        SearchWaiting(
                            request_id=timing.request_id,
                            attempt=attempt,
                            state="waiting",
                            waiting_for="embedding_drain",
                            elapsed_ms=timing.elapsed_ms(),
                            preference_limit_ms=int(_SEARCH_ENRICHMENT_WAIT_SECONDS * 1000),
                        ),
                        min(preference_end, request_end),
                    )
                    reason: Literal["drain_idle", "preference_elapsed"] = "drain_idle"
                    try:
                        async with asyncio.timeout_at(
                            asyncio.get_running_loop().time()
                            + max(0, preference_end - time.monotonic())
                        ):
                            await self._background_enrichment_idle.wait()
                    except TimeoutError:
                        reason = "preference_elapsed"
                    wait_ms = int((time.monotonic() - began) * 1000)
                    timing.enter(SearchPhase.SCAN_ADMISSION)
                    await notify(
                        SearchReleased(
                            request_id=timing.request_id,
                            attempt=attempt,
                            state="released",
                            waiting_for=None,
                            elapsed_ms=timing.elapsed_ms(),
                            wait_ms=wait_ms,
                            reason=reason,
                        ),
                        request_end,
                    )
                timing.enter(SearchPhase.SCAN_ADMISSION)

                async def admitted() -> Any:
                    timing.enter(SearchPhase.READ_QUEUE)
                    return await self._run_readonly(_do_search)

                return await self._run_duckdb_scan(admitted)
            timing.enter(SearchPhase.READ_QUEUE)
            return await self._run_readonly(_do_search)

        try:
            return await read_search(1)
        except (RuntimeError, duckdb.Error) as err:
            if not _is_recoverable_search_error(err):
                raise
            if _is_search_fts_missing(err):
                self._defer_search_fts_repair(err)
                raise
            if not _search_needs_shared_conn_recovery(err):
                raise
            timing.enter(SearchPhase.RECOVERY)
            await self._recover_search_shared_conn(err)
            try:
                return await read_search(2)
            except (RuntimeError, duckdb.Error) as retry_err:
                if _is_search_fts_missing(retry_err):
                    self._defer_search_fts_repair(retry_err)
                raise

    async def _handle_list(self, params: dict[str, Any], client: ClientConnection | None) -> Any:
        from recall.core.time import parse_since
        from recall.services.sessions import list_sessions

        source_value = params.get("source")
        source = parse_source(source_value) if source_value else None
        since_value = params.get("since")
        since = parse_since(since_value) if since_value else None
        project = params.get("project")
        host_value = params.get("host")
        host = str(host_value) if host_value is not None else None
        limit = int(params.get("limit", 50))

        return await self._run_readonly(
            lambda conn: list_sessions(
                source=source,
                since=since,
                project=project,
                host=host,
                session=params.get("session"),
                limit=limit,
                config=self._config,
                conn=conn,
            )
        )

    async def _handle_show(self, params: dict[str, Any], client: ClientConnection | None) -> Any:
        """Serve one session, windowed on request, always saying how stale it is.

        Freshness rides on every `show`, not just `--fresh` ones (REQ-LIVE-002):
        the session may still be running, and a caller cannot tell a finished
        transcript from a lagging one by looking at the messages. Every answer
        also carries a `cursor` naming the newest message it contains, so a
        monitor loop can hand it straight back as `after` (REQ-LIVE-004).
        """
        from recall.services.live import (
            catalog_progress_for_session,
            decode_cursor,
            derive_freshness,
            encode_cursor,
        )
        from recall.services.sessions import (
            load_session,
            load_session_tail,
            resolve_session_header,
        )

        session_id = params.get("session_id")
        if not session_id:
            raise RpcError(code=INVALID_PARAMS, message="session_id is required")
        include_tools = bool(params.get("tools", False))
        message_limit_value = params.get("message_limit")
        message_limit = int(message_limit_value) if message_limit_value is not None else None
        tail_value = params.get("tail")
        tail = int(tail_value) if tail_value is not None else None
        after = params.get("after")
        fresh = bool(params.get("fresh", False))

        windowed = tail is not None or after is not None
        if windowed and message_limit is not None:
            raise RpcError(
                code=INVALID_PARAMS,
                message=(
                    "message_limit reads from the start of the session;"
                    " tail and after read from the end"
                ),
            )
        if tail is not None and tail <= 0:
            raise RpcError(code=INVALID_PARAMS, message="tail must be positive")

        supplied_cursor = None
        after_idx: int | None = None
        cursor_session: str | None = None
        if after is not None:
            try:
                supplied_cursor = decode_cursor(str(after))
                cursor_session, after_idx = supplied_cursor
            except ValueError as err:
                raise RpcError(code=INVALID_PARAMS, message=str(err)) from err

        try:
            # One header read serves both pre-read jobs, and only when one is
            # asked for: `--fresh` needs the transcript path, `--after` needs the
            # resolved id to prove the cursor names *this* session.
            if fresh or cursor_session is not None:
                header = await self._run_readonly(
                    lambda conn: resolve_session_header(session_id, conn=conn)
                )
                if cursor_session is not None and cursor_session != header.id:
                    raise RpcError(
                        code=INVALID_PARAMS,
                        message=f"cursor belongs to another session: {cursor_session}",
                    )
                if fresh:
                    await self._refresh_now(
                        [SourcePath(header.source, header.source_path)],
                        timeout=self._config.live.fresh_timeout,
                    )

            def read_window(conn: Any) -> Any:
                header = resolve_session_header(session_id, conn=conn)
                progress = catalog_progress_for_session(
                    conn, session_id=header.id, source_path=header.source_path
                )
                epoch = progress.content_epoch if progress is not None else 0
                reset = (
                    supplied_cursor is not None and (supplied_cursor.content_epoch or 0) != epoch
                )
                if windowed:
                    session = load_session_tail(
                        session_id,
                        tail=(tail or 100) if reset else tail,
                        after_idx=None if reset else after_idx,
                        tools=include_tools,
                        config=self._config,
                        conn=conn,
                    )
                else:
                    session = load_session(
                        session_id,
                        include_tools=include_tools,
                        message_limit=message_limit,
                        config=self._config,
                        conn=conn,
                    )
                return session, progress, epoch, reset

            session, progress, epoch, cursor_reset = await self._run_readonly_snapshot(read_window)
        except ValueError as err:
            err_msg = str(err)
            if err_msg.startswith(("session not found", "session not indexed")):
                raise RpcError(code=APP_NOT_FOUND, message=err_msg) from err
            raise

        # An empty delta hands back the cursor it was given rather than a lower
        # one: nothing arrived, so the caller has still seen everything it had.
        last_seen = max(
            (message.idx for message in session.messages),
            default=after_idx if after_idx is not None and not cursor_reset else -1,
        )
        return {
            **session.model_dump(),
            "freshness": derive_freshness(
                session.source_path,
                indexed_mtime=session.file_mtime,
                indexed_size=session.file_size,
                catalog=progress,
            ),
            "cursor": encode_cursor(session.id, last_seen, epoch),
            "cursor_reset": cursor_reset,
            "cursor_reset_reason": "content_rewritten" if cursor_reset else None,
        }

    async def _handle_stats(self, params: dict[str, Any], client: ClientConnection | None) -> Any:
        from recall.services.analytics import overview

        def _do_stats(conn: Any) -> Any:
            stats = asdict(overview(config=self._config, conn=conn))
            stats.update(_runtime_context_stats(conn))
            return stats

        return await self._run_readonly(_do_stats)

    async def _handle_stats_tools(
        self, params: dict[str, Any], client: ClientConnection | None
    ) -> Any:
        from recall.services.analytics import tool_usage

        return await self._run_readonly(lambda conn: tool_usage(config=self._config, conn=conn))

    async def _handle_stats_bash(
        self, params: dict[str, Any], client: ClientConnection | None
    ) -> Any:
        from recall.services.analytics import bash_breakdown, bash_suggestions

        suggest = bool(params.get("suggest", False))

        def _do(conn: Any) -> Any:
            if suggest:
                sug, skip = bash_suggestions(config=self._config, conn=conn)
                return {"suggestions": sug, "skipped": skip}
            return bash_breakdown(config=self._config, conn=conn)

        return await self._run_readonly(_do)

    async def _handle_stats_tokens(
        self, params: dict[str, Any], client: ClientConnection | None
    ) -> Any:
        from recall.services.analytics import token_usage

        return await self._run_readonly(lambda conn: token_usage(config=self._config, conn=conn))

    async def _handle_stats_usage(
        self, params: dict[str, Any], client: ClientConnection | None
    ) -> Any:
        from dataclasses import asdict

        from recall.core.time import parse_since
        from recall.services.analytics import usage_by_source

        since_value = params.get("since")
        since = parse_since(str(since_value)) if since_value else None

        def _do(conn: Any) -> Any:
            rows = usage_by_source(since=since, config=self._config, conn=conn)
            return [asdict(row) for row in rows]

        return await self._run_readonly(_do)

    async def _handle_stats_skills(
        self, params: dict[str, Any], client: ClientConnection | None
    ) -> Any:
        from dataclasses import asdict

        from recall.core.time import parse_since
        from recall.services.analytics import skill_usage

        since_value = params.get("since")
        since = parse_since(str(since_value)) if since_value else None
        source_values = params.get("source")
        if source_values is None:
            sources = None
        elif isinstance(source_values, list) and all(
            isinstance(value, str) for value in source_values
        ):
            try:
                sources = tuple(dict.fromkeys(parse_source(value) for value in source_values))
            except ValueError as err:
                raise RpcError(code=INVALID_PARAMS, message=str(err)) from err
        else:
            raise RpcError(code=INVALID_PARAMS, message="source must be an array of strings")

        return await self._run_readonly(
            lambda conn: asdict(
                skill_usage(
                    sources=sources,
                    since=since,
                    config=self._config,
                    conn=conn,
                )
            )
        )

    async def _handle_set_installed_scheduler(
        self, params: dict[str, Any], client: ClientConnection | None
    ) -> None:
        from recall.core.types import SchedulerKind
        from recall.services.runtime_state import set_installed_scheduler

        if "scheduler" not in params:
            raise ValueError("scheduler is required")
        value = params["scheduler"]
        scheduler = SchedulerKind(value) if value is not None else None
        if scheduler == SchedulerKind.AUTO:
            raise ValueError("installed scheduler must be launchd, systemd, cron, or null")

        def write() -> None:
            set_installed_scheduler(self._get_conn(), scheduler)

        await self._writer_call(write, "installed scheduler")

    async def _handle_daemon_status(
        self, params: dict[str, Any], client: ClientConnection | None
    ) -> Any:
        return await self._responsive_read(params, lambda: self._read_daemon_status(params))

    def _storage_maintenance_running(self) -> bool:
        task = self._storage_maintenance_task
        return task is not None and not task.done()

    def _write_admission(self) -> WriteAdmissionStatus:
        running = self._storage_maintenance_running()
        return {
            "allowed": not running,
            "reason": "storage_maintenance" if running else None,
            "action": "recall.migrate_storage" if running else None,
        }

    def _require_write_admission(self) -> None:
        if self._storage_maintenance_running():
            raise RpcError(
                code=APP_LOCKED,
                message=(
                    "storage maintenance is running; inspect recall.daemon_status before retrying"
                ),
                data={"reason": "storage_maintenance", "action": "recall.migrate_storage"},
            )

    def _storage_progress(self, *, snapshot: bool) -> StorageMaintenanceStatus | None:
        if self._storage_maintenance_task is None:
            return None
        if self._storage_maintenance_running():
            phase: Literal["running", "failed", "succeeded"] = "running"
        elif self._storage_maintenance_error is not None:
            phase = "failed"
        else:
            phase = "succeeded"
        return {
            "phase": phase,
            "snapshot": snapshot,
            "operation_id": (self._storage_plan_snapshot or {}).get("operation_id"),
            "error": self._storage_maintenance_error,
        }

    async def _read_daemon_status(self, params: dict[str, Any]) -> Any:
        if self._storage_maintenance_running() and self._storage_status_snapshot is not None:
            return {
                **self._storage_status_snapshot,
                "storage_maintenance": self._storage_progress(snapshot=True),
                "write_admission": self._write_admission(),
            }

        from recall.services.coordinator import is_paused, reconciliation_status
        from recall.services.daemon import daemon_status, fts_rebuild_backoff_status

        limit = int(params.get("limit", 100))
        if not 1 <= limit <= 256:
            raise RpcError(code=INVALID_PARAMS, message="coverage limit must be 1..256")
        runtime = self._watch_runtime
        generation = runtime.observation_generation() if runtime is not None else None

        def read_status(conn: Any) -> Any:
            result = daemon_status(
                config=self._config,
                conn=conn,
                fts_sidecar_startup=self._fts_sidecar_startup,
                service_probes=self._service_probes,
            )
            coverage = reconciliation_status(
                self._config,
                conn=conn,
                limit=limit,
                cursor=_issued_catalog_cursor(conn, params.get("cursor")),
            )
            return result, coverage

        # Embedding counts and raw/keyword coverage must describe the same
        # committed state. A raw publication between independent reads could
        # otherwise make old zero embedding counts appear drained.
        result, coverage = await self._cohort_snapshot(
            ("status", "final", limit, params.get("cursor")),
            self._database_epoch,
            read_status,
            status=True,
        )
        # Overlay live embed state if available
        from dataclasses import asdict as dc_asdict
        from dataclasses import replace as dc_replace

        binary_version = _package_version()
        # binary_version is None when importlib.metadata can no longer resolve
        # the recall distribution from the daemon's running process. That happens
        # after `uv tool install --reinstall` rewrites the venv out from under
        # the live daemon: package metadata is gone, but the daemon keeps
        # serving until restarted. Treat that missing metadata as drift so the
        # CLI surfaces a restart prompt instead of silently reporting healthy.
        overrides: dict[str, Any] = {
            # Offline status leaves daemon_pid null because a pid file proves
            # no ownership; RPC identifies the process serving this response.
            "daemon_pid": os.getpid(),
            "daemon_version": self._daemon_version,
            "binary_version": binary_version,
            "version_drift": self._daemon_version is not None
            and (binary_version is None or self._daemon_version != binary_version),
            "bloat_ratio": self._last_bloat_ratio,
            "bloat_ratio_threshold": self._config.compaction.bloat_ratio_threshold,
            "bloat_auto_trigger": self._config.compaction.auto_trigger,
            "index_divergence": (
                self._last_index_probe.to_payload() if self._last_index_probe is not None else None
            ),
            "live_fresh_requests": self.live_fresh_requests,
            "live_fresh_timeouts": self.live_fresh_timeouts,
            "follow_subscriptions": self._session_indexed.total_subscriptions,
        }
        from recall.db.connection import resolve_memory_limit_settings

        memory_limit = resolve_memory_limit_settings(self._config)
        overrides.update(
            duckdb_memory_limit_requested=memory_limit.requested,
            duckdb_memory_limit_applied=memory_limit.applied,
            duckdb_memory_limit_clamped=memory_limit.clamped,
        )
        if self._embed_state is not None:
            es: EmbedPhaseState = self._embed_state
            cooldown_count, cooldown_until = es.cooldown_status()
            overrides.update(
                {
                    "embed_model_loaded": es.backend is not None,
                    "embed_last_batch_at": es.last_batch_at if es.last_batch_at > 0 else None,
                    "embed_last_batch_size": es.last_batch_size,
                    "embed_last_batch_duration": es.last_batch_duration,
                    "embed_loop_iterations": es.loop_iterations,
                    "embed_loop_last_iteration_at": es.last_iteration_at or None,
                    "embed_loop_last_trigger": es.last_trigger or None,
                    "embed_loop_stage": es.stage or None,
                    "embed_loop_stage_at": es.stage_at or None,
                    "embed_loop_last_outcome": es.last_outcome or None,
                    "embed_loop_next_interval": es.next_interval,
                    "embed_requested_cycles": es.requested_cycles,
                    "embed_requested_at": es.last_requested_at or None,
                    "embed_requested_stage": es.requested_stage or None,
                    "embed_requested_stage_at": es.requested_stage_at or None,
                    "embed_deferred_reason": es.deferred_reason,
                    "embed_deferred_at": es.deferred_at or None,
                    "embed_last_error": es.last_error,
                    "embed_cooldown_sessions": cooldown_count,
                    "embed_cooldown_until": cooldown_until,
                }
            )
            # `embed_pending` stays the loop's cached eligible snapshot; the
            # fresh split read above is what readiness is decided on.
            if es.last_pending is not None:
                overrides["embed_pending"] = es.last_pending.total
                overrides["embed_pending_at"] = es.last_pending_at or None
        # Overlay live watch index metrics if available
        if self._watch_metrics is not None:
            from recall.services.watch_metrics import WatchIndexMetrics

            wm: WatchIndexMetrics = self._watch_metrics
            overrides.update(wm.snapshot())
            overrides["watch_recent_events"] = tuple(dc_asdict(e) for e in wm.recent_events())
        if self._watch_runtime is not None:
            overrides.update(fts_rebuild_backoff_status(self._watch_runtime.fts_debouncer))
        overrides["write_admission"] = self._write_admission()
        overrides["storage_maintenance"] = self._storage_progress(snapshot=False)
        result = dc_replace(result, **overrides)
        now = time.monotonic()
        read_starts = [execution.started_at for execution in self._read_executions]
        # Writes the watcher reported that no observation has reached yet. The
        # catalog cannot show them, so every readiness flag waits on them too
        # (REQ-DAEMON-079).
        noticed_pending = runtime.noticed_pending() if runtime is not None else 0
        # Settlement may remove a mark after this snapshot read its old catalog
        # state. Validate both queues, including marks created and settled
        # entirely during collection; an empty final queue alone proves nothing.
        observation_stable = self._watch_runtime is runtime and (
            runtime is None or runtime.observation_generation() == generation
        )
        # The catalog's scan records outlive the process that wrote them, so a
        # restarted daemon has seen nothing written while it was down until it
        # has walked every root itself (REQ-DAEMON-079).
        for row in coverage["coverage"]:
            row["scan_current_process"] = row["configuration"] == "disabled" or (
                row["scan_started_at"] is not None
                and row["scan_started_at"] >= self._process_started_at
            )
        walked_since_start = all(row["scan_current_process"] for row in coverage["coverage"])
        observed = observation_stable and walked_since_start and noticed_pending == 0
        raw_indexing_ready = coverage["raw_indexing_ready"] and observed
        reconciliation: ReconciliationStatus = {
            **coverage,
            "noticed_pending": noticed_pending,
            "raw_indexing_ready": raw_indexing_ready,
            "keyword_search_ready": coverage["keyword_search_ready"]
            and observed
            and not self._keyword_dirty
            and self._fts_repair_error is None,
            "keyword_repair_error": self._fts_repair_error,
            "paused": is_paused(self._config),
            "runtime_mode": "watch" if self._runtime_watch else "poll",
            "rpc_ready": self._server is not None,
            "live_observation_ready": self._inventory_complete
            and self._active_observation_error is None,
            "active_observation_error": self._active_observation_error,
            # Drained means nothing eligible and nothing held, read at this call:
            # the loop's snapshot is eligible-only and can be minutes old, so it
            # reads 0 while a fresh session waits out its idle hold (REQ-ADAPT-022).
            # Rows not indexed yet are not counted as pending embeddings at all,
            # so enrichment is only drained once raw indexing is (REQ-DAEMON-079).
            "enrichment_ready": not self._config.daemon.embed
            or (
                raw_indexing_ready
                and self._embed_state is not None
                and result.embed_pending_eligible == 0
                and result.embed_pending_held == 0
                and self._embed_state.last_error is None
            ),
            "enrichment_error": self._embed_state.last_error if self._embed_state else None,
            "enrichment_deferred": self._embed_state.deferred_reason if self._embed_state else None,
            "error": "; ".join(
                error
                for error in (
                    self._inventory_error,
                    runtime.observation_error() if runtime is not None else None,
                )
                if error
            )
            or None,
            "checkpoint_maintenance": dict(self._checkpoint_status),
            "checkpoint_contention_retries": self._checkpoint_contention_retries,
            "wal_pressure_blocked": self._wal_pressure_blocked,
            "activity_hints_saturated": self._activity_hints_saturated,
            "raw_preparation_reserved_bytes": self._raw_reserved_bytes,
            "raw_preparation_historical_reserved_bytes": (self._raw_historical_reserved_bytes),
            "ordinary_read_active": len(read_starts),
            "ordinary_read_oldest_age_seconds": (now - min(read_starts) if read_starts else None),
            "recovery_waiting": len(self._recovery_waiters),
            "recovery_waiting_oldest_age_seconds": (
                now - min(self._recovery_waiters) if self._recovery_waiters else None
            ),
        }
        return {**dc_asdict(result), "reconciliation": reconciliation}

    async def _handle_daemon_pause(
        self, params: dict[str, Any], client: ClientConnection | None
    ) -> dict[str, bool]:
        from recall.services.coordinator import set_paused

        return {"paused": set_paused(self._config, True)}

    async def _handle_daemon_resume(
        self, params: dict[str, Any], client: ClientConnection | None
    ) -> dict[str, bool]:
        from recall.services.coordinator import set_paused

        paused = set_paused(self._config, False)
        self._raw_wakeup.set()
        return {"paused": paused}

    async def _handle_migrate_storage(
        self, params: dict[str, Any], client: ClientConnection | None
    ) -> dict[str, Any]:
        from recall.services.index_migration import plan_storage_migration

        if set(params) - {"dry_run", "plan_id"}:
            raise RpcError(code=INVALID_PARAMS, message="unknown storage maintenance parameter")
        dry_run = params.get("dry_run", False)
        expected = params.get("plan_id")
        if not isinstance(dry_run, bool) or (
            expected is not None and not isinstance(expected, str)
        ):
            raise RpcError(code=INVALID_PARAMS, message="dry_run must be boolean; plan_id a string")
        if self._storage_maintenance_running() and self._storage_plan_snapshot is not None:
            if expected is not None and expected != self._storage_maintenance_plan_id:
                raise RpcError(
                    code=INVALID_PARAMS, message="storage maintenance plan changed; inspect again"
                )
            return {
                "schema_version": 1,
                **self._storage_plan_snapshot,
                "status": "running",
                "accepted": False,
                "dry_run": dry_run,
                "error": None,
            }
        plan = await self._run_readonly(lambda conn: plan_storage_migration(conn, self._config))
        task = self._storage_maintenance_task
        running = task is not None and not task.done()
        duplicate = running and expected == self._storage_maintenance_plan_id
        if expected is not None and expected != plan.plan_id and not duplicate:
            raise RpcError(
                code=INVALID_PARAMS, message="storage maintenance plan changed; inspect again"
            )
        migration = plan.migration
        pending = migration.phase != "idle" and migration.storage_target is not None
        accepted = False
        if not dry_run and (migration.storage_eligible or pending or running):
            if self._shutdown_event.is_set():
                raise RpcError(code=APP_LOCKED, message="daemon is shutting down")
            if not running:
                self._storage_status_snapshot = None
                self._storage_plan_snapshot = asdict(plan)
                self._storage_maintenance_error = None
                self._storage_maintenance_plan_id = plan.plan_id

                async def execute() -> None:
                    try:
                        self._storage_status_snapshot = await self._read_daemon_status({})
                        await self._run_storage_maintenance(expected_plan_id=plan.plan_id)
                    except asyncio.CancelledError:
                        raise
                    except Exception as err:
                        self._storage_maintenance_error = f"{type(err).__name__}: {err}"
                        logger.exception(
                            "storage maintenance failed operation=%s", plan.operation_id
                        )
                    finally:
                        self._inventory_wakeup.set()
                        self._raw_wakeup.set()

                # Reserve ownership before snapshot I/O yields to another apply
                # or shutdown; both must observe the same task.
                self._storage_maintenance_task = asyncio.create_task(execute())
                accepted = True
                running = True
        error = self._storage_maintenance_error or (migration.error if pending else None)
        if running:
            state = "running" if pending else "accepted"
        elif error:
            state = "failed"
        elif pending:
            state = "running"
        elif migration.storage_eligible:
            state = "planned"
        else:
            state = "succeeded" if migration.storage_attempt else "unchanged"
        return {
            "schema_version": 1,
            **asdict(plan),
            "status": state,
            "accepted": accepted,
            "dry_run": dry_run,
            "error": error,
        }

    async def _enrichment_writer_call(
        self,
        operation: Callable[[], Any],
        label: str,
        *,
        affects_enrichment: bool = True,
    ) -> Any:
        return await self._writer_call(
            operation, label, affects_enrichment=affects_enrichment, scan=True
        )

    async def _writer_call(
        self,
        operation: Callable[[], Any],
        label: str,
        *,
        checkpoint_wakeup: bool = True,
        affects_enrichment: bool = True,
        scan: bool = False,
        storage_maintenance: bool = False,
    ) -> Any:
        if not storage_maintenance:
            self._require_write_admission()

        def run() -> Any:
            if not storage_maintenance:
                self._require_write_admission()
            try:
                with Span(Stage.WRITER_RUN):
                    return operation()
            except duckdb.Error as err:
                # Recovery belongs to the failed writer's turn: the next job
                # must never inherit an aborted transaction or invalid handle.
                self._recover_shared_conn(err, label)
                raise

        started = time.monotonic()
        with Span(Stage.WRITER_WAIT, thread_cpu=False):
            await self._write_lock.acquire(affects_enrichment=affects_enrichment, scan=scan)
        try:
            lock_held = time.monotonic()
            result = await self._await_shared_conn_work(
                asyncio.get_running_loop().run_in_executor(self._executor, run), label
            )
        finally:
            self._write_lock.release()
        logger.info(
            "writer %s: wait=%.3fs op=%.3fs",
            label,
            lock_held - started,
            time.monotonic() - lock_held,
        )
        if checkpoint_wakeup:
            self._checkpoint_wakeup.set()
        return result

    async def _checkpoint_monitor(self) -> None:
        """Run durability maintenance even while source indexing is paused."""
        from recall.db import checkpoint_wal_if_due
        from recall.db.connection import WAL_CHECKPOINT_HIGH_WATER_BYTES, wal_size_bytes

        while not self._shutdown_event.is_set():
            if self._checkpoint_contention_retries >= 3:
                if self._read_executions:
                    shutdown = asyncio.create_task(self._shutdown_event.wait())
                    drained = asyncio.create_task(self._reader_drained.wait())
                    try:
                        await asyncio.wait({shutdown, drained}, return_when=asyncio.FIRST_COMPLETED)
                    finally:
                        shutdown.cancel()
                        drained.cancel()
                        await asyncio.gather(shutdown, drained, return_exceptions=True)
                    if self._shutdown_event.is_set():
                        break
                self._checkpoint_contention_retries = 0
                self._checkpoint_retry_at = 0.0
                # Historical sources were held back for the checkpoint.
                self._raw_wakeup.set()
            retry_delay = self._checkpoint_retry_at - time.monotonic()
            if retry_delay > 0:
                with suppress(TimeoutError):
                    await asyncio.wait_for(self._shutdown_event.wait(), timeout=retry_delay)
                continue
            with suppress(TimeoutError):
                await asyncio.wait_for(self._checkpoint_wakeup.wait(), timeout=1.0)
            self._checkpoint_wakeup.clear()
            if self._shutdown_event.is_set():
                break
            await asyncio.sleep(0)
            try:
                wal_before = wal_size_bytes(self._config.db_path)
            except OSError as err:
                now = time.time()
                self._checkpoint_retry_at = time.monotonic() + self._checkpoint_retry_seconds
                self._checkpoint_status.update(
                    attempts=self._checkpoint_status["attempts"] + 1,
                    running=False,
                    last_started_at=now,
                    last_completed_at=now,
                    last_duration_seconds=0.0,
                    last_wal_bytes_before=None,
                    last_wal_bytes_after=None,
                    last_error=f"{type(err).__name__}: {err}",
                )
                logger.warning("WAL checkpoint inspection failed: %s", err)
                continue
            if wal_before < WAL_CHECKPOINT_HIGH_WATER_BYTES:
                continue
            started = time.monotonic()
            previous_status = dict(self._checkpoint_status)
            self._checkpoint_status.update(
                running=True,
                last_started_at=time.time(),
                last_completed_at=None,
                last_duration_seconds=None,
                last_wal_bytes_before=wal_before,
                last_wal_bytes_after=None,
                last_error=None,
            )
            try:
                result = await self._writer_call(
                    lambda: checkpoint_wal_if_due(self._get_conn(), self._config.db_path),
                    "WAL checkpoint maintenance",
                    checkpoint_wakeup=False,
                )
            except asyncio.CancelledError:
                try:
                    cancelled_wal_after: int | None = wal_size_bytes(self._config.db_path)
                except OSError:
                    cancelled_wal_after = None
                self._checkpoint_status.update(
                    running=False,
                    last_completed_at=time.time(),
                    last_duration_seconds=time.monotonic() - started,
                    last_wal_bytes_after=cancelled_wal_after,
                    last_error="checkpoint monitor cancelled after in-flight work completed",
                )
                raise
            except Exception as err:
                try:
                    wal_after: int | None = wal_size_bytes(self._config.db_path)
                except OSError:
                    wal_after = None
                self._checkpoint_retry_at = time.monotonic() + self._checkpoint_retry_seconds
                self._checkpoint_status.update(
                    attempts=self._checkpoint_status["attempts"] + 1,
                    running=False,
                    last_completed_at=time.time(),
                    last_duration_seconds=time.monotonic() - started,
                    last_wal_bytes_after=wal_after,
                    last_error=f"{type(err).__name__}: {err}",
                )
                logger.warning("WAL checkpoint maintenance failed: %s", err)
                continue
            if result.contended:
                self._checkpoint_contention_retries += 1
                self._checkpoint_retry_at = time.monotonic() + self._checkpoint_retry_seconds
                self._wal_pressure_blocked = result.wal_bytes_after >= _WAL_BLOCK_ALL_BYTES
                self._checkpoint_status.update(
                    attempts=self._checkpoint_status["attempts"] + 1,
                    running=False,
                    last_completed_at=time.time(),
                    last_duration_seconds=result.duration_seconds,
                    last_wal_bytes_before=result.wal_bytes_before,
                    last_wal_bytes_after=result.wal_bytes_after,
                    last_error="checkpoint deferred by active read transaction",
                )
                continue
            if result.attempted:
                self._checkpoint_retry_at = 0.0
                self._checkpoint_contention_retries = 0
                self._wal_pressure_blocked = False
                self._raw_wakeup.set()
                self._checkpoint_status.update(
                    attempts=self._checkpoint_status["attempts"] + 1,
                    successes=self._checkpoint_status["successes"] + 1,
                    running=False,
                    last_completed_at=time.time(),
                    last_duration_seconds=result.duration_seconds,
                    last_wal_bytes_before=result.wal_bytes_before,
                    last_wal_bytes_after=result.wal_bytes_after,
                    last_error=None,
                )
            else:
                self._checkpoint_status.clear()
                self._checkpoint_status.update(previous_status)

    def _commit_raw_sources(
        self,
        prepared: tuple[PreparedRawSource, ...],
        config: AppConfig,
        request: RawIndexRequest | None = None,
    ) -> int:
        from recall.core.daemon_status import IndexRunCounts
        from recall.core.types import RunKind
        from recall.db.source_files import SourceCatalog
        from recall.services.coordinator import commit_prepared_raw_sources, is_paused
        from recall.services.runtime_state import record_run_attempt, record_run_success

        if is_paused(config):
            return 0
        conn = self._get_conn()
        run_kind = RunKind.DAEMON_WATCH if self._runtime_watch else RunKind.DAEMON_SCHEDULED
        attempted_at = record_run_attempt(conn, run_kind=run_kind)
        started = time.monotonic()
        result = commit_prepared_raw_sources(prepared, config, conn=conn, request=request)
        committed = result.changed
        if result.searchable_changed and config.fts.backend == "duckdb":
            self._keyword_dirty = True
        duration = time.monotonic() - started
        catalog = SourceCatalog(conn, clock=time.time)
        failed = 0
        for capture in prepared:
            item = catalog.get(capture.item.source, capture.item.source_path)
            if item is None:
                error: str | None = "source disappeared from catalog"
            elif item.committed_generation >= capture.item.desired_generation:
                # Published. A park the acknowledgement recorded (an unsupported
                # record, a torn tail) describes the source, not this attempt.
                error = None
            else:
                error = item.last_error
            failed += int(error is not None)
            if self._watch_metrics is not None:
                self._watch_metrics.record_index(
                    capture.item.source_path,
                    capture.item.source,
                    success=error is None,
                    duration=duration,
                    error=error,
                )
        record_run_success(
            conn,
            run_kind=run_kind,
            attempted_at=attempted_at,
            index_summary=IndexRunCounts(
                total=len(prepared),
                indexed=committed,
                changed=committed,
                skipped=max(0, len(prepared) - committed - failed),
                failed=failed,
                total_seconds=duration,
            ),
        )
        return committed

    async def index_session_now(
        self,
        path: Path,
        *,
        request: RawIndexRequest | None = None,
        await_service: bool = True,
        source: str | None = None,
    ) -> None:
        """Serve a known source, or offer a generic path to every eligible parser."""
        from recall.parsers import all_parsers
        from recall.services.watcher import _resolve_parsers_for_path

        if request is not None:
            keys = [SourcePath.resolve(request.parser.source.value, path)]
        elif source is not None:
            keys = [SourcePath.resolve(source, path)]
        else:
            config = self._load_runtime_config()
            async with self._raw_slots:
                parsers = await self._await_shared_conn_work(
                    asyncio.get_running_loop().run_in_executor(
                        self._executor,
                        lambda: _resolve_parsers_for_path(str(path), all_parsers(config.sources)),
                    ),
                    "source routing",
                )
            keys = [SourcePath.resolve(parser.source.value, path) for parser in parsers]
            if not keys:
                raise NoParserForPath(f"no parser for {path}")
        await self._index_sessions_now(keys, request=request, await_service=await_service)

    async def _index_sessions_now(
        self,
        paths: Sequence[SourcePath],
        *,
        request: RawIndexRequest | None = None,
        await_service: bool = True,
        skip_unclaimed: bool = False,
    ) -> None:
        from recall.services.reconciler import READY_LIMIT

        if not 1 <= len(paths) <= READY_LIMIT:
            raise ValueError("fresh batch must contain 1..256 paths")
        keys = tuple(dict.fromkeys(paths))
        assert request is None or len(keys) == 1
        if request is not None:
            # Options cannot join an earlier request, nor overwrite another
            # option-bearing waiter admitted while this one was suspended.
            while (preceding := self._fresh_jobs.get(keys[0])) is not None:
                await self._await_shared_conn_work(preceding, "preceding fresh request")
        owned = tuple(key for key in keys if key not in self._fresh_jobs)
        if len(self._fresh_jobs) + len(owned) > READY_LIMIT:
            raise RpcError(code=APP_LOCKED, message="fresh request capacity reached; retry later")
        for key in owned:
            future = asyncio.get_running_loop().create_future()
            self._fresh_jobs[key] = future
            if request is not None:
                self._raw_requests[key] = request

            future.add_done_callback(
                lambda completed, key=key: self._report_source_result(completed, path=key.path)
            )
        # Reserve every path before yielding: overlapping requests share the
        # same target, and capacity is admitted for the entire batch or none.
        futures = tuple(self._fresh_jobs[key] for key in keys)
        if owned:
            jobs = {key: self._fresh_jobs[key] for key in owned}
            task = asyncio.create_task(
                self._index_requested_sources(jobs, request=request, await_service=await_service)
            )
            self._fresh_batches.add(task)
            task.add_done_callback(self._fresh_batches.discard)
        results = await self._await_shared_conn_work(
            asyncio.gather(*futures, return_exceptions=True), "fresh request"
        )
        for result in results:
            if isinstance(result, NoParserForPath) and skip_unclaimed:
                continue
            if isinstance(result, BaseException):
                raise result

    def _finish_fresh_job(
        self, path: SourcePath, future: asyncio.Future[None], error: BaseException | None = None
    ) -> None:
        if future.done():
            assert self._fresh_jobs.get(path) is not future
            return
        assert self._fresh_jobs.get(path) is future
        # Remove ownership before waking anyone. A caller may run before a
        # completion callback and must never join a finished target or spin
        # waiting for that callback to release an option-bearing predecessor.
        self._fresh_jobs.pop(path)
        self._raw_requests.pop(path, None)
        if isinstance(error, asyncio.CancelledError):
            future.cancel()
        elif error is not None:
            future.set_exception(error)
        else:
            future.set_result(None)

    async def _observe_requested_sources(
        self, paths: tuple[SourcePath, ...], cfg: AppConfig, request: RawIndexRequest | None
    ) -> dict[SourcePath, SourceProgress | OSError | NoParserForPath]:
        """Capture metadata in one preparation slot; publish one catalog transaction."""
        from recall.parsers import all_parsers
        from recall.services.coordinator import capture_path
        from recall.services.reconciler import CapturedSource

        def capture() -> dict[SourcePath, CapturedSource | OSError | NoParserForPath]:
            parsers = {parser.source.value: parser for parser in all_parsers(cfg.sources)}
            captured: dict[SourcePath, CapturedSource | OSError | NoParserForPath] = {}
            for path in paths:
                parser = request.parser if request else parsers.get(path.source)
                assert parser is None or parser.source.value == path.source
                if parser is None:
                    captured[path] = NoParserForPath(f"no parser for {path}")
                    continue
                try:
                    root, resolved, signature = capture_path(parser, Path(path.path))
                except OSError as err:
                    captured[path] = err
                else:
                    captured[path] = CapturedSource(parser.source.value, root, resolved, signature)
            return captured

        async with self._raw_slots:
            captures = await self._await_shared_conn_work(
                asyncio.get_running_loop().run_in_executor(self._executor, capture),
                "source observation",
            )

        def observe() -> dict[SourcePath, SourceProgress | OSError | NoParserForPath]:
            valid = {
                key: item for key, item in captures.items() if isinstance(item, CapturedSource)
            }
            results: dict[SourcePath, SourceProgress | OSError | NoParserForPath] = {
                key: item for key, item in captures.items() if not isinstance(item, CapturedSource)
            }
            if not valid:
                return results
            conn = self._get_conn()
            catalog = SourceCatalog(conn, clock=time.time)
            keys = tuple((item.source, item.source_path) for item in valid.values())
            conn.execute("BEGIN")
            try:
                catalog.observe_batch(
                    [
                        (item.source, item.root_path, item.source_path, item.signature)
                        for item in valid.values()
                    ]
                )
                if request is not None and request.full:
                    catalog.force_reconcile_batch(keys)
                progress = catalog.progress(keys)
                pending = tuple(key for key, item in progress.items() if not item.current)
                if pending:
                    catalog.retry_now_batch(pending)
                for key, item in valid.items():
                    results[key] = progress[item.source, item.source_path]
                conn.execute("COMMIT")
            except BaseException:
                conn.execute("ROLLBACK")
                raise
            if request is not None:
                request.changed = bool(pending)
            return results

        async with self._raw_turn_lock:
            return await self._writer_call(
                observe, "source observation commit", affects_enrichment=False
            )

    async def _index_requested_sources(
        self,
        jobs: dict[SourcePath, asyncio.Future[None]],
        *,
        request: RawIndexRequest | None,
        await_service: bool,
    ) -> None:
        from recall.services.coordinator import is_paused

        try:
            cfg = request.config if request else self._load_runtime_config()
            if is_paused(cfg) or self._shutdown_event.is_set():
                return
            epoch = self._database_epoch
            runtime = self._watch_runtime
            observed_from = runtime.clock.monotonic() if runtime is not None else 0.0
            observations = await self._observe_requested_sources(tuple(jobs), cfg, request)
            pending: dict[SourcePath, _FreshTarget] = {}
            for path, item in observations.items():
                if isinstance(item, Exception):
                    self._finish_fresh_job(path, jobs[path], item)
                    continue
                if runtime is not None:
                    runtime.settle(path.path, observed_from)
                if item.current or not await_service:
                    self._finish_fresh_job(path, jobs[path])
                else:
                    pending[path] = _FreshTarget(
                        item.source,
                        item.source_path,
                        epoch,
                        item.desired_generation,
                        item.last_serviced_seq,
                    )
            self._raw_wakeup.set()
            loop = asyncio.get_running_loop()
            stall_deadline = loop.time() + _REQUESTED_SERVICE_STALL_SECONDS
            served = self._raw_turns_served
            while pending and not self._shutdown_event.is_set():
                if epoch != self._database_epoch or is_paused(self._load_runtime_config()):
                    return
                self._raw_wakeup.clear()
                worked = await self._serve_raw_turn()
                keys = tuple((item.source, item.source_path) for item in pending.values())
                progress = await self._run_readonly(
                    lambda conn, keys=keys: SourceCatalog(conn, clock=time.time).progress(keys)
                )
                for path, target in tuple(pending.items()):
                    item = progress.get((target.source, target.source_path))
                    fulfilled = (
                        target.database_epoch != self._database_epoch
                        or item is None
                        or item.committed_generation >= target.generation
                        or (
                            item.last_error is not None
                            and item.last_serviced_seq > target.serviced_seq
                        )
                    )
                    if fulfilled:
                        self._finish_fresh_job(path, jobs[path])
                        pending.pop(path)
                progressed = self._raw_turns_served != served or bool(self._raw_claims)
                served = self._raw_turns_served
                if progressed:
                    stall_deadline = loop.time() + _REQUESTED_SERVICE_STALL_SECONDS
                elif pending and loop.time() >= stall_deadline:
                    raise RpcError(
                        code=APP_LOCKED,
                        message=(
                            "the reconciliation scheduler served no source and prepared none for "
                            f"{_REQUESTED_SERVICE_STALL_SECONDS:g}s while {next(iter(pending))} "
                            "is still pending; inspect "
                            "`recall daemon status --json --fields reconciliation`"
                        ),
                    )
                if pending and not worked:
                    # Level-triggered claim release wakes the batch. The bounded
                    # poll also covers retry eligibility without notifications.
                    with suppress(TimeoutError):
                        await asyncio.wait_for(self._raw_wakeup.wait(), timeout=0.5)
        except asyncio.CancelledError as err:
            for path, future in jobs.items():
                self._finish_fresh_job(path, future, err)
            raise
        except Exception as err:
            for path, future in jobs.items():
                self._finish_fresh_job(path, future, err)
        finally:
            for path, future in jobs.items():
                self._finish_fresh_job(path, future)

    async def _serve_raw_turn(self) -> bool:
        """Claim briefly, prepare independently, and publish through the sole writer."""
        from functools import partial

        from recall.db.source_files import source_key
        from recall.parsers import all_parsers
        from recall.services.coordinator import is_paused, prepare_raw_sources, select_raw_sources

        async with self._raw_slots:
            async with self._raw_turn_lock:
                config = self._load_runtime_config()
                if is_paused(config) or self._shutdown_event.is_set():
                    return False
                parsers = {parser.source.value: parser for parser in all_parsers(config.sources)}
                active = self._active_source_paths(config)
                epoch = self._database_epoch
                selected = await self._writer_call(
                    lambda: select_raw_sources(
                        parsers,
                        conn=self._get_conn(),
                        scheduler=self._raw_scheduler,
                        active_paths=active,
                        active_since_ns=int(
                            (time.time() - config.daemon.live_idle_threshold) * 1e9
                        ),
                        requested_paths=tuple(self._raw_requests) + tuple(self._fresh_jobs),
                        excluded_keys=tuple(self._raw_claims),
                        can_admit=lambda item, priority: self._raw_candidate_admitted(
                            item.source,
                            is_active=item.is_active,
                            priority=priority,
                        ),
                    ),
                    "raw scheduling",
                    affects_enrichment=False,
                )
                if not selected:
                    return False
                assert len(selected) == 1
                item = selected[0].source
                key = source_key(item.source, item.source_path)
                assert key not in self._raw_claims and len(self._raw_claims) < 2
                self._raw_claims[key] = self._reserve_raw_preparation(
                    item, historical=not selected[0].is_active
                )
                request = self._raw_requests.get(SourcePath(item.source, item.source_path))
                if request is not None:
                    config = request.config
                    parsers[item.source] = request.parser
            try:
                prepared = await self._await_shared_conn_work(
                    asyncio.get_running_loop().run_in_executor(
                        self._executor,
                        partial(
                            prepare_raw_sources,
                            (item,),
                            parsers,
                            full=request is not None and request.full,
                        ),
                    ),
                    "raw preparation",
                )
                await self._writer_call(
                    lambda: (
                        self._commit_raw_sources(prepared, config, request)
                        if epoch == self._database_epoch
                        else 0
                    ),
                    "raw commit",
                )
                if epoch == self._database_epoch:
                    await self._publish_session_indexed(item.source_path, source=item.source)
                self._note_raw_turn_served()
                return True
            finally:
                # Executor cancellation is drained before this point, so the
                # reservation includes prepared-but-uncommitted payloads too.
                reservation = self._raw_claims.pop(key)
                self._release_raw_preparation(reservation)
                self._raw_wakeup.set()

    async def _refresh_now(self, paths: Sequence[SourcePath], *, timeout: float) -> bool:
        """Await one bounded batch of captured generations within one shared budget.

        Returns whether all of them finished. The budget spans the whole
        request, not each path: a fleet listing of twenty sessions must answer
        in `live.fresh_timeout`, not twenty times it.

        A path that runs out of budget is left running rather than cancelled.
        `index_session_now` holds the write lock across its executor work and
        `_await_shared_conn_work` only honours cancellation once that work
        finishes (REQ-RESIL-021), so cancelling would extend the wait instead of
        bounding it -- and abandoning the lock mid-statement is the very
        interleaving REQ-RESIL-021 rules out. Shielding lets the answer return on time while
        the index completes on its own.

        A path no parser claims is skipped: one unreadable transcript must not
        deny the rest of the listing its refresh.
        """
        if timeout <= 0:
            raise ValueError("fresh timeout must be positive")
        if not paths:
            return True
        self.live_fresh_requests += 1
        task = asyncio.create_task(self._index_sessions_now(paths, skip_unclaimed=True))
        try:
            await asyncio.wait_for(asyncio.shield(task), timeout)
        except TimeoutError:
            logger.info("fresh read timed out with %s paths still indexing", len(paths))
            self.live_fresh_timeouts += 1
            task.add_done_callback(self._consume_refresh_result)
            return False
        except asyncio.CancelledError:
            task.add_done_callback(self._consume_refresh_result)
            raise
        return True

    @staticmethod
    def _consume_refresh_result(task: asyncio.Task[None]) -> None:
        if not task.cancelled():
            task.exception()

    def _report_source_result(self, task: asyncio.Future[None], *, path: str) -> None:
        """Report once per coalesced source job, even after every waiter leaves.

        Missing source paths are normal while projects move. Other failures
        retain their traceback; fatal database invalidation stops the daemon
        (REQ-RESIL-011). Reporting never changes what awaiting callers receive.
        """
        if task.cancelled():
            return
        err = task.exception()
        if err is None:
            return
        if isinstance(err, NoParserForPath):
            logger.debug("fresh read skipped unparseable path %s", path)
            return
        if self._stop_on_fatal_db(err, "fresh_index"):
            return
        if (
            isinstance(err, FileNotFoundError)
            and err.filename is not None
            and Path(err.filename) == Path(path)
        ):
            logger.info("fresh index source disappeared: %s", path)
            return
        logger.warning(
            "fresh index failed: %s",
            path,
            exc_info=(type(err), err, err.__traceback__),
        )

    async def _publish_session_indexed(self, path: str, *, source: str | None = None) -> None:
        """Announce that a transcript's new rows are committed and readable.

        The high-water read is skipped when nothing is listening for this path:
        the drain loop calls this for every path it indexes, and an idle daemon
        must not pay for a query no one reads.
        """
        if not self._session_indexed.has_subscribers(path):
            return
        from recall.services.live import session_indexed_event

        event = await self._run_readonly(
            lambda conn: session_indexed_event(path, source=source, conn=conn)
        )
        self._session_indexed.publish(event)

    async def _handle_live_sessions(
        self, params: dict[str, Any], client: ClientConnection | None
    ) -> Any:
        """Serve the live set joined to the index (REQ-LIVE-001).

        The live set exists only in watch mode, so a daemon without one reports
        `watching: false` rather than an empty list: it cannot tell a caller
        that nothing is active, only that it is not in a position to know.
        """
        from recall.services.live import LiveSessionsView, live_session_rows

        runtime = self._watch_runtime
        if runtime is None:
            return LiveSessionsView(
                watching=False,
                idle_threshold_seconds=float(self._config.daemon.live_idle_threshold),
            )
        members = runtime.live_set.members()
        rows = await self._run_readonly(lambda conn: live_session_rows(members, conn=conn))
        return LiveSessionsView(
            watching=True,
            idle_threshold_seconds=float(runtime.config.daemon.live_idle_threshold),
            sessions=rows,
        )

    async def _handle_live(self, params: dict[str, Any], client: ClientConnection | None) -> Any:
        return await self._responsive_read(params, lambda: self._read_live(params, client))

    async def _read_live(self, params: dict[str, Any], client: ClientConnection | None) -> Any:
        """Serve `recall live` rows (REQ-LIVE-002).

        `watching` identifies the notification accelerator. Poll observations
        still establish activity; coverage and continuation make incomplete or
        filtered pages explicit.
        """
        from recall.services.coordinator import reconciliation_status
        from recall.services.live import (
            live_fresh_catch_up_paths,
            live_metadata_coverage,
            live_roster_query,
            live_view_page,
        )
        from recall.services.session_ids import resolve_session_identifier

        runtime = self._watch_runtime
        watched_paths = (
            [str(member.path) for member in runtime.live_set.members()]
            if runtime is not None
            else []
        )
        include_idle = bool(params.get("all", False))
        source_value = params.get("source")
        source = parse_source(source_value).value if source_value else None
        project = params.get("project")
        host_value = params.get("host")
        host = str(host_value) if host_value is not None else None
        limit = int(params.get("limit", 50))
        if not 1 <= limit <= 256:
            raise RpcError(code=INVALID_PARAMS, message="limit must be 1..256")
        cursor = params.get("cursor")
        if cursor is not None and not isinstance(cursor, str):
            raise RpcError(code=INVALID_PARAMS, message="live page cursor must be a string")
        epoch = self._database_epoch
        key = (
            "live",
            limit,
            cursor,
            include_idle,
            source,
            project,
            host,
            params.get("session"),
            tuple(sorted(watched_paths)),
            runtime is not None,
        )

        def read(conn: Any, *, probe: bool = False) -> Any:
            now = datetime.now()
            options = {
                "watched_paths": watched_paths,
                "include_idle": include_idle,
                "now": now,
                "idle_window_seconds": float(self._config.live.idle_window),
                "active_window_seconds": float(self._config.daemon.live_idle_threshold),
                "source": source,
                "project": project,
                "host": host,
            }
            session_id = (
                resolve_session_identifier(conn, params["session"], source=source, host=host).id
                if params.get("session") is not None
                else None
            )
            page = live_view_page(
                conn=conn,
                limit=limit,
                cursor=cursor,
                local_host=default_session_host(),
                session_id=session_id,
                _hydrate_turn=not probe,
                **options,
            )
            if probe:
                return _LiveProbe(page)
            status = reconciliation_status(self._config, conn=conn, limit=1)
            coverage = live_metadata_coverage(
                conn,
                query=live_roster_query(session_id=session_id, **options),
                watched_count=len(watched_paths),
                catalog_scan_complete=bool(status["catalog_scan_complete"]),
            )
            return _LiveResponse(page, coverage, runtime is not None)

        # `--fresh` catches up already-indexed behind rows on this page, then
        # re-reads. Never-indexed rows stay on the coordinator. Reading first
        # keeps the catch-up set exact. The probe needs membership and freshness,
        # so leave turn payloads and coverage for the final response snapshot.
        if bool(params.get("fresh", False)):
            probe = await self._cohort_snapshot(
                (*key, "probe"), epoch, lambda conn: read(conn, probe=True)
            )
            behind = live_fresh_catch_up_paths(probe.page.sessions)
            if behind:
                budget = SERVER_READ_BUDGET.get()
                assert budget is not None
                timeout = min(
                    self._config.live.fresh_timeout,
                    _ROSTER_CATCH_UP_SECONDS,
                    budget.remaining(reserve=_FINAL_SNAPSHOT_RESERVE_SECONDS),
                )
                if timeout > 0:
                    await self._refresh_now(behind, timeout=timeout)

        return await self._cohort_snapshot((*key, "final"), epoch, read)

    async def _handle_live_mark(
        self, params: dict[str, Any], client: ClientConnection | None
    ) -> Any:
        """Record the writer pid a harness hook stamped on a session (REQ-LIVE-008).

        The write goes through the daemon like every other one, so the hook
        never takes the database lock itself and never blocks the harness it
        runs inside. Nothing here requires the session to be indexed: a
        `SessionStart` hook fires before the first byte is written, and a mark
        that had to wait for an index pass would be absent exactly when the
        session is most interesting.
        """
        from recall.services.live_marks import pid_alive, upsert_live_mark
        from recall.services.session_ids import resolve_session_identifier

        session_id = params.get("session")
        if not session_id:
            raise RpcError(code=INVALID_PARAMS, message="session is required")
        source_value = params.get("source")
        try:
            source = parse_source(source_value).value if source_value else None
        except ValueError as err:
            raise RpcError(code=INVALID_PARAMS, message=str(err)) from err
        try:
            pid = int(params["pid"])
        except (KeyError, TypeError, ValueError) as err:
            raise RpcError(code=INVALID_PARAMS, message="pid must be an integer") from err
        if pid <= 0:
            raise RpcError(code=INVALID_PARAMS, message="pid must be positive")
        host = default_session_host()
        # Naive local, the same wall clock every other timestamp in this
        # database is stored in (see the DuckDB TIMESTAMP down-conversion).
        marked_at = datetime.now()

        def _write() -> tuple[str, str, str | None]:
            conn = self._get_conn()
            try:
                identity = resolve_session_identifier(
                    conn, str(session_id), source=source, host=host
                )
            except UnknownSessionError:
                # SessionStart can run before the transcript exists.
                mark_source = source or "claude_code"
                source_id = str(session_id)
                source_path = None
            else:
                if not identity.source_session_id:
                    raise ValueError(f"session has no source session id: {identity.id}")
                mark_source = identity.source
                source_id = identity.source_session_id
                source_path = identity.source_path
            upsert_live_mark(
                conn,
                source=mark_source,
                source_session_id=source_id,
                host=host,
                pid=pid,
                marked_at=marked_at,
            )
            return mark_source, source_id, source_path

        mark_source, source_id, source_path = await self._writer_call(_write, "live mark")
        if source_path is not None and pid_alive(pid):
            self._note_source_activity(source_path)
        return {
            "marked": True,
            "source": mark_source,
            "source_session_id": source_id,
            "host": host,
            "pid": pid,
            "marked_at": marked_at,
        }

    async def _handle_show_follow(
        self, params: dict[str, Any], client: ClientConnection | None
    ) -> Any:
        """Stream a session's new messages until a deadline (REQ-LIVE-004/011).

        The stream reads from its **own** cursor and lets the rows decide what
        is new; the event only says when to look. The channel's slot is depth
        one, so two writes landing back to back collapse into a single wake, and
        a follower that trusted the event's `high_water_idx` would drop the
        message in between.

        The frame is bounded by the caller's own cursor, the same way
        `show --after` is: what landed since the last wake is small, and a
        deliberate replay from far back is the caller's choice to make.

        Closing is evented on both conditions that can end a stream -- the next
        indexed event and the daemon shutting down -- so a follower on a 60s
        deadline does not hold shutdown open for the rest of it.
        """
        from recall.services.live import decode_cursor, encode_cursor
        from recall.services.live_events import live_path_key
        from recall.services.sessions import resolve_session_header

        if client is None:
            raise RpcError(
                code=INVALID_PARAMS,
                message="follow requires a streaming connection",
            )
        session_id = params.get("session_id")
        if not session_id:
            raise RpcError(code=INVALID_PARAMS, message="session_id is required")
        timeout = float(params.get("timeout", DEFAULT_FOLLOW_TIMEOUT))
        if timeout <= 0:
            raise RpcError(code=INVALID_PARAMS, message="timeout must be positive")
        include_tools = bool(params.get("tools", False))
        after = params.get("after")

        try:
            header = await self._run_readonly(
                lambda conn: resolve_session_header(session_id, conn=conn)
            )
        except ValueError as err:
            err_msg = str(err)
            if err_msg.startswith(("session not found", "session not indexed")):
                raise RpcError(code=APP_NOT_FOUND, message=err_msg) from err
            raise
        resolved_id = header.id
        path = header.source_path

        epoch = await self._follow_epoch(resolved_id, path)
        if after is not None:
            try:
                supplied_cursor = decode_cursor(str(after))
                cursor_session, last_idx = supplied_cursor
            except ValueError as err:
                raise RpcError(code=INVALID_PARAMS, message=str(err)) from err
            if cursor_session != resolved_id:
                raise RpcError(
                    code=INVALID_PARAMS,
                    message=f"cursor belongs to another session: {cursor_session}",
                )
            if (supplied_cursor.content_epoch or 0) != epoch:
                return {
                    "event": "closed",
                    "reason": "content_rewritten",
                    "cursor_reset": True,
                    "cursor": encode_cursor(resolved_id, -1, epoch),
                }
        else:
            # Follow from here. A caller that wanted the history behind it would
            # have passed the cursor naming where to start.
            last_idx = await self._follow_high_water(resolved_id)

        runtime = self._watch_runtime
        watched_key = live_path_key(path)
        watching = runtime is not None and any(
            live_path_key(str(member.path)) == watched_key for member in runtime.live_set.members()
        )

        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout
        reason = "timeout"
        with self._session_indexed.subscribe(path) as subscription:
            while True:
                messages, observed_epoch = await self._follow_snapshot(
                    resolved_id, path, after_idx=last_idx, tools=include_tools
                )
                if observed_epoch != epoch:
                    epoch = observed_epoch
                    last_idx = -1
                    reason = "content_rewritten"
                    break
                if messages:
                    last_idx = messages[-1].idx
                    await client.send_notification(
                        "live.delta",
                        {
                            "event": "delta",
                            "cursor": encode_cursor(resolved_id, last_idx, epoch),
                            "messages": _serialize(messages),
                        },
                    )
                if self._shutdown_event.is_set():
                    reason = "daemon_stopped"
                    break
                remaining = deadline - loop.time()
                if remaining <= 0:
                    break
                if await self._wait_for_delta_or_stop(subscription, timeout=remaining):
                    reason = "daemon_stopped"
                    break

        return {
            "event": "closed",
            "reason": reason,
            "cursor_reset": reason == "content_rewritten",
            "cursor": encode_cursor(resolved_id, last_idx, epoch),
            "watching": watching,
        }

    async def _follow_epoch(self, session_id: str, path: str) -> int:
        from recall.services.live import catalog_progress_for_session

        def read(conn: Any) -> int:
            progress = catalog_progress_for_session(conn, session_id=session_id, source_path=path)
            return progress.content_epoch if progress else 0

        return await self._run_readonly(read)

    async def _follow_snapshot(
        self, session_id: str, path: str, *, after_idx: int, tools: bool
    ) -> Any:
        from recall.services.live import catalog_progress_for_session
        from recall.services.sessions import load_session_tail

        def read(conn: Any) -> Any:
            progress = catalog_progress_for_session(conn, session_id=session_id, source_path=path)
            messages = load_session_tail(
                session_id, after_idx=after_idx, tools=tools, conn=conn
            ).messages
            return messages, progress.content_epoch if progress else 0

        return await self._run_readonly_snapshot(read)

    async def _follow_delta(self, session_id: str, *, after_idx: int, tools: bool) -> list[Any]:
        """Messages indexed after `after_idx`, read fresh at each wake.

        A method rather than a closure over the stream's loop variable: the
        cursor is what decides the window, so it is passed, not captured.
        """
        from recall.services.sessions import load_session_tail

        return await self._run_readonly_snapshot(
            lambda conn: (
                load_session_tail(
                    session_id,
                    after_idx=after_idx,
                    tools=tools,
                    config=self._config,
                    conn=conn,
                ).messages
            )
        )

    async def _follow_high_water(self, session_id: str) -> int:
        """Highest message idx currently indexed, or -1 for a session with none.

        -1 rather than 0 so `after_idx > -1` selects the whole session, matching
        what `encode_cursor` writes for an empty one.
        """
        from recall.services.sessions import load_session_tail

        messages = await self._run_readonly_snapshot(
            lambda conn: (
                load_session_tail(session_id, tail=1, config=self._config, conn=conn).messages
            )
        )
        return messages[-1].idx if messages else -1

    async def _wait_for_delta_or_stop(self, subscription: Any, *, timeout: float) -> bool:
        """Wait for the next indexed event or for shutdown. True means shutdown.

        Both are events, so both are awaited as events. Polling `_shutdown_event`
        between indexed events would make a quiet follower's exit depend on how
        often the session it watches happens to be written to.
        """
        waiters = {
            asyncio.ensure_future(subscription.wait(timeout=timeout)),
            asyncio.ensure_future(self._shutdown_event.wait()),
        }
        try:
            done, _pending = await asyncio.wait(waiters, return_when=asyncio.FIRST_COMPLETED)
        finally:
            for waiter in waiters:
                waiter.cancel()
        for waiter in done:
            with suppress(asyncio.CancelledError, Exception):
                waiter.result()
        return self._shutdown_event.is_set()

    async def _handle_check_indexes(
        self, params: dict[str, Any], client: ClientConnection | None
    ) -> Any:
        """Run the index/table divergence probe live on a read cursor (REQ-RESIL-018)."""
        from recall.db.maintenance import probe_index_divergence

        sample = _resolve_probe_sample(params.get("sample"))
        report = await self._run_readonly(lambda conn: probe_index_divergence(conn, sample=sample))
        self._last_index_probe = report
        return report.to_payload()

    # ---- Connection handling ----

    def _connection_protocol(self) -> asyncio.StreamReaderProtocol:
        reader = _PeerStreamReader()
        return _PeerStreamProtocol(
            reader, lambda _reader, writer: self._handle_connection(reader, writer)
        )

    async def _handle_connection(
        self, reader: _PeerStreamReader, writer: asyncio.StreamWriter
    ) -> None:
        if self._active_connections >= _MAX_CONNECTIONS:
            writer.close()
            with suppress(Exception):
                await writer.wait_closed()
            return
        task = asyncio.current_task()
        assert task is not None
        self._client_tasks.add(task)
        self._active_connections += 1
        client = ClientConnection(writer)
        try:
            while not self._shutdown_event.is_set():
                try:
                    line = await asyncio.wait_for(reader.readline(), timeout=1.0)
                except TimeoutError:
                    if reader.at_eof():
                        break
                    continue
                if not line:
                    break
                self._last_request_time = asyncio.get_running_loop().time()
                await self._process_request(line, writer, client, reader=reader)
        except ConnectionResetError:
            pass
        except Exception as err:
            logger.error("connection handler error: %s", err)
        finally:
            self._client_tasks.discard(task)
            self._active_connections -= 1
            writer.close()
            with suppress(Exception):
                await writer.wait_closed()

    async def _wait_for_request_end(
        self,
        departure: asyncio.Event | None,
        *,
        timeout: float | None,
        cancel_on_shutdown: bool,
        timing: SearchRequest | None = None,
    ) -> str:
        """Await peer departure or the read deadline without reading input.

        Streams own shutdown: a connected stream must finish its terminal
        response before the connection loop closes its writer. Reads and
        resumable writes retain their shutdown cancellation boundary.
        """
        waiters: dict[asyncio.Task[bool], str] = {}
        if departure is not None:
            waiters[asyncio.create_task(departure.wait())] = "disconnect"
        if cancel_on_shutdown:
            waiters[asyncio.create_task(self._shutdown_event.wait())] = "shutdown"
        try:
            async with asyncio.timeout(timeout):
                if not waiters:
                    await asyncio.Future()
                done, _pending = await asyncio.wait(waiters, return_when=asyncio.FIRST_COMPLETED)
                return waiters[done.pop()]
        except TimeoutError:
            if timing is not None:
                timing.freeze()
            return "timeout"
        finally:
            for waiter in waiters:
                waiter.cancel()
            await asyncio.gather(*waiters, return_exceptions=True)

    async def _run_request_handler(
        self,
        method: str,
        handler: MethodHandler,
        params: dict[str, Any],
        client: ClientConnection,
        reader: _PeerStreamReader | None,
        *,
        timeout: float | None,
        request_id: object = None,
    ) -> Any:
        """Cancel a handler on deadline or peer departure and drain its cleanup."""
        timing = SearchRequest(request_id, timeout) if method == "recall.search" else None

        async def invoke() -> Any:
            token = search_request.set(timing)
            try:
                return await handler(params, client)
            finally:
                search_request.reset(token)

        request = asyncio.create_task(invoke())
        departure = None
        if reader is not None:
            departure = reader.input_closed if method in _STREAMING_METHODS else reader.disconnected
        request_end = asyncio.create_task(
            self._wait_for_request_end(
                departure,
                timeout=timeout,
                cancel_on_shutdown=method in _BOUNDED_READ_METHODS
                or method in _WATCHED_WRITE_METHODS,
                timing=timing,
            )
        )
        try:
            done, _pending = await asyncio.wait(
                {request, request_end}, return_when=asyncio.FIRST_COMPLETED
            )
            if request in done:
                return request.result()
            reason = request_end.result()
            snapshot = timing.freeze() if timing is not None and reason == "timeout" else None
            cleanup_started = time.monotonic()
            request.cancel()
            await asyncio.gather(request, return_exceptions=True)
            cleanup_ms = int((time.monotonic() - cleanup_started) * 1000)
            if reason in {"disconnect", "shutdown"}:
                raise ClientDisconnected(method)
            assert timeout is not None, "a handler with no deadline cannot time out"
            message = f"{method} timed out after {timeout:g} seconds"
            if snapshot is not None:
                message += (
                    f" (phase: {snapshot.phase.value}; "
                    f"largest phase: {snapshot.largest_phase.value})"
                )
            raise RpcError(
                code=INTERNAL_ERROR,
                message=message,
                data=snapshot.details(timeout, cleanup_ms) if snapshot is not None else None,
            )
        finally:
            request.cancel()
            request_end.cancel()
            await asyncio.gather(request, request_end, return_exceptions=True)

    async def _process_request(
        self,
        line: bytes,
        writer: asyncio.StreamWriter,
        client: ClientConnection,
        *,
        reader: _PeerStreamReader | None = None,
    ) -> None:
        request_id: Any = None
        try:
            data = json.loads(line)
        except json.JSONDecodeError as err:
            await self._send_error(writer, None, PARSE_ERROR, f"parse error: {err}")
            return

        if not isinstance(data, dict) or data.get("jsonrpc") != "2.0":
            await self._send_error(writer, None, INVALID_REQUEST, "invalid JSON-RPC 2.0 request")
            return

        request_id = data.get("id")
        method = data.get("method")
        params = data.get("params", {})

        if not isinstance(method, str):
            await self._send_error(writer, request_id, INVALID_REQUEST, "method must be a string")
            return

        if not isinstance(params, dict):
            await self._send_error(writer, request_id, INVALID_PARAMS, "params must be an object")
            return

        handler = self._methods.get(method)
        if handler is None:
            await self._send_error(
                writer,
                request_id,
                METHOD_NOT_FOUND,
                f"method not found: {method}",
            )
            return

        try:
            if method in _WATCHED_WRITE_METHODS and not _departure_ends_write(method, params):
                logger.info(
                    "%s runs to completion whether or not its client stays: "
                    "watch mode cannot resume this work (REQ-RPC-018)",
                    method,
                )
                result = await handler(params, client)
            else:
                # Every request, including a silent stream, shares the peer's
                # lifetime. Only ordinary reads own the fixed read deadline;
                # streams and writes keep their own completion conditions.
                timeout = _READ_REQUEST_TIMEOUT if method in _BOUNDED_READ_METHODS else None
                result = await self._run_request_handler(
                    method, handler, params, client, reader, timeout=timeout, request_id=request_id
                )
            try:
                await self._send_result(writer, request_id, result)
            except (BrokenPipeError, ConnectionError) as err:
                # A write that ran on for a departed client has nobody to
                # answer. Only the send is guarded: a connection failure raised
                # by the handler itself is the caller's error to hear.
                logger.info("client for %s departed before its result: %s", method, err)
        except ClientDisconnected:
            # A streaming handler noticed its peer is gone. There is nobody to
            # answer, and the handler's own cleanup already ran on the way out,
            # so this is a clean end to the request rather than an error.
            logger.info("client disconnected during %s", method)
        except RpcError as err:
            await self._send_error(writer, request_id, err.code, err.message, err.data)
        except ValueError as err:
            await self._send_error(writer, request_id, INVALID_PARAMS, str(err))
        except RecallLockError as err:
            await self._send_error(writer, request_id, APP_LOCKED, str(err))
        except Exception as err:
            # Mirror _fail_on_watch_task_exit: answer the client (best
            # effort), then stop so the scheduler restarts us with a fresh
            # DB instance. The shutdown decision lives in a finally because
            # the client that exposes a wedged DB is exactly the one likely
            # to have timed out and disconnected — a failed error-send must
            # not leave the daemon wedged (REQ-RESIL-011).
            try:
                await self._send_error(writer, request_id, INTERNAL_ERROR, str(err))
            finally:
                self._stop_on_fatal_db(err, method)

    async def _send_result(
        self, writer: asyncio.StreamWriter, request_id: Any, result: Any
    ) -> None:
        if isinstance(result, _LiveResponse):
            # Only the per-request metadata differs among this frozen cohort's
            # replies. Encode the bounded roster once on its read worker.
            metadata = json.dumps(
                {"jsonrpc": "2.0", "id": request_id, "_config_fp": self._config_fp},
                separators=(",", ":"),
            ).encode("utf-8")
            writer.write(metadata[:-1] + b',"result":' + result.encoded + b"}\n")
            await writer.drain()
            return
        response: dict[str, Any] = {
            "jsonrpc": "2.0",
            "result": _serialize(result),
            "id": request_id,
            "_config_fp": self._config_fp,
        }
        line = json.dumps(response, separators=(",", ":")) + "\n"
        writer.write(line.encode("utf-8"))
        await writer.drain()

    async def _send_error(
        self,
        writer: asyncio.StreamWriter,
        request_id: Any,
        code: int,
        message: str,
        data: Any = None,
    ) -> None:
        error: dict[str, Any] = {"code": code, "message": message}
        if data is not None:
            error["data"] = _serialize(data)
        response: dict[str, Any] = {
            "jsonrpc": "2.0",
            "error": error,
            "id": request_id,
        }
        line = json.dumps(response, separators=(",", ":")) + "\n"
        writer.write(line.encode("utf-8"))
        await writer.drain()

    # ---- PID / socket management ----

    def _write_pid_file(self) -> None:
        create_private_dir(self._pid_path.parent)
        self._pid_path.write_text(str(os.getpid()), encoding="utf-8")

    def _remove_pid_file(self) -> None:
        self._pid_path.unlink(missing_ok=True)

    def _remove_socket(self) -> None:
        self._socket_path.unlink(missing_ok=True)

    def _cleanup_stale_socket(self) -> None:
        if not self._socket_path.exists():
            return
        if self._pid_path.exists():
            try:
                pid = int(self._pid_path.read_text(encoding="utf-8").strip())
                os.kill(pid, 0)
                raise RuntimeError(
                    f"daemon already running (pid {pid}). Use `recall daemon stop` first."
                )
            except (ValueError, ProcessLookupError, PermissionError):
                pass
        self._remove_socket()
        self._remove_pid_file()

    # ---- Idle monitor ----

    async def _idle_monitor(self) -> None:
        if self._idle_timeout is None:
            return
        # Poll at most every second, or half the timeout for short values
        poll_interval = min(1.0, self._idle_timeout / 2)
        while not self._shutdown_event.is_set():
            await asyncio.sleep(poll_interval)
            if self._active_connections > 0:
                continue
            elapsed = asyncio.get_running_loop().time() - self._last_request_time
            if elapsed >= self._idle_timeout:
                logger.info("idle timeout reached (%.0fs), shutting down", elapsed)
                self._shutdown_event.set()
                break

    async def _service_probe_monitor(self) -> None:
        """Measure status's service view off the request path (REQ-DAEMON-081)."""
        from recall.services.runtime_state import load_runtime_status

        try:
            record = await self._run_readonly(
                lambda conn: load_runtime_status(self._config, conn=conn).installed_scheduler
            )
            await asyncio.to_thread(self._service_probes.refresh, record)
        except (OSError, RuntimeError, duckdb.Error):
            # Status then measures on its first request, as it did before.
            logger.warning("initial service probe failed", exc_info=True)
        while not self._shutdown_event.is_set():
            try:
                await asyncio.wait_for(
                    self._shutdown_event.wait(), timeout=self._service_probes.refresh_delay()
                )
                break
            except TimeoutError:
                pass
            try:
                await asyncio.to_thread(self._service_probes.refresh_current)
            except (OSError, RuntimeError):
                # The previous measurement stays served, dated by its measured-at.
                logger.warning("service probe refresh failed", exc_info=True)

    async def _compact_monitor(self) -> None:
        """Run scheduled auto-compaction checks away from the asyncio thread."""
        from recall.services.daemon import _maybe_run_auto_compact

        interval = self._config.compaction.check_interval_hours * 3600
        logger.info("auto-compact monitor started, interval=%ds", interval)
        try:
            while not self._shutdown_event.is_set():
                try:
                    await asyncio.wait_for(self._shutdown_event.wait(), timeout=interval)
                    break
                except TimeoutError:
                    pass
                if self._shutdown_event.is_set():
                    break
                try:
                    async with self._index_request_lock, self._write_lock:
                        if self._shutdown_event.is_set():
                            break
                        tick = asyncio.get_running_loop().run_in_executor(
                            self._executor,
                            _maybe_run_auto_compact,
                            self,
                            self._config,
                        )
                        try:
                            await asyncio.shield(tick)
                        except asyncio.CancelledError:
                            # The executor work cannot be interrupted once it
                            # starts. Keep the write lock held until DuckDB has
                            # been reopened so shutdown cannot close the shared
                            # connection while file-level compaction still owns it.
                            logger.info(
                                "auto-compact monitor cancellation requested; "
                                "waiting for active tick"
                            )
                            try:
                                await asyncio.shield(tick)
                            except Exception:
                                logger.exception("auto-compact monitor tick failed during shutdown")
                            raise
                except Exception:
                    logger.exception("auto-compact monitor tick failed; daemon continuing")
        finally:
            logger.info("auto-compact monitor stopping")

    # ---- Server lifecycle ----

    def _invalidate_moved_history(self) -> None:
        self._write_lock.invalidate_enrichment()
        if self._config.fts.backend == "duckdb":
            # Copy cleanup retires keyword identities and can carry stored context.
            self._keyword_dirty = True

    async def _init_fts(self, *, reason: str) -> None:
        """Refresh DuckDB keyword coverage; sidecar repair has its own writer path."""
        fts = self._config.fts
        if fts.backend == "sqlite_sidecar" or not fts.fields:
            logger.debug(
                "FTS refresh skipped: reason=%s backend=%s skip=%s",
                reason,
                fts.backend,
                "sidecar maintained incrementally"
                if fts.backend == "sqlite_sidecar"
                else "no fields",
            )
            self._keyword_dirty = False
            return

        from recall.db import create_fts_indexes

        def _do() -> None:
            try:
                conn = self._get_conn()
                create_fts_indexes(conn, fts)
                logger.debug("FTS indexes refreshed: reason=%s backend=%s", reason, fts.backend)
            except Exception as err:
                self._recover_shared_conn(err, "init_fts")
                raise

        async with self._write_lock:
            await self._await_shared_conn_work(
                asyncio.get_running_loop().run_in_executor(self._executor, _do),
                "FTS init",
            )
            self._keyword_dirty = False

    async def start(
        self,
        *,
        idle_timeout: float | None = None,
        watch: bool = False,
    ) -> None:
        from recall.core.keyword_maintenance import require_keyword_writes_enabled
        from recall.db.ownership import daemon_ownership

        require_keyword_writes_enabled(self._config.data_dir)
        validate_socket_path(self._socket_path)
        async with daemon_ownership(self._config.db_path):
            try:
                await self._start_owned(idle_timeout=idle_timeout, watch=watch)
            finally:
                # Cancellation must drain executor work and close every handle
                # before the OS ownership descriptor is released.
                cleanup = asyncio.create_task(self.stop())
                cancelled = False
                while not cleanup.done():
                    try:
                        await asyncio.shield(cleanup)
                    except asyncio.CancelledError:
                        cancelled = True
                cleanup.result()
                if cancelled:
                    raise asyncio.CancelledError

    async def _start_owned(self, *, idle_timeout: float | None, watch: bool) -> None:
        self._idle_timeout = idle_timeout
        self._runtime_watch = watch
        from recall.db.connection import resolve_memory_limit_settings

        memory_limit = resolve_memory_limit_settings(self._config)
        if memory_limit.clamped:
            logger.warning(
                "DuckDB memory_limit clamped: requested=%s applied=%s; "
                "set [duckdb] memory_limit_exceeds_floor = true to honor the override "
                "(whole-process RSS may exceed the 4 GiB floor)",
                memory_limit.requested,
                memory_limit.applied,
            )
        self._cleanup_stale_socket()
        create_private_dir(self._socket_path.parent)

        # Bind a server-owned executor to the running loop so all
        # `run_in_executor(self._executor, ...)` calls below have a stable
        # target, and so shutdown is driven by `self.stop()` rather than
        # asyncio.run()'s implicit `shutdown_default_executor` step.
        if self._executor is None:
            self._executor = DiagnosticExecutor(
                max_workers=_WRITER_WORKERS, thread_name_prefix="recall-rpc"
            )

        # Start listening BEFORE FTS init so the socket exists for clients.
        # Without this ordering, a schema-version mismatch in the DB causes
        # _init_fts → _get_conn → ensure_schema to crash, and the socket
        # never appears — deadlocking `recall index --recreate` which needs
        # the daemon to be listening in order to send the recreate RPC.
        bound_socket = _bind_owner_only_socket(self._socket_path)
        self._owns_runtime_artifacts = True
        try:
            self._server = await asyncio.get_running_loop().create_unix_server(
                self._connection_protocol, sock=bound_socket
            )
        except BaseException:
            bound_socket.close()
            raise
        self._write_pid_file()
        self._last_request_time = asyncio.get_running_loop().time()
        logger.info(
            "RPC server listening on %s (pid %d)",
            self._socket_path,
            os.getpid(),
        )

        from recall.services.daemon import run_startup_fts_sidecar_sync, run_startup_snapshot_gc
        from recall.services.self_repair import DaemonStartupRefused

        # Seed the bloat ratio once at startup — before self-repair opens the
        # shared connection, since the same file cannot be opened read-only once
        # it exists — so CLI health notices have a value immediately rather than
        # only after the first (6h) auto-compact check. Best-effort: a failed
        # estimate must never block the daemon from listening.
        try:
            from recall.services.compaction import estimate_bloat_ratio

            def _estimate_bloat() -> float:
                # The socket is already bound. Keep early requests from opening
                # a read-write handle while this probe holds a read-only one.
                self._conn_lifecycle_gate.acquire_write()
                try:
                    return estimate_bloat_ratio(self._config.db_path, self._config).ratio
                finally:
                    self._conn_lifecycle_gate.release_write()

            async with self._write_lock:
                self._last_bloat_ratio = await self._await_shared_conn_work(
                    asyncio.get_running_loop().run_in_executor(self._executor, _estimate_bloat),
                    "startup bloat estimate",
                )
        except Exception:
            logger.debug("startup bloat estimate failed", exc_info=True)

        # Self-repair runs before any other DuckDB work (sidecar sync, FTS init,
        # catch-up all write) so a database left diverged by a failed checkpoint
        # is healed -- or the start refused -- before the first write can kill
        # us again (REQ-RESIL-015/016). It opens the shared connection and holds
        # it exclusively: the socket is already bound, and a request that reaches
        # `_get_conn()` mid-repair must wait rather than open a second connection.
        try:
            await self._run_startup_self_repair()
        except DaemonStartupRefused:
            raise
        except Exception as err:
            # A dead instance is not a step failure to listen past: the fatal
            # funnel spools it for the next start and the process leaves so
            # the scheduler can relaunch it (REQ-RESIL-015, INV-RESIL-004).
            if self._stop_on_fatal_db(err, "startup_self_repair"):
                raise
            logger.warning("startup self-repair failed; daemon is listening", exc_info=True)

        await self._await_shared_conn_work(
            asyncio.get_running_loop().run_in_executor(
                self._executor, run_startup_snapshot_gc, self._config
            ),
            "startup snapshot GC",
        )

        async def sync_sidecar() -> Any:
            return await self._await_shared_conn_work(
                asyncio.get_running_loop().run_in_executor(
                    self._executor, run_startup_fts_sidecar_sync, self._config
                ),
                "startup sidecar sync",
            )

        self._fts_sidecar_startup = await self._run_duckdb_scan(sync_sidecar)

        try:
            await self._reopen_unsupported_parks()
        except Exception as err:
            if self._stop_on_fatal_db(err, "startup_reopen_unsupported_parks"):
                raise
            logger.warning("startup reopen of unsupported parks failed", exc_info=True)

        # Build FTS indexes after the socket is live.  If this fails (e.g.
        # stale schema version), the daemon stays up so --recreate can fix
        # the DB.  Missing search coverage queues a background keyword repair.
        try:
            await self._init_fts(reason="startup")
            if self._fts_sidecar_startup.error is None:
                logger.info(
                    "FTS indexes initialized on startup (backend=%s)", self._config.fts.backend
                )
        except Exception:
            logger.warning(
                "FTS init failed on startup (schema mismatch?); "
                "daemon is listening — run `recall index --recreate --yes` to fix",
                exc_info=True,
            )

        loop = asyncio.get_running_loop()
        if threading.current_thread() is threading.main_thread():
            loop.add_signal_handler(signal.SIGTERM, self._signal_shutdown)
            loop.add_signal_handler(signal.SIGINT, self._signal_shutdown)

        tasks: list[asyncio.Task[Any]] = []
        watch_runtime = None
        try:
            tasks.extend(
                [
                    asyncio.create_task(self._idle_monitor()),
                    asyncio.create_task(self._checkpoint_monitor()),
                    asyncio.create_task(self._service_probe_monitor()),
                ]
            )
            if self._config.compaction.auto_trigger:
                tasks.append(asyncio.create_task(self._compact_monitor()))
            if watch:
                watch_runtime, watch_task = await self._start_watch_mode()
                tasks.append(watch_task)
            tasks.append(asyncio.create_task(self._run_reconciliation_poll_loop()))
            embed_task = await self._start_embed_phase()
            if embed_task is not None:
                tasks.append(embed_task)
            await self._shutdown_event.wait()
        finally:
            self._shutdown_event.set()
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            if watch_runtime is not None:
                await self._stop_watch_mode(watch_runtime)

    async def _run_inventory_loop(self) -> None:
        from recall.db.source_files import PresentSource, SourceCatalog
        from recall.services.coordinator import persist_prepared_inventory, prepare_raw_cycle
        from recall.services.reconciler import InventoryBatch, unobserved

        loop = asyncio.get_running_loop()
        while not self._shutdown_event.is_set():
            self._inventory_wakeup.clear()
            cfg = self._load_runtime_config()
            try:
                # Only one captured batch is retained across writer turns.
                epoch = self._database_epoch
                runtime = self._watch_runtime
                topology_generation = runtime.topology.generation() if runtime else 0
                events = prepare_raw_cycle(cfg, cancelled=self._shutdown_event.is_set)
                generation = None
                # The walked root's catalog, read once per walk: filtering batches
                # against it keeps an unchanged root away from the writer, and one
                # read costs less than a per-batch lookup across the table.
                present: dict[str, PresentSource] = {}
                while not self._shutdown_event.is_set():
                    event = await self._await_shared_conn_work(
                        loop.run_in_executor(self._executor, next, events, None),
                        "inventory capture",
                    )
                    if event is None:
                        break
                    if isinstance(event.event, InventoryBatch):
                        event = replace(event, event=unobserved(event.event, present))
                        if not event.event.files:
                            continue

                    def persist_event(
                        event=event,
                        generation=generation,
                        expected_epoch: int = epoch,
                    ) -> tuple[int | None, tuple[str, ...]]:
                        if expected_epoch != self._database_epoch:
                            return None, ()
                        return persist_prepared_inventory(
                            event,
                            generation,
                            conn=self._get_conn(),
                            on_enrichment_change=self._invalidate_moved_history,
                        )

                    generation, _ = await self._writer_call(
                        persist_event, "inventory batch", affects_enrichment=False
                    )
                    # The catalog's modification time supplies inventory activity
                    # evidence. Observing old metadata changes is catch-up, not
                    # a live event, even when the source was captured before.
                    if isinstance(event.event, InventoryBatch):
                        # Only sources the catalog did not already hold reach
                        # this point, and a newly catalogued one is pending.
                        self._raw_wakeup.set()
                    if epoch != self._database_epoch:
                        break
                    if event.event is None:
                        present = await self._run_readonly(
                            lambda conn, source=event.parser.source.value, root=str(event.root): (
                                SourceCatalog(conn, clock=time.time).present_states(source, root)
                            )
                        )
                if epoch != self._database_epoch:
                    continue
                if runtime is not None:
                    runtime.topology.settle(topology_generation)
                self._inventory_complete = True
                self._inventory_error = None
            except asyncio.CancelledError:
                raise
            except Exception as err:
                self._inventory_complete = False
                self._inventory_error = f"{type(err).__name__}: {err}"
                logger.warning("reconciliation inventory failed: %s", err)
            # Measured from the end of the walk, so a slow walk never runs back to back.
            timeout = min(cfg.daemon.interval, _INVENTORY_RESCAN_SECONDS_MAX)
            shutdown = asyncio.create_task(self._shutdown_event.wait())
            rescan = asyncio.create_task(self._inventory_wakeup.wait())
            try:
                await asyncio.wait(
                    {shutdown, rescan}, timeout=timeout, return_when=asyncio.FIRST_COMPLETED
                )
            finally:
                shutdown.cancel()
                rescan.cancel()
                await asyncio.gather(shutdown, rescan, return_exceptions=True)

    async def _run_active_observation_loop(self) -> None:
        from recall.db.source_files import SourceCatalog
        from recall.services.coordinator import active_observation_page, capture_active_observations

        loop = asyncio.get_running_loop()
        cursor = None
        failures: list[str] = []
        while not self._shutdown_event.is_set():
            config = self._load_runtime_config()
            epoch = self._database_epoch
            try:
                page = await self._run_readonly(
                    lambda conn, config=config, cursor=cursor: active_observation_page(
                        config, conn=conn, cursor=cursor, now=time.time()
                    )
                )
                batch, errors = await self._await_shared_conn_work(
                    loop.run_in_executor(self._executor, capture_active_observations, config, page),
                    "active observation capture",
                )
                if batch.files:

                    def persist_observations(
                        batch=batch, expected_epoch: int = epoch
                    ) -> tuple[str, ...]:
                        if expected_epoch != self._database_epoch:
                            return ()
                        catalog = SourceCatalog(self._get_conn(), clock=time.time)
                        return catalog.observe_batch(
                            [
                                (item.source, item.root_path, item.source_path, item.signature)
                                for item in batch.files
                            ]
                        )

                    changed = await self._writer_call(
                        persist_observations, "active observation batch", affects_enrichment=False
                    )
                    for path in changed:
                        self._note_source_activity(path)
                cursor = page.next_cursor if epoch == self._database_epoch else None
                failures.extend(errors[: max(0, 32 - len(failures))])
                self._active_observation_error = "; ".join(failures) or None
                if cursor is not None:
                    continue
                failures.clear()
            except asyncio.CancelledError:
                raise
            except Exception as error:
                cursor = None
                failures.clear()
                self._active_observation_error = f"{type(error).__name__}: {error}"
                logger.warning("active observation failed: %s", error)
            with suppress(TimeoutError):
                await asyncio.wait_for(self._shutdown_event.wait(), timeout=1.0)

    async def _repair_keyword_batch(self) -> int:
        from contextlib import closing

        from recall.db import open_sidecar, sidecar_path
        from recall.services.coordinator import is_paused
        from recall.services.fts_sidecar_reconcile import repair_pending_batch

        config = self._load_runtime_config()
        if (
            is_paused(config)
            or self._shutdown_event.is_set()
            or time.monotonic() < self._next_fts_repair_at
        ):
            return 0
        self._next_fts_repair_at = time.monotonic() + max(1, config.daemon.fts_debounce)
        try:
            if config.fts.backend == "duckdb":
                if self._keyword_dirty:
                    debouncer = (
                        self._watch_runtime.fts_debouncer
                        if self._watch_runtime is not None
                        else None
                    )
                    try:
                        await self._init_fts(reason="keyword dirty")
                    except FtsRebuildOutOfMemoryError as err:
                        if debouncer is not None:
                            backoff = debouncer.mark_oom(str(err))
                            self._next_fts_repair_at = time.monotonic() + backoff
                        raise
                    if debouncer is not None:
                        debouncer.mark_rebuilt()
                    self._fts_repair_error = None
                    return 1
                return 0

            def repair():
                with closing(open_sidecar(sidecar_path(config.data_dir))) as sidecar:
                    return repair_pending_batch(
                        self._get_conn(),
                        sidecar,
                        cursor=self._fts_repair_cursor,
                        fts_fields=config.fts.fields,
                    )

            result = await self._writer_call(repair, "keyword repair", affects_enrichment=False)
            self._fts_repair_cursor = result.cursor
            if result.error:
                self._fts_repair_error = result.error
            elif result.remaining == 0:
                self._fts_repair_error = None
            if result.remaining > 0 and result.cursor is not None:
                # Finish this scan before retrying failed rows from its start.
                self._next_fts_repair_at = 0
            return result.applied
        except Exception as err:
            self._fts_repair_error = f"{type(err).__name__}: {err}"
            raise

    async def _reopen_unsupported_parks(self) -> int:
        """Give unsupported parks an older build left behind one reparse.

        This build always stores ``detail`` and acknowledges past unsupported
        records, so no such park appears while it runs and one pass per start
        reaches every one. Raw scheduling stays free of catalog writes when
        nothing changed (REQ-RECON-026, -029).
        """
        from recall.services.unsupported_diagnostics import reopen_unsupported_parks

        reopened = await self._writer_call(
            lambda: reopen_unsupported_parks(self._get_conn(), time.time()),
            "reopen unsupported parks",
            checkpoint_wakeup=False,
        )
        if reopened:
            logger.info("reopened %d unsupported parks for one reparse", reopened)
            self._raw_wakeup.set()
        return reopened

    async def _maybe_begin_index_migration(self, config: AppConfig | None = None) -> None:
        from recall.services.coordinator import is_paused
        from recall.services.index_migration import (
            begin_migration,
            begin_storage_migration,
            is_eligible,
            migration_status,
        )

        cfg = config or self._load_runtime_config()
        if self._shutdown_event.is_set():
            return
        status = await self._run_readonly(migration_status)
        storage_pending = status.phase != "idle" and status.storage_target is not None
        if not storage_pending and (is_paused(cfg) or not status.eligible):
            return

        def _begin() -> None:
            conn = self._get_conn()
            if not isinstance(conn, duckdb.DuckDBPyConnection):
                return
            try:
                current = migration_status(conn)
                if current.phase != "idle" and current.storage_target is not None:
                    begin_storage_migration(conn, cfg)
                    self._finish_storage_maintenance(cfg)
                    conn = self._get_conn()
                if not is_paused(cfg) and is_eligible(conn):
                    begin_migration(conn, cfg)
                    self._finish_storage_maintenance(cfg)
            except Exception as err:
                self._record_maintenance_failure(err)
                raise

        await self._writer_call(_begin, "index migration begin")

    async def _run_storage_maintenance(self, *, expected_plan_id: str | None = None) -> None:
        """Run explicit storage maintenance through the existing writer/handle owner."""
        from recall.services.index_migration import begin_storage_migration, plan_storage_migration

        cfg = self._load_runtime_config()

        def operation() -> None:
            if expected_plan_id is not None:
                plan = plan_storage_migration(self._get_conn(), cfg)
                if plan.plan_id != expected_plan_id:
                    raise ValueError(
                        "storage maintenance plan changed before execution; inspect again"
                    )
            try:
                begin_storage_migration(self._get_conn(), cfg)
                self._finish_storage_maintenance(cfg)
            except Exception as err:
                self._record_maintenance_failure(err)
                raise

        await self._writer_call(operation, "storage maintenance", storage_maintenance=True)

    def _finish_storage_maintenance(self, config: AppConfig) -> None:
        """The caller owns the writer; only handle replacement excludes read cursors."""
        from recall.services.index_migration import (
            migration_status,
            perform_storage_transition,
            verify_and_complete,
        )

        conn = self._get_conn()
        status = migration_status(conn)
        if status.storage_target is None or status.phase == "idle":
            return
        if status.storage_eligible:
            # Flush the legacy intent WAL while read cursors can still run.
            # Closing a dirty legacy file inside the lifecycle gate would add
            # that entire checkpoint to every queued reader's latency.
            conn.execute("FORCE CHECKPOINT")
            self.close_db_connection()
            try:
                perform_storage_transition(config)
            finally:
                self.reopen_db_connection()
            conn = self._get_conn()
        status = migration_status(conn)
        if status.storage_version is None or status.storage_eligible:
            raise RuntimeError("persisted storage format has not been verified after reopen")
        if status.captured == 0:
            verify_and_complete(conn, verified=True)

    def _record_maintenance_failure(self, error: Exception) -> None:
        if self._conn is not None:
            self._conn.execute(
                """UPDATE index_migration_jobs SET phase='failed', error=?
                   WHERE singleton AND phase != 'idle'""",
                [f"{type(error).__name__}: {error}"],
            )

    async def _run_reconciliation_poll_loop(self) -> None:
        """Inventory and shared raw service progress independently of model latency."""
        inventory_task = asyncio.create_task(self._run_inventory_loop())
        active_task = asyncio.create_task(self._run_active_observation_loop())
        try:
            begin_task = asyncio.create_task(self._maybe_begin_index_migration())
            while not self._shutdown_event.is_set():
                self._raw_wakeup.clear()
                worked = False
                try:
                    worked = await self._serve_raw_turn()
                    repaired = await self._repair_keyword_batch()
                    worked = worked or repaired > 0
                except asyncio.CancelledError:
                    raise
                except Exception as err:
                    logger.warning("raw reconciliation failed: %s", err)
                if worked:
                    with suppress(TimeoutError):
                        await asyncio.wait_for(self._shutdown_event.wait(), timeout=0.01)
                    continue
                # An idle turn costs a scheduling query against the whole
                # catalog, so an idle scheduler waits for an edge instead:
                # source activity, a catalogued source, a released claim, a
                # request left to the drain, or lifted write pressure. The
                # rescan covers what becomes eligible with time alone -- an
                # expiring retry backoff -- and the keyword repair's debounce.
                until_repair = self._next_fts_repair_at - time.monotonic()
                timeout = min(_RAW_IDLE_RESCAN_SECONDS, max(0.01, until_repair))
                shutdown = asyncio.create_task(self._shutdown_event.wait())
                wakeup = asyncio.create_task(self._raw_wakeup.wait())
                try:
                    await asyncio.wait(
                        {shutdown, wakeup}, timeout=timeout, return_when=asyncio.FIRST_COMPLETED
                    )
                finally:
                    shutdown.cancel()
                    wakeup.cancel()
                    await asyncio.gather(shutdown, wakeup, return_exceptions=True)
        finally:
            begin_task.cancel()
            inventory_task.cancel()
            active_task.cancel()
            for task in (begin_task, inventory_task, active_task):
                with suppress(asyncio.CancelledError):
                    await task

    async def _run_startup_self_repair(self) -> None:
        from recall.services import self_repair

        def _repair() -> self_repair.StartupRepairOutcome:
            # Exclusive lifecycle access for the whole step: a request that
            # reaches `_get_conn()` meanwhile blocks until the repair releases
            # it (REQ-RESIL-015). Two connections bootstrapping the schema at
            # once fail with `Catalog write-write conflict on create ...
            # schema_version`, which is what a second connection here did.
            self._conn_lifecycle_gate.acquire_write()
            try:
                return self_repair.run_startup_self_repair_on(
                    self._config, self._open_conn_unlocked()
                )
            finally:
                self._conn_lifecycle_gate.release_write()

        # The repair is shared-connection work (DDL and CHECKPOINT on the
        # handle), so it takes the write lock like every other executor job;
        # the lifecycle gate inside `_repair` additionally parks `_get_conn()`
        # callers that would otherwise open a second connection (REQ-RESIL-021).
        async with self._write_lock:
            outcome = await self._await_shared_conn_work(
                asyncio.get_running_loop().run_in_executor(self._executor, _repair),
                "startup self-repair",
            )
        self._last_index_probe = outcome.probe
        if outcome.rebuild is not None:
            logger.warning(
                "startup index rebuild dropped=%d created=%d healed=%s elapsed=%.3fs",
                outcome.rebuild.dropped,
                outcome.rebuild.created,
                ",".join(outcome.rebuild.healed) or "-",
                outcome.rebuild.elapsed_seconds,
            )

    async def stop(self) -> None:
        self._shutdown_event.set()
        self._background_enrichment_idle.set()
        await self._snapshot_cohorts.close()
        jobs = tuple(self._fresh_batches) + (
            (self._storage_maintenance_task,) if self._storage_maintenance_task is not None else ()
        )
        for job in jobs:
            job.cancel()
        if jobs:
            await asyncio.gather(*jobs, return_exceptions=True)
        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()
            self._server = None
        # Handlers observe shutdown and streams send their terminal frame
        # before their sockets close; cancellation here would cut that off.
        clients = tuple(self._client_tasks)
        if clients:
            await asyncio.gather(*clients, return_exceptions=True)
        # Interrupt admitted reads, then join every worker before closing the
        # parent connection or releasing database ownership. wait=False would
        # let a snapshot or startup worker outlive its owner.
        for execution in tuple(self._read_executions):
            execution.cancel()
        async with self._write_lock:
            for executor in (
                self._read_executor,
                self._query_model_executor,
                self._enrichment_executor,
                self._executor,
            ):
                if executor is not None:
                    await asyncio.to_thread(executor.shutdown, wait=True, cancel_futures=True)
            self._read_executor = None
            self._query_model_executor = None
            self._enrichment_executor = None
            self._executor = None
            self._conn_lifecycle_gate.acquire_write()
            try:
                if self._conn is not None:
                    self._conn.close()
                    self._conn = None
            finally:
                self._conn_lifecycle_gate.release_write()
        if self._owns_runtime_artifacts:
            self._remove_socket()
            self._remove_pid_file()
            self._owns_runtime_artifacts = False
        logger.info("RPC server stopped")

    def _signal_shutdown(self) -> None:
        logger.info("received shutdown signal")
        self._shutdown_event.set()
        try:
            terminated = self._terminate_codex_processes()
            if terminated:
                logger.info("terminated %d active codex subprocess(es)", terminated)
        except Exception:  # pragma: no cover - shutdown must never raise
            logger.exception("failed to terminate codex subprocesses during shutdown")

    def request_shutdown(self) -> None:
        self._shutdown_event.set()

    # ---- Watch mode integration ----

    async def _observe_watch_paths(self, runtime: LiveWatchRuntime, paths: tuple[str, ...]) -> bool:
        """Publish metadata before allowing notifications to leave readiness accounting."""
        from recall.db.source_files import SourceCatalog
        from recall.services.coordinator import capture_watch_observations
        from recall.services.reconciler import persist_inventory_batch
        from recall.services.watcher import WATCH_CAPTURE_ATTEMPTS, WATCH_CAPTURE_RETRY_SECONDS

        config = self._load_runtime_config()
        epoch = self._database_epoch
        observed_any = False
        for attempt in range(WATCH_CAPTURE_ATTEMPTS):
            if self._shutdown_event.is_set():
                return observed_any
            observed_from = runtime.clock.monotonic()
            batch, failures = await self._await_shared_conn_work(
                asyncio.get_running_loop().run_in_executor(
                    self._executor, capture_watch_observations, config, paths
                ),
                "watch observation capture",
            )

            def persist(batch=batch) -> tuple[str, ...] | None:
                if epoch != self._database_epoch:
                    return None
                return persist_inventory_batch(
                    SourceCatalog(self._get_conn(), clock=time.time), batch, notification=True
                )

            changed = await self._writer_call(persist, "watch observation batch")
            if changed is None:
                return observed_any
            for item in batch.files:
                runtime.settle(item.source_path, observed_from)
            for path in changed:
                self._note_source_activity(path)
            if batch.files:
                observed_any = True
                self._raw_wakeup.set()
            if not failures:
                return observed_any
            for failure in failures:
                logger.warning(
                    "watch observation failed path=%s attempt=%d/%d reason=%s",
                    failure.path,
                    attempt + 1,
                    WATCH_CAPTURE_ATTEMPTS,
                    failure.detail,
                )
            if attempt + 1 == WATCH_CAPTURE_ATTEMPTS:
                for failure in failures:
                    runtime.settle(
                        failure.path,
                        observed_from,
                        failure=f"watch capture gave up after {WATCH_CAPTURE_ATTEMPTS} attempts: "
                        f"{failure.detail}",
                    )
            else:
                paths = tuple(failure.path for failure in failures)
                with suppress(TimeoutError):
                    await asyncio.wait_for(
                        self._shutdown_event.wait(),
                        timeout=WATCH_CAPTURE_RETRY_SECONDS * (attempt + 1),
                    )
        return observed_any

    async def _start_watch_mode(self) -> tuple[LiveWatchRuntime, asyncio.Task[None]]:
        from recall.services.watch_metrics import WatchIndexMetrics
        from recall.services.watcher import (
            build_live_watch_runtime,
            run_live_discovery_tick,
            start_live_watch_runtime,
        )

        cfg = self._load_runtime_config()
        loop = asyncio.get_running_loop()
        runtime = await loop.run_in_executor(
            self._executor,
            lambda: build_live_watch_runtime(config=cfg, source=cfg.daemon.source),
        )
        runtime.cancelled = self._shutdown_event.is_set
        self._watch_runtime = runtime
        self._watch_metrics = WatchIndexMetrics()

        async def _discovery_loop() -> None:
            await self._await_shared_conn_work(
                loop.run_in_executor(self._executor, start_live_watch_runtime, runtime),
                "watch start",
            )
            while not self._shutdown_event.is_set():
                with suppress(TimeoutError):
                    await asyncio.wait_for(
                        self._shutdown_event.wait(),
                        timeout=max(0.1, cfg.daemon.live_discovery_interval),
                    )
                if self._shutdown_event.is_set():
                    break
                try:
                    await self._await_shared_conn_work(
                        loop.run_in_executor(self._executor, run_live_discovery_tick, runtime),
                        "live observation",
                    )
                except asyncio.CancelledError:
                    raise
                except Exception as err:
                    logger.warning("watch discovery failed: %s", err)
                await self._harvest_usage_once(None, None, "usage harvest")

        async def _drain_loop() -> None:
            notified_topology_generation = 0
            while not self._shutdown_event.is_set():
                with suppress(TimeoutError):
                    await asyncio.wait_for(self._shutdown_event.wait(), timeout=0.25)
                from recall.services.coordinator import WATCH_OBSERVATION_LIMIT, is_paused

                if is_paused(self._load_runtime_config()):
                    continue
                topology_generation = runtime.topology.generation()
                if topology_generation != notified_topology_generation:
                    self._inventory_wakeup.set()
                    notified_topology_generation = topology_generation
                # Observation never waits for service: otherwise a burst's first
                # pending source prevents later notices from reaching the scheduler.
                ready = sorted(set(runtime.discovered.ready()) | set(runtime.queue.ready()))
                for start in range(0, len(ready), WATCH_OBSERVATION_LIMIT):
                    if self._shutdown_event.is_set():
                        break
                    try:
                        await self._observe_watch_paths(
                            runtime, tuple(ready[start : start + WATCH_OBSERVATION_LIMIT])
                        )
                    except asyncio.CancelledError:
                        raise
                    except Exception as err:
                        logger.warning("watch observation batch failed: %s", err)

        drain_task = asyncio.create_task(_drain_loop())
        discovery_task = asyncio.create_task(_discovery_loop())
        drain_task.add_done_callback(lambda task: self._fail_on_watch_task_exit("drain", task))
        discovery_task.add_done_callback(
            lambda task: self._fail_on_watch_task_exit("discovery", task)
        )
        self._watch_drain_task = drain_task
        self._watch_discovery_task = discovery_task
        return runtime, drain_task

    async def _stop_watch_mode(self, runtime: LiveWatchRuntime) -> None:
        from recall.services.watcher import stop_live_watch_runtime

        task = self._watch_discovery_task
        if task is not None:
            task.cancel()
            with suppress(asyncio.CancelledError):
                await task
            self._watch_discovery_task = None
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(self._executor, stop_live_watch_runtime, runtime)
        self._watch_runtime = None
        self._watch_drain_task = None
        logger.info("watch mode stopped")

    # ---- Embed phase integration ----

    @measured_async(Stage.EMBED_CYCLE)
    async def _run_embed_batch(self, config: AppConfig, state: EmbedPhaseState) -> int:
        if self._wal_pressure_blocked:
            state.deferred_reason = "WAL pressure is waiting for checkpoint maintenance"
            state.record_outcome("wal-pressure")
            return 0
        # A manual cycle and the background timer must not snapshot/generate the
        # same pending inputs or operate a shared model concurrently.
        state.enter_stage("enrichment-lock")
        async with self._enrichment_lock:
            return await self._run_owned_embed_batch(config, state)

    async def _run_owned_embed_batch(self, config: AppConfig, state: EmbedPhaseState) -> int:
        """Snapshot and commit under the writer; models own no database handle."""
        from recall.services.coordinator import is_paused
        from recall.services.embed_phase import (
            find_eligible_pending_embeds,
            generate_prepared_embed_cycle,
            prepare_embed_cycle,
            publish_prepared_embed_cycle,
        )
        from recall.services.system_state import check_load, resolve_load_threshold

        if is_paused(config) or self._shutdown_event.is_set():
            state.record_outcome("paused")
            return 0
        state.enter_stage("configure")
        await self._await_shared_conn_work(
            asyncio.get_running_loop().run_in_executor(
                self._enrichment_model_executor(), state.configure, config
            ),
            "model configuration",
        )
        state.deferred_reason = None

        def preconditions():
            # The reason names which of the two ceilings applied and the power
            # state that selected it; the phase dates it (REQ-ADAPT-006).
            return check_load(
                resolve_load_threshold(
                    config.daemon.load_threshold, config.daemon.battery_threshold
                )
            )

        state.enter_stage("preconditions")
        allowed = await asyncio.get_running_loop().run_in_executor(self._executor, preconditions)
        if not allowed.ok:
            state.deferred_reason = allowed.reason
            state.record_outcome("deferred")
            return -1

        state.enter_stage("snapshot")
        started = time.monotonic()

        async def snapshot():
            return await self._enrichment_writer_call(
                lambda: prepare_embed_cycle(
                    config,
                    state,
                    conn=self._get_conn(),
                    max_sessions=1,
                    should_stop=self._shutdown_event.is_set,
                    work_generation=self._write_lock.enrichment_generation,
                ),
                "embed snapshot",
                affects_enrichment=False,
            )

        async def unload_and_return(outcome: str) -> int:
            state.record_outcome(outcome)
            state.enter_stage("unload")
            await asyncio.get_running_loop().run_in_executor(
                self._enrichment_model_executor(), state.maybe_unload
            )
            return 0

        prepared = await snapshot()
        if prepared.error is not None:
            state.last_error = prepared.error
            state.record_outcome("capacity-error")
            return -1
        if not prepared.pending.session_ids:
            # An empty roster has no stall to pace on and no offender to
            # remember, whichever guard below ends the cycle.
            state.note_nothing_pending()
        if is_paused(config) or self._shutdown_event.is_set():
            return await unload_and_return("paused")
        if not prepared.sessions:
            outcome = (
                "capacity-scan"
                if prepared.selection_scanning
                else "cooldown-wait"
                if state.cooldown_status()[0]
                else "no-eligible-sessions"
            )
            return await unload_and_return(outcome)
        paced_for = state.stalled_for(prepared.pending)
        if paced_for > 0:
            # Pure pacing. This cycle attempts no commit, so it is not evidence
            # against the head session and must not extend the backoff either
            # (`REQ-ADAPT-016`).
            state.deferred_reason = (
                f"pending enrichment made no progress; retrying in {paced_for:.0f}s"
            )
            return await unload_and_return("stalled")

        state.enter_stage("generate")
        prepared = await self._await_shared_conn_work(
            asyncio.get_running_loop().run_in_executor(
                self._enrichment_model_executor(),
                lambda: generate_prepared_embed_cycle(
                    config, state, prepared, should_stop=self._shutdown_event.is_set
                ),
            ),
            "embed generation",
        )
        state.last_error = prepared.error
        if prepared.error:
            state.record_outcome("generation-error")
            return -1
        if is_paused(config) or self._shutdown_event.is_set():
            state.record_outcome("paused")
            return 0

        state.enter_stage("commit")

        def commit():
            conn = self._get_conn()
            # Freeze eligibility across both counts. Neither new raw writes nor
            # sessions crossing the idle threshold can manufacture drained work.
            # The roster is filtered exactly as the next cycle's snapshot will
            # filter it, so the stall signature taken from it can match.
            now = time.time()
            cooled_now = state.cooled_session_ids()
            before = find_eligible_pending_embeds(
                conn,
                config.daemon.embed_idle_session,
                max_sessions=1,
                cooled=cooled_now,
                now=now,
            )
            publish_prepared_embed_cycle(
                prepared, config, conn=conn, should_stop=self._shutdown_event.is_set
            )
            if prepared.searchable_changed and config.fts.backend == "duckdb":
                self._keyword_dirty = True
            remaining = find_eligible_pending_embeds(
                conn,
                config.daemon.embed_idle_session,
                max_sessions=1,
                cooled=cooled_now,
                now=now,
            )
            return max(0, before.total - remaining.total), remaining

        drained, remaining = await self._enrichment_writer_call(commit, "embed commit")
        count(Counter.DRAINED, drained)
        state.record_pending(remaining)
        state.record_batch(drained, time.monotonic() - started)
        backoff = float(config.daemon.embed_backoff)
        cooled = state.note_commit_rejections(
            committed=prepared.committed_session_ids,
            rejected=[
                item.session.id
                for item in prepared.sessions
                if item.session.id in prepared.generated_session_ids
                and item.session.id not in prepared.committed_session_ids
            ],
            cooldown=backoff,
        )
        if cooled:
            logger.warning(
                "deprioritized %d session(s) for %.0fs after repeatedly discarding stale "
                "embed results: %s",
                len(cooled),
                backoff,
                ", ".join(cooled),
            )
        if not remaining.session_ids:
            state.note_nothing_pending()
        elif drained:
            state.clear_stalled()
        else:
            state.mark_stalled(remaining, backoff)
            state.deferred_reason = (
                f"deprioritized {len(cooled)} uncommittable session(s); cooling down"
                if cooled
                else "pending enrichment made no progress; cooling down"
            )
        state.record_outcome("drained" if drained else "no-progress")
        return drained

    async def _yield_to_index_requests(self, state: EmbedPhaseState, *, timeout: float) -> None:
        """Stand down while an operator index request is in flight (`REQ-ADAPT-017`).

        An index request enriches each session it indexes under the same
        enrichment lock the drain holds, and the drain re-arms 0.01s after a
        draining cycle. Yielding here is what keeps the request's wait at one
        in-flight cycle instead of one cycle per changed session. The wait is
        bounded so a request that never finishes cannot disable enrichment.

        The yield is expected behaviour, so it is announced once per run of
        requests and escalates only when the scheduler served nothing while the
        drain stood down: a `--full` run otherwise wrote 15-30 WARNINGs
        describing a healthy daemon.
        """
        if not self._index_turns.in_flight:
            self._announced_index_yield = False
            return
        self._background_enrichment_idle.set()
        state.enter_stage("index-request-yield")
        state.deferred_reason = "yielding to an in-flight index request"
        if not self._announced_index_yield:
            self._announced_index_yield = True
            logger.info("embed loop yielding to %d index request(s)", self._index_turns.in_flight)
        served = self._raw_turns_served
        if await self._index_turns.wait_idle(timeout):
            self._announced_index_yield = False
        else:
            drained = self._raw_turns_served - served
            if drained:
                logger.info(
                    "embed loop resumed after %.0fs; %d index request(s) still working "
                    "(%d source(s) served meanwhile)",
                    timeout,
                    self._index_turns.in_flight,
                    drained,
                )
            else:
                logger.warning(
                    "embed loop resumed after %.0fs with %d index request(s) still in flight "
                    "and no source served in that window",
                    timeout,
                    self._index_turns.in_flight,
                )
        state.deferred_reason = None

    async def _start_embed_phase(self) -> asyncio.Task[None] | None:
        from recall.services.embed_phase import EmbedPhaseState

        state: EmbedPhaseState = self._embed_state or EmbedPhaseState(
            _config=self._load_runtime_config()
        )
        self._embed_state = state

        async def _embed_loop() -> None:
            interval = 10
            while not self._shutdown_event.is_set():
                state.next_interval = interval
                state.enter_stage("wait")
                with suppress(TimeoutError):
                    await asyncio.wait_for(self._shutdown_event.wait(), timeout=interval)
                if self._shutdown_event.is_set():
                    break
                state.begin_timer_cycle()
                state.enter_stage("config-load")
                config = self._load_runtime_config()
                if not config.daemon.embed:
                    self._background_enrichment_idle.set()
                    async with self._enrichment_lock:
                        state.deferred_reason = "background enrichment disabled"
                        state.record_outcome("disabled")
                        await self._await_shared_conn_work(
                            asyncio.get_running_loop().run_in_executor(
                                self._enrichment_model_executor(), state.maybe_unload
                            ),
                            "disabled model unload",
                        )
                    interval = min(config.daemon.embed_interval, 10)
                    continue
                await self._yield_to_index_requests(
                    state, timeout=float(config.daemon.embed_backoff)
                )
                if self._shutdown_event.is_set():
                    break
                try:
                    result = await self._run_embed_batch(config, state)
                    if result < 0:
                        self._background_enrichment_idle.set()
                    elif result > 0:
                        self._background_enrichment_idle.clear()
                    else:
                        self._background_enrichment_idle.set()
                    interval = next_embed_interval(result, state.last_pending, config)
                except asyncio.CancelledError:
                    raise
                except Exception as err:
                    self._background_enrichment_idle.set()
                    state.last_error = f"{type(err).__name__}: {err}"
                    state.record_outcome("error")
                    logger.error("embed phase error: %s", err)
                    if self._stop_on_fatal_db(err, "embed_loop"):
                        return
                    interval = config.daemon.embed_backoff
                state.next_interval = interval
                if state.last_outcome != "drained":
                    logger.info(
                        "embed loop iteration=%d outcome=%s interval=%.2fs pending=%s",
                        state.loop_iterations,
                        state.last_outcome,
                        interval,
                        state.last_pending.total if state.last_pending else None,
                    )

        task = asyncio.create_task(_embed_loop())
        task.add_done_callback(lambda _: self._background_enrichment_idle.set())
        return task


async def run_rpc_server(
    *,
    config: AppConfig | None = None,
    idle_timeout: float | None = None,
    watch: bool = False,
    shutdown_event: threading.Event | None = None,
) -> None:
    server = RpcServer(config=config)
    shutdown_task: asyncio.Task[None] | None = None
    if shutdown_event is not None:

        async def _bridge_shutdown() -> None:
            await asyncio.to_thread(shutdown_event.wait)
            server.request_shutdown()

        shutdown_task = asyncio.create_task(_bridge_shutdown())

    try:
        await server.start(idle_timeout=idle_timeout, watch=watch)
    finally:
        if shutdown_task is not None:
            shutdown_task.cancel()
            with suppress(asyncio.CancelledError):
                await shutdown_task


def start_server_blocking(
    *,
    config: AppConfig | None = None,
    idle_timeout: float | None = None,
    watch: bool = False,
    shutdown_event: threading.Event | None = None,
) -> None:
    """Start the RPC server, blocking until shutdown."""
    asyncio.run(
        run_rpc_server(
            config=config,
            idle_timeout=idle_timeout,
            watch=watch,
            shutdown_event=shutdown_event,
        )
    )
