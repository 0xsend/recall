"""Admitted standalone writes own or explicitly borrow their complete lifetime."""

from collections.abc import Iterator
from contextlib import ExitStack, contextmanager

import duckdb

from recall.core.config import AppConfig
from recall.db.connection import advisory_lock, connect_owned
from recall.db.lease import DbLease, StandaloneLifetime, owned_connection
from recall.db.ownership import offline_ownership
from recall.db.schema import check_schema_compatibility
from recall.services.recovery_admission import RecoveryAdmission
from recall.services.self_repair import preflight_recovery_files, write_refusal_marker


@contextmanager
def standalone_write_lease(
    config: AppConfig, connection: duckdb.DuckDBPyConnection | None = None
) -> Iterator[DbLease]:
    lifetime = StandaloneLifetime()
    admission = RecoveryAdmission(config, lifetime)
    with ExitStack() as stack:
        if connection is None:
            stack.enter_context(offline_ownership(config.db_path))
            stack.enter_context(advisory_lock(config.lock_path))
        preflight_recovery_files(config)
        try:
            with ExitStack() as native:
                if connection is None:
                    lease = native.enter_context(
                        owned_connection(
                            lifetime,
                            "standalone_write",
                            lambda: connect_owned(
                                config, lifetime=lifetime, admission=admission.preserving
                            ),
                        )
                    )
                else:
                    lease = native.enter_context(lifetime.lease(connection))
                    version = lease.call("borrowed_write_schema", check_schema_compatibility)
                    admission.preserving(lease, version)
                yield lease
        except BaseException as primary:
            lifetime.observe(primary, "standalone_write")
            failure = lifetime.pending_completion
            if failure is not None and connection is None:
                if not lease.connection_closed:
                    message = (
                        "failed completion has no acknowledged native close; "
                        "retain for manual diagnosis"
                    )
                    lifetime.reject_admission(primary, "completion:close")
                    write_refusal_marker(config.data_dir, message)
                else:
                    try:
                        admission.resolve_completion(failure)
                        lifetime.resolve_completion(failure)
                    except BaseException as cleanup:
                        lifetime.reject_admission(cleanup, "completion:resolution")
                        write_refusal_marker(
                            config.data_dir, f"fresh completion diagnosis failed: {cleanup}"
                        )
                        raise BaseExceptionGroup(
                            "standalone work and fresh completion diagnosis failed",
                            [primary, cleanup],
                        ) from None
            raise
