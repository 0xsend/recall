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
from recall.services.self_repair import preflight_recovery_files


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
        if connection is None:
            lease = stack.enter_context(
                owned_connection(
                    lifetime,
                    "standalone_write",
                    lambda: connect_owned(
                        config, lifetime=lifetime, admission=admission.preserving
                    ),
                )
            )
        else:
            lease = stack.enter_context(lifetime.lease(connection))
            version = lease.call("borrowed_write_schema", check_schema_compatibility)
            admission.preserving(lease, version)
        try:
            yield lease
        except BaseException as error:
            lifetime.observe(error, "standalone_write")
            raise
