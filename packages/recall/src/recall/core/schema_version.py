"""Logical database compatibility errors shared across application layers."""

RECREATE_PROHIBITION = (
    "Never run the released 0.46.1 `recall index --recreate --yes` against this "
    "database: it bypasses the schema fence and destroys recovery state."
)
"""Appended to every diagnostic that tells an operator how to proceed."""

UNFENCED_SCHEMA_REMEDY = (
    "Recover with offline `recall db rebuild-indexes`, a confirmed manual compaction, "
    f"or a matched restore of the pre-upgrade database files. {RECREATE_PROHIBITION}"
)


class SchemaTooNewError(RuntimeError):
    """This build cannot safely use a database created by a newer release."""

    def __init__(self, database_version: int, supported_version: int) -> None:
        self.database_version = database_version
        self.supported_version = supported_version
        super().__init__(
            f"Database schema {database_version} is newer than this build supports "
            f"(schema {supported_version}). Upgrade recall to a release that supports "
            f"database schema {database_version}, or restore the matched pre-upgrade "
            "backup (database, write-ahead log, keyword sidecar and recovery files "
            f"together). Automatic downgrade is not supported. {RECREATE_PROHIBITION}"
        )


class RecoveryStateUnavailableError(RuntimeError):
    """The logical schema is current but its recovery singleton cannot be trusted."""
