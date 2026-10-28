"""Logical database compatibility errors shared across application layers."""


class SchemaTooNewError(RuntimeError):
    """This build cannot safely use a database created by a newer release."""

    def __init__(self, database_version: int, supported_version: int) -> None:
        self.database_version = database_version
        self.supported_version = supported_version
        super().__init__(
            f"Database schema {database_version} is newer than this build supports "
            f"(schema {supported_version}). Upgrade recall to a release that supports "
            f"database schema {database_version}, or restore the pre-upgrade backup. "
            "Automatic downgrade is not supported."
        )
