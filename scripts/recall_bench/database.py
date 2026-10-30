"""A verified template with the database required by comparison and paired runs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .fs import Cancellation, Refused
from .isolation import FileRecord, Policy, Template

DATABASE = "recall.duckdb"
WAL = DATABASE + ".wal"


@dataclass(frozen=True)
class Database:
    """A held template, its mandatory database record and optional WAL record."""

    template: Template
    database: FileRecord
    wal: FileRecord | None

    @classmethod
    def open(
        cls, policy: Policy, argument: str, raw: str | Path, cancellation: Cancellation
    ) -> Database:
        template = Template.open(policy, argument, raw, cancellation)
        try:
            records = {record.path: record for record in template.manifest.files}
            database = records.get(f"payload/{DATABASE}")
            if database is None:
                raise Refused(argument, template.path, f"manifest has no payload/{DATABASE}")
            return cls(template, database, records.get(f"payload/{WAL}"))
        except BaseException:
            template.close()
            raise

    @property
    def files(self) -> tuple[FileRecord, ...]:
        return (self.database, self.wal) if self.wal is not None else (self.database,)

    def __enter__(self) -> Database:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.template.close()
