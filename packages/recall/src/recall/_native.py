"""Lazy native constructors, independent of the storage package initializer."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sqlite3 import connect as sqlite_connect

    from duckdb import connect as duckdb_connect


__all__ = ["duckdb_connect", "sqlite_connect"]


def __getattr__(name: str) -> object:
    if name == "duckdb_connect":
        import duckdb

        return duckdb.connect
    if name == "sqlite_connect":
        import sqlite3

        return sqlite3.connect
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
