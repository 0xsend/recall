"""The id-set predicate trades bound parameters for literals only when it is safe.

DuckDB 1.5.5 pays two failed ``pandas`` import searches per bound non-NULL
parameter (REQ-RECON-027), so a writer-owned id set is rendered into the
statement instead of bound. That is only sound for ids drawn from a validated
token set; every other value keeps the parameterized form, which is what makes
the change a representation choice rather than a new escaping rule.
"""

from __future__ import annotations

import duckdb
import pytest
from recall.db.queries import id_set_predicate

HEX_ID = "a" * 32


def test_generated_ids_render_as_literals_and_bind_nothing() -> None:
    predicate, params = id_set_predicate("message_id", [HEX_ID, "noop-message-1"])

    assert predicate == f"message_id IN ('{HEX_ID}', 'noop-message-1')"
    assert params == []


@pytest.mark.parametrize(
    "unsafe",
    [
        "x' OR '1'='1",
        'quote"id',
        "back\\slash",
        "with space",
        "new\nline",
        "semi;colon",
        "",
        "z" * 129,
    ],
)
def test_an_id_outside_the_token_set_keeps_binding(unsafe: str) -> None:
    predicate, params = id_set_predicate("message_id", [HEX_ID, unsafe])

    assert predicate == "message_id IN (?, ?)"
    assert params == [HEX_ID, unsafe]


def test_an_empty_id_set_is_rejected_rather_than_rendered() -> None:
    """``IN ()`` is not valid SQL; every caller already guards for emptiness."""
    with pytest.raises(AssertionError):
        id_set_predicate("message_id", [])


@pytest.mark.parametrize("identifier", [HEX_ID, "x' OR '1'='1", "quote'id"])
def test_either_form_selects_exactly_the_named_row(identifier: str) -> None:
    conn = duckdb.connect(":memory:")
    try:
        conn.execute("CREATE TABLE t (id VARCHAR)")
        conn.executemany("INSERT INTO t VALUES (?)", [[identifier], ["other"]])

        predicate, params = id_set_predicate("id", [identifier])
        assert conn.execute(f"SELECT id FROM t WHERE {predicate}", params).fetchall() == [
            (identifier,)
        ]
    finally:
        conn.close()
