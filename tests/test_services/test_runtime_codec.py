"""Owned writers retain legacy history while excluding new Roaring segments."""

from __future__ import annotations

import asyncio
from pathlib import Path

from lane_harness import lane_config
from recall.db.connection import connect
from recall.services.compaction import compact
from recall.services.rpc_server import RpcServer


def test_legacy_roaring_stays_writable_and_compaction_reopens_without_roaring(
    tmp_path: Path,
) -> None:
    cfg = lane_config(tmp_path)
    with connect(cfg) as conn:
        conn.execute("SET disabled_compression_methods=''")
        conn.execute("CREATE TABLE historical(id VARCHAR PRIMARY KEY, v FLOAT[384])")
        conn.execute(
            "INSERT INTO historical SELECT md5(i::VARCHAR), "
            "list_transform(range(384), j -> CASE WHEN i%4=0 THEN NULL::FLOAT "
            "ELSE .25::FLOAT END)::FLOAT[384] FROM range(2000)t(i)"
        )
        conn.execute("CHECKPOINT")
        assert any(
            row[8] == "Roaring"
            for row in conn.execute("PRAGMA storage_info('historical')").fetchall()
        )
    server = RpcServer(config=cfg)

    async def scenario() -> None:
        try:
            await server._writer_call(
                lambda: server._get_conn().execute(
                    "INSERT INTO historical VALUES ('append', array_value(.5::FLOAT) || "
                    "list_transform(range(383), j -> .5::FLOAT))"
                ),
                "append",
            )
        finally:
            await server.stop()

    asyncio.run(scenario())
    result = compact(cfg)
    assert result.replaced
    with connect(cfg) as conn:
        assert conn.execute("SELECT count(*) FROM historical").fetchone() == (2001,)
        assert conn.execute("SELECT v[1] FROM historical WHERE id='append'").fetchone() == (0.5,)
        assert not any(
            row[8] == "Roaring"
            for row in conn.execute("PRAGMA storage_info('historical')").fetchall()
        )
        conn.execute("INSERT INTO historical VALUES ('after', NULL)")
    with connect(cfg) as conn:
        assert conn.execute("SELECT count(*) FROM historical").fetchone() == (2002,)
