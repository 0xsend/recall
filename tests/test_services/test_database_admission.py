"""Joint enrichment turns cannot starve between raw and search handoffs."""

from __future__ import annotations

import asyncio

from recall.services.rpc_server import _DatabaseAdmission


def test_enrichment_gets_the_next_complete_turn_after_raw_publication() -> None:
    admission = _DatabaseAdmission()

    async def scenario() -> None:
        current_scan = await admission.acquire("scan")
        enrichment = asyncio.create_task(admission.acquire("enrichment"))
        await asyncio.sleep(0)
        publication = await admission.acquire("raw")
        admission.release(current_scan)
        next_scan = asyncio.create_task(admission.acquire("scan"))
        await asyncio.sleep(0)
        assert not next_scan.done()
        admission.release(publication)
        combined = await asyncio.wait_for(enrichment, 1)
        assert not next_scan.done()
        admission.release(combined)
        admission.release(await asyncio.wait_for(next_scan, 1))

    asyncio.run(scenario())


def test_ready_enrichment_does_not_wait_behind_the_next_search() -> None:
    admission = _DatabaseAdmission()

    async def scenario() -> None:
        current_scan = await admission.acquire("scan")
        next_scan = asyncio.create_task(admission.acquire("scan"))
        enrichment = asyncio.create_task(admission.acquire("enrichment"))
        await asyncio.sleep(0)
        admission.release(current_scan)
        await asyncio.sleep(0)
        assert enrichment.done()
        assert not next_scan.done()
        admission.release(await enrichment)
        admission.release(await asyncio.wait_for(next_scan, 1))

    asyncio.run(scenario())
