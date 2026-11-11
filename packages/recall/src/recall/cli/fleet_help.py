"""Shared fleet query help and introspection text."""

FLEET_TEXT_NOTE = (
    "Fleet text list/search/live rows and the show header append host=NAME from the JSON "
    "host field, matching stats usage. Nonempty ASCII letters/digits/._- stay plain; "
    "other host names are quoted JSON strings with non-ASCII/control characters escaped. "
    "Usage and skill-census host text share this encoding, including ordinary usage; "
    "other ordinary text is unchanged."
)

FLEET_QUERY_HELP = (
    "Query the calling daemon and fleet inventory hosts; merge results. "
    "With no inventory, query the caller alone. Set inventory hostname to the caller's "
    "exact short hostname for self aliases; undeclared aliases remain separate endpoints. "
    + FLEET_TEXT_NOTE
)
INVENTORY_ONLY_HELP = (
    "With --fleet, select only inventory endpoints; "
    "an explicitly listed caller still uses local RPC."
)
FLEET_SCOPE_NOTE = (
    "Fleet includes the calling daemon by default; "
    "partial endpoint failures are reported on stderr. "
    "The calling CLI never starts its daemon or uses a poll snapshot, "
    "including for guidance/notices. "
    "Set hostname to the caller's exact short hostname for self aliases, FQDNs or different case. "
    "Without that declaration or a canonical rename, an alias remains a separate SSH endpoint "
    "and can count the same machine twice. No DNS, domain removal or case folding occurs. "
    "Foreign machines sharing a short hostname need distinct display/effective identities. "
    "--fleet-inventory-only restores inventory membership, including empty-inventory refusal; "
    "explicitly listed self still uses local RPC. It does not restore SSH-to-self failures, "
    "tie order or fallback labels. "
    "Default --fleet with missing/empty inventory reads caller alone. "
    "List/search/live use one global limit; a selected endpoint need not have a returned row. "
    "List/live/show --host selects an endpoint, including a declared self alias, "
    "not stored origins. "
    "Stored origin hosts are preserved; missing/empty/local labels use the endpoint name."
)
CENSUS_SCOPE_NOTE = (
    "Default local-plus-fleet scope counts a recognized inventoried caller once and requires "
    "a nonempty inventory and complete configured coverage. If the local daemon is unavailable, "
    "default scope fails with exit 1 and no census data; it does not start a daemon or use a "
    "poll snapshot. --local keeps ordinary local-only read lifecycle behavior. "
    "Set inventory hostname to the caller's exact short hostname for self aliases; "
    "undeclared aliases, FQDNs and case variants remain separate endpoints."
)
