"""Search preference and caller silence budgets (REQ-RPC-029)."""

# Yield at most one normal writer turn, leaving useful time for the scan.
SEARCH_ENRICHMENT_WAIT_SECONDS = 5.0
# Scheduling and transport headroom after the deliberate silent preference.
SEARCH_SILENCE_MARGIN_SECONDS = 1.0
SEARCH_IDLE_TIMEOUT_SECONDS = 30.0

assert SEARCH_IDLE_TIMEOUT_SECONDS > SEARCH_ENRICHMENT_WAIT_SECONDS + SEARCH_SILENCE_MARGIN_SECONDS
