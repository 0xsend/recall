"""The external sorter that turns one table scan into one canonical row stream."""

from __future__ import annotations

# Every file the sorter creates besides the final stream lives in this one
# directory of its working container, so a validator that knows the directory
# knows every temporary the sorter can leave after a kill.
SORT_DIRECTORY = "sort"
# Fail-closed ceiling on what the validator accepts in that directory. The
# sorter's own bound (levels times fan-in plus one output) is far below it.
SORT_FILES_MAX = 1024
