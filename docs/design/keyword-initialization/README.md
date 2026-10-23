# Keyword initialization operation

Operators and agents need to initialize a legacy contentless keyword index
without guessing its old content, losing canonical enrichment, or resuming
writers into an incomplete index. The human approved PR #141 revision 2 and its
explicit reconstruction entry point on 2026-10-23 via coordinator addendum 1.
That approved flow is implemented as an offline Python module beside bootstrap
and reconciliation; normal daemon startup does not initiate data migration.

The flow and states are specified in [the operator guide](../../keyword-initialization.md).
`inspect` returns an absent or captured operation; `begin` captures backup and
maintenance; `build` is resumable and reports bounded progress; `install` changes
only the verified standalone sidecar; `rollback` restores the retained original
before release; `finish` enables ordinary writers. The operation targets and build
are explicit in every call. JSON stdout carries stable states and failure
context; human progress goes to stderr. There are no interactive prompts,
implicit service operations, or implicit target selection.

Empty canonical selections produce an empty verified index. Partial construction
retains whole committed batches. Changed snapshot, fields or build is a conflict.
Missing inputs, lock contention, disk exhaustion, integrity failures and changed
hashes retain the original and report failure. A disconnected caller resumes by
operation directory. Replacement crash recovery distinguishes the old and complete
candidate hashes. Released operations prohibit sidecar-only rollback.

The existing general `recall db` surface remains unchanged. A module entry point
keeps this explicitly approved offline maintenance separate from daemon RPC and
its ordinary command manifest. A nullable first-touch transition was rejected in
the approved plan because it cannot preserve unchanged legacy statistics.
