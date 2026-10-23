# Storage codec maintenance

People and automation use the existing daemon storage command to preview,
apply, disconnect and inspect an explicit conversion. The approved direction is
the #93 physical re-encode plan, retained in
[the experiment record](../../../experiments/checkpoint_bytes/plan.md).
The implementation preserves the selected base's logical schema and index version.

The existing surface reports a version-1 response with operation/plan digests,
fixed-header target, migration job, acceptance and a terminal state. It currently
converts an old header, and reports unchanged at the current header even when
historical segments contain Roaring. The extended surface adds the separate
`desired_codec_profile`, `codec_eligible`, `codec_phase` and `codec_backup_path`
fields; existing response fields and CLI arguments retain their meanings.

| State | Observed response / next action |
| --- | --- |
| Fresh/current, no Roaring | `unchanged`; ordinary writes admitted with Roaring excluded |
| Legacy header or unfinished format/index job | Finish the original operation and backup first, then inspect a new codec plan |
| Current header, persisted Roaring | `planned`, `codec_eligible: true`; inspect digest, then apply |
| Source or job changes after preview | Conflict; inspect again |
| Accepted/draining | `accepted` or `running`; disconnect/reconnect without cancelling the operation |
| Copy/validation/publication | `running` with durable codec phase; status reports a marked frozen snapshot |
| Verified activation/reopen | `succeeded`, `codec_phase: complete`, `codec_eligible: false`; writes resume |
| Backup/space/copy/validation/recovery error | `failed` with reason; complete authoritative file and attributed artifacts retained |
| Interrupted before activation | Source remains authoritative; inspect and explicitly resume with the same verified backup |
| Interrupted during publication | Identify source/candidate hash before reopening; unknown or later WAL progress holds for recovery |
| Caller deadline | `timed_out`; operation continues, inspect before retrying |

A daemon holding legacy masks exposes `write_admission.allowed: false` and
`reason: codec_maintenance_required`. Mutating RPC requests use the existing
`APP_LOCKED` envelope with the same typed reason and maintenance action. No
ordinary open starts a physical copy. The runtime policy remains automatic
compression with only Roaring excluded; forced uncompressed storage and compact
IDs were rejected by the approved plan's measurements and identity contract.

The backup/journal are retained by the existing snapshots recovery owner.
Only the DuckDB path is atomically replaced. SQLite stays in place and is bound
by its quiesced hash. Local accounts/installed executables remain trusted;
accidental interruption, ENOSPC, stale plans and mismatched artifacts are the
risk scenarios. A specialist review and platform/capacity qualification remain
required before rollout.
