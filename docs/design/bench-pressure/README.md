# Arrival and historical backlog bench profiles

The maintained development bench serves people and agents measuring daemon
reconciliation. Its existing prepare → daemon-run → summarize / pairs flow
uses explicit isolated paths, a final JSON outcome, separate progress and
owned child cleanup. Idle, append and cohort profiles settle before sampling;
they cannot represent post-boot arrivals or history that is still draining.
The baseline invalid-choice outcomes and author observations are retained in
the task evidence.

The authorized lane brief specifies two additional profiles on that surface.
The [command reference](../../../scripts/recall_bench/README.md#arrival-and-historical-backlog)
defines their options, outputs and limits. The decision is to reuse the
existing profile parser, supervisor, CLI observers, pair runner and typed
result boundary. A separate copied harness would duplicate isolation and
cleanup. A product scheduler change would exceed the tooling scope.

| State | Observable outcome / next action |
| --- | --- |
| Invalid schedule, unsafe paths, unsupported build | Failed JSON with cause; correct inputs and use fresh destinations. |
| Starting | Expected-PID RPC readiness, then settle for arrival or pending-history admission for backlog. |
| Sampling | Fresh bursts followed by delayed probes, or positive historical pending lower bounds and observed historical commits. Optional append/arrival schedules run concurrently. |
| Draining | Every written arrival is witnessed by keyword search; pending/noticed raw work reaches zero and configured enrichment is ready. Background embeddings remain disabled; their queues may remain pending. |
| Missing progress, expired deadline or history drained during sampling | Incomplete JSON and partial evidence; increase history or shorten the window. |
| CLI failure, missing witness, generation error, breached floor or trace obligation | Failed JSON and raw calls/observations; diagnose the named failure. |
| Complete | Final status/stats, exact discovered count and measurements; passed only for the stated profile scope. |
| Guest hold | Existing hold/done-file protocol keeps the owned daemon available until bounded release. |
| Cancellation | Existing supervisor stops/reaps owned children before publishing incomplete evidence. |

Generated fixture content is repeatable; actual write times and daemon choices
remain observed inputs. Optional bounded monitoring attributes the existing
daemon's decisions without replacing product functions. Inclusive attribution
and instrumented timing are explicitly distinguished from performance
acceptance. The evidence record carries author and fresh guest CLI observations;
there is no graphical UI.
