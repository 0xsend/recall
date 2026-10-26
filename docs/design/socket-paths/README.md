# Daemon socket path errors

People and calling agents start Recall against a selected data directory. Its
Unix socket remains at `<data_dir>/recall.sock` (REQ-RPC-001). When that address
cannot fit, startup reports the path, its filesystem-encoded byte length, the
platform limit, and `Choose a shorter RECALL_DATA_DIR.`

Foreground startup uses the existing validation channel: one diagnostic and
exit 2. Detached startup rejects the address before creating data or log
state. Auto-fork reports the same reason through its existing RPC connection
error channel. Direct `RpcServer.start` callers receive `ValueError` before
resources are initialized. Supported addresses retain their startup behavior,
owner-only socket mode, and location.

Linux permits 107 pathname bytes and macOS 103, leaving room for the terminating
NUL. Unicode counts encoded bytes. Both the maximum supported length and one
byte over it are boundary cases. A shorter data directory is the selected
remedy; relocating sockets would change the documented discovery contract.

The previous foreground error was an `AF_UNIX path too long` traceback after
creating the data directory. The validation diagnostic uses the same form as
other foreground configuration errors. No new command, flag, or output schema
is introduced. Runtime captures and platform gaps belong in the change's
evidence record.
