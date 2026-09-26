# Security Policy

## Supported versions

Security fixes land in the latest release only. Upgrade to the newest
`vX.Y.Z` tag before reporting, and confirm the issue still reproduces there.

## Reporting a vulnerability

Report vulnerabilities privately. Do not open a public issue.

- Preferred: GitHub private vulnerability reporting, from this repository's
  **Security** tab ("Report a vulnerability").
- Alternative: email allen@send.it.

Include the affected version (`recall --version`), the platform, steps to
reproduce, and the impact you observed. Reports are acknowledged as soon as
practical, and a fix or mitigation is coordinated with the reporter before
public disclosure.

## Scope

In scope:

- The `recall` CLI.
- The recall daemon, including indexing, reconciliation, and its on-disk data
  directory.
- The daemon's local RPC socket and anything that can reach it.
- The install script, `scripts/install.sh`.
- The Claude Code, Codex, Kimi Code, and Pi plugins and skills shipped in this
  repository.

Out of scope: vulnerabilities in the agent tools whose sessions recall indexes,
and in third-party model APIs that optional contextual-retrieval modes call.
