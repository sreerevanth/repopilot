# Security Policy

## Reporting a vulnerability

Please report security issues privately rather than in a public issue.

Use GitHub's [private vulnerability reporting](https://github.com/sreerevanth/repopilot/security/advisories/new)
on this repository. If that is unavailable, open an issue titled "Security
contact request" containing no details, and a maintainer will arrange a private
channel.

Please include what you can: the version or commit, the flags in use, and a
minimal reproduction. A reproduction matters more than a full write-up — most of
what follows is easier to confirm than to describe.

You can expect an acknowledgement within a few days and an assessment of whether
the report is in scope. This is a volunteer-maintained project, so please do not
expect a same-day response.

## What this tool does with your machine

RepoPilot executes model-authored code and writes to your repository, so the
trust boundaries are worth stating plainly. A report that shows one of these
being crossed is in scope.

**Model output is untrusted.** Anything the model produces — file contents,
paths, shell arguments — is treated as hostile input, not as instructions.

**Writes stay inside the repository.** File changes are confined to the
`--repo` root. A path that escapes it, including through a symlink, a rename
destination or a `..` sequence, is a vulnerability.

**Tests run in a sandbox when Docker is available.** Networking is disabled,
memory is capped, and the container is removed afterwards. When Docker is not
available the run falls back to a subprocess, which is **not** isolated. The log
says so, but there is currently no flag to refuse the fallback — `DockerSandbox`
accepts `strict=True` internally and it is not wired to the CLI.

An escape from the Docker sandbox is in scope. The subprocess fallback is a
documented limitation rather than a defect: if you need isolation guaranteed,
check the log, or run in a container yourself.

**API keys are not written to logs.** `--verbose` prints prompts and replies
with known secret patterns masked by `modules/secret_scanner.py`. A key shape
that reaches a log unmasked is in scope, and the pattern list is the thing most
likely to fall behind.

**Destructive operations ask first.** `--clean` removes only files matching the
shapes this tool writes, and refuses to recurse. `--undo` deletes only branches
carrying the agent's prefix, and asks before doing it. A path that removes
something outside those rules is in scope.

## Out of scope

- the subprocess fallback not being isolated — documented above, and the lack
  of a flag to refuse it is tracked separately as a feature gap
- the model producing incorrect or low-quality code, which is a correctness
  matter rather than a security one
- vulnerabilities in the provider SDKs themselves; report those upstream
- anything requiring an attacker to already have write access to your
  repository or shell

## Supported versions

The `main` branch. This project has no released versions yet, so fixes land
there and are available immediately.
