# Changelog

All notable changes to RepoPilot are recorded here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and
the project aims to follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

Entries are grouped under `Added`, `Changed`, `Fixed`, `Removed`, `Deprecated`
and `Security`. Each links the pull request it came from, so the reasoning is one
click away rather than compressed into a line.

## [Unreleased]

### Added

- Run several tasks concurrently, each in its own git worktree and branch, so
  they can touch the same files without interfering ([#210])
- Run the repository's pre-commit hooks before the test suite, telling an
  auto-fix apart from a genuine failure by re-running ([#208])
- Load per-repository conventions from `.agentcontext`, prepended to every
  prompt so they survive a tight context budget ([#200])
- Record per-phase timing for each iteration — ingest, context, LLM, apply and
  execution — in the run log ([#190])
- Derive the context character budget from the model's context window instead of
  a flat 60,000 ([#194])
- `--verbose` dumps the exact prompt and raw reply, with known secret patterns
  masked ([#188])

### Changed

- `sandbox.py` gained a `Sandbox` base class, so a backend implements `run` and
  inherits the rest rather than diverging silently ([#209])
- Runner configuration moved into one registry; the five parallel tables are now
  derived from it and cannot disagree ([#211])
- Ingestion reads in bounded batches, so peak memory tracks the size budget
  rather than the repository — a 141 MB checkout dropped from 166 MB to 55 MB
  ([#207])
- Import scanning during context scoring is bounded to the head of each file,
  making a minified bundle roughly 4.6x cheaper to score ([#195])

### Fixed

- `--max-cost` was registered but never enforced; `--plan-first` raised
  `TypeError`; `--resume` restarted from iteration 1 and never checkpointed
  ([#204])
- `--quiet` and `--api-base-url` were registered and never read ([#206])
- CLI flags and config fields lost in earlier merges, which left `main.py`
  unable to complete a run at all ([#196])
- Module imports broken by merge damage ([#186])
- A missing `pytest` was reported as a failing test suite rather than a missing
  runner ([#189])
- Files created during a run survived a rollback, leaving untracked files behind
  ([#191])

### CI

- The import guard now also runs the CLI, and checks that every `args.X` is
  registered and that `AgentConfig` accepts everything `main.py` passes ([#199])
- Fail the build when a module cannot be imported ([#187])

## Earlier work

Before this file existed, the project already gained multi-provider support
(OpenAI, Gemini and Ollama), token and cost tracking, streaming output, secret
scanning, parallel file processing, GitHub Actions integration, documentation
lookups from an allowlist, and resumable runs. Those are recorded in the git
history rather than reconstructed here, since doing so accurately after the fact
is guesswork and a changelog nobody can trust is worse than none.

## Keeping this current

Add an entry under `Unreleased` in the pull request that makes the change, in
the same commit. A changelog written afterwards from `git log` records what
changed but not why, which is the part worth having.

Write for someone deciding whether an upgrade affects them: name the behaviour
that changed rather than the function that changed, and say so plainly when a
default moves.

[unreleased]: https://github.com/sreerevanth/repopilot/commits/main
[#186]: https://github.com/sreerevanth/repopilot/pull/186
[#187]: https://github.com/sreerevanth/repopilot/pull/187
[#188]: https://github.com/sreerevanth/repopilot/pull/188
[#189]: https://github.com/sreerevanth/repopilot/pull/189
[#190]: https://github.com/sreerevanth/repopilot/pull/190
[#191]: https://github.com/sreerevanth/repopilot/pull/191
[#194]: https://github.com/sreerevanth/repopilot/pull/194
[#195]: https://github.com/sreerevanth/repopilot/pull/195
[#196]: https://github.com/sreerevanth/repopilot/pull/196
[#199]: https://github.com/sreerevanth/repopilot/pull/199
[#200]: https://github.com/sreerevanth/repopilot/pull/200
[#204]: https://github.com/sreerevanth/repopilot/pull/204
[#206]: https://github.com/sreerevanth/repopilot/pull/206
[#207]: https://github.com/sreerevanth/repopilot/pull/207
[#208]: https://github.com/sreerevanth/repopilot/pull/208
[#209]: https://github.com/sreerevanth/repopilot/pull/209
[#210]: https://github.com/sreerevanth/repopilot/pull/210
[#211]: https://github.com/sreerevanth/repopilot/pull/211
