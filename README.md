# Autonomous AI Developer Agent

A production-grade autonomous code modification system that reads a repository,
understands a task, modifies code using an LLM, executes tests in a sandbox,
and iteratively self-corrects until tests pass — then commits the result to Git.

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     CLI  (main.py)                              │
│                AgentConfig → AutonomousAgent                    │
└──────────────────────────┬──────────────────────────────────────┘
                           │
           ┌───────────────▼──────────────────┐
           │         AGENT LOOP               │
           │      (modules/agent_loop.py)     │
           └──┬──────┬────────┬──────┬────────┘
              │      │        │      │
    ┌─────────▼─┐ ┌──▼───┐ ┌─▼───┐ ┌▼──────────┐
    │  Repo     │ │Ctx   │ │ LLM │ │ Code      │
    │ Ingestion │ │Build │ │     │ │ Modifier  │
    └─────────┬─┘ └──┬───┘ └─┬───┘ └┬──────────┘
              │      │       │      │
    ┌─────────▼──────▼───────▼──────▼──────────┐
    │              Sandbox Executor             │
    │         (subprocess / Docker)            │
    └──────────────────┬────────────────────────┘
                       │ stdout/stderr/exit_code
                       ▼
              ┌────────────────┐
              │  Git Integration│
              │  Logger / JSONL │
              └────────────────┘
```

### Data Flow (Single Iteration)

```
repo_root ──► ingest_repository()
                 └── FileRecord[]
                        │
                        ▼
              build_context(task)           # score + select files
                 └── BuiltContext.render()  # XML-tagged source
                        │
                        ▼
              LLMClient.initial_request()   # or retry_request()
                 └── LLMResponse
                        │  .changes: FileChange[]
                        │  .analysis, .confidence, .done
                        ▼
              CodeModificationEngine        # backup → write
                 └── ApplyResult[]
                        │
                        ▼
              SubprocessSandbox.run_tests() # pytest / npm / go test
                 └── ExecutionResult
                        │
              ┌─────────┴──────────────┐
              │ success?               │ failure?
              ▼                       ▼
          git commit             feed error → LLM
          (optional push/PR)     next iteration
```

---

## Folder Structure

```
autonomous_agent/
├── main.py                    # CLI entry point
├── demo_run.py                # Offline demo (MockLLM)
├── requirements.txt
├── modules/
│   ├── __init__.py
│   ├── repo_ingestion.py      # Module 1: File scanner
│   ├── context_builder.py     # Module 2: Relevance scoring
│   ├── llm_client.py          # Module 3: Anthropic API + prompts
│   ├── code_modifier.py       # Module 4: Safe file writes + rollback
│   ├── sandbox.py             # Module 5: Subprocess/Docker execution
│   ├── agent_loop.py          # Module 6: Autonomous loop (CORE)
│   ├── git_integration.py     # Module 7: Branch/commit/push/PR
│   └── logger.py              # Module 8: JSONL + human logs
└── [runtime directories, created automatically]
    ├── logs/
    └── backups/
```

---

## Installation

```bash
git clone <this-repo>
cd autonomous_agent
pip install -r requirements.txt
export ANTHROPIC_API_KEY=sk-ant-...
```

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for an architecture walkthrough, how to
run the pipeline locally without an API key, and the project's conventions.

---

## Usage

### Basic — fix failing tests

```bash
python main.py \
  --repo /path/to/your/project \
  --task "Fix the TypeError in tests/test_parser.py line 42"
```

### With full Git pipeline

```bash
python main.py \
  --repo /path/to/your/project \
  --task "Add input validation to the user_signup() function" \
  --runner pytest \
  --max-iter 6 \
  --push \
  --pr \
  --base-branch main
```

### Run a specific file (not test suite)

```bash
python main.py \
  --repo . \
  --task "Fix the script so it processes all rows without crashing" \
  --run-file scripts/etl.py \
  --run-file-runner python
```

### Offline demo (no API key needed)

```bash
python demo_run.py /path/to/sample_repo
```

### All CLI flags

```
--repo          Path to git repository (required)
--task          Task description (required)
--runner        Test runner: pytest|npm_test|vitest|jest|go|cargo|... (default: pytest)
--runner-args   Extra args for runner
--run-file      Run a specific file instead of test suite
--timeout       Sandbox timeout in seconds (default: 120)
--max-iter      Max LLM iterations (default: 5)
--no-git        Disable all git operations
--push          Push branch to remote after success
--pr            Create GitHub PR after push (needs GITHUB_TOKEN env var)
--base-branch   Base branch name (default: main)
--include       Force-include specific files in context
--log-dir       Log output dir (default: logs/)
--backup-dir    Backup dir (default: backups/)
--quiet         Suppress verbose output
--yes, -y       Auto-approve changes (bypasses every prompt; implied by CI=true)
--interactive   After tests pass, print the diff and confirm before committing
```

### Reviewing before a commit

By default the agent commits as soon as the suite goes green. `--interactive`
inserts a review gate at that point — it prints the working-tree diff the agent
produced and asks:

```
Commit these 3 change(s)? [Y/n]:
```

Declining leaves the changes on disk, uncommitted and unpushed, so they can be
inspected and committed by hand — or discarded with `--rollback`. The run exits
with outcome `aborted`.

This is distinct from the existing prompt, which fires _before_ files are
written and shows a manifest of paths rather than a diff. `--yes` (and `CI=true`,
which sets it automatically) bypasses both, so unattended runs never block on
stdin.

---

## Task Sources

`--task` normally carries a description. It can instead carry a GitHub issue
URL, in which case the title, body and comments are fetched and composed into
the task the agent works from:

```bash
python main.py --repo . --task https://github.com/sreerevanth/repopilot/issues/75
```

Public repositories work unauthenticated; `GITHUB_TOKEN` raises the rate limit
and is required for private ones. Long bodies and threads are trimmed so the
issue text does not crowd source files out of the context budget.

If the URL cannot be resolved the run stops with an explanation. Passing it
through as a literal task would send the agent off to implement a URL.

---

## Updating RepoPilot

```bash
python main.py --update
```

Fast-forwards RepoPilot's own checkout to the latest upstream commit. Fetching
and then running new code is remote code execution by definition, so this is
deliberately conservative:

- it updates a **git checkout** rather than unpacking a downloaded archive, so
  every change is attributable to a commit and reversible with `git reset`
- it **fast-forwards only** — a diverged local branch is reported, never
  overwritten
- it **refuses on a dirty tree**, so nothing uncommitted is lost
- it **shows the incoming commits and asks** before moving anything (`--yes`
  skips the prompt)
- it **prints the pip command rather than running it** when `requirements.txt`
  changes

`--repo` is not required with `--update`.

---

---

## Module Reference

### Module 1 — `repo_ingestion.py`

- Recursively walks a directory, skipping `.git`, `node_modules`, `__pycache__`, etc.
- Respects the repository's `.gitignore` and `.git/info/exclude` via `pathspec`,
  so build output, caches and anything else the project already ignores stays
  out of the context budget.
- Never ingests credential-bearing filenames (`.env*`, `*.pem`, `*.key`,
  `id_rsa`, `.npmrc`, `.netrc`, `credentials*`, …). Ingested content is sent
  verbatim to the model, so this holds even when the project has no
  `.gitignore`.
- Ignores binary files, files > 512KB, and enforces an 8MB total repo budget.
- Returns a `Repository` with `FileRecord[]` — path, content, language, checksum.

> Only root-level ignore files are read; nested `.gitignore` files are not
> resolved. If `pathspec` is not installed the walker degrades to the built-in
> `IGNORE_DIRS`/`IGNORE_EXTENSIONS` rules — the secret-filename filter is
> independent of it and always applies.

### Module 2 — `context_builder.py`

- Scores every file against the task using: language priority, path keyword match,
  content keyword frequency, entry-point bonus, import graph hints.
- Fills a configurable character budget (~60K chars / ~15K tokens).
- A Python file too large to include whole is reduced to a signature-only
  outline — imports, constants, class fields and `def` lines with their
  annotations, bodies omitted — instead of being dropped. That costs roughly a
  seventh of the space, so the model still learns the module exists and what it
  exposes. Outlined files are flagged in `BuiltContext.outlined` and carry an
  `# OUTLINE ONLY` header. Non-Python files are not outlined; that would need
  tree-sitter.
- Returns `BuiltContext.render()` — XML-tagged source ready for the LLM prompt.

### Module 3 — `llm_client.py`

- Wraps the Anthropic API with structured JSON I/O.
- Prompts live in `prompts/*.txt` (`system`, `initial`, `retry`) so they can be
  edited and diffed without touching Python. Point `REPOPILOT_PROMPT_DIR` at
  another directory to try an alternative set. A missing, empty or unreadable
  file falls back to the built-in text, so a bad checkout degrades to the
  previous behaviour rather than leaving the agent with no prompt.
- System prompt enforces a machine-parseable output schema.
- `initial_request()` for first pass; `retry_request()` for error-fed retries.
- Parses `FileChange[]` from JSON; gracefully handles malformed output.

#### Streaming

Responses stream by default, so a 20-30 second call shows progress instead of
sitting silent. Only a character count is echoed, and only to a tty — the
response is one JSON object whose largest field is complete file contents, so
printing the deltas verbatim would dump the rewritten source into the terminal.

The text is reassembled and parsed as a single object at the end. Deltas split
wherever the API decides, including mid-token, so incremental parsing would be
fragile for no real gain. `LLMClient(stream=False)` restores the blocking call,
and an SDK without `messages.stream()` falls back to it automatically.

### Module 4 — `code_modifier.py`

- Validates paths (prevents directory traversal).
- Backs up every file before modification.
- Supports `modify`, `create`, `delete` and `rename` actions (`move` is accepted
  as a synonym). A rename takes a `new_path`; include `content` only if the
  file's contents change too. Renames refuse to overwrite an existing file, and
  rolling one back removes the moved copy as well as restoring the original.
- `rollback()` restores all backups on failure.

### Module 5 — `sandbox.py`

- `SubprocessSandbox`: runs commands via `subprocess.run()` with timeout, output
  capture, and environment sanitization (blocks cloud credentials from leaking).
- `DockerSandbox`: wraps Docker with `--network=none`, memory/CPU caps, read-only
  volume mount. Falls back to subprocess if Docker is unavailable.

### Module 6 — `agent_loop.py` (CORE)

- `AutonomousAgent.run()` orchestrates all modules.

#### Linting before tests

`--lint ruff` (or `flake8`, `pyflakes`, `eslint`, `tsc`, `govet`, `clippy`) runs
a linter before the suite. A failure short-circuits the iteration and feeds the
lint output back to the model, so a syntax error or an undefined name is
corrected in under a second instead of arriving as a pytest collection error.

The Python linters select error-class rules only (`E9,F`). A default rule set
fails on style the model did not write — ruff's `I001` flags unsorted imports in
otherwise valid code — which would make the gate fail every iteration regardless
of what the model produced. Widen it per project with `--lint-args`.

```bash
python main.py --repo . --task "Fix the parser" --lint ruff
```

- Iterates up to `max_iterations` times.
- On success: commits (optionally pushes + opens PR).
- On failure: rolls back all file changes.
- Produces a `AgentRunResult` with outcome, branch, PR URL, iteration count.

### Module 7 — `git_integration.py`

- Wraps `git` subprocess calls: `create_branch`, `stage_files`, `commit`, `push`.
- GitHub PR creation via REST API (no extra dependencies — uses `urllib`).
- `rollback` is handled by `code_modifier.py`; git ops are only for success path.
- `create_branch` refuses to reuse an existing branch that does not already
  contain the base branch, rather than checking it out and running the agent
  against a stale tree.
- A push rejected because the remote branch moved is retried once after
  `git rebase`. A rebase that conflicts is aborted, leaving the working tree
  untouched — an unattended agent cannot resolve conflicts, and a repository
  left mid-rebase is worse than a failed push. Pass `retry_with_rebase=False`
  to skip it. Other push failures (missing remote, auth, network, protected
  branch) are classified and get a specific remedy appended to the error.

### Module 9 — `notify.py`

- Posts the run outcome to a webhook when `WEBHOOK_URL` is set. A six-iteration
  run takes ten minutes or more, so people walk away from the terminal.
- Detects Slack and Discord from the URL and uses each one's payload key; any
  other endpoint receives the structured fields as JSON.
- Cannot fail a run. A broken URL, unreachable host or rejected request is
  reported through the log and the return value, never raised — the agent's
  work is already finished by the time this fires.
- Only `http`/`https` URLs are sent. Uses `urllib`, so no new dependency.

```bash
export WEBHOOK_URL=https://hooks.slack.com/services/T000/B000/xxxx
python main.py --repo . --task "Fix the failing parser test"
```

### Module 8 — `logger.py`

- Every iteration appended to `<run_id>.jsonl` (structured, machine-readable).
- Human-readable log at `<run_id>_human.log`.
- Final `<run_id>_summary.json` with full run record.

---


### `dashboard.py` — run viewer

Renders a run log as a readable timeline — the model's analysis, its file
changes, and the test output for each iteration:

```bash
python -m modules.dashboard logs/agent_20260807_abc.jsonl
python -m modules.dashboard logs/agent_20260807_abc.jsonl --follow
```

`--follow` tails the log while a run is in progress, so a long run is visible as
it happens rather than scrolling past in CLI output.

It reads the JSONL `logger.py` already writes rather than being instrumented
into the loop. That keeps it decoupled: it works on a finished run, on a run in
another terminal, and needs no changes to `agent_loop.py`. It also means no web
server and no new dependency. Colour is disabled off a tty and by `NO_COLOR`.

## Container Cleanup

Every container `DockerSandbox` starts is given a unique `--name` and a
`com.repopilot.sandbox` label, and is force-removed if the run does not end
cleanly.

`--rm` alone is not enough. The daemon honours it when a container _exits_, and
on timeout `subprocess.run` kills the docker **CLI** with `SIGKILL` — a signal
that cannot be caught, so it is never proxied to the container. The container
keeps running, still holding its memory and CPU reservation, and never exits, so
`--rm` never fires. Ctrl+C has the same shape: `KeyboardInterrupt` is a
`BaseException`, which an `except Exception` would miss.

Both paths now issue an explicit `docker rm --force`. A successful run does not,
since `--rm` has already handled it.

`DockerSandbox` also works as a context manager:

```python
with DockerSandbox(repo) as sandbox:
    result = sandbox.run_tests("pytest")
# any container this instance started is removed on the way out
```

For the case nothing in-process can cover — SIGKILL, or the machine losing
power — there is an explicit sweep:

```python
DockerSandbox.sweep_orphaned_containers()   # returns the ids removed
```

It is deliberately not automatic. A sweep cannot tell a leaked container from
one belonging to another agent running right now, so the decision stays with the
caller.


---

## Execution Loop Detail

```
for iteration in 1..max_iterations:
    repo   = ingest_repository(repo_root)          # fresh read each iter
    ctx    = build_context(repo, task)             # score & select files
    resp   = llm.initial_request(task, ctx)        # or retry_request(errors)

    if resp.confidence < min_threshold:
        continue                                   # skip, try again

    errors = modifier.verify_changes(resp.changes) # path validation
    results = modifier.apply_changes(resp.changes) # backup + write

    exec = sandbox.run_tests(runner)               # run test suite

    if exec.success:
        git.stage_all()
        git.commit(message)
        if push:   git.push(branch)
        if pr:     git.create_github_pr(...)
        return SUCCESS

    # failure: loop continues, error fed to LLM in next iter

# exhausted iterations
modifier.rollback(results)
return MAX_RETRIES
```

---

## Container Hardening

`DockerSandbox` runs each container with `--cap-drop ALL`,
`--security-opt no-new-privileges`, a `--pids-limit` (256 by default) and, on
POSIX hosts, `--user $(id -u):$(id -g)`.

`--user` matters beyond privilege reduction: the workspace is a read-write bind
mount, and bind mounts preserve UIDs, so a root container leaves root-owned
files inside the user's own project. It is omitted on Windows and macOS, where
`os.getuid` does not exist and Docker Desktop maps bind-mount ownership itself.

Because `--user` leaves the process without a passwd entry, the container also
gets a `--tmpfs /tmp` with `HOME` pointing at it — otherwise pytest's cache,
npm and go all fail on an unwritable home.

```python
DockerSandbox(repo, pids_limit=1024)   # parallel runners on a large machine
DockerSandbox(repo, read_only=True)    # read-only root filesystem
```

`read_only` is off by default: it breaks any runner that writes outside
`/workspace` and `/tmp`. The workspace mount itself stays read-write, which is
deliberate — plenty of suites create fixtures beside their tests.

To check the flags against a real daemon rather than trusting the argv:

```bash
python scripts/verify_docker_sandbox.py
```

---

## Failure Cases & Mitigations

| Failure                      | Cause                               | Mitigation                                                     |
| ---------------------------- | ----------------------------------- | -------------------------------------------------------------- |
| `JSONDecodeError` from LLM   | Model adds markdown fences or prose | Regex strips fences; parse error fed back as context next iter |
| Path traversal in LLM output | LLM outputs `../../etc/passwd`      | `_safe_abs_path()` validates all paths against repo root       |
| Empty content for modify     | LLM returns `""` for file content   | Validation rejects before apply; error logged                  |
| Infinite test loop           | Test hangs                          | `timeout_seconds` in sandbox kills process                     |
| Repo too large               | Monorepo with 10K files             | 8MB total budget + per-file 512KB cap; budget exhausted = skip |
| Git merge conflict           | Branch already exists               | `create_branch` falls back to checkout if branch exists        |
| LLM low confidence           | Ambiguous task                      | `min_confidence_to_apply` threshold; skip without applying     |
| Test runner not found        | `pytest` not installed              | `sandbox.py` checks `shutil.which()`; returns exit_code 127    |
| All apply ops fail           | Wrong paths, permission error       | Agent breaks loop, returns `error` outcome                     |
| Push auth failure            | Missing SSH key / token             | Logged as non-fatal; outcome still `success` locally           |
| Ctrl+C mid-apply             | User interrupts during file writes  | `run()` catches the interrupt and rolls back applied files     |

---

### Knowing whether a run was isolated

`DockerSandbox` falls back to `SubprocessSandbox` when Docker is unavailable,
which means `--network=none`, the 512MB memory cap and the 1-CPU limit **do not
apply to that run**. The fallback logs a warning when it happens, and every
`ExecutionResult` records which executor produced it — `docker` (with
`result.isolated` set), `subprocess`, or `subprocess-fallback` for the case
where isolation was asked for and quietly not delivered.

Pass `strict=True` to make that a hard failure instead:

```python
DockerSandbox(repo, strict=True).run_tests("pytest")
# raises SandboxUnavailableError rather than running unisolated
```

"Unavailable" means either no `docker` on PATH or no daemon answering — an
installed-but-not-running Docker Desktop is the common case, and it leaves the
binary on PATH.

---

## Extending the System

**Add a new test runner:**

```python
# In sandbox.py, add to ALLOWED_RUNNERS:
"deno": ["deno", "test"],
```

**JavaScript / TypeScript runners:** `vitest` and `jest` go through
`npx --no-install`, which resolves the target project's own `node_modules/.bin`
and fails if the package is missing rather than fetching it from the registry —
so the sandbox stays hermetic and `DockerSandbox`'s `--network=none` is not
quietly bypassed. Install the runner as a dev dependency of the repo under test
first. `vitest` is invoked as `vitest run`: bare `vitest` starts a watch server
when it believes it is interactive, which under the sandbox would sit until
`timeout_seconds` elapsed and be reported as a test timeout.

**Add a new file type to context scoring:**

```python
# In context_builder.py LANGUAGE_PRIORITY:
"lua": 7,
```

**Swap LLM provider (e.g., OpenAI):**
Implement the same `initial_request()` / `retry_request()` interface in a new
`OpenAIClient` class and pass it to `AutonomousAgent` — the loop is provider-agnostic.

**Add PR reviewer assignment:**

```python
# In git_integration.py create_github_pr():
payload["reviewers"] = ["alice", "bob"]
```
