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
--runner        Test runner: pytest|npm_test|go|cargo|... (default: pytest)
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
```

---

## Module Reference

### Module 1 — `repo_ingestion.py`
- Recursively walks a directory, skipping `.git`, `node_modules`, `__pycache__`, etc.
- Ignores binary files, files > 512KB, and enforces an 8MB total repo budget.
- Returns a `Repository` with `FileRecord[]` — path, content, language, checksum.

### Module 2 — `context_builder.py`
- Scores every file against the task using: language priority, path keyword match,
  content keyword frequency, entry-point bonus, import graph hints.
- Fills a configurable character budget (~60K chars / ~15K tokens).
- Returns `BuiltContext.render()` — XML-tagged source ready for the LLM prompt.

### Module 3 — `llm_client.py`
- Wraps the Anthropic API with structured JSON I/O.
- System prompt enforces a machine-parseable output schema.
- `initial_request()` for first pass; `retry_request()` for error-fed retries.
- Parses `FileChange[]` from JSON; gracefully handles malformed output.

### Module 4 — `code_modifier.py`
- Validates paths (prevents directory traversal).
- Backs up every file before modification.
- Supports `modify`, `create`, `delete` actions.
- `rollback()` restores all backups on failure.

### Module 5 — `sandbox.py`
- `SubprocessSandbox`: runs commands via `subprocess.run()` with timeout, output
  capture, and environment sanitization (blocks cloud credentials from leaking).
- `DockerSandbox`: wraps Docker with `--network=none`, memory/CPU caps, read-only
  volume mount. Falls back to subprocess if Docker is unavailable.

### Module 6 — `agent_loop.py` (CORE)
- `AutonomousAgent.run()` orchestrates all modules.
- Iterates up to `max_iterations` times.
- On success: commits (optionally pushes + opens PR).
- On failure: rolls back all file changes.
- Produces a `AgentRunResult` with outcome, branch, PR URL, iteration count.

### Module 7 — `git_integration.py`
- Wraps `git` subprocess calls: `create_branch`, `stage_files`, `commit`, `push`.
- GitHub PR creation via REST API (no extra dependencies — uses `urllib`).
- `rollback` is handled by `code_modifier.py`; git ops are only for success path.

### Module 8 — `logger.py`
- Every iteration appended to `<run_id>.jsonl` (structured, machine-readable).
- Human-readable log at `<run_id>_human.log`.
- Final `<run_id>_summary.json` with full run record.

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

## Failure Cases & Mitigations

| Failure | Cause | Mitigation |
|---------|-------|------------|
| `JSONDecodeError` from LLM | Model adds markdown fences or prose | Regex strips fences; parse error fed back as context next iter |
| Path traversal in LLM output | LLM outputs `../../etc/passwd` | `_safe_abs_path()` validates all paths against repo root |
| Empty content for modify | LLM returns `""` for file content | Validation rejects before apply; error logged |
| Infinite test loop | Test hangs | `timeout_seconds` in sandbox kills process |
| Repo too large | Monorepo with 10K files | 8MB total budget + per-file 512KB cap; budget exhausted = skip |
| Git merge conflict | Branch already exists | `create_branch` falls back to checkout if branch exists |
| LLM low confidence | Ambiguous task | `min_confidence_to_apply` threshold; skip without applying |
| Test runner not found | `pytest` not installed | `sandbox.py` checks `shutil.which()`; returns exit_code 127 |
| All apply ops fail | Wrong paths, permission error | Agent breaks loop, returns `error` outcome |
| Push auth failure | Missing SSH key / token | Logged as non-fatal; outcome still `success` locally |

---

## Extending the System

**Add a new test runner:**
```python
# In sandbox.py, add to ALLOWED_RUNNERS:
"deno": ["deno", "test"],
```

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
