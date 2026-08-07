# Contributing to RepoPilot

Thanks for taking the time. This guide covers the architecture, how to run
things locally without an API key, and the conventions the codebase follows.

---

## Getting set up

```bash
git clone https://github.com/sreerevanth/repopilot.git
cd repopilot
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

An `ANTHROPIC_API_KEY` is only needed to run the real agent. Everything in the
next section works without one.

---

## Running locally without an API key

`demo_run.py` drives the whole pipeline with a `MockLLMClient` that returns a
real, correct fix on the first call. It exercises ingestion, context building,
modification, sandbox execution, git and logging — every layer except the
network call.

```bash
python demo_run.py sample_repo
```

You should see `FINAL OUTCOME : SUCCESS`.

**Pass the path explicitly.** The default argument is `../sample_repo`, which
resolves to a directory _outside_ the repository. Running `python demo_run.py`
with no argument creates an empty directory there, collects zero tests, and
reports `MAX_RETRIES` — the pipeline is fine, it is just pointed at nothing.

**It has side effects on your checkout.** `sample_repo/` lives inside this git
repository, so the demo's git integration operates on _this_ repo: it creates
and checks out a branch named `agent/demo-fix-<id>`, and writes `logs/` and
`backups/` under `sample_repo/`. Clean up afterwards:

```bash
git checkout <your-branch>
git branch -D agent/demo-fix-<id>
rm -rf sample_repo/logs sample_repo/backups
```

---

## Running the tests

Run test files by path:

```bash
python -m pytest tests/test_utils.py -q
```

**A bare `python -m pytest` from the repo root currently fails collection.**
Three files share the basename `test_utils.py` (`tests/`, `modules/`,
`sample_repo/`) with no `__init__.py` and no pytest configuration, so pytest
reports an import file mismatch before running anything. This is a known
pre-existing issue, not something you broke.

Because of it, **give new test files a unique basename** — `test_sandbox_env.py`
rather than another `test_utils.py`.

Formatting is checked by Prettier (Markdown, JSON, YAML — not Python):

```bash
npm ci
npm run format:check
```

Some files on `main` already fail this check. If `format:check` is red on files
you did not touch, that is expected.

---

## Architecture

Eight modules, run in order by the agent loop. Each is independently testable.

| #   | Module               | Responsibility                                            |
| --- | -------------------- | --------------------------------------------------------- |
| 1   | `repo_ingestion.py`  | Walk the repo, read text files, apply size budgets        |
| 2   | `context_builder.py` | Score and select files that fit the context budget        |
| 3   | `llm_client.py`      | Prompt construction, Anthropic call, response parsing     |
| 4   | `code_modifier.py`   | Validate paths, back up, apply `modify`/`create`/`delete` |
| 5   | `sandbox.py`         | Run the test suite with a timeout, bounded output         |
| 6   | `agent_loop.py`      | Orchestrates 1–5 and 7–8; the core loop                   |
| 7   | `git_integration.py` | Branch, stage, commit, push, open a PR                    |
| 8   | `logger.py`          | JSONL run records plus a human-readable log               |

### The loop

```
ingest → build context → ask the LLM → validate → apply → run tests
   ↑                                                          │
   └──────────── feed errors back, next iteration ────────────┘
```

On success it commits, optionally pushes and opens a PR. On failure it rolls
back every file it touched, restoring from the backups taken before each write.

### Boundaries worth knowing

The code the LLM writes is **not reviewed before it runs**. Two places treat it
as untrusted, and changes near them deserve extra care:

- `code_modifier._safe_abs_path()` validates every path against the repo root,
  so the model cannot write outside it.
- `sandbox._build_safe_env()` filters the environment handed to executed code.

`DockerSandbox` provides real isolation (`--network=none`, memory and CPU caps).
`SubprocessSandbox` does not — it is a timeout and output cap, nothing more.

---

## Conventions

**Read the live code before writing.** Issue descriptions in this repo have
sometimes described behaviour that no longer exists, or proposed fixes that
would break something else. Reproduce the problem first.

**Say what you verified.** If you could not exercise a path — no Docker daemon,
no Windows machine — say so in the PR rather than implying coverage. A stated
gap is far more useful than a confident claim that turns out to be wrong.

**Keep pre-existing problems separate.** If you notice something broken that is
not your issue, mention it in the PR description or open a new issue. Do not
fold unrelated fixes into a change, and do not claim credit for behaviour that
was already there.

**One issue, one branch, one PR.** Branch from the default branch, not from
another feature branch. Several modules — `sandbox.py` especially — have
multiple open PRs at once, so check whether your change overlaps before starting.

**Tests go beside the behaviour they pin.** Prefer assertions on observable
behaviour over implementation detail, and give each test file a unique basename.

**Python style:** 4-space indent, type hints on public functions, docstrings
that explain _why_ rather than restating the signature. Lines under 88
characters where practical, though `main.py` and `agent_loop.py` already exceed
that in places.

---

## Opening a pull request

Include, at minimum:

- what the problem was, and how you reproduced it
- what you changed
- what you ran to verify, and what you could not verify
- anything pre-existing you noticed but deliberately left alone

Screenshots or terminal output showing before-and-after behaviour help a great
deal on bug fixes.
