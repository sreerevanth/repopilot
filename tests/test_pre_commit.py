"""
Tests for pre-commit integration (modules/sandbox.py, modules/agent_loop.py).

When a repository has `.pre-commit-config.yaml`, its hooks run against the
changed files before the test suite.

The distinction that makes this work: pre-commit reports both auto-fixes and
genuine failures as "Failed". Treating the first as a failure would send the
agent back to fix code that a hook just fixed for it.
"""

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from modules.agent_loop import AgentConfig  # noqa: E402
from modules.sandbox import PRE_COMMIT_CONFIG, SubprocessSandbox  # noqa: E402

AGENT_LOOP = Path(__file__).resolve().parents[1] / "modules" / "agent_loop.py"

needs_pre_commit = pytest.mark.skipif(
    shutil.which("pre-commit") is None, reason="pre-commit not installed"
)

CONFIG = """\
repos:
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.5.0
    hooks:
      - id: trailing-whitespace
      - id: end-of-file-fixer
      - id: check-yaml
"""


@pytest.fixture
def repo(tmp_path):
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    (tmp_path / PRE_COMMIT_CONFIG).write_text(CONFIG)
    return tmp_path


def staged(repo):
    subprocess.run(["git", "-C", str(repo), "add", "-A"], check=True)
    return SubprocessSandbox(str(repo), timeout_seconds=300)


# ── refusals and no-ops ───────────────────────────────────────────────────


def test_no_files_is_a_pass(tmp_path):
    """An iteration that changed nothing has nothing for hooks to check."""
    result = SubprocessSandbox(str(tmp_path)).run_pre_commit([])

    assert result.success is True


def test_a_missing_binary_is_reported_not_raised(tmp_path, monkeypatch):
    monkeypatch.setattr(shutil, "which", lambda name: None)

    result = SubprocessSandbox(str(tmp_path)).run_pre_commit(["a.py"])

    assert result.exit_code == -3
    assert "pip install pre-commit" in result.stderr


def test_the_config_name_is_the_conventional_one():
    assert PRE_COMMIT_CONFIG == ".pre-commit-config.yaml"


# ── the distinction that matters ──────────────────────────────────────────


@needs_pre_commit
def test_an_auto_fix_counts_as_success(repo):
    """
    trailing-whitespace rewrites the file and exits non-zero to say so. Nothing
    is wrong — treating it as a failure would send the agent back to fix code a
    hook just fixed.
    """
    (repo / "messy.py").write_text("x = 1   \ny = 2")

    assert staged(repo).run_pre_commit(["messy.py"]).success is True


@needs_pre_commit
def test_the_file_is_actually_rewritten(repo):
    (repo / "messy.py").write_text("x = 1   \ny = 2")
    staged(repo).run_pre_commit(["messy.py"])

    assert (repo / "messy.py").read_text() == "x = 1\ny = 2\n"


@needs_pre_commit
def test_the_reformatting_is_reported(repo):
    """
    Silence here would hide the fact that the tree no longer matches what the
    model wrote.
    """
    (repo / "messy.py").write_text("x = 1   \ny = 2")

    stdout = staged(repo).run_pre_commit(["messy.py"]).stdout

    assert "reformatted" in stdout
    assert "will be tested" in stdout


@needs_pre_commit
def test_a_genuine_failure_fails(repo):
    """check-yaml cannot fix invalid YAML, so a re-run fails too."""
    (repo / "bad.yaml").write_text("not: valid: yaml: [\n")

    assert staged(repo).run_pre_commit(["bad.yaml"]).success is False


@needs_pre_commit
def test_the_failure_reason_reaches_the_model(repo):
    (repo / "bad.yaml").write_text("not: valid: yaml: [\n")

    result = staged(repo).run_pre_commit(["bad.yaml"])

    assert "yaml" in (result.stdout + result.stderr).lower()


@needs_pre_commit
def test_clean_files_pass_without_a_second_run(repo):
    (repo / "clean.py").write_text("x = 1\n")

    assert staged(repo).run_pre_commit(["clean.py"]).success is True


# ── wiring ────────────────────────────────────────────────────────────────


def test_it_is_on_by_default():
    """The issue asks for it whenever the repo configures hooks."""
    assert AgentConfig(repo_root=".", task="t").run_pre_commit is True


def test_it_runs_before_the_test_suite():
    """
    Hooks that reformat leave the tree different from what the model wrote, so
    tests measured first would be measuring code the hooks then rewrite.
    """
    source = AGENT_LOOP.read_text(encoding="utf-8")

    assert source.index("run_pre_commit(changed)") < \
        source.index("# ── Step 5: Execute ──")


def test_nothing_happens_without_a_config():
    """Repositories that do not use pre-commit are unaffected."""
    source = AGENT_LOOP.read_text(encoding="utf-8")
    block = source[source.index("if cfg.run_pre_commit and last_changes:"):][:400]

    assert "os.path.isfile(config_path)" in block


def test_only_changed_files_are_checked():
    """Running every hook over a whole repository would be far slower."""
    source = AGENT_LOOP.read_text(encoding="utf-8")
    block = source[source.index("if cfg.run_pre_commit and last_changes:"):][:500]

    assert "[c.path for c in last_changes]" in block


def test_a_missing_binary_does_not_fail_the_iteration():
    """
    -3 means pre-commit is not installed, which is the user's environment
    rather than a problem with the model's output.
    """
    source = AGENT_LOOP.read_text(encoding="utf-8")
    block = source[source.index("if cfg.run_pre_commit and last_changes:"):][:900]

    assert "hook_result.exit_code != -3" in block
