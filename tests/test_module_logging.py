"""
Tests for module-level logging (modules/code_modifier.py, modules/agent_loop.py).

A library that calls print() writes to stdout unconditionally: it cannot be
routed to a file, cannot be silenced by --quiet, and does not appear in the run
log. These pin the conversion — and, just as deliberately, pin the places where
print() is the right call and should stay.
"""

import logging
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from modules.code_modifier import CodeModificationEngine  # noqa: E402

MODULES = Path(__file__).resolve().parents[1] / "modules"


def source(name: str) -> str:
    # Explicit encoding: the default is locale-dependent and mangles the
    # box-drawing section comments on a default Windows install.
    return (MODULES / name).read_text(encoding="utf-8")


def git(cwd, *args):
    return subprocess.run(
        ["git", "-C", str(cwd), *args], capture_output=True, text=True
    )


@pytest.fixture
def engine(tmp_path):
    git(tmp_path, "init", "-q")
    git(tmp_path, "config", "user.email", "t@example.com")
    git(tmp_path, "config", "user.name", "Test")
    (tmp_path / "f.txt").write_text("v1\n")
    git(tmp_path, "add", "-A")
    git(tmp_path, "commit", "-qm", "initial")
    (tmp_path / "f.txt").write_text("v2\n")

    return CodeModificationEngine(
        repo_root=str(tmp_path), backup_dir=str(tmp_path / ".backups")
    )


# ── the conversion ────────────────────────────────────────────────────────


def test_code_modifier_no_longer_prints():
    assert "print(" not in source("code_modifier.py")


def test_agent_loops_dry_run_lines_go_to_the_logger():
    """The two [DRY RUN] lines had a logger to hand and were not using it."""
    text = source("agent_loop.py")
    assert 'print(f"\\n[DRY RUN]' not in text
    assert 'self.logger.info("[DRY RUN] No files were modified.")' in text


def test_the_review_prompt_still_prints():
    """
    _confirm_commit renders the diff a user is being asked to approve, for the
    same reason dry_run's manifest does. A log level must not be able to hide it.
    """
    text = source("agent_loop.py")
    assert "TESTS PASSED - REVIEW BEFORE COMMIT" in text
    assert 'print("=" * 60)' in text


def test_the_logger_is_in_the_agent_namespace():
    """
    So it inherits the handlers AgentLogger configures on `agent.<run_id>` and
    appears in the run log rather than only on the terminal.
    """
    assert 'logging.getLogger("agent.code_modifier")' in source("code_modifier.py")


def test_stash_messages_go_to_the_logger(engine, caplog):
    with caplog.at_level(logging.INFO, logger="agent.code_modifier"):
        engine.git_stash_before_apply(engine.repo_root)

    assert any("stash" in record.message.lower() for record in caplog.records)


def test_nothing_reaches_stdout(engine, capsys):
    """A library must not write to stdout uninvited."""
    engine.git_stash_before_apply(engine.repo_root)
    engine.git_stash_pop(engine.repo_root)

    assert capsys.readouterr().out == ""


def test_info_is_suppressed_at_warning_level(engine, caplog):
    """This is what --quiet buys; print() could not be silenced at all."""
    with caplog.at_level(logging.WARNING, logger="agent.code_modifier"):
        engine.git_stash_before_apply(engine.repo_root)

    assert [r for r in caplog.records if r.levelno < logging.WARNING] == []


def test_a_failed_pop_is_logged_as_an_error(engine, caplog):
    """Nothing to pop: the failure should be an error, not silence."""
    with caplog.at_level(logging.DEBUG, logger="agent.code_modifier"):
        engine.git_stash_pop(engine.repo_root)

    assert any(r.levelno >= logging.ERROR for r in caplog.records)


def test_a_clean_tree_is_reported_at_info(engine, caplog):
    engine.git_stash_before_apply(engine.repo_root)  # stashes the pending edit
    with caplog.at_level(logging.DEBUG, logger="agent.code_modifier"):
        engine.git_stash_before_apply(engine.repo_root)

    assert any("clean" in r.message.lower() for r in caplog.records)


def test_lazy_formatting_is_used():
    """
    %-style arguments rather than f-strings, so a suppressed message costs no
    string interpolation.
    """
    text = source("code_modifier.py")
    assert '_LOG.warning("git stash failed: %s"' in text
    assert '_LOG.error("Failed to pop the stash: %s"' in text


# ── what should stay a print ──────────────────────────────────────────────


def test_dry_run_still_prints_the_manifest():
    """
    print_manifest renders a table the user is being asked to approve, and
    ask_confirmation drives an interactive prompt. Routing those through the
    logger would let a log level hide the thing the user must read.
    """
    assert "print(" in source("dry_run.py")


def test_main_still_prints_cli_output():
    """
    The banner, the final summary and errors to stderr are the CLI's output,
    not diagnostics. They belong on stdout unconditionally.
    """
    root = Path(__file__).resolve().parents[1]
    assert "print(" in (root / "main.py").read_text(encoding="utf-8")
