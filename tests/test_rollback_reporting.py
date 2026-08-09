"""
Tests for rollback failure reporting (modules/code_modifier.py).

Two sites in the rollback path caught `except Exception: pass` — removing a
renamed file's destination, and restoring a backup. A failure at either left
the tree partly reverted while the run reported success.

That is worse than a rollback that fails outright. Rollback runs when something
has already gone wrong, and it is the last thing between a bad run and a tree
the user has to repair by hand. Told the changes were undone, they do not check;
the leftover file surfaces later with nothing linking it to the run.
"""

import io
import logging
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from modules.code_modifier import ApplyResult, CodeModificationEngine  # noqa: E402


@pytest.fixture
def engine(tmp_path):
    repo = tmp_path / "repo"
    backups = tmp_path / "backups"
    repo.mkdir()
    backups.mkdir()
    return CodeModificationEngine(str(repo), str(backups)), repo, backups


@pytest.fixture
def captured_log():
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    log = logging.getLogger("agent.code_modifier")
    log.addHandler(handler)
    previous = log.level
    log.setLevel(logging.WARNING)
    yield stream
    log.removeHandler(handler)
    log.setLevel(previous)


def modified(path, backup):
    return ApplyResult(path=path, action="modify", success=True,
                       backup_path=str(backup), error=None, new_path=None)


# ── a failed restore ──────────────────────────────────────────────────────


def test_a_failed_restore_is_logged(engine, captured_log):
    agent, repo, backups = engine
    (repo / "a.py").write_text("modified by agent\n")
    backup = backups / "a.py.bak"
    backup.write_text("original\n")

    with patch("shutil.copy2", side_effect=OSError("permission denied")):
        agent.rollback([modified("a.py", backup)])

    assert "could not restore a.py" in captured_log.getvalue()


def test_a_failed_restore_names_the_reason(engine, captured_log):
    """"Failed" without the errno sends the reader to the wrong place."""
    agent, repo, backups = engine
    (repo / "a.py").write_text("modified\n")
    backup = backups / "a.py.bak"
    backup.write_text("original\n")

    with patch("shutil.copy2", side_effect=OSError("permission denied")):
        agent.rollback([modified("a.py", backup)])

    assert "permission denied" in captured_log.getvalue()


def test_an_incomplete_rollback_says_so(engine, captured_log):
    """
    The summary is the part that matters: a per-file warning can scroll past,
    and the caller's count of reverted paths says nothing about the rest.
    """
    agent, repo, backups = engine
    (repo / "a.py").write_text("modified\n")
    backup = backups / "a.py.bak"
    backup.write_text("original\n")

    with patch("shutil.copy2", side_effect=OSError("nope")):
        agent.rollback([modified("a.py", backup)])

    output = captured_log.getvalue()
    assert "Rollback incomplete" in output
    assert "by hand" in output


def test_a_failed_path_is_not_counted_as_reverted(engine, captured_log):
    """Reporting it as reverted is what made the old behaviour a lie."""
    agent, repo, backups = engine
    (repo / "a.py").write_text("modified\n")
    backup = backups / "a.py.bak"
    backup.write_text("original\n")

    with patch("shutil.copy2", side_effect=OSError("nope")):
        reverted = agent.rollback([modified("a.py", backup)])

    assert "a.py" not in reverted


# ── the happy path is unchanged ───────────────────────────────────────────


def test_a_successful_restore_still_works(engine, captured_log):
    agent, repo, backups = engine
    (repo / "a.py").write_text("modified by agent\n")
    backup = backups / "a.py.bak"
    backup.write_text("original\n")

    reverted = agent.rollback([modified("a.py", backup)])

    assert (repo / "a.py").read_text() == "original\n"
    assert "a.py" in reverted


def test_a_clean_rollback_reports_nothing(engine, captured_log):
    """No warning when there is nothing to warn about."""
    agent, repo, backups = engine
    (repo / "a.py").write_text("modified\n")
    backup = backups / "a.py.bak"
    backup.write_text("original\n")

    agent.rollback([modified("a.py", backup)])

    assert "Rollback incomplete" not in captured_log.getvalue()


def test_one_failure_does_not_stop_the_others(engine, captured_log):
    """
    A rollback that raises partway leaves things worse than one that continues,
    so a failure is recorded and the remaining files are still attempted.
    """
    agent, repo, backups = engine
    for name in ("a.py", "b.py"):
        (repo / name).write_text("modified\n")
        (backups / f"{name}.bak").write_text("original\n")

    real = __import__("shutil").copy2
    calls = {"n": 0}

    def fail_first(src, dst):
        calls["n"] += 1
        if calls["n"] == 1:
            raise OSError("nope")
        return real(src, dst)

    with patch("shutil.copy2", side_effect=fail_first):
        reverted = agent.rollback([
            modified("a.py", backups / "a.py.bak"),
            modified("b.py", backups / "b.py.bak"),
        ])

    assert reverted == ["b.py"]
    assert (repo / "b.py").read_text() == "original\n"


# ── the mechanism ─────────────────────────────────────────────────────────


def test_no_bare_swallow_remains_in_rollback():
    """
    `except Exception: pass` is what made a partial rollback invisible. Its
    absence in this method is the change.
    """
    source = (ROOT / "modules" / "code_modifier.py").read_text(encoding="utf-8")
    start = source.index("def rollback(self")
    end = source.index("def verify_changes(self")
    body = source[start:end]

    assert "except Exception:\n                    pass" not in body


def test_the_handlers_are_narrowed():
    """
    `except Exception` here would also swallow a TypeError from
    _safe_abs_path, which would be a bug in this module rather than a
    filesystem condition.
    """
    source = (ROOT / "modules" / "code_modifier.py").read_text(encoding="utf-8")
    start = source.index("def rollback(self")
    end = source.index("def verify_changes(self")

    assert source[start:end].count("except (OSError, ValueError) as exc:") == 2
