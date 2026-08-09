"""
Tests for --clean (modules/housekeeping.py).

This is the one part of the tool whose job is deleting things, so most of what
follows is about what it must *not* remove. A log directory is a plausible place
for someone to have put a file of their own, and `--log-dir` pointing somewhere
unexpected should cost nothing.
"""

import os
import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from modules.housekeeping import (  # noqa: E402
    BACKUP_PATTERN,
    LOG_PATTERNS,
    clean,
    clean_directory,
    render_clean_summary,
)

OURS = [
    "agent_20260419_192447_57f5a9.jsonl",
    "agent_20260419_192447_57f5a9_human.log",
    "demo_20260419_183148_ae2caf_summary.json",
    "agent_20260419_192447_57f5a9_state.json",
]

NOT_OURS = ["notes.md", "README.md", "config.json", "important.log", ".gitkeep"]


@pytest.fixture
def logs(tmp_path):
    directory = tmp_path / "logs"
    directory.mkdir()
    for name in OURS + NOT_OURS:
        (directory / name).write_text("x" * 100)
    return directory


def remaining(directory):
    return sorted(p.name for p in directory.iterdir())


# ── what it removes ───────────────────────────────────────────────────────


@pytest.mark.parametrize("name", OURS)
def test_our_own_files_are_removed(tmp_path, name):
    (tmp_path / name).write_text("x")

    clean_directory(str(tmp_path), LOG_PATTERNS)

    assert not (tmp_path / name).exists()


def test_backups_are_removed(tmp_path):
    (tmp_path / "20260419_192447_main.py.bak").write_text("x")

    clean_directory(str(tmp_path), (BACKUP_PATTERN,))

    assert remaining(tmp_path) == []


def test_the_freed_size_is_reported(tmp_path):
    (tmp_path / OURS[0]).write_text("x" * 2048)

    assert clean_directory(str(tmp_path), LOG_PATTERNS).freed_bytes == 2048


# ── what it must not remove ───────────────────────────────────────────────


@pytest.mark.parametrize("name", NOT_OURS)
def test_other_peoples_files_are_left_alone(logs, name):
    """Only names matching the shapes this tool writes are removed."""
    clean_directory(str(logs), LOG_PATTERNS)

    assert (logs / name).exists()


def test_directories_are_left_alone(tmp_path):
    """
    Never recurses. Walking subdirectories means deciding what to do about a
    directory someone else created — a judgement this should not make.
    """
    nested = tmp_path / "archive"
    nested.mkdir()
    (nested / OURS[0]).write_text("x")

    clean_directory(str(tmp_path), LOG_PATTERNS)

    assert (nested / OURS[0]).exists()


def test_symlinks_are_left_alone(tmp_path):
    target = tmp_path / "real.txt"
    target.write_text("x")
    link = tmp_path / OURS[0]
    try:
        link.symlink_to(target)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks unavailable")

    clean_directory(str(tmp_path), LOG_PATTERNS)

    assert link.exists()
    assert target.exists()


def test_a_missing_directory_is_not_an_error(tmp_path):
    result = clean_directory(str(tmp_path / "nope"), LOG_PATTERNS)

    assert result.removed == []


def test_an_unrelated_directory_loses_nothing(tmp_path):
    """`--log-dir /etc` should cost nothing, not everything."""
    for name in ("passwd", "hosts", "fstab"):
        (tmp_path / name).write_text("x")

    clean_directory(str(tmp_path), LOG_PATTERNS)

    assert len(remaining(tmp_path)) == 3


# ── dry run ───────────────────────────────────────────────────────────────


def test_a_dry_run_deletes_nothing(logs):
    clean_directory(str(logs), LOG_PATTERNS, dry_run=True)

    assert len(remaining(logs)) == len(OURS) + len(NOT_OURS)


def test_a_dry_run_still_reports_what_would_go(logs):
    result = clean_directory(str(logs), LOG_PATTERNS, dry_run=True)

    assert len(result.removed) == len(OURS)


# ── age filter ────────────────────────────────────────────────────────────


def test_recent_files_are_kept(tmp_path):
    (tmp_path / OURS[0]).write_text("x")

    result = clean_directory(str(tmp_path), LOG_PATTERNS, older_than_days=7)

    assert result.removed == []
    assert (tmp_path / OURS[0]).exists()


def test_old_files_are_removed(tmp_path):
    path = tmp_path / OURS[0]
    path.write_text("x")
    old = time.time() - (30 * 86400)
    os.utime(path, (old, old))

    result = clean_directory(str(tmp_path), LOG_PATTERNS, older_than_days=7)

    assert result.removed == [OURS[0]]


def test_no_age_filter_removes_everything_matching(logs):
    result = clean_directory(str(logs), LOG_PATTERNS)

    assert len(result.removed) == len(OURS)


# ── both directories ──────────────────────────────────────────────────────


def test_both_directories_are_cleaned(tmp_path):
    (tmp_path / "logs").mkdir()
    (tmp_path / "backups").mkdir()
    (tmp_path / "logs" / OURS[0]).write_text("x")
    (tmp_path / "backups" / "20260419_192447_a.py.bak").write_text("x")

    results = clean(str(tmp_path))

    assert len(results["logs"].removed) == 1
    assert len(results["backups"].removed) == 1


# ── reporting ─────────────────────────────────────────────────────────────


def test_nothing_to_clean_says_so(tmp_path):
    assert render_clean_summary(clean(str(tmp_path))) == "Nothing to clean."


def test_the_summary_reports_counts_and_size(tmp_path):
    (tmp_path / "logs").mkdir()
    (tmp_path / "logs" / OURS[0]).write_text("x" * 4096)

    summary = render_clean_summary(clean(str(tmp_path)))

    assert "1 file(s)" in summary
    assert "KB" in summary


def test_a_dry_run_summary_uses_the_conditional(tmp_path):
    (tmp_path / "logs").mkdir()
    (tmp_path / "logs" / OURS[0]).write_text("x")

    summary = render_clean_summary(clean(str(tmp_path), dry_run=True), dry_run=True)

    assert summary.startswith("Would remove")


def test_files_left_alone_are_mentioned(tmp_path):
    """
    So it is visible that something was skipped, rather than looking like the
    directory was already clean.
    """
    (tmp_path / "logs").mkdir()
    (tmp_path / "logs" / OURS[0]).write_text("x")
    (tmp_path / "logs" / "notes.md").write_text("x")

    assert "not ours" in render_clean_summary(clean(str(tmp_path)))
