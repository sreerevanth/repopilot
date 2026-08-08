"""
Tests for rollback leaving nothing behind (modules/code_modifier.py).

`_backup` returns None when a file does not already exist, so a "create" has
nothing to restore from — and rollback, which only restored backups, left the
new file in the tree. `git status` then showed untracked files after a rollback
that reported success, which is what #139 describes.
"""

import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from modules.code_modifier import CodeModificationEngine  # noqa: E402
from modules.llm_client import FileChange  # noqa: E402


@pytest.fixture
def engine(tmp_path):
    (tmp_path / "existing.py").write_text("original\n")
    return CodeModificationEngine(str(tmp_path), str(tmp_path / ".backups"))


def visible(root):
    return sorted(p for p in os.listdir(root) if not p.startswith("."))


def change(path, action="create", content="new\n"):
    return FileChange(path=path, action=action, content=content, explanation="e")


# ── the reported bug ──────────────────────────────────────────────────────


def test_a_created_file_is_removed(engine, tmp_path):
    results = engine.apply_changes([change("brand_new.py")])
    assert (tmp_path / "brand_new.py").exists()

    engine.rollback(results)

    assert not (tmp_path / "brand_new.py").exists()


def test_nothing_is_left_behind(engine, tmp_path):
    """The whole point: `git status` clean after a rollback."""
    results = engine.apply_changes([
        change("existing.py", action="modify", content="edited\n"),
        change("brand_new.py"),
        change("pkg/nested_new.py", content="nested\n"),
    ])

    engine.rollback(results)

    assert visible(tmp_path) == ["existing.py"]


def test_a_modified_file_is_still_restored(engine, tmp_path):
    """The existing behaviour must not regress."""
    results = engine.apply_changes(
        [change("existing.py", action="modify", content="edited\n")]
    )

    engine.rollback(results)

    assert (tmp_path / "existing.py").read_text() == "original\n"


# ── directories ───────────────────────────────────────────────────────────


def test_a_directory_created_for_the_file_is_pruned(engine, tmp_path):
    """`create` makes intermediate directories; an empty pkg/ is still litter."""
    results = engine.apply_changes([change("pkg/new.py")])

    engine.rollback(results)

    assert not (tmp_path / "pkg").exists()


def test_nested_directories_are_pruned(engine, tmp_path):
    results = engine.apply_changes([change("a/b/c/new.py")])

    engine.rollback(results)

    assert not (tmp_path / "a").exists()


def test_a_directory_that_still_has_files_is_kept(engine, tmp_path):
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "keep.py").write_text("keep me\n")
    results = engine.apply_changes([change("pkg/new.py")])

    engine.rollback(results)

    assert (tmp_path / "pkg" / "keep.py").exists()


def test_the_repo_root_is_never_removed(engine, tmp_path):
    results = engine.apply_changes([change("top_level.py")])

    engine.rollback(results)

    assert tmp_path.is_dir()


# ── reporting ─────────────────────────────────────────────────────────────


def test_removed_files_are_reported(engine):
    """
    agent_loop logs "Rolled back N file(s)". Omitting removals would understate
    what was actually undone.
    """
    results = engine.apply_changes([
        change("existing.py", action="modify", content="edited\n"),
        change("brand_new.py"),
    ])

    reverted = engine.rollback(results)

    assert sorted(reverted) == ["brand_new.py", "existing.py"]


def test_rollback_returns_paths_not_none(engine):
    assert isinstance(engine.rollback(engine.apply_changes([change("x.py")])), list)


# ── it does not overreach ─────────────────────────────────────────────────


def test_a_failed_create_does_not_remove_an_existing_file(engine, tmp_path):
    """
    A create that failed validation must not cause rollback to delete whatever
    happens to sit at that path.
    """
    from modules.code_modifier import ApplyResult

    fake = ApplyResult(
        path="existing.py", action="create", success=False,
        backup_path=None, error="refused",
    )

    engine.rollback([fake])

    assert (tmp_path / "existing.py").read_text() == "original\n"


def test_rollback_is_safe_to_run_twice(engine, tmp_path):
    """The second pass finds the file already gone."""
    results = engine.apply_changes([change("brand_new.py")])

    engine.rollback(results)
    engine.rollback(results)

    assert not (tmp_path / "brand_new.py").exists()


def test_other_actions_still_roll_back(engine, tmp_path):
    """delete restores from its backup; create is the only new path."""
    results = engine.apply_changes(
        [FileChange(path="existing.py", action="delete", content="", explanation="e")]
    )
    assert not (tmp_path / "existing.py").exists()

    engine.rollback(results)

    assert (tmp_path / "existing.py").read_text() == "original\n"
