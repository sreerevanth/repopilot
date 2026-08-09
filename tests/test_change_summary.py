"""
Tests for the end-of-run file summary (modules/change_summary.py).

A run has several iterations and a file can be touched in more than one, so the
table reports net effect against the state the run started from rather than a
log of every intermediate step. A file created in iteration 1 and edited again
in iteration 3 is an addition, not a modification.
"""

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from modules.change_summary import (  # noqa: E402
    ADDED,
    DELETED,
    MODIFIED,
    RENAMED,
    render_change_summary,
    summarise_changes,
)


def applied(path, action, success=True, new_path=None):
    return SimpleNamespace(
        path=path, action=action, success=success, new_path=new_path
    )


def actions(results):
    return {c.path: c.action for c in summarise_changes(results)}


# ── one action per file ───────────────────────────────────────────────────


@pytest.mark.parametrize(
    "action,expected",
    [("create", ADDED), ("modify", MODIFIED), ("delete", DELETED),
     ("patch", MODIFIED), ("rename", RENAMED)],
)
def test_each_action_maps_to_an_outcome(action, expected):
    assert actions([applied("f.py", action)]) == {"f.py": expected}


def test_a_failed_change_is_not_reported():
    """It changed nothing, so listing it would send someone looking for a diff."""
    assert actions([applied("f.py", "modify", success=False)]) == {}


# ── net effect across iterations ──────────────────────────────────────────


def test_created_then_edited_is_an_addition():
    """
    The case that makes this more than a log replay: the file is new to the
    repository however many times it was edited afterwards.
    """
    result = actions([applied("new.py", "create"), applied("new.py", "modify")])

    assert result == {"new.py": ADDED}


def test_edited_repeatedly_is_one_modification():
    result = actions([applied("f.py", "modify")] * 4)

    assert result == {"f.py": MODIFIED}


def test_created_then_deleted_cancels_out():
    """
    The repository ends as it began. Reporting it would send someone looking
    for a change that is not there.
    """
    assert actions([applied("t.py", "create"), applied("t.py", "delete")]) == {}


def test_modified_then_deleted_is_a_deletion():
    result = actions([applied("f.py", "modify"), applied("f.py", "delete")])

    assert result == {"f.py": DELETED}


def test_deleted_then_recreated_is_a_modification():
    """
    The path existed before and exists now with different content — which is
    what a modification is, even though the steps were delete then create.
    """
    result = actions([applied("f.py", "delete"), applied("f.py", "create")])

    assert result == {"f.py": MODIFIED}


def test_a_rename_survives_later_edits():
    result = actions([
        applied("a.py", "rename", new_path="b.py"),
        applied("a.py", "modify"),
    ])

    assert result == {"a.py": RENAMED}


# ── the table ─────────────────────────────────────────────────────────────


@pytest.fixture
def mixed():
    return summarise_changes([
        applied("added.py", "create"),
        applied("edited.py", "modify"),
        applied("gone.py", "delete"),
        applied("moved.py", "rename", new_path="elsewhere.py"),
    ])


def test_every_file_appears(mixed):
    rendered = render_change_summary(mixed)

    for name in ("added.py", "edited.py", "gone.py", "moved.py"):
        assert name in rendered


def test_the_counts_are_reported(mixed):
    rendered = render_change_summary(mixed)

    assert "1 added" in rendered
    assert "1 modified" in rendered
    assert "1 deleted" in rendered


def test_a_rename_shows_both_paths(mixed):
    assert "moved.py -> elsewhere.py" in render_change_summary(mixed)


def test_files_are_grouped_by_action(mixed):
    rendered = render_change_summary(mixed)

    assert rendered.index("+ added.py") < rendered.index("~ edited.py")
    assert rendered.index("~ edited.py") < rendered.index("- gone.py")


def test_paths_are_sorted():
    changed = summarise_changes([
        applied("z.py", "modify"), applied("a.py", "modify"),
    ])

    assert [c.path for c in changed] == ["a.py", "z.py"]


def test_no_changes_says_so():
    assert render_change_summary([]) == "No files were changed."


# ── wiring ────────────────────────────────────────────────────────────────


def test_the_loop_accumulates_across_iterations():
    """
    last_apply_results is overwritten each iteration, so summarising it would
    report only the final one.
    """
    source = (
        Path(__file__).resolve().parents[1] / "modules" / "agent_loop.py"
    ).read_text(encoding="utf-8")

    assert "all_apply_results.extend(apply_results)" in source


def test_the_summary_is_printed_before_the_run_closes():
    source = (
        Path(__file__).resolve().parents[1] / "modules" / "agent_loop.py"
    ).read_text(encoding="utf-8")

    assert source.index("render_change_summary(changed)") < \
        source.index("self.logger.finish_run(outcome, self.branch_name, self.pr_url)")
