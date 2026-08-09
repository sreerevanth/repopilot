"""
Tests for the rename/move action (modules/code_modifier.py).

Without it the model's only route to moving a file is a `create` at the new
path plus a `delete` at the old one — two independent operations that can half
apply. The rollback half matters as much as the apply half: restoring the
backup to the original path leaves the moved copy behind unless the destination
is removed too.
"""

import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from modules.code_modifier import (  # noqa: E402
    RENAME_ACTIONS,
    VALID_ACTIONS,
    CodeModificationEngine,
)
from modules.llm_client import BaseLLMClient, FileChange  # noqa: E402

ORIGINAL = "def add(a, b):\n    return a + b\n"


@pytest.fixture
def engine(tmp_path):
    (tmp_path / "old.py").write_text(ORIGINAL)
    return CodeModificationEngine(
        repo_root=str(tmp_path), backup_dir=str(tmp_path / ".backups")
    )


def rename(path="old.py", new_path="new.py", content="", action="rename"):
    return FileChange(
        path=path, action=action, content=content,
        explanation="moving it", new_path=new_path,
    )


# ── applying ──────────────────────────────────────────────────────────────


def test_file_moves_to_the_new_path(engine, tmp_path):
    result = engine.apply_changes([rename()])[0]

    assert result.success is True
    assert (tmp_path / "new.py").exists()
    assert not (tmp_path / "old.py").exists()


def test_contents_survive_a_plain_rename(engine, tmp_path):
    engine.apply_changes([rename()])
    assert (tmp_path / "new.py").read_text() == ORIGINAL


def test_destination_directories_are_created(engine, tmp_path):
    engine.apply_changes([rename(new_path="pkg/sub/new.py")])
    assert (tmp_path / "pkg" / "sub" / "new.py").read_text() == ORIGINAL


def test_rename_can_also_change_content(engine, tmp_path):
    """Moving and editing in one step, rather than a move then a modify."""
    engine.apply_changes([rename(content="def add(a, b):\n    return a + b + 0\n")])
    assert "a + b + 0" in (tmp_path / "new.py").read_text()


def test_move_is_accepted_as_an_alias(engine, tmp_path):
    result = engine.apply_changes([rename(action="move")])[0]
    assert result.success is True
    assert (tmp_path / "new.py").exists()


def test_result_records_the_destination(engine):
    """rollback needs it; without it the move cannot be undone."""
    assert engine.apply_changes([rename()])[0].new_path == "new.py"


def test_a_backup_is_taken(engine):
    result = engine.apply_changes([rename()])[0]
    assert result.backup_path is not None
    assert os.path.exists(result.backup_path)


# ── refusals ──────────────────────────────────────────────────────────────


def test_missing_source_is_refused(engine):
    result = engine.apply_changes([rename(path="nope.py")])[0]
    assert result.success is False
    assert "missing file" in result.error


def test_missing_new_path_is_refused(engine, tmp_path):
    result = engine.apply_changes([rename(new_path=None)])[0]
    assert result.success is False
    assert "new_path" in result.error
    assert (tmp_path / "old.py").exists()  # untouched


def test_renaming_onto_itself_is_refused(engine):
    result = engine.apply_changes([rename(new_path="old.py")])[0]
    assert result.success is False
    assert "same" in result.error


def test_existing_destination_is_not_clobbered(engine, tmp_path):
    """No backup here would cover a file the task never mentioned."""
    (tmp_path / "occupied.py").write_text("do not lose me\n")

    result = engine.apply_changes([rename(new_path="occupied.py")])[0]

    assert result.success is False
    assert (tmp_path / "occupied.py").read_text() == "do not lose me\n"
    assert (tmp_path / "old.py").exists()


@pytest.mark.parametrize("escape", ["../../etc/passwd", "/etc/passwd"])
def test_destination_cannot_escape_the_repo(engine, tmp_path, escape):
    """new_path is model output and gets the same validation as path."""
    result = engine.apply_changes([rename(new_path=escape)])[0]

    assert result.success is False
    assert (tmp_path / "old.py").exists()


def test_a_refused_rename_leaves_no_partial_state(engine, tmp_path):
    engine.apply_changes([rename(new_path="occupied.py")])  # dest does not exist
    # It should have succeeded; now try a genuinely bad one on the moved file.
    result = engine.apply_changes([rename(path="occupied.py", new_path="../out.py")])[0]

    assert result.success is False
    assert (tmp_path / "occupied.py").exists()
    assert not (tmp_path.parent / "out.py").exists()


# ── rollback ──────────────────────────────────────────────────────────────


def test_rollback_restores_the_original_path(engine, tmp_path):
    results = engine.apply_changes([rename()])
    engine.rollback(results)

    assert (tmp_path / "old.py").exists()
    assert (tmp_path / "old.py").read_text() == ORIGINAL


def test_rollback_removes_the_moved_copy(engine, tmp_path):
    """
    The half that is easy to miss. Restoring the backup alone would leave the
    file at both paths, which is worse than not rolling back at all.
    """
    results = engine.apply_changes([rename()])
    engine.rollback(results)

    assert not (tmp_path / "new.py").exists()


def test_rollback_after_a_nested_rename(engine, tmp_path):
    results = engine.apply_changes([rename(new_path="pkg/sub/new.py")])
    engine.rollback(results)

    assert (tmp_path / "old.py").read_text() == ORIGINAL
    assert not (tmp_path / "pkg" / "sub" / "new.py").exists()


def test_rollback_of_a_content_changing_rename(engine, tmp_path):
    results = engine.apply_changes([rename(content="changed\n")])
    engine.rollback(results)

    assert (tmp_path / "old.py").read_text() == ORIGINAL
    assert not (tmp_path / "new.py").exists()


def test_rollback_still_works_for_modify(engine, tmp_path):
    """The rename branch must not disturb the existing actions."""
    results = engine.apply_changes(
        [FileChange(path="old.py", action="modify", content="broken\n", explanation="e")]
    )
    engine.rollback(results)
    assert (tmp_path / "old.py").read_text() == ORIGINAL


# ── pre-flight validation ─────────────────────────────────────────────────


def test_rename_is_a_valid_action():
    for action in RENAME_ACTIONS:
        assert action in VALID_ACTIONS


def test_verify_accepts_a_well_formed_rename(engine):
    assert engine.verify_changes([rename()]) == []


def test_verify_rejects_a_missing_new_path(engine):
    errors = engine.verify_changes([rename(new_path=None)])
    assert any("new_path" in e for e in errors)


def test_verify_rejects_an_escaping_new_path(engine):
    errors = engine.verify_changes([rename(new_path="../../etc/passwd")])
    assert any("new_path" in e for e in errors)


def test_verify_errors_name_the_source_path(engine):
    """agent_loop filters changes by matching path substrings against errors."""
    errors = engine.verify_changes([rename(new_path=None)])
    assert all("old.py" in e for e in errors)


def test_rename_does_not_require_content(engine):
    """Only modify and create need content; a plain move has none."""
    assert engine.verify_changes([rename(content="")]) == []


def test_unknown_actions_are_still_rejected(engine):
    errors = engine.verify_changes(
        [FileChange(path="old.py", action="teleport", content="x", explanation="e")]
    )
    assert any("Invalid action" in e for e in errors)


# ── the model has to know the action exists ───────────────────────────────


def test_new_path_is_parsed_from_the_response():
    # BaseLLMClient is where _parse_response lives; LLMClient is a facade that
    # delegates to a provider. The second argument is the input token count,
    # added when cost tracking moved onto the base class.
    response = BaseLLMClient()._parse_response(
        '{"analysis":"a","changes":[{"path":"old.py","action":"rename",'
        '"new_path":"new.py","explanation":"e"}],"confidence":0.9,"done":true}',
        0,
    )
    assert response.changes[0].new_path == "new.py"


def test_absent_new_path_parses_as_none():
    # BaseLLMClient is where _parse_response lives; LLMClient is a facade that
    # delegates to a provider. The second argument is the input token count,
    # added when cost tracking moved onto the base class.
    response = BaseLLMClient()._parse_response(
        '{"analysis":"a","changes":[{"path":"a.py","action":"modify",'
        '"content":"x","explanation":"e"}],"confidence":0.9,"done":true}',
        0,
    )
    assert response.changes[0].new_path is None


def test_prompt_documents_the_action():
    """Unreachable otherwise — the model only emits what the schema describes."""
    from modules.llm_client import SYSTEM_PROMPT

    assert "rename" in SYSTEM_PROMPT
    assert "new_path" in SYSTEM_PROMPT
