"""
Tests for state file validation (modules/run_state.py).

`load_state` checked the version and filtered unknown keys, then splatted the
rest into the dataclass. Annotations are not enforced at runtime, so a file with
the right keys and wrong values loaded cleanly and failed later, elsewhere.

Three distinct outcomes, all from a corrupt `iteration`:

- a string reached `range()` and raised deep inside the loop
- a negative value did not raise at all — `range(-4, 4)` ran eight iterations
  where `--max-iter 3` was asked for, each one a paid API call
- a missing field raised `TypeError` from `__init__` rather than `ResumeError`,
  so it escaped the handler that prints a remedy and became a traceback
"""

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from modules.errors import AgentError  # noqa: E402
from modules.run_state import STATE_VERSION, ResumeError, load_state  # noqa: E402

VALID = {
    "version": STATE_VERSION,
    "run_id": "r1",
    "iteration": 2,
    "branch_name": "agent/x",
    "task": "do a thing",
    "repo_root": ".",
    "outcome": "failed",
}


@pytest.fixture
def write(tmp_path):
    def inner(payload):
        (tmp_path / "r1_state.json").write_text(json.dumps(payload))
        return str(tmp_path)
    return inner


# ── files that should load ────────────────────────────────────────────────


def test_a_valid_state_loads(write):
    state = load_state(write(VALID), "r1")

    assert state.iteration == 2
    assert state.run_id == "r1"


def test_unknown_keys_are_still_filtered(write):
    """Existing behaviour, pinned: a newer build's extra field must not break."""
    state = load_state(write({**VALID, "field_from_the_future": 1}), "r1")

    assert state.iteration == 2


def test_iteration_zero_is_allowed(write):
    """A run that checkpointed before its first iteration is legitimate."""
    assert load_state(write({**VALID, "iteration": 0}), "r1").iteration == 0


# ── iteration ─────────────────────────────────────────────────────────────


def test_a_string_iteration_is_refused(write):
    with pytest.raises(ResumeError, match="not an integer"):
        load_state(write({**VALID, "iteration": "not a number"}), "r1")


def test_a_negative_iteration_is_refused(write):
    """
    The one that failed silently. `state.iteration + 1` became -4, so the loop
    ran from -4 and exceeded --max-iter with nothing reported.
    """
    with pytest.raises(ResumeError, match="negative"):
        load_state(write({**VALID, "iteration": -5}), "r1")


def test_the_negative_message_names_the_consequence(write):
    """"Invalid" alone does not tell the user why it matters."""
    with pytest.raises(ResumeError, match="max-iter"):
        load_state(write({**VALID, "iteration": -1}), "r1")


def test_a_boolean_iteration_is_refused(write):
    """`isinstance(True, int)` is True in Python, so bool needs excluding."""
    with pytest.raises(ResumeError):
        load_state(write({**VALID, "iteration": True}), "r1")


# ── the other fields ──────────────────────────────────────────────────────


@pytest.mark.parametrize("field", ["run_id", "task", "repo_root"])
def test_a_non_string_field_is_refused(write, field):
    with pytest.raises(ResumeError, match=field):
        load_state(write({**VALID, field: ["a", "list"]}), "r1")


def test_a_non_list_last_changes_is_refused(write):
    with pytest.raises(ResumeError, match="last_changes"):
        load_state(write({**VALID, "last_changes": "not a list"}), "r1")


def test_a_missing_field_raises_resume_error_not_type_error(write):
    """
    The distinction that matters: ResumeError reaches the handler added in
    #260 and prints a remedy; TypeError escapes as a traceback.
    """
    payload = {k: v for k, v in VALID.items() if k != "iteration"}

    with pytest.raises(ResumeError, match="missing a required field"):
        load_state(write(payload), "r1")


# ── malformed last_changes ────────────────────────────────────────────────


def test_a_malformed_change_entry_raises_resume_error():
    """
    `FileChange(**c)` raised a bare TypeError for an entry with a missing or
    unexpected key. Same class of escape as above, one line further on.
    """
    from modules.agent_loop import _restore_changes

    with pytest.raises(ResumeError, match="malformed last_changes"):
        _restore_changes([{"path": "a.py"}], "r1")


def test_an_unexpected_key_in_a_change_entry_is_refused():
    from modules.agent_loop import _restore_changes

    with pytest.raises(ResumeError):
        _restore_changes([{"path": "a.py", "action": "modify",
                           "content": "x", "explanation": "e", "bogus": 1}], "r1")


def test_a_well_formed_change_entry_still_restores():
    from modules.agent_loop import _restore_changes

    restored = _restore_changes(
        [{"path": "a.py", "action": "modify", "content": "x", "explanation": "e"}],
        "r1",
    )

    assert restored[0].path == "a.py"


# ── the error class ───────────────────────────────────────────────────────


def test_resume_error_is_an_agent_error():
    """This is what routes it to the clean handler rather than a traceback."""
    assert issubclass(ResumeError, AgentError)


def test_every_corrupted_shape_raises_resume_error(write):
    """
    No corrupted file should leak a different exception type — that is the
    whole defect being fixed.
    """
    corruptions = [
        {**VALID, "iteration": "x"},
        {**VALID, "iteration": -1},
        {**VALID, "iteration": True},
        {**VALID, "task": 5},
        {**VALID, "last_changes": {}},
        {k: v for k, v in VALID.items() if k != "run_id"},
        {**VALID, "version": 999},
    ]

    for payload in corruptions:
        with pytest.raises(ResumeError):
            load_state(write(payload), "r1")
