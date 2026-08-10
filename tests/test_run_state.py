"""
Tests for resumable runs (modules/run_state.py, modules/agent_loop.py).

A run that dies at iteration 4 throws away three iterations of paid API calls.
`--resume <run_id>` picks up from the last completed iteration.

The properties worth pinning are the refusals: resuming into the wrong
repository or the wrong task would silently apply one run's half-finished work
to another run's problem, which is worse than starting over.
"""

import json
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from modules.run_state import (  # noqa: E402
    STATE_VERSION,
    ResumeError,
    RunState,
    check_resumable,
    clear_state,
    list_resumable,
    load_state,
    save_state,
    state_path,
)

AGENT_LOOP = Path(__file__).resolve().parents[1] / "modules" / "agent_loop.py"


def source() -> str:
    # Explicit encoding: the default is locale-dependent and mangles this
    # module's box-drawing section comments on a default Windows install.
    return AGENT_LOOP.read_text(encoding="utf-8")


def make_state(tmp_path, **overrides):
    base = dict(
        run_id="agent_20260807_abc",
        task="Fix the parser",
        repo_root=str(tmp_path),
        iteration=3,
        branch_name="agent/fix-abc",
        last_changes=[{"path": "p.py", "action": "modify",
                       "content": "x", "explanation": "e"}],
        last_exit_code=1,
        last_stdout="1 failed",
        last_stderr="AssertionError",
    )
    base.update(overrides)
    return RunState(**base)


# ── round trip ────────────────────────────────────────────────────────────


def test_state_survives_a_round_trip(tmp_path):
    original = make_state(tmp_path)
    save_state(original, str(tmp_path))

    restored = load_state(str(tmp_path), original.run_id)

    assert restored.iteration == 3
    assert restored.branch_name == "agent/fix-abc"
    assert restored.last_changes == original.last_changes
    assert restored.last_stderr == "AssertionError"


def test_the_write_is_atomic(tmp_path):
    """
    Saved after every iteration, so a crash mid-write is the case this exists
    to survive. A temp file plus rename means the reader sees whole or nothing.
    """
    save_state(make_state(tmp_path), str(tmp_path))

    leftovers = [n for n in os.listdir(tmp_path) if n.endswith(".tmp")]
    assert leftovers == []
    json.loads(Path(state_path(str(tmp_path), "agent_20260807_abc")).read_text())


def test_saving_never_raises(tmp_path):
    """Failing to checkpoint must not fail a run that is otherwise going fine."""
    blocker = tmp_path / "not-a-directory"
    blocker.write_text("a file where a log dir should be")

    # makedirs cannot create a directory inside a regular file, on any platform
    # and regardless of privileges.
    assert save_state(make_state(tmp_path), str(blocker / "logs")) is None


def test_clearing_a_missing_state_is_not_an_error(tmp_path):
    clear_state(str(tmp_path), "never-existed")


def test_clearing_removes_the_file(tmp_path):
    state = make_state(tmp_path)
    save_state(state, str(tmp_path))
    clear_state(str(tmp_path), state.run_id)

    assert not os.path.exists(state_path(str(tmp_path), state.run_id))


# ── refusing to load ──────────────────────────────────────────────────────


def test_a_missing_state_explains_itself(tmp_path):
    with pytest.raises(ResumeError) as excinfo:
        load_state(str(tmp_path), "nope")

    assert "No resumable state" in str(excinfo.value)


def test_a_corrupt_state_is_refused(tmp_path):
    Path(state_path(str(tmp_path), "broken")).write_text("{not json")

    with pytest.raises(ResumeError):
        load_state(str(tmp_path), "broken")


def test_an_older_schema_is_refused(tmp_path):
    """
    Resuming from a misread state would apply changes the model never proposed,
    so an unrecognised version is refused rather than half-understood.
    """
    path = Path(state_path(str(tmp_path), "old"))
    path.write_text(json.dumps({"version": STATE_VERSION - 1, "run_id": "old"}))

    with pytest.raises(ResumeError) as excinfo:
        load_state(str(tmp_path), "old")

    assert "version" in str(excinfo.value)


def test_unknown_fields_are_ignored(tmp_path):
    """A newer build's extra keys should not crash an older reader."""
    state = make_state(tmp_path)
    save_state(state, str(tmp_path))
    path = Path(state_path(str(tmp_path), state.run_id))
    data = json.loads(path.read_text())
    data["something_from_the_future"] = True
    path.write_text(json.dumps(data))

    assert load_state(str(tmp_path), state.run_id).iteration == 3


# ── refusing to resume ────────────────────────────────────────────────────


def test_a_different_repository_is_refused(tmp_path):
    state = make_state(tmp_path)

    with pytest.raises(ResumeError) as excinfo:
        check_resumable(state, "/some/other/repo")

    assert "not" in str(excinfo.value)


def test_a_different_task_is_refused(tmp_path):
    state = make_state(tmp_path)

    with pytest.raises(ResumeError) as excinfo:
        check_resumable(state, str(tmp_path), task="Something else entirely")

    assert "different task" in str(excinfo.value)


def test_the_same_task_is_allowed(tmp_path):
    check_resumable(make_state(tmp_path), str(tmp_path), task="Fix the parser")


def test_no_task_means_continue_the_original(tmp_path):
    """--resume without --task is the normal way to continue."""
    check_resumable(make_state(tmp_path), str(tmp_path), task=None)


def test_whitespace_differences_do_not_block_a_resume(tmp_path):
    check_resumable(make_state(tmp_path), str(tmp_path), task="  Fix the parser  ")


# ── listing ───────────────────────────────────────────────────────────────


def test_resumable_runs_are_listed(tmp_path):
    for run_id in ("run_a", "run_b"):
        save_state(make_state(tmp_path, run_id=run_id), str(tmp_path))

    assert set(list_resumable(str(tmp_path))) == {"run_a", "run_b"}


def test_only_state_files_are_listed(tmp_path):
    save_state(make_state(tmp_path, run_id="run_a"), str(tmp_path))
    (tmp_path / "run_a.jsonl").write_text("{}")
    (tmp_path / "run_a_summary.json").write_text("{}")

    assert list_resumable(str(tmp_path)) == ["run_a"]


def test_a_missing_log_dir_lists_nothing(tmp_path):
    assert list_resumable(str(tmp_path / "nope")) == []


# ── wiring ────────────────────────────────────────────────────────────────


def test_resume_is_off_by_default():
    from modules.agent_loop import AgentConfig

    assert AgentConfig(repo_root=".", task="t").resume_from is None


def test_the_loop_starts_from_the_saved_iteration():
    text = source()

    assert "start_iteration = state.iteration + 1" in text
    assert "for iteration in range(start_iteration," in text


def test_state_is_saved_every_iteration():
    """A checkpoint written only at the end would never survive a crash."""
    text = source()
    # Bound to the helper rather than a character window: every path that ends
    # an iteration calls _checkpoint, including the lint-failure and parse-error
    # paths that CodeRabbit flagged as previously unsaved.
    assert "def _checkpoint(" in text

    finalisers = text.count("self.logger.record_iteration(iter_record)")
    checkpoints = text.count("_checkpoint(self, cfg, iteration")

    assert checkpoints == finalisers, (
        f"{finalisers} iteration-ending paths but {checkpoints} checkpoints"
    )


def test_a_finished_run_clears_its_state():
    """Otherwise --list-resumable would offer to continue a completed run."""
    text = source()
    block = text[text.index("clear_state(cfg.log_dir"):]

    assert "clear_state" in block
    assert 'outcome not in ("failed", "max_retries", "error")' in text


def test_previous_output_is_restored_for_the_retry_prompt():
    """Without it the resumed iteration would ask as though nothing had failed."""
    text = source()
    block = text[text.index("if cfg.resume_from:"):][:1400]

    assert "last_exec = ExecutionResult(" in block
    assert "state.last_stderr" in block
    # Matched on the call rather than on `FileChange(**c)`: the reconstruction
    # moved into _restore_changes so a malformed entry raises ResumeError
    # instead of a bare TypeError. What matters here is that the restore still
    # happens on the resume path, not how it is spelled.
    assert "_restore_changes(state.last_changes" in block
