"""
Tests for the run dashboard (modules/dashboard.py).

The dashboard reads the JSONL that logger.py already writes, rather than being
instrumented into the loop. That decoupling is what makes it work on a finished
run, on a run in another terminal, and without touching agent_loop.py — so the
parsing has to tolerate whatever state the file is in when it is read.
"""

import json
import sys
import threading
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from modules.dashboard import (  # noqa: E402
    follow_records,
    main,
    paint,
    read_records,
    read_summary,
    render_iteration,
    render_summary,
    show,
    summary_path_for,
)

ITERATION = {
    "type": "iteration",
    "iteration": 1,
    "timestamp": "2026-08-07T13:12:22",
    "context_files": ["utils.py", "test_utils.py"],
    "context_chars": 3182,
    "llm_analysis": "The functions lack input validation.",
    "llm_confidence": 0.97,
    "llm_done": True,
    "changes_attempted": [{"path": "utils.py", "action": "modify"}],
    "apply_results": [{"path": "utils.py", "success": True, "error": None}],
    "execution_command": "python -m pytest",
    "execution_exit_code": 0,
    "execution_stdout": "4 passed",
    "execution_stderr": "",
    "execution_timed_out": False,
    "execution_success": True,
    "parse_error": None,
}


def write_log(tmp_path, *records, name="run.jsonl"):
    path = tmp_path / name
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")
    return str(path)


# ── parsing ───────────────────────────────────────────────────────────────


def test_records_are_read_in_order(tmp_path):
    path = write_log(tmp_path, {"iteration": 1}, {"iteration": 2}, {"iteration": 3})
    assert [r["iteration"] for r in read_records(path)] == [1, 2, 3]


def test_a_truncated_final_line_is_skipped(tmp_path):
    """
    A run killed mid-write leaves a partial line. Showing the rest is more
    useful than refusing to open the file.
    """
    path = tmp_path / "run.jsonl"
    path.write_text('{"iteration": 1}\n{"iteration": 2}\n{"iterat')

    assert [r["iteration"] for r in read_records(str(path))] == [1, 2]


def test_blank_lines_are_ignored(tmp_path):
    path = tmp_path / "run.jsonl"
    path.write_text('{"iteration": 1}\n\n\n{"iteration": 2}\n')

    assert len(read_records(str(path))) == 2


def test_an_empty_log_yields_nothing(tmp_path):
    path = tmp_path / "run.jsonl"
    path.write_text("")
    assert read_records(str(path)) == []


# ── the summary ───────────────────────────────────────────────────────────


def test_summary_path_is_derived_from_the_log():
    assert summary_path_for("logs/run_abc.jsonl") == "logs/run_abc_summary.json"


def test_a_missing_summary_is_not_an_error(tmp_path):
    """It is only written when a run finishes; mid-run there is none."""
    assert read_summary(write_log(tmp_path, ITERATION)) is None


def test_a_malformed_summary_is_not_an_error(tmp_path):
    path = write_log(tmp_path, ITERATION)
    Path(summary_path_for(path)).write_text("{not json")

    assert read_summary(path) is None


def test_the_summary_is_read_when_present(tmp_path):
    path = write_log(tmp_path, ITERATION)
    Path(summary_path_for(path)).write_text(json.dumps({"outcome": "success"}))

    assert read_summary(path)["outcome"] == "success"


# ── rendering ─────────────────────────────────────────────────────────────


@pytest.fixture
def rendered():
    return render_iteration(ITERATION, colour=False)


def test_analysis_is_shown(rendered):
    assert "The functions lack input validation." in rendered


def test_context_and_changes_are_shown(rendered):
    assert "2 files" in rendered
    assert "3,182 chars" in rendered
    assert "modify" in rendered and "utils.py" in rendered


def test_confidence_is_shown(rendered):
    assert "0.97" in rendered


def test_a_passing_run_is_marked_pass(rendered):
    assert "PASS" in rendered


def test_a_failing_run_shows_its_output():
    failing = dict(
        ITERATION, execution_success=False, execution_exit_code=1,
        execution_stderr="AssertionError: expected 3, got 4",
    )
    text = render_iteration(failing, colour=False)

    assert "FAIL" in text
    assert "AssertionError: expected 3, got 4" in text


def test_a_timeout_is_distinguished_from_a_failure():
    text = render_iteration(
        dict(ITERATION, execution_success=False, execution_timed_out=True), colour=False
    )
    assert "TIMEOUT" in text


def test_failed_file_writes_are_surfaced():
    text = render_iteration(
        dict(ITERATION, apply_results=[
            {"path": "x.py", "success": False, "error": "Path traversal detected"}
        ]),
        colour=False,
    )
    assert "FAILED" in text and "Path traversal detected" in text


def test_a_parse_error_is_surfaced():
    record = dict(ITERATION, parse_error="No JSON object found")
    text = render_iteration(record, colour=False)
    assert "No JSON object found" in text


def test_a_sparse_record_still_renders():
    """Records from an aborted iteration carry very few fields."""
    assert "Iteration 2" in render_iteration({"iteration": 2}, colour=False)


def test_outcome_is_shown_in_the_summary():
    summary = {"outcome": "success", "run_id": "r", "task": "t"}
    text = render_summary(summary, colour=False)
    assert "SUCCESS" in text


# ── colour ────────────────────────────────────────────────────────────────


def test_colour_can_be_disabled():
    assert paint("hello", "red", enabled=False) == "hello"


def test_colour_wraps_when_enabled():
    assert paint("hello", "red", enabled=True) != "hello"


def test_rendering_without_colour_has_no_escape_codes(rendered):
    assert "\033[" not in rendered


# ── following a live run ──────────────────────────────────────────────────


def test_follow_yields_records_as_they_are_written(tmp_path):
    path = str(tmp_path / "live.jsonl")

    def writer():
        time.sleep(0.2)  # the file does not exist yet when following starts
        with open(path, "w", encoding="utf-8") as handle:
            for i in (1, 2, 3):
                handle.write(json.dumps({"iteration": i}) + "\n")
                handle.flush()
                time.sleep(0.1)

    threading.Thread(target=writer, daemon=True).start()

    seen = []
    for record in follow_records(path, poll=0.05):
        seen.append(record["iteration"])
        if len(seen) == 3:
            break

    assert seen == [1, 2, 3]


def test_follow_buffers_a_partial_line(tmp_path):
    """
    A record being written when we read is complete a moment later. Discarding
    it would silently drop an iteration from a live view.
    """
    path = str(tmp_path / "live.jsonl")

    def writer():
        with open(path, "w", encoding="utf-8") as handle:
            handle.write('{"iteration": 1, "llm_conf')
            handle.flush()
            time.sleep(0.3)
            handle.write('idence": 0.9}\n')
            handle.flush()

    threading.Thread(target=writer, daemon=True).start()

    for record in follow_records(path, poll=0.05):
        assert record["iteration"] == 1
        assert record["llm_confidence"] == 0.9
        break


# ── the command line ──────────────────────────────────────────────────────


def test_a_missing_log_reports_rather_than_crashing(tmp_path, capsys):
    assert show(str(tmp_path / "nope.jsonl")) == 1
    assert "No such run log" in capsys.readouterr().err


def test_showing_a_finished_run_succeeds(tmp_path, capsys):
    path = write_log(tmp_path, ITERATION)
    Path(summary_path_for(path)).write_text(
        json.dumps({"outcome": "success", "run_id": "r"})
    )

    assert show(path, colour=False) == 0
    out = capsys.readouterr().out
    assert "Iteration 1" in out and "SUCCESS" in out


def test_main_accepts_no_color(tmp_path, capsys):
    path = write_log(tmp_path, ITERATION)
    assert main([path, "--no-color"]) == 0
    assert "\033[" not in capsys.readouterr().out
