"""
Tests for per-iteration timing (modules/agent_loop.py, modules/logger.py).

Records how long each phase took — ingest, context, LLM, apply, execution — so
a slow run can be attributed rather than guessed at. The breakdown goes into
the JSONL; the terminal gets one summary line.
"""

import json
import sys
import time
from dataclasses import fields
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from modules.agent_loop import _stamp_timings  # noqa: E402
from modules.logger import AgentLogger, IterationRecord  # noqa: E402

AGENT_LOOP = Path(__file__).resolve().parents[1] / "modules" / "agent_loop.py"

PHASES = [
    "duration_ingest",
    "duration_context",
    "duration_llm",
    "duration_apply",
    "duration_execution",
    "duration_total",
]


def make_record():
    return IterationRecord(
        iteration=1, timestamp="2026-08-08T00:00:00", context_files=[], context_chars=0,
        llm_analysis="", llm_confidence=0.9, llm_done=True, changes_attempted=[],
        apply_results=[], execution_command=None, execution_exit_code=None,
        execution_stdout=None, execution_stderr=None, execution_timed_out=False,
        execution_success=True, parse_error=None,
    )


# ── the record ────────────────────────────────────────────────────────────


@pytest.mark.parametrize("name", PHASES)
def test_every_phase_has_a_field(name):
    assert name in {f.name for f in fields(IterationRecord)}


@pytest.mark.parametrize("name", PHASES)
def test_timing_fields_default_to_zero(name):
    """
    record_iteration is constructed in several places. A required field would
    break every one of them.
    """
    assert getattr(make_record(), name) == 0.0


# ── stamping ──────────────────────────────────────────────────────────────


@pytest.fixture
def stamped():
    record = make_record()
    _stamp_timings(record, time.time() - 5.0, 0.4, 0.15, 1.9, 0.05, time.time() - 2.0)
    return record


def test_measured_phases_are_recorded(stamped):
    assert stamped.duration_ingest == 0.4
    assert stamped.duration_context == 0.15
    assert stamped.duration_llm == 1.9
    assert stamped.duration_apply == 0.05


def test_execution_is_measured_from_its_phase_start(stamped):
    assert 1.9 <= stamped.duration_execution <= 2.2


def test_total_covers_the_whole_iteration(stamped):
    assert 4.9 <= stamped.duration_total <= 5.2


def test_total_is_at_least_the_sum_of_its_parts(stamped):
    parts = (
        stamped.duration_ingest + stamped.duration_context + stamped.duration_llm
        + stamped.duration_apply + stamped.duration_execution
    )
    assert stamped.duration_total >= parts - 0.01


def test_values_are_rounded(stamped):
    """Raw floats make the JSONL noisy without adding precision anyone uses."""
    for name in PHASES:
        value = getattr(stamped, name)
        assert round(value, 3) == value


# ── it reaches the log ────────────────────────────────────────────────────


def test_timings_are_written_to_the_jsonl(tmp_path):
    logger = AgentLogger(str(tmp_path), "run_abc", verbose=False)
    record = make_record()
    _stamp_timings(record, time.time() - 3.0, 0.1, 0.2, 1.0, 0.3, time.time() - 1.0)
    logger.record_iteration(record)

    line = json.loads((tmp_path / "run_abc.jsonl").read_text().splitlines()[0])

    assert line["duration_llm"] == 1.0
    assert line["duration_total"] >= 2.9


def test_a_summary_line_is_logged(tmp_path, capsys):
    """
    AgentLogger attaches its own handlers rather than propagating, so the line
    is asserted on captured output rather than through caplog.
    """
    logger = AgentLogger(str(tmp_path), "run_def", verbose=False)
    record = make_record()
    _stamp_timings(record, time.time() - 3.0, 0.1, 0.2, 1.0, 0.3, time.time() - 1.0)
    logger.record_iteration(record)

    captured = capsys.readouterr()
    assert "Timing:" in captured.out + captured.err


def test_the_summary_names_each_phase(tmp_path, capsys):
    logger = AgentLogger(str(tmp_path), "run_xyz", verbose=False)
    record = make_record()
    _stamp_timings(record, time.time() - 3.0, 0.1, 0.2, 1.0, 0.3, time.time() - 1.0)
    logger.record_iteration(record)

    output = capsys.readouterr().out + capsys.readouterr().err
    for label in ("llm", "tests", "ingest", "total"):
        assert label in output


def test_no_summary_when_nothing_was_measured(tmp_path, capsys):
    """An unstamped record should not print a line of zeroes."""
    logger = AgentLogger(str(tmp_path), "run_ghi", verbose=False)
    logger.record_iteration(make_record())

    captured = capsys.readouterr()
    assert "Timing:" not in captured.out + captured.err


# ── wiring ────────────────────────────────────────────────────────────────


def test_every_phase_boundary_is_instrumented():
    source = AGENT_LOOP.read_text(encoding="utf-8")

    for name in ("duration_ingest =", "duration_context =",
                 "duration_llm =", "duration_apply ="):
        assert name in source


def test_every_record_write_is_stamped():
    """
    Seven call sites write a record -- parse errors, validation failures and
    budget stops each have their own. One missed site logs zeroes.
    """
    source = AGENT_LOOP.read_text(encoding="utf-8")

    assert source.count("self.logger.record_iteration(iter_record)") == \
        source.count("_stamp_timings(")- 1   # -1 for the def itself


def test_durations_are_initialised_before_any_early_return():
    """
    Several paths write a record before the later phases run -- a parse error
    returns before the LLM phase completes. Without an up-front initialisation
    those paths raise NameError instead of logging.
    """
    source = AGENT_LOOP.read_text(encoding="utf-8")
    init = source.index("duration_ingest = duration_context")

    first_call = source.index("_stamp_timings(\n", init)

    assert init < first_call


def test_the_iteration_clock_starts_at_step_one():
    source = AGENT_LOOP.read_text(encoding="utf-8")
    assert source.index("iter_started = time.time()") < source.index("# ── Step 2")
