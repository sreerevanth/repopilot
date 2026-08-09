"""
Tests for human-readable log formatting (modules/logger.py).

Every line went through one formatter, so an iteration separator came out as

    2026-04-19 18:31:48 [INFO] ------------------------------------------

indented by thirty characters of metadata and no longer separating anything.
Scanning a run meant reading every line to find where one iteration ended.

Structural lines — rules and headers — are now emitted bare. Everything else
keeps its timestamp and level.
"""

import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from modules.logger import AgentLogger  # noqa: E402

TIMESTAMPED = re.compile(r"^\d{2}:\d{2}:\d{2} \[(DEBUG|INFO|WARNING|ERROR)\]")


@pytest.fixture
def logger(tmp_path):
    return AgentLogger(str(tmp_path), "run_test", verbose=True)


def written(tmp_path):
    return (tmp_path / "run_test_human.log").read_text(encoding="utf-8").splitlines()


# ── structural lines are bare ─────────────────────────────────────────────


def test_a_rule_carries_no_prefix(logger, tmp_path):
    logger.start_iteration(1)

    rules = [line for line in written(tmp_path) if set(line.strip()) == {"-"}]

    assert rules
    for rule in rules:
        assert not TIMESTAMPED.match(rule)


def test_the_iteration_header_carries_no_prefix(logger, tmp_path):
    logger.start_iteration(3)

    header = next(line for line in written(tmp_path) if line.startswith("Iteration 3"))

    assert not TIMESTAMPED.match(header)


def test_a_header_is_bounded_above_and_below(logger, tmp_path):
    """A rule on one side only reads as decoration rather than a boundary."""
    logger.start_iteration(1)
    lines = written(tmp_path)
    index = next(i for i, line in enumerate(lines) if line.startswith("Iteration 1"))

    assert set(lines[index - 1].strip()) == {"-"}
    assert set(lines[index + 1].strip()) == {"-"}


def test_a_blank_line_precedes_the_header(logger, tmp_path):
    logger.start_iteration(1)

    assert written(tmp_path)[0] == ""


# ── ordinary lines keep their prefix ──────────────────────────────────────


def test_messages_are_still_timestamped(logger, tmp_path):
    logger.info("  Applying 2 change(s)")

    assert TIMESTAMPED.match(written(tmp_path)[0])


def test_the_level_is_still_recorded(logger, tmp_path):
    logger.warning("  something looks off")

    assert "[WARNING]" in written(tmp_path)[0]


def test_debug_lines_still_reach_the_file(logger, tmp_path):
    """The file handler is DEBUG regardless of --quiet; it is the record."""
    logger.log_context(["a.py"], 100)

    assert any("[DEBUG]" in line for line in written(tmp_path))


# ── the date ──────────────────────────────────────────────────────────────


def test_the_date_appears_in_the_header(logger, tmp_path):
    logger.start_iteration(1)

    header = next(line for line in written(tmp_path) if line.startswith("Iteration 1"))

    assert re.search(r"\d{4}-\d{2}-\d{2}", header)


def test_the_date_is_not_repeated_on_every_line(logger, tmp_path):
    """
    A run happens on one day. Repeating the date on 200 lines pushed the
    message itself off to the right for no information.
    """
    logger.info("  a message")

    assert not re.search(r"\d{4}-\d{2}-\d{2}", written(tmp_path)[0])


# ── boundaries are findable ───────────────────────────────────────────────


def test_each_iteration_is_separated(logger, tmp_path):
    for iteration in (1, 2, 3):
        logger.start_iteration(iteration)
        logger.info(f"  work for {iteration}")

    lines = written(tmp_path)
    headers = [line for line in lines if line.startswith("Iteration ")]

    assert len(headers) == 3


def test_a_custom_section_can_be_written(logger, tmp_path):
    logger.section("Summary")

    assert "Summary" in written(tmp_path)


def test_a_section_rule_character_can_be_changed(logger, tmp_path):
    logger.section("Done", rule="=")

    assert any(set(line.strip()) == {"="} for line in written(tmp_path))
