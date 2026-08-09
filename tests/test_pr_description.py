"""
Tests for --describe-pr (modules/llm_client.py, modules/agent_loop.py).

`--pr` opened a PR with a templated body: the task, the run id and a diffstat.
That says what was asked for, not what was done. With --describe-pr the model
writes the title and body from the actual diff.

Opt-in, because it is one more paid call on top of the run that produced the
diff — and it falls back to the template on any failure, since a PR that opens
with a plain description beats a run that succeeded and then died writing prose
about itself.
"""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from modules.agent_loop import AgentConfig  # noqa: E402
from modules.llm_client import (  # noqa: E402
    PR_DIFF_CHARS,
    BaseLLMClient,
    BudgetExceededError,
    _parse_pr_description,
)

WRITTEN = json.dumps({
    "title": "Guard against an empty list in calculate_average",
    "body": "## What\n\nAdds a guard and a regression test.",
})


class Stub(BaseLLMClient):
    def __init__(self, reply=WRITTEN, error=None, **kwargs):
        super().__init__(**kwargs)
        self.reply = reply
        self.error = error
        self.prompts = []

    def _call(self, prompt):
        self.prompts.append(prompt)
        if self.error:
            raise self.error
        return self.reply


# ── parsing ───────────────────────────────────────────────────────────────


def test_a_well_formed_response_parses():
    title, body = _parse_pr_description(WRITTEN)

    assert title.startswith("Guard against")
    assert "## What" in body


def test_prose_around_the_json_is_tolerated():
    """Models preface things. The run should not fail over a "Sure!"."""
    title, _ = _parse_pr_description(f"Sure, here you go:\n{WRITTEN}\nHope that helps.")

    assert title.startswith("Guard against")


@pytest.mark.parametrize(
    "raw,label",
    [
        ("no json here", "no json"),
        ("{not valid json", "malformed"),
        (json.dumps({"title": "", "body": "x"}), "empty title"),
        (json.dumps({"body": "x"}), "missing title"),
        (json.dumps({"title": "x"}), "missing body"),
        ("", "empty response"),
    ],
)
def test_anything_unusable_yields_nothing(raw, label):
    """Every failure funnels to the same place, so the caller has one case."""
    assert _parse_pr_description(raw) == (None, None)


def test_an_over_long_title_is_truncated():
    """GitHub accepts long titles; a 200-character one is unreadable in a list."""
    long = json.dumps({"title": "x" * 200, "body": "b"})

    assert len(_parse_pr_description(long)[0]) == 72


# ── the request ───────────────────────────────────────────────────────────


def test_it_returns_a_title_and_body():
    title, body = Stub().pr_description_request("fix the average", "a diff")

    assert title and body


def test_the_diff_reaches_the_prompt():
    """
    The point of the feature: describing what changed rather than restating the
    task, which the template already did.
    """
    stub = Stub()
    stub.pr_description_request("fix the average", "DISTINCTIVE DIFF MARKER")

    assert "DISTINCTIVE DIFF MARKER" in stub.prompts[0]


def test_the_diff_is_bounded():
    """One extra paid call; sending a megabyte of diff would not be free."""
    stub = Stub()
    stub.pr_description_request("task", "x" * (PR_DIFF_CHARS * 3))

    assert len(stub.prompts[0]) < PR_DIFF_CHARS * 2


def test_the_call_is_costed():
    """
    It goes through _accounted_call, so --pr --max-cost cannot overshoot
    because of the description.
    """
    stub = Stub()
    stub.pr_description_request("task", "diff")

    assert stub.total_cost > 0


def test_a_budget_stop_is_not_swallowed():
    """
    Every other failure degrades to the template. This one must not: a run at
    its limit should stop, not quietly spend one more call's worth of attempt.
    """
    stub = Stub(max_cost=0.0000001)
    stub.total_cost = 999.0

    with pytest.raises(BudgetExceededError):
        stub.pr_description_request("task", "diff")


def test_a_provider_error_degrades():
    result = Stub(error=RuntimeError("503 Service Unavailable")).pr_description_request(
        "task", "diff"
    )

    assert result == (None, None)


def test_an_unusable_response_degrades():
    assert Stub(reply="rambling, no json").pr_description_request("t", "d") == (None, None)


# ── wiring ────────────────────────────────────────────────────────────────


def test_it_is_off_by_default():
    assert AgentConfig(repo_root=".", task="t").describe_pr is False


def test_the_template_remains_the_fallback():
    source = (
        Path(__file__).resolve().parents[1] / "modules" / "agent_loop.py"
    ).read_text(encoding="utf-8")

    assert 'title = f"[Agent] {cfg.task[:72]}"' in source
    assert "Autonomous Agent PR" in source


def test_the_fallback_is_announced():
    """
    Silently opening a templated PR after being asked for a written one looks
    like the flag was ignored.
    """
    source = (
        Path(__file__).resolve().parents[1] / "modules" / "agent_loop.py"
    ).read_text(encoding="utf-8")

    assert "using the standard template" in source


def test_a_written_body_records_the_run():
    """
    The template carried the run id; a written body would lose it, and that is
    the only link back to the log.
    """
    source = (
        Path(__file__).resolve().parents[1] / "modules" / "agent_loop.py"
    ).read_text(encoding="utf-8")

    assert "Written by RepoPilot, run" in source


def test_main_registers_and_passes_the_flag():
    source = (Path(__file__).resolve().parents[1] / "main.py").read_text(encoding="utf-8")

    assert '"--describe-pr"' in source
    assert "describe_pr=args.describe_pr" in source
