"""
Tests for the planning pass (modules/llm_client.py, modules/agent_loop.py).

`--plan-first` asks the model how it would approach the task before it writes
any code, then appends that plan to the execution prompt.

A planning pass costs an API call, so the properties worth pinning are that it
is off by default, that a bad plan degrades rather than derails, and that the
execution prompt is byte-identical when planning is off.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from modules.agent_loop import AgentConfig  # noqa: E402
from modules.llm_client import (  # noqa: E402
    BaseLLMClient,
    PLAN_PROMPT_TEMPLATE,
    TASK_PROMPT_TEMPLATE,
    LLMClient,
    Plan,
)

AGENT_LOOP = Path(__file__).resolve().parents[1] / "modules" / "agent_loop.py"

GOOD_PLAN = (
    '{"plan":["read the parser","add a guard"],'
    '"files_to_change":["parser.py"],'
    '"risks":["tests may not cover it"],"confidence":0.8}'
)


def client():
    return BaseLLMClient()


def parse(raw):
    return client()._parse_plan(raw)


# ── parsing ───────────────────────────────────────────────────────────────


def test_a_well_formed_plan_parses():
    plan = parse(GOOD_PLAN)

    assert plan.steps == ["read the parser", "add a guard"]
    assert plan.files_to_change == ["parser.py"]
    assert plan.risks == ["tests may not cover it"]
    assert plan.confidence == 0.8
    assert plan.usable is True


def test_surrounding_prose_is_tolerated():
    """Same leniency the change parser already applies."""
    assert parse(f"Here is my plan:\n{GOOD_PLAN}\nHope that helps.").usable is True


@pytest.mark.parametrize("raw", ["not json at all", "", "{unclosed", "[]"])
def test_malformed_plans_are_not_usable(raw):
    plan = parse(raw)

    assert plan.usable is False
    assert plan.confidence == 0.0 or plan.steps == []


def test_a_plan_with_no_steps_is_not_usable():
    """Nothing to add to the next prompt."""
    assert parse('{"plan":[],"confidence":0.9}').usable is False


def test_wrong_types_do_not_raise():
    plan = parse('{"plan":"a string not a list","risks":42,"confidence":0.5}')

    assert plan.steps == []
    assert plan.risks == []


def test_missing_fields_default_cleanly():
    plan = parse('{"plan":["one step"]}')

    assert plan.steps == ["one step"]
    assert plan.files_to_change == []
    assert plan.confidence == 0.5


# ── rendering ─────────────────────────────────────────────────────────────


def test_steps_are_numbered():
    text = parse(GOOD_PLAN).render()

    assert "1. read the parser" in text
    assert "2. add a guard" in text


def test_files_and_risks_are_included():
    text = parse(GOOD_PLAN).render()

    assert "parser.py" in text
    assert "tests may not cover it" in text


def test_empty_sections_are_omitted():
    text = parse('{"plan":["one step"]}').render()

    assert "Files you expect" not in text
    assert "Risks" not in text


# ── the planning prompt ───────────────────────────────────────────────────


def test_the_plan_prompt_forbids_code():
    """
    Asking for a plan and changes together produces changes with a plan-shaped
    preamble, which is the opposite of planning.
    """
    assert "Do NOT write any code" in PLAN_PROMPT_TEMPLATE


def test_the_plan_prompt_asks_for_risks():
    assert "risks" in PLAN_PROMPT_TEMPLATE


def test_the_plan_prompt_has_its_placeholders():
    PLAN_PROMPT_TEMPLATE.format(task="t", context="c")


def test_the_plan_prompt_file_matches_its_builtin():
    """
    #59 moved prompts into prompts/*.txt with a built-in fallback. A drift here
    would change what the model sees while looking like a pure addition.
    """
    from pathlib import Path

    from modules.llm_client import _BUILTIN_PLAN_PROMPT

    path = Path(__file__).resolve().parents[1] / "prompts" / "plan.txt"
    assert path.read_text(encoding="utf-8") == _BUILTIN_PLAN_PROMPT


def test_a_missing_plan_file_falls_back(monkeypatch, tmp_path):
    """A bad checkout degrades to the built-in prompt, not to no prompt."""
    from modules.llm_client import (
        PROMPT_DIR_ENV_VAR,
        _BUILTIN_PLAN_PROMPT,
        load_prompt,
    )

    monkeypatch.setenv(PROMPT_DIR_ENV_VAR, str(tmp_path))
    assert load_prompt("plan", _BUILTIN_PLAN_PROMPT) == _BUILTIN_PLAN_PROMPT


# ── the execution prompt ──────────────────────────────────────────────────


# LLMClient is a facade that delegates to a provider in
# self.underlying_client. Test doubles subclass BaseLLMClient, which is
# where _call, _parse_response and the request methods actually live.
class Recorder(BaseLLMClient):
    def __init__(self):
        # super() sets model, verbose, system_prompt and the token counters that
        # the request methods read. Skipping it leaves the double half-built.
        super().__init__()
        self.prompts = []

    def _call(self, prompt):
        self.prompts.append(prompt)
        return '{"analysis":"a","changes":[],"confidence":0.9,"done":true}'


def test_no_plan_leaves_the_prompt_unchanged():
    """Planning is opt-in; the existing path must be untouched."""
    recorder = Recorder()
    recorder.initial_request("task", "context")

    expected = TASK_PROMPT_TEMPLATE.format(task="task", context="context")
    assert recorder.prompts[0] == expected


def test_a_usable_plan_is_appended():
    recorder = Recorder()
    recorder.initial_request("task", "context", plan=parse(GOOD_PLAN))

    assert "## Your Plan" in recorder.prompts[0]
    assert "1. read the parser" in recorder.prompts[0]


def test_an_unusable_plan_is_not_appended():
    """A failed planning call should not corrupt the execution prompt."""
    recorder = Recorder()
    recorder.initial_request("task", "context", plan=parse("garbage"))

    expected = TASK_PROMPT_TEMPLATE.format(task="task", context="context")
    assert recorder.prompts[0] == expected


def test_plan_request_uses_the_plan_prompt():
    recorder = Recorder()
    recorder.plan_request("task", "context")

    assert "Do NOT write any code" in recorder.prompts[0]


# ── wiring ────────────────────────────────────────────────────────────────


def test_planning_is_off_by_default():
    assert AgentConfig(repo_root=".", task="t").plan_first is False


def test_planning_only_runs_on_the_first_iteration():
    """
    Later iterations already carry real test output, which is a stronger signal
    than a re-plan and does not cost an extra call.
    """
    source = AGENT_LOOP.read_text(encoding="utf-8")
    start = source.index("if cfg.plan_first")

    assert "iteration == 1" in source[start:start + 80]


def test_an_unusable_plan_is_reported_not_fatal():
    source = AGENT_LOOP.read_text(encoding="utf-8")
    start = source.index("if cfg.plan_first")
    block = source[start:start + 1600]

    assert "continuing without it" in block


def test_the_plan_is_logged():
    source = AGENT_LOOP.read_text(encoding="utf-8")
    start = source.index("if cfg.plan_first")
    block = source[start:start + 1600]

    assert "plan.steps" in block
    assert "risk" in block


def test_plan_is_a_typed_result_not_a_dict():
    """The loop reads .usable and .steps; a dict would fail silently."""
    assert isinstance(parse(GOOD_PLAN), Plan)
