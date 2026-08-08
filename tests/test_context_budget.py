"""
Tests for the model-derived context budget (modules/context_builder.py).

`CONTEXT_CHAR_BUDGET` was a flat 60,000 characters regardless of model. That
underfills a 200k-token window and overflows an 8k one — llama3's budget works
out at 8,400 characters, well under the old default, so a small-window model was
being sent more than it could hold.

The conversion rests on an estimate, and these tests pin that it is a
conservative one.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from modules.agent_loop import AgentConfig  # noqa: E402
from modules.context_builder import (  # noqa: E402
    CHARS_PER_TOKEN,
    CONTEXT_CHAR_BUDGET,
    CONTEXT_WINDOW_FRACTION,
    MODEL_CONTEXT_TOKENS,
    budget_for_model,
    build_context,
)
from modules.repo_ingestion import ingest_repository  # noqa: E402


@pytest.fixture
def repo(tmp_path):
    for i in range(12):
        (tmp_path / f"mod_{i:02d}.py").write_text(
            f"# module {i}\n" + "def parse(x):\n    return x\n" * 200
        )
    return ingest_repository(str(tmp_path))


# ── the table ─────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "model", ["claude-sonnet-4-20250514", "gpt-4o", "gemini-1.5-pro", "llama3"]
)
def test_known_models_have_a_window(model):
    assert MODEL_CONTEXT_TOKENS[model] > 0


def test_a_larger_window_gives_a_larger_budget():
    assert budget_for_model("claude-sonnet-4-20250514") > budget_for_model("gpt-4o")
    assert budget_for_model("gpt-4o") > budget_for_model("llama3")


def test_an_unknown_model_falls_back_to_the_default():
    """
    A new or self-hosted model keeps today's behaviour rather than having its
    window guessed at, which is the failure that would actually cost money.
    """
    assert budget_for_model("some-model-released-tomorrow") == CONTEXT_CHAR_BUDGET


@pytest.mark.parametrize("value", [None, ""])
def test_no_model_falls_back_to_the_default(value):
    assert budget_for_model(value) == CONTEXT_CHAR_BUDGET


def test_the_fallback_is_overridable():
    assert budget_for_model("unknown", default=123) == 123


# ── the conversion is conservative ────────────────────────────────────────


def test_chars_per_token_is_below_the_prose_estimate():
    """
    Source code tokenises worse than prose -- punctuation, indentation and short
    identifiers all cost tokens. Using the prose figure of ~4 would overestimate
    how much fits.
    """
    assert CHARS_PER_TOKEN < 4.0


def test_only_part_of_the_window_is_used_for_context():
    """
    The rest is for the system prompt, the task, prior error output on a retry,
    and the reply -- which has to carry complete file contents.
    """
    assert 0 < CONTEXT_WINDOW_FRACTION < 0.5


def test_a_budget_never_exceeds_its_window():
    for model, window in MODEL_CONTEXT_TOKENS.items():
        assert budget_for_model(model) < window * 4.0, model


def test_a_small_window_gets_less_than_the_old_default():
    """The case the flat constant got wrong: llama3 holds 8k tokens, not 60k chars."""
    assert budget_for_model("llama3") < CONTEXT_CHAR_BUDGET


# ── it changes what is sent ───────────────────────────────────────────────


def test_a_small_budget_selects_fewer_files(repo):
    small = build_context(repo, "fix parse", char_budget=budget_for_model("llama3"))
    large = build_context(repo, "fix parse", char_budget=budget_for_model("gpt-4o"))

    assert len(small.files) < len(large.files)


def test_the_budget_is_respected(repo):
    budget = budget_for_model("llama3")

    assert build_context(repo, "fix parse", char_budget=budget).total_chars <= budget


def test_the_default_behaviour_is_unchanged(repo):
    """An unknown model must produce exactly what the flat constant produced."""
    derived = build_context(repo, "fix parse", char_budget=budget_for_model(None))
    flat = build_context(repo, "fix parse", char_budget=CONTEXT_CHAR_BUDGET)

    assert [f.path for f in derived.files] == [f.path for f in flat.files]


# ── wiring ────────────────────────────────────────────────────────────────


def test_the_override_is_off_by_default():
    assert AgentConfig(repo_root=".", task="t").context_budget is None


def test_an_explicit_override_wins_over_the_model():
    source = (
        Path(__file__).resolve().parents[1] / "modules" / "agent_loop.py"
    ).read_text(encoding="utf-8")
    block = source[source.index("char_budget=("):][:220]

    assert block.index("cfg.context_budget") < block.index("budget_for_model")


def test_the_loop_derives_the_budget_from_the_model():
    source = (
        Path(__file__).resolve().parents[1] / "modules" / "agent_loop.py"
    ).read_text(encoding="utf-8")

    assert "budget_for_model(cfg.model)" in source
