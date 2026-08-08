"""
Tests for `.agentcontext` (modules/project_rules.py).

A repository can carry rules that apply to every task in it — a language
version to stay within, a directory never to touch, a test layout that differs
from the default.

Distinct from `.repopilot.json`, which sets CLI defaults: that file answers
"how should the tool be invoked", this one answers "what should the model know
before it writes anything".
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from modules.agent_loop import AgentConfig  # noqa: E402
from modules.project_rules import (  # noqa: E402
    MAX_RULES_CHARS,
    RULES_FILENAME,
    load_project_rules,
    render_project_rules,
    rules_path,
)

RULES = (
    "- This project targets Python 3.9. Do not use match statements.\n"
    "- Tests live in spec/, not tests/.\n"
)


@pytest.fixture
def repo(tmp_path):
    (tmp_path / RULES_FILENAME).write_text(RULES, encoding="utf-8")
    return tmp_path


# ── loading ───────────────────────────────────────────────────────────────


def test_rules_are_read_from_the_repo_root(repo):
    assert load_project_rules(str(repo)) == RULES.strip()


def test_no_file_means_no_rules(tmp_path):
    """Every repository without one must behave exactly as it does today."""
    assert load_project_rules(str(tmp_path)) is None


def test_an_empty_file_means_no_rules(tmp_path):
    (tmp_path / RULES_FILENAME).write_text("\n\n   \n")

    assert load_project_rules(str(tmp_path)) is None


def test_a_directory_in_place_of_the_file_is_ignored(tmp_path):
    (tmp_path / RULES_FILENAME).mkdir()

    assert load_project_rules(str(tmp_path)) is None


def test_an_unreadable_file_does_not_raise(tmp_path, monkeypatch):
    """
    A broken rules file must not stop a run that would otherwise work — the
    agent proceeds without the guidance, which is today's behaviour anyway.
    """
    (tmp_path / RULES_FILENAME).write_text(RULES)

    def explode(*args, **kwargs):
        raise OSError("permission denied")

    monkeypatch.setattr("builtins.open", explode)

    assert load_project_rules(str(tmp_path)) is None


def test_utf8_is_read_correctly(tmp_path):
    """The default encoding is cp1252 on Windows and would mangle this."""
    (tmp_path / RULES_FILENAME).write_text("- Répondez précisément — no waffle\n",
                                           encoding="utf-8")

    assert "précisément" in load_project_rules(str(tmp_path))


def test_undecodable_bytes_do_not_raise(tmp_path):
    (tmp_path / RULES_FILENAME).write_bytes(b"- rule \xff\xfe invalid\n")

    assert load_project_rules(str(tmp_path)) is not None


def test_a_custom_filename_is_honoured(tmp_path):
    (tmp_path / "RULES.md").write_text(RULES)

    assert load_project_rules(str(tmp_path), "RULES.md") == RULES.strip()


def test_the_path_is_built_from_the_repo_root(tmp_path):
    assert rules_path(str(tmp_path)).endswith(RULES_FILENAME)


# ── the size cap ──────────────────────────────────────────────────────────


def test_a_long_file_is_truncated(tmp_path):
    """
    Rules are sent on every iteration, so their cost is paid repeatedly. The cap
    catches a file pasted here by mistake rather than being billed for it.
    """
    (tmp_path / RULES_FILENAME).write_text("- rule\n" * (MAX_RULES_CHARS // 7 + 500))

    assert len(load_project_rules(str(tmp_path))) <= MAX_RULES_CHARS


def test_a_normal_file_is_not_truncated(repo):
    assert load_project_rules(str(repo)) == RULES.strip()


def test_the_cap_is_generous_enough_for_real_rules():
    assert MAX_RULES_CHARS >= 4_000


# ── rendering ─────────────────────────────────────────────────────────────


def test_rules_are_tagged():
    """Matches the <file> blocks the context already emits."""
    rendered = render_project_rules(RULES)

    assert rendered.startswith("<project_rules>")
    assert rendered.endswith("</project_rules>")


def test_the_rules_text_survives_rendering():
    assert "Python 3.9" in render_project_rules(RULES)


def test_the_block_says_these_are_rules_not_content():
    """
    Without this the model may treat the block as material to edit — which
    matters most when a rule says never to touch something.
    """
    rendered = render_project_rules(RULES)

    assert "rules" in rendered.lower()
    assert "apply to every task" in rendered


def test_nothing_renders_without_rules():
    assert render_project_rules(None) == ""
    assert render_project_rules("") == ""


# ── wiring ────────────────────────────────────────────────────────────────


def test_the_default_filename_is_agentcontext():
    assert AgentConfig(repo_root=".", task="t").project_rules_file == ".agentcontext"


def test_rules_are_prepended_not_appended():
    """
    When the budget is tight, files get dropped or outlined. Putting the rules
    first keeps them out of what gets squeezed.
    """
    source = (
        Path(__file__).resolve().parents[1] / "modules" / "agent_loop.py"
    ).read_text(encoding="utf-8")

    assert 'render_project_rules(rules) + "\\n\\n" + context_str' in source


def test_rules_are_read_after_the_context_is_built():
    source = (
        Path(__file__).resolve().parents[1] / "modules" / "agent_loop.py"
    ).read_text(encoding="utf-8")

    assert source.index("context_str = context.render()") < \
        source.index("load_project_rules(cfg.repo_root")


def test_an_empty_setting_disables_the_feature():
    source = (
        Path(__file__).resolve().parents[1] / "modules" / "agent_loop.py"
    ).read_text(encoding="utf-8")

    assert "if cfg.project_rules_file:" in source
