"""
Tests for --config merging (main.py).

The previous implementation guarded each key with

    getattr(args, k) in (None, "anthropic", "claude-sonnet-4-20250514", False)

trying to express "apply this only where the user did not pass a flag".
argparse does not record where a value came from, so that guard approximated
the question by comparing against four literals — and failed both ways.

Fourteen settings could never be configured, because their defaults are not in
that tuple. And a config value could beat an explicit flag, because "anthropic"
is in it, so `--provider anthropic` was indistinguishable from the default.
"""

import argparse
import contextlib
import io
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import main as cli  # noqa: E402


@pytest.fixture
def parser():
    """The real parser, captured by intercepting parse_args."""
    real = argparse.ArgumentParser.parse_args
    captured = {}

    def spy(self, *args, **kwargs):
        captured["parser"] = self
        raise SystemExit(0)

    argparse.ArgumentParser.parse_args = spy
    try:
        with contextlib.suppress(SystemExit), contextlib.redirect_stdout(io.StringIO()):
            cli.main()
    finally:
        argparse.ArgumentParser.parse_args = real

    return captured["parser"]


@pytest.fixture
def apply(parser, tmp_path, capsys):
    """Write a config, apply it to a fresh namespace, return (args, stderr)."""
    def run(config, argv=("--repo", ".", "--task", "t")):
        path = tmp_path / "config.json"
        path.write_text(json.dumps(config) if not isinstance(config, str) else config)
        args = parser.parse_args(list(argv))
        cli.apply_config_file(str(path), args, parser)
        return args, capsys.readouterr().err

    return run


# ── settings that could never be configured ───────────────────────────────


@pytest.mark.parametrize(
    "key,value,default",
    [
        ("max_iter", 7, 5),
        ("workers", 3, 10),
        ("timeout", 600, 120),
        ("runner", "make", "pytest"),
        ("base_branch", "develop", "main"),
        ("branch_prefix", "bot", "agent"),
        ("log_dir", "mylogs", "logs"),
        ("max_parallel_tasks", 8, 4),
    ],
)
def test_a_non_falsy_default_can_now_be_configured(apply, key, value, default):
    """
    Each of these has a default outside the old literal tuple, so a config file
    setting it did nothing at all — silently.
    """
    args, _ = apply({key: value})

    assert getattr(args, key) == value, f"{key} stayed at {default}"


def test_a_falsy_default_still_works(apply):
    """The cases the old guard did handle must keep working."""
    args, _ = apply({"verbose_payloads": True})

    assert args.verbose_payloads is True


# ── precedence ────────────────────────────────────────────────────────────


def test_the_command_line_beats_the_config(apply):
    """
    An explicit flag differing from the default must win. Under the old guard
    a config value could beat one, because the guard compared against literals
    rather than against what the parser would have produced.
    """
    args, _ = apply(
        {"provider": "gemini"},
        argv=("--repo", ".", "--task", "t", "--provider", "openai"),
    )

    assert args.provider == "openai"


def test_the_config_beats_the_default(apply):
    """The other half of the precedence: defaults lose to the file."""
    args, _ = apply({"provider": "openai"})

    assert args.provider == "openai"


def test_an_explicit_flag_matching_a_default_is_still_respected(apply):
    """
    Passing --runner pytest explicitly is the same value as the default, so
    this is the one case the parser genuinely cannot distinguish. Documented
    rather than claimed as fixed: the config wins here.
    """
    args, _ = apply(
        {"runner": "make"},
        argv=("--repo", ".", "--task", "t", "--runner", "pytest"),
    )

    assert args.runner == "make"


# ── wrong types ───────────────────────────────────────────────────────────


def test_a_wrong_type_is_reported_and_ignored(apply):
    """
    Previously the string reached AgentConfig and the failure surfaced as a
    TypeError inside the iteration loop, mentioning neither the file nor the key.
    """
    args, err = apply({"max_iter": "not a number"})

    assert args.max_iter == 5
    assert "max_iter" in err and "int" in err


def test_a_coercible_value_is_coerced(apply):
    """JSON has no int/float distinction the way argparse does."""
    args, _ = apply({"max_iter": "7"})

    assert args.max_iter == 7


def test_one_bad_key_does_not_stop_the_others(apply):
    args, _ = apply({"max_iter": "nonsense", "workers": 3})

    assert args.max_iter == 5
    assert args.workers == 3


# ── unknown keys ──────────────────────────────────────────────────────────


def test_an_unknown_key_is_reported(apply):
    """
    Silence made a typo indistinguishable from a setting the build does not
    support, and both needed completely different fixes.
    """
    _, err = apply({"max_iteration": 7})

    assert "max_iteration" in err
    assert "matches no option" in err


def test_an_unknown_key_is_a_warning_not_an_error(apply):
    """A config shared across versions may carry a key an older build predates."""
    args, _ = apply({"future_setting": 1, "workers": 3})

    assert args.workers == 3


# ── malformed files ───────────────────────────────────────────────────────


def test_invalid_json_is_reported(apply):
    _, err = apply("{not json")

    assert "failed to load" in err.lower()


def test_a_non_object_config_is_refused(apply):
    """A list would iterate as keys and produce nonsense warnings."""
    args, err = apply([1, 2, 3])

    assert "JSON object" in err
    assert args.workers == 10


def test_a_malformed_file_does_not_stop_the_run(apply):
    """A bad config should not be fatal; defaults are a usable fallback."""
    args, _ = apply("{not json")

    assert args.max_iter == 5


# ── the mechanism ─────────────────────────────────────────────────────────


def test_the_literal_tuple_is_gone():
    """
    The specific construct that caused both failures. Its absence is what this
    change is; a future refactor reintroducing it would reintroduce both bugs.
    """
    source = (ROOT / "main.py").read_text(encoding="utf-8")

    # Checked against the code rather than the whole file: the replacement
    # quotes the old guard in its docstring to explain what went wrong, and
    # matching prose would make this pass or fail on wording.
    code = "\n".join(
        line for line in source.splitlines()
        if not line.strip().startswith(("#", "*"))
    )
    guard = 'if hasattr(args, k) and getattr(args, k) in ('

    assert guard not in code


def test_parse_args_returns_the_parser():
    """
    Merging needs each flag's default and declared type, and neither is
    recoverable from the Namespace alone.
    """
    source = (ROOT / "main.py").read_text(encoding="utf-8")

    assert "return parser.parse_args(), parser" in source
