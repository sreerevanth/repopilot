"""
Tests for numeric flag bounds (main.py).

Seven numeric flags accepted anything argparse could convert. Two failed in
ways that pointed away from the cause:

`--max-cost -1` stopped the run before its first request with "Spent $0.0000,
which reaches the --max-cost limit of $-1.00" — a message about budgets, for
what is an invalid argument.

`--workers 0` escaped as an unhandled `ValueError: max_workers must be greater
than 0` from inside concurrent.futures, naming a parameter the user never typed.

Validating in `type=` rather than after parsing is what puts the flag name in
the message, which is the information both failures were missing.
"""

import argparse
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from main import non_negative_float, positive_float, positive_int  # noqa: E402


def cli(*flags):
    """Run the CLI and return combined output."""
    result = subprocess.run(
        [sys.executable, "main.py", "--repo", ".", "--task", "t", "--context-only", *flags],
        capture_output=True, text=True, cwd=ROOT,
    )
    return result.stdout + result.stderr


# ── the validators ────────────────────────────────────────────────────────


@pytest.mark.parametrize("text,expected", [("1", 1), ("10", 10), ("9999", 9999)])
def test_positive_int_accepts_one_and_above(text, expected):
    assert positive_int(text) == expected


@pytest.mark.parametrize("text", ["0", "-1", "-100"])
def test_positive_int_refuses_zero_and_below(text):
    with pytest.raises(argparse.ArgumentTypeError):
        positive_int(text)


@pytest.mark.parametrize("text,expected", [("0.01", 0.01), ("1.5", 1.5), ("100", 100.0)])
def test_positive_float_accepts_above_zero(text, expected):
    assert positive_float(text) == pytest.approx(expected)


@pytest.mark.parametrize("text", ["0", "0.0", "-0.01", "-5"])
def test_positive_float_refuses_zero_and_below(text):
    """A limit of nothing is not a limit; it is a run that cannot start."""
    with pytest.raises(argparse.ArgumentTypeError):
        positive_float(text)


def test_non_negative_float_accepts_zero():
    """`--clean-older-than 0` is a reasonable way to say "everything"."""
    assert non_negative_float("0") == 0.0


def test_non_negative_float_refuses_negatives():
    with pytest.raises(argparse.ArgumentTypeError):
        non_negative_float("-1")


def test_the_message_states_the_bound():
    """Otherwise the user knows the value is wrong but not what would be right."""
    with pytest.raises(argparse.ArgumentTypeError, match="1 or greater"):
        positive_int("0")


def test_the_message_includes_the_offending_value():
    with pytest.raises(argparse.ArgumentTypeError, match="-5"):
        positive_int("-5")


def test_a_non_numeric_value_still_raises():
    """Delegated to int()/float(); argparse catches ValueError the same way."""
    with pytest.raises(ValueError):
        positive_int("abc")


# ── through the CLI ───────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "flag,value",
    [
        ("--max-cost", "-1"),
        ("--max-cost", "0"),
        ("--workers", "0"),
        ("--timeout", "-10"),
        ("--context-budget", "-100"),
        ("--max-iter", "0"),
        ("--clean-older-than", "-1"),
    ],
)
def test_the_cli_refuses_an_invalid_value(flag, value):
    output = cli(flag, value)

    assert f"argument {flag}" in output, "the error should name the flag the user typed"
    assert "must be" in output


def test_the_budget_message_no_longer_masquerades_as_a_budget_problem():
    """
    The specific regression: `--max-cost -1` used to report reaching a limit,
    which sends the reader to look at their spending rather than their command.
    """
    output = cli("--max-cost", "-1")

    assert "Spent $0.0000" not in output
    assert "argument --max-cost: must be greater than 0" in output


def test_workers_zero_no_longer_leaks_a_thread_pool_error():
    output = cli("--workers", "0")

    assert "Traceback" not in output
    assert "argument --workers: must be 1 or greater" in output


@pytest.mark.parametrize(
    "flags",
    [
        ("--max-cost", "1.5"),
        ("--workers", "4"),
        ("--timeout", "600"),
        ("--max-iter", "1"),
        ("--clean-older-than", "0"),
        ("--context-budget", "50000"),
    ],
)
def test_valid_values_are_still_accepted(flags):
    """The bounds must not narrow what legitimately worked before."""
    output = cli(*flags)

    # Matched against argparse's own error shape rather than the words "must
    # be": --context-only prints the compiled context, which now contains
    # main.py including the validator docstrings that quote those words.
    assert "error: argument" not in output
    assert "Compiled context printed" in output


# ── with the config file (#291) ───────────────────────────────────────────


def _parser():
    """The real parser, captured by intercepting parse_args."""
    import argparse as ap
    import contextlib
    import io as _io

    import main as cli

    real = ap.ArgumentParser.parse_args
    captured = {}

    def spy(self, *args, **kwargs):
        captured["parser"] = self
        raise SystemExit(0)

    ap.ArgumentParser.parse_args = spy
    try:
        with contextlib.suppress(SystemExit), contextlib.redirect_stdout(_io.StringIO()):
            cli.main()
    finally:
        ap.ArgumentParser.parse_args = real
    return captured["parser"]


def _apply(tmp_path, config):
    """Apply a config and return (args, stderr)."""
    import contextlib
    import io as _io
    import json

    import main as cli

    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))
    parser = _parser()
    args = parser.parse_args(["--repo", ".", "--task", "t"])
    err = _io.StringIO()
    with contextlib.redirect_stderr(err):
        cli.apply_config_file(str(path), args, parser)
    return args, err.getvalue()


def test_a_negative_in_a_config_is_reported_not_raised(tmp_path):
    """
    #291 coerces config values through each flag's declared type. These
    validators raise ArgumentTypeError, which inherits from Exception rather
    than from TypeError or ValueError — so without widening that handler, a
    negative in a config file escaped as a traceback instead of a warning.
    """
    args, err = _apply(tmp_path, {"workers": -5})

    assert "workers" in err
    assert args.workers == 10


def test_a_negative_float_in_a_config_is_reported(tmp_path):
    args, err = _apply(tmp_path, {"max_cost": -1})

    assert "max_cost" in err
    assert args.max_cost is None


def test_a_valid_config_value_still_applies(tmp_path):
    """The bounds must not stop legitimate config values working."""
    args, err = _apply(tmp_path, {"timeout": 600})

    assert args.timeout == 600
    assert err == ""


def test_one_bad_config_value_does_not_stop_the_others(tmp_path):
    args, _ = _apply(tmp_path, {"workers": -5, "timeout": 600})

    assert args.workers == 10
    assert args.timeout == 600
