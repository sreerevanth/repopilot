"""
Tests for TLS verification (modules/llm_client.py).

The OpenAI, Gemini and Ollama request paths each hard-coded:

    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE

which accepts any certificate, including a self-signed one, and never matches
the hostname. Two of those three are public APIs reached over the internet.

The consequence is worse here than a credential leak. An attacker on the
network path could read the API key from the Authorization header, read the
prompt — which carries the user's source — and rewrite the response, which is
code this tool writes to disk and runs.
"""

import ssl
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from modules.llm_client import _tls_context, set_insecure_tls  # noqa: E402


@pytest.fixture(autouse=True)
def reset():
    """Module-level state; leaking it would make other tests unverified."""
    set_insecure_tls(False)
    yield
    set_insecure_tls(False)


# ── the default ───────────────────────────────────────────────────────────


def test_certificates_are_verified_by_default():
    context = _tls_context()

    assert context.verify_mode == ssl.CERT_REQUIRED


def test_the_hostname_is_checked_by_default():
    """
    Verification without hostname checking accepts any valid certificate for
    any domain, which is most of the way to no verification at all.
    """
    assert _tls_context().check_hostname is True


def test_no_provider_disables_verification_inline():
    """
    The three hard-coded sites are what this replaces. Their absence is the
    change; a future provider copying the old pattern would reintroduce it.
    """
    source = (ROOT / "modules" / "llm_client.py").read_text(encoding="utf-8")
    code = "\n".join(
        line for line in source.splitlines() if not line.strip().startswith("#")
    )

    assert "ctx.verify_mode = ssl.CERT_NONE" not in code
    assert "ctx.check_hostname = False" not in code


def test_every_provider_uses_the_shared_context():
    source = (ROOT / "modules" / "llm_client.py").read_text(encoding="utf-8")

    assert source.count("ctx = _tls_context()") == 3


# ── the opt-out ───────────────────────────────────────────────────────────


def test_insecure_tls_disables_verification():
    """
    The capability stays: a local Ollama with a self-signed certificate is a
    real case. It is opt-in now, so the person disabling it decided to.
    """
    set_insecure_tls(True)
    context = _tls_context()

    assert context.verify_mode == ssl.CERT_NONE
    assert context.check_hostname is False


def test_it_can_be_turned_back_off():
    set_insecure_tls(True)
    set_insecure_tls(False)

    assert _tls_context().verify_mode == ssl.CERT_REQUIRED


def test_each_call_returns_a_fresh_context():
    """A shared context mutated by one provider would affect the others."""
    assert _tls_context() is not _tls_context()


# ── the CLI ───────────────────────────────────────────────────────────────


def cli(*flags):
    """Combined output."""
    result = subprocess.run(
        [sys.executable, "main.py", "--repo", ".", "--task", "t", "--context-only", *flags],
        capture_output=True, text=True, cwd=ROOT,
    )
    return result.stdout + result.stderr


def cli_stderr(*flags):
    """
    stderr only. The warning is checked here rather than in combined output
    because --context-only prints the compiled context, which now contains
    main.py including the literal warning string.
    """
    return subprocess.run(
        [sys.executable, "main.py", "--repo", ".", "--task", "t", "--context-only", *flags],
        capture_output=True, text=True, cwd=ROOT,
    ).stderr


def test_the_flag_exists():
    assert "--insecure-tls" in cli("--help")


def test_using_it_warns():
    """
    Every run, not once. Someone who set it for a local server and forgot is
    otherwise running two public APIs unverified with nothing to remind them.
    """
    output = cli_stderr("--insecure-tls")

    assert "WARNING: --insecure-tls is set" in output


def test_the_warning_says_what_is_at_risk():
    """"Verification disabled" alone does not convey that the key is exposed."""
    output = cli_stderr("--insecure-tls")

    assert "API key" in output


def test_a_normal_run_is_silent_about_tls():
    assert "WARNING: --insecure-tls is set" not in cli_stderr()


def test_the_help_text_names_the_safer_alternative():
    """A corporate CA is the common reason to reach for this, and it has a
    correct answer that keeps verification on."""
    assert "SSL_CERT_FILE" in cli("--help")
