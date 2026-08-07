"""
Tests for documentation lookup (modules/doc_lookup.py).

The sandbox runs with --network=none because model-written code is untrusted.
This does not breach that — the fetch happens in the agent process — but a
model-supplied string now decides what URL is requested from the machine
running the agent, which on a CI runner can reach cloud metadata endpoints.

Almost everything below is about what is refused. No test touches the network.
"""

import socket
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from modules import doc_lookup  # noqa: E402
from modules.doc_lookup import (  # noqa: E402
    ALLOWED_DOC_HOSTS,
    MAX_LOOKUPS_PER_ITERATION,
    LookupRefused,
    LookupResult,
    address_is_public,
    extract_text,
    fetch_doc,
    host_allowed,
    perform_lookups,
    render_lookups,
    validate_url,
)


@pytest.fixture
def resolves_to(monkeypatch):
    def _apply(address):
        monkeypatch.setattr(
            socket, "getaddrinfo",
            lambda host, port, **kw: [(2, 1, 6, "", (address, 443))],
        )
    return _apply


@pytest.fixture(autouse=True)
def public_by_default(resolves_to):
    resolves_to("93.184.216.34")


# ── the host allowlist ────────────────────────────────────────────────────


@pytest.mark.parametrize("host", ["docs.python.org", "boto3.amazonaws.com",
                                  "api.docs.aws.amazon.com"])
def test_documentation_hosts_are_allowed(host):
    assert host_allowed(host) is True


@pytest.mark.parametrize(
    "host",
    [
        "docs.python.org.attacker.com",   # suffix that merely contains the name
        "evil-docs.python.org",           # not a subdomain, just similar
        "github.com",
        "169.254.169.254",
        "localhost",
        "",
    ],
)
def test_everything_else_is_refused(host):
    assert host_allowed(host) is False


def test_a_trailing_dot_does_not_bypass_the_check():
    """"docs.python.org." is the same host to a resolver."""
    assert host_allowed("docs.python.org.") is True
    assert host_allowed("attacker.com.") is False


# ── URL validation ────────────────────────────────────────────────────────


def test_an_allowlisted_https_url_passes():
    assert validate_url("https://docs.python.org/3/library/os.html")


@pytest.mark.parametrize(
    "url,reason",
    [
        ("http://docs.python.org/3/", "https"),
        ("file:///etc/passwd", "https"),
        ("ftp://docs.python.org/x", "https"),
        ("https://github.com/evil", "allowed documentation host"),
        ("https://169.254.169.254/latest/meta-data/", "allowed documentation host"),
        ("https://localhost/admin", "allowed documentation host"),
    ],
)
def test_bad_urls_are_refused_with_a_reason(url, reason):
    """The message goes back to the model, so it must say what was wrong."""
    with pytest.raises(LookupRefused) as excinfo:
        validate_url(url)

    assert reason in str(excinfo.value)


# ── the address check ─────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "address",
    ["169.254.169.254", "127.0.0.1", "10.0.0.5", "192.168.1.1", "172.16.0.1",
     "0.0.0.0", "224.0.0.1"],
)
def test_an_allowlisted_name_resolving_privately_is_refused(resolves_to, address):
    """
    The interesting attack: a name that passes the allowlist but points inside
    the network. A name-only check would let this through.
    """
    resolves_to(address)

    assert address_is_public("docs.python.org") is False
    with pytest.raises(LookupRefused):
        validate_url("https://docs.python.org/3/")


def test_a_public_address_passes(resolves_to):
    resolves_to("93.184.216.34")
    assert address_is_public("docs.python.org") is True


def test_an_unresolvable_host_is_refused(monkeypatch):
    monkeypatch.setattr(
        socket, "getaddrinfo",
        lambda *a, **k: (_ for _ in ()).throw(socket.gaierror("nope")),
    )
    assert address_is_public("docs.python.org") is False


# ── fetching ──────────────────────────────────────────────────────────────


class _Response:
    def __init__(self, body=b"<h1>os.path</h1>", status=200, headers=None):
        self._body = body
        self.status = status
        self.headers = _Headers(headers or {"Content-Type": "text/html"})

    def read(self, n=None):
        return self._body[:n] if n else self._body

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class _Headers(dict):
    def get_content_charset(self):
        return "utf-8"


def test_a_page_is_fetched_as_text(monkeypatch):
    monkeypatch.setattr(
        doc_lookup.urllib.request, "build_opener",
        lambda *a: type("O", (), {"open": lambda s, r, timeout=None: _Response()})(),
    )
    result = fetch_doc("https://docs.python.org/3/library/os.html")

    assert result.ok is True
    assert "os.path" in result.text


def test_a_refusal_is_returned_not_raised():
    """The loop puts the reason in the next prompt; an exception helps nobody."""
    result = fetch_doc("https://github.com/evil")

    assert result.ok is False
    assert "not an allowed" in result.error


def test_a_network_error_is_returned_not_raised(monkeypatch):
    def explode(*a, **k):
        raise OSError("connection reset")

    monkeypatch.setattr(
        doc_lookup.urllib.request, "build_opener",
        lambda *a: type("O", (), {"open": lambda s, r, timeout=None: explode()})(),
    )
    assert fetch_doc("https://docs.python.org/3/").ok is False


def test_the_response_is_size_capped():
    assert doc_lookup.MAX_BYTES <= 1_000_000
    assert doc_lookup.MAX_CHARS_RETURNED <= 20_000


def test_redirects_are_revalidated():
    """A redirect off the allowlist would otherwise walk straight around it."""
    import inspect

    source = inspect.getsource(fetch_doc)
    assert "validate_url(current)" in source
    assert "MAX_REDIRECTS" in source


# ── text extraction ───────────────────────────────────────────────────────


def test_scripts_and_styles_are_dropped():
    text = extract_text(
        "<html><script>evil()</script><style>a{}</style>"
        "<body><h1>Title</h1><p>Body text.</p></body></html>",
        "text/html",
    )
    assert "evil()" not in text
    assert "Title" in text and "Body text." in text


def test_plain_text_passes_through():
    assert extract_text("just some text", "text/plain") == "just some text"


def test_malformed_html_does_not_raise():
    extract_text("<div><p>unclosed", "text/html")


# ── batching and rendering ────────────────────────────────────────────────


def test_the_number_of_lookups_is_capped(monkeypatch):
    """Otherwise one response could request a hundred fetches."""
    calls = []
    monkeypatch.setattr(
        doc_lookup, "fetch_doc",
        lambda url, timeout=None: calls.append(url) or LookupResult(url, True, "x"),
    )
    perform_lookups([f"https://docs.python.org/{i}" for i in range(20)])

    assert len(calls) == MAX_LOOKUPS_PER_ITERATION


def test_no_urls_means_no_fetches():
    assert perform_lookups([]) == []
    assert perform_lookups(None) == []


def test_rendered_output_labels_each_url():
    text = render_lookups([LookupResult("https://docs.python.org/3/", True, "content")])

    assert "https://docs.python.org/3/" in text
    assert "content" in text


def test_a_failure_is_rendered_so_the_model_learns_why():
    text = render_lookups([LookupResult("https://evil/", False, error="not allowed")])

    assert "not available" in text
    assert "not allowed" in text


def test_nothing_renders_for_no_results():
    assert render_lookups([]) == ""


# ── the allowlist itself ──────────────────────────────────────────────────


def test_the_allowlist_contains_only_documentation_hosts():
    """A code-hosting or paste host would make this a general fetch tool."""
    for host in ("github.com", "gist.github.com", "pastebin.com", "raw.githubusercontent.com"):
        assert host not in ALLOWED_DOC_HOSTS
