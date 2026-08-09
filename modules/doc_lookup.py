"""
Module: Documentation lookup.

The model sometimes invents third-party API surface — a boto3 method that does
not exist, a parameter that was renamed two versions ago. Letting it read the
real documentation when a task depends on an unfamiliar library removes a class
of failure that no amount of retrying fixes, because the retry prompt only ever
says "that method does not exist" without saying what does.

This is off by default and deliberately narrow. The sandbox runs with
`--network=none` on the premise that model-written code is untrusted; a lookup
tool does not breach that — the fetch happens in the agent process, not the
sandbox — but a model-supplied string now decides what URL is requested from
the machine running the agent. On a CI runner that machine can reach cloud
metadata endpoints, so the URL is constrained rather than trusted:

- HTTPS only
- host must be on ALLOWED_DOC_HOSTS
- the resolved address must be public, which blocks a hostname on the allowlist
  that resolves to 169.254.169.254 or a private range
- redirects are re-validated at every hop, not followed blindly
- response size and time are capped
"""

import html.parser
import ipaddress
import logging
import re
import socket
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from modules.errors import ConfigurationError

_LOG = logging.getLogger("agent.doc_lookup")

# Documentation hosts only. Subdomains are matched, so "docs.python.org" also
# permits "docs.python.org" itself and anything under it, but never a host that
# merely ends with the same characters ("evildocs.python.org.attacker.com").
ALLOWED_DOC_HOSTS = (
    "docs.python.org",
    "docs.aws.amazon.com",
    "boto3.amazonaws.com",
    "developer.mozilla.org",
    "docs.djangoproject.com",
    "flask.palletsprojects.com",
    "docs.pytest.org",
    "numpy.org",
    "pandas.pydata.org",
    "docs.rs",
    "pkg.go.dev",
    "nodejs.org",
    "typescriptlang.org",
    "docs.oracle.com",
    "docs.microsoft.com",
    "learn.microsoft.com",
    "kubernetes.io",
    "docs.docker.com",
    "git-scm.com",
    "peps.python.org",
)

MAX_BYTES = 200_000          # a doc page, not a download
MAX_CHARS_RETURNED = 8_000   # what actually reaches the prompt
MAX_REDIRECTS = 3
DEFAULT_TIMEOUT = 15
MAX_LOOKUPS_PER_ITERATION = 3


class LookupRefused(ConfigurationError, ValueError):
    """Raised when a URL is not permitted. The message is shown to the model."""


@dataclass
class LookupResult:
    url: str
    ok: bool
    text: str = ""
    error: str = ""


def host_allowed(host: str) -> bool:
    host = (host or "").lower().strip().rstrip(".")
    if not host:
        return False
    return any(
        host == allowed or host.endswith("." + allowed)
        for allowed in ALLOWED_DOC_HOSTS
    )


def address_is_public(host: str) -> bool:
    """
    Resolve the host and require every address to be publicly routable.

    An allowlisted name that resolves into a private range is the interesting
    attack: it defeats a name-only check while still reaching the metadata
    service or an internal host. There is a small window between this check and
    the connection, which the host allowlist is what really closes.
    """
    try:
        infos = socket.getaddrinfo(host, 443, proto=socket.IPPROTO_TCP)
    except (socket.gaierror, UnicodeError, OSError):
        return False

    for info in infos:
        address = ipaddress.ip_address(info[4][0])
        if (
            address.is_private
            or address.is_loopback
            or address.is_link_local
            or address.is_reserved
            or address.is_multicast
            or address.is_unspecified
        ):
            return False
    return bool(infos)


def validate_url(url: str) -> str:
    """Return the URL if it may be fetched, else raise LookupRefused."""
    parsed = urllib.parse.urlparse((url or "").strip())

    if parsed.scheme.lower() != "https":
        raise LookupRefused(
            f"Only https:// URLs may be looked up (got '{parsed.scheme or 'none'}')."
        )
    if not host_allowed(parsed.hostname or ""):
        raise LookupRefused(
            f"'{parsed.hostname}' is not an allowed documentation host. "
            f"Allowed: {', '.join(ALLOWED_DOC_HOSTS[:6])}, and others."
        )
    if not address_is_public(parsed.hostname or ""):
        raise LookupRefused(
            f"'{parsed.hostname}' does not resolve to a public address."
        )
    return url


class _TextExtractor(html.parser.HTMLParser):
    """Strip tags without pulling in a parser dependency."""

    SKIP = {"script", "style", "nav", "footer", "head", "svg"}

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self._parts: list[str] = []
        self._skipping = 0

    def handle_starttag(self, tag, attrs):
        if tag in self.SKIP:
            self._skipping += 1

    def handle_endtag(self, tag):
        if tag in self.SKIP and self._skipping:
            self._skipping -= 1

    def handle_data(self, data):
        if not self._skipping and data.strip():
            self._parts.append(data.strip())

    def text(self) -> str:
        return re.sub(r"\n{3,}", "\n\n", "\n".join(self._parts))


def extract_text(body: str, content_type: str = "") -> str:
    if "html" in content_type.lower() or "<" in body[:200]:
        parser = _TextExtractor()
        try:
            parser.feed(body)
        except Exception:
            return body
        return parser.text()
    return body


def fetch_doc(url: str, timeout: int = DEFAULT_TIMEOUT) -> LookupResult:
    """
    Fetch one documentation page as text.

    Never raises: a refusal or a network failure is returned so the loop can put
    the reason in the next prompt, which is more useful to the model than an
    exception is to anyone.
    """
    try:
        current = validate_url(url)
    except LookupRefused as exc:
        return LookupResult(url=url, ok=False, error=str(exc))

    opener = urllib.request.build_opener(_NoRedirect())
    for _ in range(MAX_REDIRECTS + 1):
        request = urllib.request.Request(
            current,
            headers={"User-Agent": "repopilot-agent", "Accept": "text/html,text/plain"},
            method="GET",
        )
        try:
            with opener.open(request, timeout=timeout) as response:
                status = getattr(response, "status", 200)
                if status in (301, 302, 303, 307, 308):
                    location = response.headers.get("Location", "")
                    current = urllib.parse.urljoin(current, location)
                    try:
                        # Re-validated at every hop: a redirect off the allowlist
                        # would otherwise walk straight around it.
                        current = validate_url(current)
                    except LookupRefused as exc:
                        return LookupResult(url=url, ok=False, error=str(exc))
                    continue

                raw = response.read(MAX_BYTES)
                charset = response.headers.get_content_charset() or "utf-8"
                body = raw.decode(charset, errors="replace")
                text = extract_text(body, response.headers.get("Content-Type", ""))
                return LookupResult(url=current, ok=True, text=text[:MAX_CHARS_RETURNED])
        except urllib.error.HTTPError as exc:
            return LookupResult(url=current, ok=False, error=f"HTTP {exc.code}")
        except Exception as exc:
            return LookupResult(url=current, ok=False, error=str(exc))

    return LookupResult(url=url, ok=False, error="Too many redirects.")


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    """Surface redirects instead of following them, so each hop is checked."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def render_lookups(results: list[LookupResult]) -> str:
    """Format lookup results for the next prompt."""
    if not results:
        return ""
    blocks = ["## Documentation you requested", ""]
    for result in results:
        if result.ok:
            blocks += [f"### {result.url}", "", result.text, ""]
        else:
            blocks += [f"### {result.url}", "", f"(not available: {result.error})", ""]
    return "\n".join(blocks)


def perform_lookups(
    urls: list[str],
    timeout: int = DEFAULT_TIMEOUT,
    limit: int = MAX_LOOKUPS_PER_ITERATION,
) -> list[LookupResult]:
    """Fetch up to `limit` URLs, in order. Never raises."""
    results: list[LookupResult] = []
    for url in (urls or [])[:limit]:
        result = fetch_doc(url, timeout=timeout)
        _LOG.info(
            "lookup %s -> %s", url, "ok" if result.ok else f"refused: {result.error}"
        )
        results.append(result)
    return results
