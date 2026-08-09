"""
Module: Response cache.

Stores LLM responses on disk so an identical request does not have to be paid
for twice. The case that prompted it: a dry run followed by a real run against
an unchanged repository makes the same call with the same context.

Opt-in, deliberately. Requests are made at `temperature: 0.2`, so the provider
may return different text for the same prompt. Serving a cached answer is
therefore a behaviour change rather than a pure optimisation -- useful when
debugging, because a run becomes reproducible, and wrong when the point of
re-running is to see whether the model does better this time.
"""

import hashlib
import json
import logging
import os
import time
from dataclasses import dataclass
from typing import Optional

_LOG = logging.getLogger("agent.response_cache")

CACHE_DIRNAME = ".repopilot-cache"

# Entries older than this are ignored and swept. The key already covers the
# prompt, so a changed repository produces a different key and cannot hit a
# stale entry -- this is disk hygiene rather than correctness.
DEFAULT_TTL_SECONDS = 7 * 24 * 3600

MAX_ENTRY_BYTES = 2 * 1024 * 1024


@dataclass
class CacheStats:
    hits: int = 0
    misses: int = 0
    writes: int = 0

    @property
    def total(self) -> int:
        return self.hits + self.misses


def cache_key(model: str, system_prompt: str, prompt: str) -> str:
    """
    Identity of a request.

    All three parts matter. The model changes the answer; the system prompt
    changes it too, so a run with --system-prompt must not be served an entry
    written by the default persona; and the prompt carries the task and the
    repository context, which is what makes a changed repository miss.
    """
    digest = hashlib.sha256()
    for part in (model or "", system_prompt or "", prompt or ""):
        digest.update(part.encode("utf-8", errors="replace"))
        digest.update(b"\x00")            # so ("ab", "c") != ("a", "bc")
    return digest.hexdigest()


class ResponseCache:
    """A content-addressed cache of raw provider responses."""

    def __init__(
        self,
        root: str,
        enabled: bool = False,
        ttl_seconds: int = DEFAULT_TTL_SECONDS,
    ):
        self.root = os.path.join(root, CACHE_DIRNAME)
        self.enabled = enabled
        self.ttl_seconds = ttl_seconds
        self.stats = CacheStats()

    def _path(self, key: str) -> str:
        # Two-character prefix directory: a long run can write thousands of
        # entries, and some filesystems slow down badly with that many siblings.
        return os.path.join(self.root, key[:2], f"{key}.json")

    def get(self, key: str) -> Optional[str]:
        """The cached response, or None. Never raises."""
        if not self.enabled:
            return None

        path = self._path(key)
        try:
            with open(path, "r", encoding="utf-8") as handle:
                entry = json.load(handle)
        except (OSError, json.JSONDecodeError):
            self.stats.misses += 1
            return None

        age = time.time() - entry.get("written_at", 0)
        if age > self.ttl_seconds:
            self.stats.misses += 1
            self._discard(path)
            return None

        self.stats.hits += 1
        _LOG.info("Cache hit; no API call made for this request.")
        return entry.get("response")

    def put(self, key: str, response: str) -> None:
        """Store a response. A cache that cannot be written is not an error."""
        if not self.enabled or response is None:
            return

        if len(response.encode("utf-8", errors="replace")) > MAX_ENTRY_BYTES:
            _LOG.debug("Response too large to cache; skipping.")
            return

        path = self._path(key)
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            # Written to a temporary name and moved, so an interrupted write
            # cannot leave a half-written entry that later parses as valid.
            temporary = f"{path}.tmp"
            with open(temporary, "w", encoding="utf-8") as handle:
                json.dump({"written_at": time.time(), "response": response}, handle)
            os.replace(temporary, path)
            self.stats.writes += 1
        except OSError as exc:
            _LOG.debug("Could not write cache entry: %s", exc)

    def _discard(self, path: str) -> None:
        try:
            os.remove(path)
        except OSError:
            pass

    def summary(self) -> str:
        if not self.enabled or not self.stats.total:
            return ""
        return (
            f"Cache: {self.stats.hits} hit(s), {self.stats.misses} miss(es) "
            f"of {self.stats.total} request(s)."
        )
