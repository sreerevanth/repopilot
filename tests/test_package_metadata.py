"""
Tests for package.json metadata.

`repository`, `bugs` and `homepage` all pointed at `rohitkumarnaidu/repopilot`
rather than this repository — leftovers from `npm init` in a fork. Nothing
breaks, but `npm repo`, `npm bugs` and `npm docs` all send you to the wrong
place, and Dependabot reads `repository` when it links its own pull requests.

`main` claimed `index.js`, which does not exist. This is a Python project whose
package.json exists only to pin prettier for CI.
"""

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

MANIFEST = json.loads((ROOT / "package.json").read_text(encoding="utf-8"))
REPO = "sreerevanth/repopilot"


@pytest.mark.parametrize("field", ["repository", "bugs", "homepage"])
def test_the_urls_point_at_this_repository(field):
    value = MANIFEST[field]
    url = value["url"] if isinstance(value, dict) else value

    assert REPO in url, f"{field} points elsewhere: {url}"


def test_no_stale_fork_owner_remains():
    """The specific value that was wrong, in every field at once."""
    assert "rohitkumarnaidu" not in json.dumps(MANIFEST)


def test_no_entry_point_is_claimed():
    """`main: index.js` named a file that does not exist in a Python project."""
    assert "main" not in MANIFEST
    assert not (ROOT / "index.js").exists()


def test_it_is_marked_private():
    """
    This manifest exists to pin prettier for CI, not to publish anything.
    `private` makes an accidental `npm publish` fail rather than succeed.
    """
    assert MANIFEST["private"] is True


# ── the part CI depends on ────────────────────────────────────────────────


def test_prettier_is_pinned():
    """
    `.github/workflows/prettier.yml` runs `npm ci` then `npm run format:check`,
    so a missing or unpinned prettier would fail the format job on a day the
    upstream default changed rather than on anything in this repository.
    """
    assert "prettier" in MANIFEST["devDependencies"]


def test_the_format_scripts_exist():
    scripts = MANIFEST["scripts"]

    assert scripts["format:check"].startswith("prettier --check")
    assert scripts["format"].startswith("prettier --write")


def test_the_lockfile_is_present():
    """`npm ci` requires it and fails outright without one."""
    assert (ROOT / "package-lock.json").exists()
