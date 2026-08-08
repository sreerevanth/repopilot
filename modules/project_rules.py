"""
Module: Project rules.

A `.agentcontext` file at the repository root carries instructions that apply to
every task in that repository — "this project targets Python 3.9, do not use
match statements", "tests live in spec/ not tests/", "never edit generated/".

These are different from `.repopilot.json`, which sets CLI defaults. That file
answers "how should the tool be invoked"; this one answers "what should the
model know before it writes anything".

Kept separate from the file context because the two behave differently under
pressure: when the character budget is tight, source files get dropped or
outlined, and the project's own rules should not be the thing that falls out.
"""

import logging
import os
from typing import Optional

_LOG = logging.getLogger("agent.project_rules")

RULES_FILENAME = ".agentcontext"

# Rules are prepended to every prompt, on every iteration, so their cost is paid
# repeatedly. This is generous for a page of conventions and small enough that
# a file pasted here by mistake is caught rather than silently billed for.
MAX_RULES_CHARS = 8_000


def rules_path(repo_root: str, filename: str = RULES_FILENAME) -> str:
    return os.path.join(repo_root, filename)


def load_project_rules(
    repo_root: str,
    filename: str = RULES_FILENAME,
) -> Optional[str]:
    """
    Read `.agentcontext` from the repository root, or None if there is none.

    Never raises. An unreadable rules file should not stop a run that would
    otherwise work — the agent simply proceeds without the extra guidance, which
    is what happens today for every repository that has no such file.
    """
    path = rules_path(repo_root, filename)
    if not os.path.isfile(path):
        return None

    try:
        text = open(path, "r", encoding="utf-8", errors="replace").read()
    except OSError as exc:
        _LOG.warning("could not read %s: %s", path, exc)
        return None

    text = text.strip()
    if not text:
        _LOG.debug("%s is empty; ignoring", path)
        return None

    if len(text) > MAX_RULES_CHARS:
        _LOG.warning(
            "%s is %d chars; using the first %d. Rules are sent on every "
            "iteration, so a long file is paid for repeatedly.",
            path, len(text), MAX_RULES_CHARS,
        )
        text = text[:MAX_RULES_CHARS].rstrip()

    return text


def render_project_rules(rules: Optional[str]) -> str:
    """
    Wrap the rules for inclusion in a prompt.

    Tagged like the file blocks the context already emits, so the model sees a
    consistent structure, and labelled as rules rather than as content — the
    distinction matters when a rule says "never edit generated/" and the model
    is otherwise free to treat everything it receives as material to change.
    """
    if not rules:
        return ""
    return (
        "<project_rules>\n"
        "These rules come from the repository and apply to every task in it. "
        "Follow them unless the task explicitly says otherwise.\n\n"
        f"{rules}\n"
        "</project_rules>"
    )
