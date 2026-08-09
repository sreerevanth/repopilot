"""
Module: Change summary.

A table of what a run actually did to the repository, printed when it finishes.

The subtlety is that a run has several iterations and a file can be touched in
more than one. A file created in iteration 1 and edited again in iteration 3 is
"added" as far as the repository is concerned, not "modified" -- what a reader
wants is the net effect against the state the run started from, not a log of
every intermediate step.
"""

from dataclasses import dataclass
from typing import Optional

ADDED = "added"
MODIFIED = "modified"
DELETED = "deleted"
RENAMED = "renamed"

# Column order in the rendered table, and the order sections appear in.
ACTION_ORDER = (ADDED, MODIFIED, RENAMED, DELETED)

SYMBOLS = {ADDED: "+", MODIFIED: "~", DELETED: "-", RENAMED: ">"}


@dataclass
class ChangedFile:
    path: str
    action: str
    new_path: Optional[str] = None


def _net(previous: Optional[str], current: str) -> Optional[str]:
    """
    Combine what already happened to a path with what just happened to it.

    Returns None when the two cancel out -- a file created and then deleted
    within the same run leaves the repository as it was, and listing it would
    send someone looking for a change that is not there.
    """
    if previous is None:
        return current

    if previous == ADDED:
        if current == DELETED:
            return None          # created then removed: no net change
        return ADDED             # still an addition, however often it was edited

    if previous == DELETED:
        # Deleted then written again is a modification, not an addition.
        return MODIFIED if current in (ADDED, MODIFIED) else DELETED

    if previous == MODIFIED and current == DELETED:
        return DELETED

    if previous == RENAMED:
        return RENAMED

    return current if current != MODIFIED else previous


def summarise_changes(apply_results) -> list:
    """
    Fold every applied change across a run into one entry per path.

    Takes anything with `.path`, `.action`, `.success` and optionally
    `.new_path`; failed applications are ignored, since they changed nothing.
    """
    net: dict = {}
    renames: dict = {}

    for result in apply_results:
        if not getattr(result, "success", False):
            continue

        path = result.path
        action = result.action
        if action == "create":
            action = ADDED
        elif action == "modify" or action == "patch":
            action = MODIFIED
        elif action == "delete":
            action = DELETED
        elif action == "rename":
            action = RENAMED
            renames[path] = getattr(result, "new_path", None)

        combined = _net(net.get(path), action)
        if combined is None:
            net.pop(path, None)
            renames.pop(path, None)
        else:
            net[path] = combined

    return [
        ChangedFile(path=path, action=action, new_path=renames.get(path))
        for path, action in sorted(net.items())
    ]


def render_change_summary(changed: list) -> str:
    """A grouped table of the net changes, or a line saying there were none."""
    if not changed:
        return "No files were changed."

    by_action: dict = {}
    for entry in changed:
        by_action.setdefault(entry.action, []).append(entry)

    counts = ", ".join(
        f"{len(by_action[a])} {a}" for a in ACTION_ORDER if a in by_action
    )
    lines = [
        "-" * 60,
        f"Files changed: {counts}",
        "-" * 60,
    ]

    for action in ACTION_ORDER:
        for entry in by_action.get(action, []):
            symbol = SYMBOLS[action]
            if action == RENAMED and entry.new_path:
                lines.append(f"  {symbol} {entry.path} -> {entry.new_path}")
            else:
                lines.append(f"  {symbol} {entry.path}")

    return "\n".join(lines)
