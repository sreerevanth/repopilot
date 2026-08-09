"""
Module: Housekeeping.

Removes old run logs and file backups.

Written defensively, because this is the one part of the tool whose whole job is
deleting things. Two rules it does not bend: only files matching the shapes this
tool creates are removed, and only from inside the directories it was given. A
stray `--log-dir /` should delete nothing rather than everything.
"""

import logging
import os
import re
import time
from dataclasses import dataclass, field
from typing import Optional

_LOG = logging.getLogger("agent.housekeeping")

# Names this tool writes. Anything else in these directories belongs to someone
# else and is left alone -- a log directory is a plausible place for a person to
# have put something of their own.
LOG_PATTERNS = (
    re.compile(r"^[\w.-]+_\d{8}_\d{6}_[0-9a-f]{6}\.jsonl$"),
    re.compile(r"^[\w.-]+_\d{8}_\d{6}_[0-9a-f]{6}_human\.log$"),
    re.compile(r"^[\w.-]+_\d{8}_\d{6}_[0-9a-f]{6}_summary\.json$"),
    re.compile(r"^[\w.-]+_\d{8}_\d{6}_[0-9a-f]{6}_state\.json$"),
)

BACKUP_PATTERN = re.compile(r"^\d{8}_\d{6}_")


@dataclass
class CleanResult:
    removed: list = field(default_factory=list)
    kept: list = field(default_factory=list)
    freed_bytes: int = 0
    errors: list = field(default_factory=list)

    @property
    def freed_human(self) -> str:
        size = float(self.freed_bytes)
        for unit in ("B", "KB", "MB", "GB"):
            if size < 1024 or unit == "GB":
                return f"{size:.1f} {unit}"
            size /= 1024
        return f"{size:.1f} GB"


def _is_removable(name: str, patterns) -> bool:
    return any(p.match(name) for p in patterns)


def _within(root: str, path: str) -> bool:
    """True when path really sits inside root, symlinks resolved."""
    root_real = os.path.realpath(root)
    return os.path.realpath(path).startswith(root_real + os.sep)


def clean_directory(
    directory: str,
    patterns,
    older_than_days: Optional[float] = None,
    dry_run: bool = False,
) -> CleanResult:
    """
    Remove matching files from one directory.

    Never recurses. A run writes its logs flat, and walking subdirectories would
    mean deciding what to do about a directory someone else created inside this
    one -- which is exactly the judgement this should not be making.
    """
    result = CleanResult()
    if not os.path.isdir(directory):
        return result

    cutoff = None
    if older_than_days is not None:
        cutoff = time.time() - (older_than_days * 86400)

    for name in sorted(os.listdir(directory)):
        path = os.path.join(directory, name)

        if not os.path.isfile(path) or os.path.islink(path):
            result.kept.append(name)
            continue

        if not _is_removable(name, patterns):
            result.kept.append(name)
            continue

        if not _within(directory, path):
            # A hard link or an unexpected resolution: refuse rather than guess.
            result.kept.append(name)
            continue

        try:
            stat = os.stat(path)
            if cutoff is not None and stat.st_mtime > cutoff:
                result.kept.append(name)
                continue

            if not dry_run:
                os.remove(path)
            result.removed.append(name)
            result.freed_bytes += stat.st_size
        except OSError as exc:
            result.errors.append(f"{name}: {exc}")
            _LOG.warning("could not remove %s: %s", path, exc)

    return result


def clean(
    repo_root: str,
    log_dir: str = "logs",
    backup_dir: str = "backups",
    older_than_days: Optional[float] = None,
    dry_run: bool = False,
) -> dict:
    """Clean both directories, returning a result for each."""
    return {
        "logs": clean_directory(
            os.path.join(repo_root, log_dir), LOG_PATTERNS, older_than_days, dry_run
        ),
        "backups": clean_directory(
            os.path.join(repo_root, backup_dir), (BACKUP_PATTERN,),
            older_than_days, dry_run,
        ),
    }


def render_clean_summary(results: dict, dry_run: bool = False) -> str:
    """What was removed, or what would be."""
    verb = "Would remove" if dry_run else "Removed"
    total_files = sum(len(r.removed) for r in results.values())
    total_bytes = sum(r.freed_bytes for r in results.values())

    if not total_files:
        return "Nothing to clean."

    lines = [f"{verb} {total_files} file(s), freeing {CleanResult(freed_bytes=total_bytes).freed_human}."]
    for label, result in results.items():
        if result.removed:
            lines.append(f"  {label}: {len(result.removed)} file(s)")
        if result.kept:
            lines.append(f"  {label}: {len(result.kept)} left alone (not ours)")
        for error in result.errors:
            lines.append(f"  {label}: could not remove {error}")
    return "\n".join(lines)
