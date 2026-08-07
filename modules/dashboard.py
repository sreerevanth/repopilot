"""
Module: Run dashboard.

`logger.py` already writes one JSON object per iteration to `logs/<run_id>.jsonl`.
This renders that as a readable timeline, and with --follow it tails the file
while a run is in progress — so the agent's reasoning, its file changes and its
test output are visible as they happen rather than scrolling past in CLI output.

    python -m modules.dashboard logs/agent_20260807_abc.jsonl
    python -m modules.dashboard logs/agent_20260807_abc.jsonl --follow

Reading the log rather than instrumenting the loop keeps this decoupled: it
works on a finished run, on a run in another terminal, and needs no changes to
agent_loop.py. It also means no web server and no new dependency.
"""

import argparse
import json
import os
import sys
import time
from typing import Iterator, Optional

POLL_SECONDS = 0.5

# Plain ANSI. Disabled when stdout is not a tty, so piping to a file or a CI log
# produces clean text rather than escape sequences.
_CODES = {
    "reset": "\033[0m",
    "dim": "\033[2m",
    "bold": "\033[1m",
    "red": "\033[31m",
    "green": "\033[32m",
    "yellow": "\033[33m",
    "blue": "\033[34m",
    "cyan": "\033[36m",
}


def _use_colour(stream=None) -> bool:
    stream = stream or sys.stdout
    if os.environ.get("NO_COLOR"):
        return False
    return bool(getattr(stream, "isatty", lambda: False)())


def paint(text: str, colour: str, enabled: bool = True) -> str:
    if not enabled or colour not in _CODES:
        return text
    return f"{_CODES[colour]}{text}{_CODES['reset']}"


def read_records(path: str) -> list[dict]:
    """Parse a run log. Malformed lines are skipped, not fatal."""
    records = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                # A run killed mid-write leaves a partial final line. Showing
                # the rest of the run is more useful than refusing to start.
                continue
    return records


def follow_records(path: str, poll: float = POLL_SECONDS) -> Iterator[dict]:
    """
    Yield records as they are appended, waiting for the file to appear.

    Buffers a partial trailing line rather than discarding it: a record being
    written when we read is complete a moment later.
    """
    while not os.path.exists(path):
        time.sleep(poll)

    buffer = ""
    with open(path, "r", encoding="utf-8") as handle:
        while True:
            chunk = handle.readline()
            if not chunk:
                time.sleep(poll)
                continue
            buffer += chunk
            if not buffer.endswith("\n"):
                continue
            for line in buffer.splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue
            buffer = ""


def summary_path_for(log_path: str) -> str:
    base = log_path[:-6] if log_path.endswith(".jsonl") else log_path
    return f"{base}_summary.json"


def read_summary(log_path: str) -> Optional[dict]:
    """The summary is only written when a run finishes."""
    try:
        with open(summary_path_for(log_path), "r", encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError):
        return None


def _outcome_colour(outcome: str) -> str:
    return {
        "success": "green",
        "failed": "red",
        "max_retries": "red",
        "error": "red",
        "aborted": "yellow",
        "budget_exceeded": "yellow",
        "dry_run": "cyan",
    }.get(outcome, "yellow")


def render_iteration(record: dict, colour: bool = True, width: int = 78) -> str:
    """One iteration as a block of text."""
    lines: list[str] = []
    number = record.get("iteration", "?")

    lines.append(paint("─" * width, "dim", colour))
    header = f"Iteration {number}"
    timestamp = record.get("timestamp", "")
    if timestamp:
        header += f"   {timestamp}"
    lines.append(paint(header, "bold", colour))

    files = record.get("context_files") or []
    if files:
        shown = ", ".join(files[:5])
        if len(files) > 5:
            shown += f" (+{len(files) - 5} more)"
        chars = record.get("context_chars", 0)
        summary_line = f"  context   {len(files)} files, {chars:,} chars"
        lines.append(paint(summary_line, "dim", colour))
        lines.append(paint(f"            {shown}", "dim", colour))

    analysis = (record.get("llm_analysis") or "").strip()
    if analysis:
        lines.append(paint("  analysis", "cyan", colour))
        for line in analysis.splitlines()[:6]:
            lines.append(f"    {line}")

    confidence = record.get("llm_confidence")
    if confidence is not None:
        done = "done" if record.get("llm_done") else "not done"
        lines.append(paint(f"  confidence {confidence:.2f} ({done})", "dim", colour))

    if record.get("parse_error"):
        lines.append(
            paint(f"  parse error: {record['parse_error'][:200]}", "red", colour)
        )

    for change in record.get("changes_attempted") or []:
        action = change.get("action", "?")
        path = change.get("path", "?")
        lines.append(paint(f"  {action:<7} {path}", "blue", colour))

    failed = [r for r in (record.get("apply_results") or []) if not r.get("success")]
    for result in failed:
        lines.append(
            paint(
                f"  FAILED  {result.get('path')}: {result.get('error')}", "red", colour
            )
        )

    command = record.get("execution_command")
    if command:
        success = record.get("execution_success")
        timed_out = record.get("execution_timed_out")
        status = "PASS" if success else ("TIMEOUT" if timed_out else "FAIL")
        lines.append(
            paint(f"  {status}", "green" if success else "red", colour)
            + paint(
                f"  exit={record.get('execution_exit_code')}  {command}", "dim", colour
            )
        )
        if not success:
            tail = (
                record.get("execution_stderr")
                or record.get("execution_stdout")
                or ""
            )
            for line in tail.strip().splitlines()[-8:]:
                lines.append(paint(f"    {line}", "dim", colour))

    return "\n".join(lines)


def render_summary(summary: dict, colour: bool = True) -> str:
    outcome = summary.get("outcome", "unknown")
    lines = [
        paint("=" * 78, "dim", colour),
        paint(f"OUTCOME   {outcome.upper()}", _outcome_colour(outcome), colour),
        f"RUN       {summary.get('run_id', '')}",
        f"TASK      {(summary.get('task') or '')[:200]}",
    ]
    if summary.get("branch_name"):
        lines.append(f"BRANCH    {summary['branch_name']}")
    if summary.get("pr_url"):
        lines.append(f"PR        {summary['pr_url']}")
    if summary.get("final_error"):
        lines.append(paint(f"ERROR     {summary['final_error']}", "red", colour))
    return "\n".join(lines)


def show(path: str, follow: bool = False, colour: Optional[bool] = None) -> int:
    colour = _use_colour() if colour is None else colour

    if not follow:
        if not os.path.exists(path):
            print(f"No such run log: {path}", file=sys.stderr)
            return 1
        for record in read_records(path):
            print(render_iteration(record, colour))
        summary = read_summary(path)
        if summary:
            print(render_summary(summary, colour))
        return 0

    print(paint(f"Following {path} — Ctrl+C to stop", "dim", colour))
    try:
        for record in follow_records(path):
            print(render_iteration(record, colour))
            summary = read_summary(path)
            if summary and summary.get("outcome"):
                print(render_summary(summary, colour))
                return 0
    except KeyboardInterrupt:
        print()
    return 0


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Render a RepoPilot run log as a readable timeline.",
    )
    parser.add_argument("log", help="Path to logs/<run_id>.jsonl")
    parser.add_argument(
        "--follow", "-f", action="store_true",
        help="Tail the log while a run is in progress",
    )
    parser.add_argument("--no-color", action="store_true", help="Disable ANSI colour")
    args = parser.parse_args(argv)

    return show(args.log, follow=args.follow, colour=False if args.no_color else None)


if __name__ == "__main__":
    sys.exit(main())
