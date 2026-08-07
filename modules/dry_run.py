# modules/dry_run.py
import json
import os
import datetime


def print_manifest(changes: list) -> None:
    """Print a human-readable change manifest to the console."""
    print("\n" + "=" * 60)
    print(f"  CHANGE MANIFEST  ({len(changes)} file(s) affected)")
    print("=" * 60)
    for i, c in enumerate(changes, 1):
        print(f"\n[{i}] Action : {c.get('action', 'modify').upper()}")
        print(f"    File   : {c.get('file_path', c.get('path', 'unknown'))}")
        reason = c.get('reason') or c.get('analysis', '')
        if reason:
            print(f"    Reason : {reason[:200]}")
        content = c.get('content', '')
        if content:
            preview = content[:300].replace('\n', '\n           ')
            suffix = '...' if len(content) > 300 else ''
            print(f"    Preview: {preview}{suffix}")
    print("\n" + "=" * 60)


def save_manifest(changes: list, log_dir: str, run_id: str) -> str:
    """Save the change manifest as a JSON file and return its path."""
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()
    path = os.path.join(log_dir, f"manifest_{run_id}.json")
    data = {
        "run_id": run_id,
        "timestamp": timestamp,
        "dry_run": True,
        "total_changes": len(changes),
        "changes": [
            {
                "action": c.get('action', 'modify'),
                "file": c.get('file_path', c.get('path', 'unknown')),
                "reason": c.get('reason', ''),
                "content_preview": (c.get('content', '') or '')[:500],
            }
            for c in changes
        ],
    }
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    return path


def ask_confirmation(n: int, action: str = "Apply") -> bool:
    """
    Prompt the user to confirm an action on n changes. Returns True if approved.

    `action` is the verb shown in the prompt. It defaults to "Apply" so the
    existing pre-apply prompt is unchanged; --interactive passes "Commit".
    """
    try:
        answer = input(f"\n{action} these {n} change(s)? [Y/n]: ").strip().lower()
        return answer not in ("n", "no")
    except (KeyboardInterrupt, EOFError):
        print("\nAborted.")
        return False