"""
Module 4: Code Modification Engine
Safely applies file changes from LLM output.
Creates backups, validates paths, preserves structure.
"""

import os
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

from modules.llm_client import FileChange


# "rename" is the canonical name; "move" is accepted because it is the word a
# model is just as likely to reach for, and rejecting it would send the agent
# back around the loop for a synonym.
RENAME_ACTIONS = ("rename", "move")
VALID_ACTIONS = ("modify", "create", "delete", "patch") + RENAME_ACTIONS


@dataclass
class ApplyResult:
    path: str
    action: str
    success: bool
    backup_path: Optional[str]
    error: Optional[str]
    new_path: Optional[str] = None   # destination, for "rename"


import re

def _apply_unified_diff(original: str, patch: str) -> str:
    """Apply a unified diff patch to the original text in pure Python."""
    original_lines = original.splitlines(keepends=True)
    patch_lines = patch.splitlines(keepends=True)
    
    result_lines = []
    orig_idx = 0
    
    hunk_re = re.compile(r'^@@\s+-(?P<old_start>\d+)(?:,(?P<old_len>\d+))?\s+\+(?P<new_start>\d+)(?:,(?P<new_len>\d+))?\s+@@')
    
    patch_idx = 0
    # Skip diff header lines until the first hunk
    while patch_idx < len(patch_lines) and not patch_lines[patch_idx].startswith('@@'):
        patch_idx += 1
        
    if patch_idx == len(patch_lines):
        # No hunks found, treat as full replacement fallback
        return patch
        
    while patch_idx < len(patch_lines):
        match = hunk_re.match(patch_lines[patch_idx])
        if not match:
            patch_idx += 1
            continue
            
        old_start = int(match.group('old_start'))
        # Adjust 1-based index to 0-based
        old_start = max(0, old_start - 1)
        
        # Copy original lines up to the start of this hunk
        result_lines.extend(original_lines[orig_idx:old_start])
        orig_idx = old_start
        
        patch_idx += 1
        # Process hunk lines
        while patch_idx < len(patch_lines) and not patch_lines[patch_idx].startswith('@@'):
            line = patch_lines[patch_idx]
            patch_idx += 1
            if line.startswith(' '):
                # Context line: verify and copy
                if orig_idx < len(original_lines):
                    result_lines.append(original_lines[orig_idx])
                    orig_idx += 1
            elif line.startswith('-'):
                # Deletion line: verify and skip original line
                if orig_idx < len(original_lines):
                    orig_idx += 1
            elif line.startswith('+'):
                # Addition line: append new line
                result_lines.append(line[1:])
                
    # Copy any remaining original lines
    if orig_idx < len(original_lines):
        result_lines.extend(original_lines[orig_idx:])
        
    return "".join(result_lines)


class CodeModificationEngine:
    def __init__(self, repo_root: str, backup_dir: str):
        self.repo_root = os.path.abspath(repo_root)
        self.backup_dir = os.path.abspath(backup_dir)
        os.makedirs(self.backup_dir, exist_ok=True)
        self._session_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    def _safe_abs_path(self, relative_path: str) -> str:
        """
        Resolve relative_path within repo_root.
        Raises ValueError if the resolved path would escape the repo root
        (path traversal prevention).
        """
        if not relative_path or not relative_path.strip():
            raise ValueError("Path is empty")

        candidate = Path(relative_path.replace("\\", "/"))
        if candidate.is_absolute():
            raise ValueError(f"Absolute paths are not allowed: '{relative_path}'")

        abs_path = (Path(self.repo_root) / candidate).resolve()
        repo_root = Path(self.repo_root).resolve()

        try:
            abs_path.relative_to(repo_root)
        except ValueError as exc:
            raise ValueError(
                f"Path traversal detected: '{relative_path}' resolves to '{abs_path}'"
            ) from exc
        return str(abs_path)

    def _backup(self, abs_path: str, relative_path: str) -> Optional[str]:
        """Copy file to backup dir before modification. Returns backup path."""
        if not os.path.exists(abs_path):
            return None
        safe_name = relative_path.replace("/", "__").replace("\\", "__")
        backup_path = os.path.join(
            self.backup_dir,
            f"{self._session_id}__{safe_name}"
        )
        shutil.copy2(abs_path, backup_path)
        return backup_path

    def _apply_single(self, change: FileChange) -> ApplyResult:
        try:
            abs_path = self._safe_abs_path(change.path)
        except ValueError as e:
            return ApplyResult(
                path=change.path, action=change.action,
                success=False, backup_path=None, error=str(e)
            )

        backup_path = None

        try:
            if change.action == "delete":
                backup_path = self._backup(abs_path, change.path)
                if os.path.exists(abs_path):
                    os.remove(abs_path)
                return ApplyResult(
                    path=change.path, action="delete",
                    success=True, backup_path=backup_path, error=None
                )

            elif change.action in ("modify", "create", "patch"):
                if not change.content and change.action in ("modify", "patch"):
                    return ApplyResult(
                        path=change.path, action=change.action,
                        success=False, backup_path=None,
                        error=f"LLM returned empty content for {change.action} action"
                    )

                # Backup existing file
                backup_path = self._backup(abs_path, change.path)

                # Ensure parent dirs exist
                os.makedirs(os.path.dirname(abs_path), exist_ok=True)

                if change.action == "patch":
                    # Load original text if file exists, else empty
                    original_text = ""
                    if os.path.exists(abs_path):
                        with open(abs_path, "r", encoding="utf-8") as fh:
                            original_text = fh.read()
                    patched_content = _apply_unified_diff(original_text, change.content)
                    with open(abs_path, "w", encoding="utf-8") as fh:
                        fh.write(patched_content)
                else:
                    # Write new content
                    with open(abs_path, "w", encoding="utf-8") as fh:
                        fh.write(change.content)

                return ApplyResult(
                    path=change.path, action=change.action,
                    success=True, backup_path=backup_path, error=None
                )

            elif change.action in RENAME_ACTIONS:
                if not change.new_path:
                    return ApplyResult(
                        path=change.path, action=change.action,
                        success=False, backup_path=None,
                        error=f"'{change.action}' on {change.path} requires 'new_path'"
                    )
                try:
                    dest_abs = self._safe_abs_path(change.new_path)
                except ValueError as e:
                    return ApplyResult(
                        path=change.path, action=change.action,
                        success=False, backup_path=None, error=str(e)
                    )
                if not os.path.exists(abs_path):
                    return ApplyResult(
                        path=change.path, action=change.action,
                        success=False, backup_path=None,
                        error=f"Cannot rename missing file: {change.path}"
                    )
                if os.path.normcase(dest_abs) == os.path.normcase(abs_path):
                    return ApplyResult(
                        path=change.path, action=change.action,
                        success=False, backup_path=None,
                        error=(
                            "Rename source and destination are the same: "
                            f"{change.path}"
                        )
                    )
                if os.path.exists(dest_abs):
                    # Silently clobbering a file the task never mentioned is the
                    # kind of loss no backup here would cover.
                    return ApplyResult(
                        path=change.path, action=change.action,
                        success=False, backup_path=None,
                        error=f"Refusing to overwrite existing file: {change.new_path}"
                    )

                backup_path = self._backup(abs_path, change.path)
                os.makedirs(os.path.dirname(dest_abs), exist_ok=True)
                shutil.move(abs_path, dest_abs)
                if change.content:
                    with open(dest_abs, "w", encoding="utf-8") as fh:
                        fh.write(change.content)

                return ApplyResult(
                    path=change.path, action=change.action,
                    success=True, backup_path=backup_path, error=None,
                    new_path=change.new_path
                )

            else:
                return ApplyResult(
                    path=change.path, action=change.action,
                    success=False, backup_path=None,
                    error=f"Unknown action: {change.action}"
                )

        except Exception as e:
            return ApplyResult(
                path=change.path, action=change.action,
                success=False, backup_path=backup_path,
                error=f"Unexpected error: {e}"
            )

    def apply_changes(self, changes: list[FileChange]) -> list[ApplyResult]:
        """Apply a list of file changes. Returns per-file results."""
        results = []
        for change in changes:
            result = self._apply_single(change)
            results.append(result)
        return results

    def rollback(self, results: list[ApplyResult]) -> list[str]:
        """
        Restore all backed-up files. Used if the iteration should be fully reverted.
        Returns list of restored paths.
        """
        restored = []
        for result in results:
            # A rename left a file at the destination. Restoring the backup to
            # the original path alone would leave both copies behind.
            if result.action in RENAME_ACTIONS and result.new_path:
                try:
                    dest_abs = self._safe_abs_path(result.new_path)
                    if os.path.exists(dest_abs):
                        os.remove(dest_abs)
                except Exception:
                    pass

            if result.backup_path and os.path.exists(result.backup_path):
                try:
                    abs_path = self._safe_abs_path(result.path)
                    shutil.copy2(result.backup_path, abs_path)
                    restored.append(result.path)
                except Exception:
                    pass
        return restored

    def verify_changes(self, changes: list[FileChange]) -> list[str]:
        """
        Pre-flight validation before applying.
        Returns list of error strings (empty = all good).
        """
        errors = []
        for change in changes:
            if not change.path:
                errors.append("Change has empty path")
                continue
            if change.action in RENAME_ACTIONS:
                if not change.new_path:
                    errors.append(
                        f"Missing 'new_path' for {change.action} on {change.path}"
                    )
                else:
                    try:
                        self._safe_abs_path(change.new_path)
                    except ValueError as e:
                        errors.append(f"{change.path}: invalid new_path - {e}")
            if change.action not in VALID_ACTIONS:
                errors.append(f"Invalid action '{change.action}' for {change.path}")
            if change.action in ("modify", "create", "patch") and not change.content:
                errors.append(f"Empty content for {change.action} on {change.path}")
            try:
                self._safe_abs_path(change.path)
            except ValueError as e:
                errors.append(str(e))
        return errors
    
    def git_stash_before_apply(self, repo_root: str) -> bool:
        """Stash current working state before agent applies changes."""
        import subprocess

        # Check if there's anything to stash
        check = subprocess.run(
            ["git", "stash", "push", "-m", "repopilot-pre-agent-state"],
            cwd=repo_root,
            capture_output=True,
            text=True,
        )
        if check.returncode != 0:
            print(f"[Rollback] Warning: git stash failed — {check.stderr.strip()}")
            return False

        # git stash exits 0 even on a clean tree — detect that case
        if "No local changes to save" in check.stdout:
            print("[Rollback] Nothing to stash — working tree is clean.")
            return False

        print("[Rollback] Git stash saved. Run with --rollback to undo.")
        return True

    def git_stash_pop(self, repo_root: str) -> bool:
        """Restore the pre-agent state via git stash pop."""
        import subprocess
        result = subprocess.run(
            ["git", "stash", "pop"],
            cwd=repo_root,
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            print("[Rollback] Successfully restored previous state.")
        else:
            print(f"[Rollback] Failed to pop stash — {result.stderr.strip()}")
        return result.returncode == 0