"""
Module 7: Git Integration
Branch creation, staging, committing, pushing.
Optionally creates GitHub PRs via REST API.
"""

import json
import os
import re
import subprocess
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Optional


@dataclass
class GitResult:
    command: str
    success: bool
    output: str
    error: str


class GitIntegration:
    def __init__(self, repo_root: str):
        self.repo_root = os.path.abspath(repo_root)
        self._verify_git()

    def _verify_git(self):
        result = self._run(["git", "rev-parse", "--is-inside-work-tree"])
        if not result.success:
            raise RuntimeError(f"Not a git repository: {self.repo_root}")

    def _run(self, cmd: list[str], check: bool = False) -> GitResult:
        try:
            proc = subprocess.run(
                cmd,
                cwd=self.repo_root,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=30,
            )
            success = proc.returncode == 0
            if check and not success:
                raise RuntimeError(
                    f"Git command failed: {' '.join(cmd)}\n{proc.stderr}"
                )
            return GitResult(
                command=" ".join(cmd),
                success=success,
                output=proc.stdout.strip(),
                error=proc.stderr.strip(),
            )
        except subprocess.TimeoutExpired:
            return GitResult(
                command=" ".join(cmd),
                success=False,
                output="",
                error="Git command timed out",
            )
        except FileNotFoundError:
            return GitResult(
                command=" ".join(cmd),
                success=False,
                output="",
                error="git binary not found",
            )

    def current_branch(self) -> str:
        result = self._run(["git", "branch", "--show-current"])
        return result.output or "unknown"

    def create_branch(self, branch_name: str, from_branch: str = "main") -> GitResult:
        """Create and checkout a new branch."""
        base = self._run(["git", "rev-parse", "--verify", from_branch])
        if not base.success:
            base = self._run(["git", "rev-parse", "--verify", "master"])
            from_branch = "master" if base.success else "HEAD"

        result = self._run(["git", "checkout", "-b", branch_name, from_branch])
        if not result.success:
            result = self._run(["git", "checkout", branch_name])
        return result

    def stage_all(self) -> GitResult:
        """Stage all modified and new files."""
        return self._run(["git", "add", "-A"])

    def stage_files(self, paths: list[str]) -> GitResult:
        """Stage specific files."""
        return self._run(["git", "add", "--"] + paths)

    def commit(self, message: str, author: Optional[str] = None) -> GitResult:
        """Create a commit with the given message."""
        cmd = ["git", "commit", "-m", message]
        env = os.environ.copy()
        if author:
            author_name = author.split("<")[0].strip()
            env["GIT_AUTHOR_NAME"] = author_name
            env.setdefault("GIT_COMMITTER_NAME", author_name)
            email_match = re.search(r"<(.+)>", author)
            if email_match:
                author_email = email_match.group(1)
                env["GIT_AUTHOR_EMAIL"] = author_email
                env.setdefault("GIT_COMMITTER_EMAIL", author_email)

        try:
            proc = subprocess.run(
                cmd,
                cwd=self.repo_root,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=30,
                env=env,
            )
            return GitResult(
                command=" ".join(cmd),
                success=proc.returncode == 0,
                output=proc.stdout.strip(),
                error=proc.stderr.strip(),
            )
        except Exception as exc:
            return GitResult(command=" ".join(cmd), success=False, output="", error=str(exc))

    def push(self, branch: str, remote: str = "origin", force: bool = False) -> GitResult:
        """Push branch to remote."""
        cmd = ["git", "push", remote, branch]
        if force:
            cmd.append("--force-with-lease")
        return self._run(cmd)

    def diff_staged(self) -> str:
        """Return the staged diff as a string for PR descriptions."""
        result = self._run(["git", "diff", "--cached", "--stat"])
        return result.output

    def get_remote_url(self, remote: str = "origin") -> Optional[str]:
        result = self._run(["git", "remote", "get-url", remote])
        return result.output if result.success else None

    def has_uncommitted_changes(self) -> bool:
        result = self._run(["git", "status", "--porcelain"])
        return bool(result.output.strip())

    def create_github_pr(
        self,
        title: str,
        body: str,
        head_branch: str,
        base_branch: str = "main",
        github_token: Optional[str] = None,
    ) -> Optional[str]:
        """
        Create a GitHub PR via REST API.
        Returns the PR URL on success, None on failure.
        Requires GITHUB_TOKEN env var or explicit token.
        """
        token = github_token or os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
        if not token:
            return None

        remote_url = self.get_remote_url()
        if not remote_url:
            return None

        match = re.search(r"github\.com[:/]([^/]+)/([^/.]+)", remote_url)
        if not match:
            return None
        owner, repo = match.group(1), match.group(2)

        payload = json.dumps({
            "title": title,
            "body": body,
            "head": head_branch,
            "base": base_branch,
            "draft": False,
        }).encode("utf-8")

        url = f"https://api.github.com/repos/{owner}/{repo}/pulls"
        req = urllib.request.Request(
            url,
            data=payload,
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json",
                "Accept": "application/vnd.github+json",
                "X-GitHub-Api-Version": "2022-11-28",
            },
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=15) as resp:
                data = json.loads(resp.read().decode("utf-8", errors="replace"))
                return data.get("html_url")
        except urllib.error.HTTPError as exc:
            error_body = exc.read().decode("utf-8", errors="replace")
            if exc.code == 422 and "already exists" in error_body:
                return f"PR already exists for {head_branch} -> {base_branch}"
            return None
        except Exception:
            return None
