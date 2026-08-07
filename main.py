#!/usr/bin/env python3
"""
autonomous_agent/main.py
CLI entry point for the Autonomous AI Developer Agent.

Usage:
  python main.py --repo /path/to/repo --task "Fix the failing tests in utils.py"
  python main.py --repo . --task "Add input validation to user_signup function" --runner pytest --max-iter 5
  python main.py --repo . --task "..." --push --pr
"""

import argparse
import os
import sys

# Ensure the project root is on the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from modules.agent_loop import AutonomousAgent, AgentConfig
from modules.notify import notify_run_complete
from modules.code_modifier import CodeModificationEngine
from modules.task_source import (
    TaskResolutionError,
    looks_like_issue_url,
    resolve_task,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Autonomous AI Developer Agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Fix failing tests in a Python project
  python main.py --repo /my/project --task "Fix the TypeError in tests/test_parser.py"

  # Add a feature with up to 7 iterations
  python main.py --repo . --task "Add rate limiting to the /api/login endpoint" --max-iter 7

  # Run a specific file instead of tests
  python main.py --repo . --task "Fix the script" --run-file scripts/process.py

  # Full pipeline: push branch + open PR
  python main.py --repo . --task "Fix bug #123" --push --pr --base-branch main
        """
    )

    parser.add_argument("--repo", required=True, help="Path to the git repository")
    parser.add_argument("--task", required=False, help="High-level task description, or a GitHub issue URL to pull one from (can also be provided via AGENT_TASK env var)")

    # Execution
    parser.add_argument("--runner", default="pytest",
                        choices=["pytest", "npm_test", "go", "cargo", "ruby", "rspec", "bash", "make"],
                        help="Test runner to use (default: pytest)")
    parser.add_argument("--runner-args", nargs="*", default=None,
                        help="Extra arguments to pass to the test runner")
    parser.add_argument("--run-file", default=None,
                        help="Run a specific file instead of the test suite")
    parser.add_argument("--run-file-runner", default="python",
                        choices=["python", "node", "ruby", "bash"],
                        help="Runner for --run-file (default: python)")
    parser.add_argument("--lint", dest="lint_runner", default=None,
                        choices=["ruff", "flake8", "pyflakes", "eslint", "tsc",
                                 "govet", "clippy"],
                        help="Run a linter before the test suite. A failure short-"
                             "circuits the iteration and feeds the lint output back "
                             "to the model.")
    parser.add_argument("--lint-args", nargs="*", default=[],
                        help="Extra arguments for --lint")
    parser.add_argument("--timeout", type=int, default=120,
                        help="Execution timeout in seconds (default: 120)")

    # Loop control
    parser.add_argument("--max-iter", type=int, default=5,
                        help="Maximum number of agent iterations (default: 5)")

    # Git
    parser.add_argument("--no-git", action="store_true",
                        help="Disable git operations entirely")
    parser.add_argument("--push", action="store_true",
                        help="Push the branch to remote after success")
    parser.add_argument("--pr", action="store_true",
                        help="Create a GitHub PR after pushing (requires GITHUB_TOKEN)")
    parser.add_argument("--base-branch", default="main",
                        help="Base branch for PR and branch creation (default: main)")
    parser.add_argument("--branch-prefix", default="agent",
                        help="Prefix for the auto-created branch name (default: agent)")

    # Context
    parser.add_argument("--include", nargs="*", default=None,
                        help="Force-include specific file paths in context (relative to repo root)")

    # Output
    parser.add_argument("--log-dir", default="logs",
                        help="Directory for log files (default: logs/ inside repo)")
    parser.add_argument("--backup-dir", default="backups",
                        help="Directory for file backups (default: backups/ inside repo)")
    parser.add_argument("--quiet", action="store_true",
                        help="Suppress verbose output")
    parser.add_argument("--dry-run", "-d", action="store_true",help="Preview changes without applying them. Saves a manifest to logs/.")
    parser.add_argument("--resume", dest="resume_from", default=None, metavar="RUN_ID",
                        help="Continue an interrupted run from its last completed "
                             "iteration. Use --list-resumable to see candidates.")
    parser.add_argument("--list-resumable", action="store_true",
                        help="List run ids that can be resumed, then exit.")
    parser.add_argument("--rollback",action="store_true",help="Undo the last agent run by popping the git stash.")

    # API key
    parser.add_argument("--api-key", default=None,
                        help="Anthropic API key (default: ANTHROPIC_API_KEY env var)")

    # Non-interactive / CI
    parser.add_argument("--yes", "-y", action="store_true",
                        help="Auto-approve file changes (bypass confirmation prompt)")
    parser.add_argument("--interactive", "-i", action="store_true",
                        help="After tests pass, show the diff and confirm before "
                             "committing. Ignored when --yes is set or CI=true.")

    # Config file support
    parser.add_argument("--config", default=None,
                        help="Path to configuration JSON file (default: .repopilot.json in repo root)")

    return parser.parse_args()


def write_github_output(outputs: dict[str, str]):
    github_output_path = os.environ.get("GITHUB_OUTPUT")
    if github_output_path:
        try:
            with open(github_output_path, "a", encoding="utf-8") as f:
                for k, v in outputs.items():
                    if "\n" in v:
                        delimiter = "EOF"
                        f.write(f"{k}<<{delimiter}\n{v}\n{delimiter}\n")
                    else:
                        f.write(f"{k}={v}\n")
        except Exception as e:
            print(f"Failed to write to GITHUB_OUTPUT: {e}", file=sys.stderr)


def main():
    args = parse_args()

    # Configuration file support (#22)
    config_file = args.config or os.path.join(args.repo or ".", ".repopilot.json")
    if os.path.exists(config_file):
        try:
            import json
            with open(config_file, "r", encoding="utf-8") as f:
                config_data = json.load(f)
                for k, v in config_data.items():
                    if hasattr(args, k) and getattr(args, k) in (None, "anthropic", "claude-sonnet-4-20250514", False):
                        setattr(args, k, v)
        except Exception as e:
            print(f"Warning: Failed to load config file {config_file}: {e}", file=sys.stderr)

    repo_root = os.path.abspath(args.repo)
    if args.list_resumable:
        from modules.run_state import list_resumable
        candidates = list_resumable(args.log_dir)
        if not candidates:
            print("No resumable runs found.")
        for run_id in candidates:
            print(run_id)
        sys.exit(0)

    if args.rollback:
        modifier = CodeModificationEngine(repo_root=repo_root, backup_dir="backups")
        success = modifier.git_stash_pop(repo_root)
        sys.exit(0 if success else 1)
    if not os.path.isdir(repo_root):
        print(f"ERROR: Repository path does not exist: {repo_root}", file=sys.stderr)
        sys.exit(1)

    yes_flag = args.yes or os.environ.get("CI") == "true"

    task = args.task or os.environ.get("AGENT_TASK")
    if task and looks_like_issue_url(task):
        try:
            task = resolve_task(task)
        except TaskResolutionError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            sys.exit(1)
    if not task:
        print("ERROR: Task description is required. Provide --task or set the AGENT_TASK environment variable.", file=sys.stderr)
        sys.exit(1)

    config = AgentConfig(
        repo_root=repo_root,
        task=task,
        dry_run=args.dry_run,
        yes=yes_flag,
        interactive=args.interactive,

        # Execution
        test_runner=args.runner,
        test_args=args.runner_args,
        run_file=args.run_file,
        run_file_runner=args.run_file_runner,
        lint_runner=args.lint_runner,
        lint_args=args.lint_args,
        timeout_seconds=args.timeout,

        # Loop
        max_iterations=args.max_iter,

        # Git
        git_enabled=not args.no_git,
        git_branch_prefix=args.branch_prefix,
        git_base_branch=args.base_branch,
        git_push=args.push,
        git_create_pr=args.pr,

        # Dirs
        backup_dir=args.backup_dir,
        log_dir=args.log_dir,

        # Context
        force_include_paths=args.include,

        # LLM
        anthropic_api_key=args.api_key,
        resume_from=args.resume_from,
    )

    try:
        agent = AutonomousAgent(config)
        result = agent.run()

        print(f"\n{'='*60}")
        print(f"OUTCOME   : {result.outcome.upper()}")
        print(f"RUN ID    : {result.run_id}")
        print(f"ITERATIONS: {result.iterations_used}")
        if result.branch_name:
            print(f"BRANCH    : {result.branch_name}")
        if result.pr_url:
            print(f"PR URL    : {result.pr_url}")
        print(f"MESSAGE   : {result.final_message}")
        print(f"{'='*60}")

        notify_run_complete(result, task)

        write_github_output({
            "outcome": result.outcome,
            "run_id": result.run_id,
            "iterations": str(result.iterations_used),
            "branch_name": result.branch_name or "",
            "pr_url": result.pr_url or "",
            "final_message": result.final_message,
        })

        sys.exit(0 if result.outcome == "success" else 1)

    except KeyboardInterrupt:
        # AutonomousAgent.run() catches KeyboardInterrupt raised inside the loop
        # and rolls back before returning. Reaching here means the interrupt
        # landed outside it -- during setup, or on a second Ctrl+C -- so no
        # files have been applied by the agent.
        print("\nInterrupted by user.", file=sys.stderr)
        write_github_output({
            "outcome": "aborted",
            "run_id": "",
            "iterations": "0",
            "branch_name": "",
            "pr_url": "",
            "final_message": "Interrupted by user before any changes were applied.",
        })
        sys.exit(130)
    except Exception as e:
        print(f"\nFATAL ERROR: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        write_github_output({
            "outcome": "error",
            "run_id": "",
            "iterations": "0",
            "branch_name": "",
            "pr_url": "",
            "final_message": str(e),
        })
        sys.exit(1)


if __name__ == "__main__":
    main()
