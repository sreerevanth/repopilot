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
    parser.add_argument("--task", required=True, help="High-level task description")

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
    parser.add_argument("--rollback",action="store_true",help="Undo the last agent run by popping the git stash.")

    # API key
    parser.add_argument("--api-key", default=None,
                        help="Anthropic API key (default: ANTHROPIC_API_KEY env var)")

    return parser.parse_args()


def main():
    args = parse_args()

    repo_root = os.path.abspath(args.repo)
    if args.rollback:
    modifier = CodeModificationEngine(
        repo_root=repo_root,
        backup_dir="backups"
    )
    success = modifier.git_stash_pop(repo_root)
    sys.exit(0 if success else 1)
    if not os.path.isdir(repo_root):
        print(f"ERROR: Repository path does not exist: {repo_root}", file=sys.stderr)
        sys.exit(1)

    config = AgentConfig(
        repo_root=repo_root,
        task=args.task,
        dry_run=args.dry_run,

        # Execution
        test_runner=args.runner,
        test_args=args.runner_args,
        run_file=args.run_file,
        run_file_runner=args.run_file_runner,
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

        sys.exit(0 if result.outcome == "success" else 1)

    except KeyboardInterrupt:
        print("\nInterrupted by user.", file=sys.stderr)
        sys.exit(130)
    except Exception as e:
        print(f"\nFATAL ERROR: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
