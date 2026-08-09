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

from modules.updater import run_update
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

    parser.add_argument("--repo", required=False,
                        help="Path to the git repository (not needed with --update)")
    parser.add_argument("--task", required=False, help="High-level task description, or a GitHub issue URL to pull one from (can also be provided via AGENT_TASK env var)")
    parser.add_argument("--tasks", nargs="+", default=None, metavar="TASK",
                        help="Run several tasks concurrently, each in its own git "
                             "worktree on its own branch. Implies isolation, so "
                             "tasks may touch the same files.")
    parser.add_argument("--max-parallel-tasks", type=int, default=4, metavar="N",
                        help="How many tasks to run at once (default: 4)")

    # Execution
    parser.add_argument("--runner", default="pytest",
                        choices=["pytest", "npm_test", "go", "cargo", "ruby", "rspec", "bash", "make"],
                        help="Test runner to use (default: pytest)")
    parser.add_argument("--runner-args", nargs="*", default=None,
                        help="Extra arguments to pass to the test runner")
    parser.add_argument("--skip-tests", action="store_true",
                        help="Do not run a test suite. Changes are accepted on the "
                             "model's confidence alone - suitable for refactors and "
                             "comment passes, not for behaviour changes.")
    parser.add_argument("--run-file", default=None,
                        help="Run a specific file instead of the test suite")
    parser.add_argument("--run-file-runner", default="python",
                        choices=["python", "node", "ruby", "bash"],
                        help="Runner for --run-file (default: python)")
    parser.add_argument("--coverage", action="store_true",
                        help="Measure test coverage and fail an iteration that "
                             "lowers it. Requires pytest-cov.")
    parser.add_argument("--coverage-source", default=".", metavar="PATH",
                        help="What --cov points at (default: .)")
    parser.add_argument("--no-pre-commit", dest="run_pre_commit",
                        action="store_false", default=True,
                        help="Skip the repository's pre-commit hooks even when "
                             ".pre-commit-config.yaml is present.")
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
    parser.add_argument("--plan-first", action="store_true",
                        help="Ask the model for an approach before it writes code. "
                             "Costs one extra API call on the first iteration.")
    parser.add_argument("--max-iter", type=int, default=5,
                        help="Maximum number of agent iterations (default: 5)")

    # Parallel Processing
    parser.add_argument("--parallel", action="store_true",
                        help="Enable parallel file processing (ingestion and modification)")
    parser.add_argument("--workers", type=int, default=10,
                        help="Number of worker threads for parallel processing (default: 10)")

    # Git
    parser.add_argument("--no-commit", action="store_true",
                        help="Stage successful changes but do not commit them. "
                             "Leaves them in the index for you to review and "
                             "commit yourself.")
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
    parser.add_argument("--context-budget", type=int, default=None, metavar="CHARS",
                        help="Characters of repository context to send. Derived "
                             "from the model's context window when not set.")
    parser.add_argument("--project-rules", dest="project_rules_file",
                        default=".agentcontext", metavar="PATH",
                        help="Per-repo rules file, read from the repo root and "
                             "prepended to every prompt. Pass an empty string to "
                             "disable.")
    parser.add_argument("--include", nargs="*", default=None,
                        help="Force-include specific file paths in context (relative to repo root)")

    # Output
    parser.add_argument("--log-dir", default="logs",
                        help="Directory for log files (default: logs/ inside repo)")
    parser.add_argument("--backup-dir", default="backups",
                        help="Directory for file backups (default: backups/ inside repo)")
    parser.add_argument("--verbose", dest="verbose_payloads", action="store_true",
                        help="Print the exact prompt sent to the LLM and the raw "
                             "response, to stderr. Known secret patterns are masked.")
    parser.add_argument("--quiet", action="store_true",
                        help="Suppress verbose output")
    parser.add_argument("--context-only", action="store_true",
                        help="Print the compiled LLM context and exit, before any "
                             "API call is made. Use it to check which files were "
                             "selected without spending credits.")
    parser.add_argument("--dry-run", "-d", action="store_true",help="Preview changes without applying them. Saves a manifest to logs/.")
    parser.add_argument("--update", action="store_true",
                        help="Fast-forward RepoPilot's own checkout to the latest "
                             "upstream commit. Refuses on a dirty tree or a diverged "
                             "branch, and never installs packages for you.")
    parser.add_argument("--resume", dest="resume_from", default=None, metavar="RUN_ID",
                        help="Continue an interrupted run from its last completed "
                             "iteration. Use --list-resumable to see candidates.")
    parser.add_argument("--list-resumable", action="store_true",
                        help="List run ids that can be resumed, then exit.")
    parser.add_argument("--rollback",action="store_true",help="Undo the last agent run by popping the git stash.")

    # API key
    parser.add_argument("--system-prompt", dest="system_prompt_file", default=None,
                        metavar="PATH",
                        help="Replace the default system prompt with the contents of "
                             "a file. The JSON output contract must be preserved or "
                             "responses will not parse.")
    parser.add_argument("--api-key", default=None,
                        help="API key for the LLM provider (default: provider-specific env var)")
    parser.add_argument("--model", default=None,
                        help="LLM model name (default: provider-specific, or AGENT_MODEL env var)")
    parser.add_argument("--provider", default="anthropic",
                        choices=["anthropic", "openai", "gemini", "ollama"],
                        help="LLM provider to use (default: anthropic)")
    parser.add_argument("--api-base-url", default=None,
                        help="Custom API base URL (for Ollama or self-hosted endpoints)")


    # Config file support
    parser.add_argument("--config", default=None,
                        help="Path to configuration JSON file (default: .repopilot.json in repo root)")

    # Non-interactive / CI
    parser.add_argument("--max-cost", type=float, default=None, metavar="USD",
                        help="Stop before the next LLM call once this much has been "
                             "spent (e.g. --max-cost 1.00). Off by default.")
    parser.add_argument("--yes", "-y", action="store_true",
                        help="Auto-approve file changes (bypass confirmation prompt)")
    parser.add_argument("--interactive", "-i", action="store_true",
                        help="After tests pass, show the diff and confirm before "
                             "committing. Ignored when --yes is set or CI=true.")


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


def _run_task_batch(args) -> int:
    """
    Run several tasks concurrently, each in an isolated worktree.

    Kept apart from main() because the single-task path builds one AgentConfig
    and reports one result; this one builds N and reports a table.
    """
    from modules.agent_loop import AgentConfig, AutonomousAgent
    from modules.parallel_tasks import WorktreeError, render_summary, run_tasks

    if not args.repo:
        print("ERROR: --repo is required with --tasks.", file=sys.stderr)
        return 2

    repo_root = os.path.abspath(args.repo)

    def run_one(task: str, worktree: str, branch: str):
        # Each agent is pointed at its own checkout. Git is disabled inside the
        # run because the worktree is already on the branch this task owns --
        # letting the agent create another would nest branches per task.
        config = AgentConfig(
            repo_root=worktree,
            task=task,
            git_enabled=False,
        no_commit=args.no_commit,
            yes=True,
            max_iterations=args.max_iter,
            test_runner=args.runner,
            timeout_seconds=args.timeout,
            anthropic_api_key=args.api_key,
            model=args.model,
            provider=args.provider,
        )
        return AutonomousAgent(config).run()

    try:
        outcomes = run_tasks(
            repo_root,
            args.tasks,
            run_one,
            max_workers=args.max_parallel_tasks,
            branch_prefix=args.branch_prefix,
            base_branch=args.base_branch,
        )
    except WorktreeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print(render_summary(outcomes))
    return 0 if all(o.ok for o in outcomes) else 1


def main():
    args = parse_args()

    if args.tasks:
        sys.exit(_run_task_batch(args))

    if args.update:
        sys.exit(run_update(assume_yes=args.yes))

    if not args.repo:
        print(
            "ERROR: --repo is required (except with --update).", file=sys.stderr
        )
        sys.exit(2)

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
        context_only=args.context_only,
        yes=yes_flag,
        interactive=args.interactive,

        # Execution
        plan_first=args.plan_first,
        test_runner=args.runner,
        test_args=args.runner_args,
        skip_tests=args.skip_tests,
        run_file=args.run_file,
        run_file_runner=args.run_file_runner,
        coverage=args.coverage,
        coverage_source=args.coverage_source,
        run_pre_commit=args.run_pre_commit,
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
        resume_from=args.resume_from,

        # CLI behavior

        # Context
        force_include_paths=args.include,
        project_rules_file=args.project_rules_file,
        context_budget=args.context_budget,

        # LLM
        anthropic_api_key=args.api_key,
        max_cost=args.max_cost,
        system_prompt_file=args.system_prompt_file,
        api_base_url=args.api_base_url,
        verbose_payloads=args.verbose_payloads,
        quiet=args.quiet,
        model=args.model,
        provider=args.provider,
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
