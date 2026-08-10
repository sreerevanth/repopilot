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
import json
import os
import sys
import uuid

# Ensure the project root is on the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from modules.errors import AgentError
from modules.updater import run_update
from modules.agent_loop import AutonomousAgent, AgentConfig
from modules.notify import notify_run_complete
from modules.code_modifier import CodeModificationEngine
from modules.task_source import (
    TaskResolutionError,
    looks_like_issue_url,
    resolve_task,
)


def parse_args() -> tuple[argparse.Namespace, argparse.ArgumentParser]:
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
    parser.add_argument("--max-parallel-tasks", type=positive_int, default=4, metavar="N",
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
    parser.add_argument("--timeout", type=positive_int, default=120,
                        help="Execution timeout in seconds (default: 120)")

    # Loop control
    parser.add_argument("--plan-first", action="store_true",
                        help="Ask the model for an approach before it writes code. "
                             "Costs one extra API call on the first iteration.")
    parser.add_argument("--max-iter", type=positive_int, default=5,
                        help="Maximum number of agent iterations (default: 5)")

    # Parallel Processing
    parser.add_argument("--parallel", action="store_true",
                        help="Enable parallel file processing (ingestion and modification)")
    parser.add_argument("--workers", type=positive_int, default=10,
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
    parser.add_argument("--describe-pr", action="store_true",
                        help="With --pr, ask the model to write the PR title and "
                             "body from the diff. Costs one extra call; falls back "
                             "to the standard template if it fails.")
    parser.add_argument("--base-branch", default="main",
                        help="Base branch for PR and branch creation (default: main)")
    parser.add_argument("--branch-prefix", default="agent",
                        help="Prefix for the auto-created branch name (default: agent)")

    # Context
    parser.add_argument("--context-budget", type=positive_int, default=None, metavar="CHARS",
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
    parser.add_argument("--undo", nargs="?", const="__last__", default=None,
                        metavar="RUN_ID",
                        help="Undo a previous run: leave its branch and delete "
                             "it. Defaults to the most recent run. Use "
                             "--list-runs to see them.")
    parser.add_argument("--list-runs", action="store_true",
                        help="List runs that --undo could remove, then exit.")
    parser.add_argument("--clean", action="store_true",
                        help="Remove this tool's old logs and backups, then exit. "
                             "Only files it created are touched.")
    parser.add_argument("--clean-older-than", type=non_negative_float, default=None,
                        metavar="DAYS",
                        help="With --clean, keep anything newer than this.")
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
    parser.add_argument("--fallback-provider",
                        choices=["anthropic", "openai", "gemini", "ollama"],
                        default=None,
                        help="Try this provider when the primary fails with a "
                             "transient error such as a 500 or an overload.")
    parser.add_argument("--fallback-api-key", default=None,
                        help="Key for the fallback provider (defaults to --api-key).")
    parser.add_argument("--provider", default="anthropic",
                        choices=["anthropic", "openai", "gemini", "ollama"],
                        help="LLM provider to use (default: anthropic)")
    parser.add_argument("--api-base-url", default=None,
                        help="Custom API base URL (for Ollama or self-hosted endpoints)")


    # Config file support
    parser.add_argument("--config", default=None,
                        help="Path to configuration JSON file (default: .repopilot.json in repo root)")

    # Non-interactive / CI
    parser.add_argument("--cache", dest="use_cache", action="store_true",
                        help="Reuse a stored response when the model, system "
                             "prompt and full prompt are identical. Off by "
                             "default: requests are not deterministic, so this "
                             "changes behaviour as well as saving money.")
    parser.add_argument("--max-cost", type=positive_float, default=None, metavar="USD",
                        help="Stop before the next LLM call once this much has been "
                             "spent (e.g. --max-cost 1.00). Off by default.")
    parser.add_argument("--yes", "-y", action="store_true",
                        help="Auto-approve file changes (bypass confirmation prompt)")
    parser.add_argument("--step", action="store_true",
                        help="Pause after each iteration so you can inspect the "
                             "working tree before the next one. Needs a terminal; "
                             "ignored when output is piped or --yes is set.")
    parser.add_argument("--interactive", "-i", action="store_true",
                        help="After tests pass, show the diff and confirm before "
                             "committing. Ignored when --yes is set or CI=true.")


    # The parser comes back too. Merging a config file needs to know what each
    # flag's default actually is and what type it declares, and neither is
    # recoverable from the Namespace alone.
    return parser.parse_args(), parser


def write_github_output(outputs: dict[str, str]):
    github_output_path = os.environ.get("GITHUB_OUTPUT")
    if github_output_path:
        try:
            with open(github_output_path, "a", encoding="utf-8") as f:
                for k, v in outputs.items():
                    v = str(v)
                    if "\n" in v:
                        # A random delimiter per value, which is what GitHub
                        # documents for multiline outputs.
                        #
                        # The fixed "EOF" it replaced could appear inside the
                        # value: final_message is set from str(exception), so a
                        # message containing a line that is exactly EOF closed
                        # the heredoc early and everything after it was parsed
                        # by the runner as further key=value outputs. That let
                        # a crafted message set pr_url and outcome to anything,
                        # which the workflow then posts to the issue as "a Pull
                        # Request has been created".
                        delimiter = f"ghadelimiter_{uuid.uuid4()}"
                        if delimiter in v:
                            raise ValueError(
                                f"value for {k} contains the generated delimiter"
                            )
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
        fallback_provider=args.fallback_provider,
        fallback_api_key=args.fallback_api_key,
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


def _report_expected_error(error) -> None:
    """
    Report an anticipated failure and exit.

    Shared so every path reports identically. Raised in review of #260: task
    resolution caught TaskResolutionError separately, so a bad issue URL
    printed a bare message, skipped any remedy, and wrote no GITHUB_OUTPUT --
    a CI run saw no `outcome=error` at all.
    """
    message = error.user_message() if isinstance(error, AgentError) else str(error)
    print(f"\nERROR: {message}", file=sys.stderr)
    write_github_output({
        "outcome": "error",
        "run_id": "",
        "iterations": "0",
        "branch_name": "",
        "pr_url": "",
        "final_message": message,
    })
    sys.exit(1)


def positive_int(text: str) -> int:
    """
    An integer of 1 or more.

    Used on flags where zero or negative is not a weaker setting but a broken
    one. `--workers 0` previously reached ThreadPoolExecutor and surfaced as an
    unhandled `ValueError: max_workers must be greater than 0` from inside
    concurrent.futures, naming a parameter the user never typed.

    Raising from `type=` rather than checking after parsing is deliberate:
    argparse prints the flag name alongside the message, which is exactly the
    information those failures were missing.

    These also apply to values coming from a config file, since #291 coerces
    each one through its flag's declared type -- so a negative in a config is
    now reported the same way as a negative on the command line.
    """
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError(f"must be 1 or greater, got {value}")
    return value


def positive_float(text: str) -> float:
    """
    A float greater than zero.

    `--max-cost -1` previously stopped the run before its first request, with
    "Spent $0.0000, which reaches the --max-cost limit of $-1.00" -- a message
    about budgets, for what is an invalid argument. Zero is refused for the
    same reason: a limit of nothing is not a limit, it is a run that cannot
    start.
    """
    value = float(text)
    if value <= 0:
        raise argparse.ArgumentTypeError(f"must be greater than 0, got {value}")
    return value


def non_negative_float(text: str) -> float:
    """
    A float of zero or more.

    Zero is meaningful here -- `--clean-older-than 0` is a reasonable way to
    say "everything" -- so only negatives are refused.
    """
    value = float(text)
    if value < 0:
        raise argparse.ArgumentTypeError(f"must be 0 or greater, got {value}")
    return value


def apply_config_file(
    path: str,
    args: argparse.Namespace,
    parser: argparse.ArgumentParser,
) -> None:
    """
    Merge a config file into the parsed arguments.

    Precedence is defaults, then the config file, then the command line. That
    is the order every other tool uses, and the previous implementation got it
    wrong in both directions.

    It guarded each key with

        getattr(args, k) in (None, "anthropic", "claude-sonnet-4-20250514", False)

    trying to express "apply this only where the user did not pass a flag".
    argparse does not record whether a value came from the command line or from
    `default=`, so that guard approximated the question by comparing against
    four literals -- and the approximation failed both ways.

    Fourteen settings could never be configured, because their defaults are not
    in that tuple: timeout, max_iter, workers, runner, base_branch, log_dir and
    others. A config file setting them did nothing, silently.

    And a config value could beat an explicit flag: "anthropic" is in the tuple,
    so `--provider anthropic --config c.json` with `{"provider": "openai"}` ran
    against OpenAI. The file won over the command line.

    Comparing against the parser's own defaults answers the question the old
    guard was approximating, for every flag, without a list of literals to keep
    in step.
    """
    try:
        with open(path, "r", encoding="utf-8") as handle:
            config_data = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        print(f"Warning: failed to load config file {path}: {exc}", file=sys.stderr)
        return

    if not isinstance(config_data, dict):
        print(
            f"Warning: config file {path} must contain a JSON object, "
            f"found {type(config_data).__name__}; ignoring it",
            file=sys.stderr,
        )
        return

    actions = {action.dest: action for action in parser._actions if action.dest != "help"}
    defaults = {dest: action.default for dest, action in actions.items()}

    for key, value in config_data.items():
        action = actions.get(key)

        # An unknown key was previously dropped in silence, so a typo was
        # indistinguishable from a setting the build does not support -- both
        # produced no output at all. A warning rather than an error, since a
        # config shared across versions may carry a key an older build predates.
        if action is None:
            print(
                f"Warning: config file key '{key}' matches no option, ignoring",
                file=sys.stderr,
            )
            continue

        # The command line wins. A value differing from the parser's default is
        # one the user supplied, which is the question the old literal tuple was
        # trying and failing to answer.
        if getattr(args, key, None) != defaults[key]:
            continue

        # Coerce through the flag's own type. Without this a string reaches a
        # field the loop uses as an integer, and the failure surfaces as a
        # TypeError several hundred lines away with no mention of the file or
        # the key that caused it.
        if action.type is not None and value is not None:
            try:
                value = action.type(value)
            except (TypeError, ValueError, argparse.ArgumentTypeError):
                # ArgumentTypeError is included deliberately: the validators
                # added for #280 raise it rather than ValueError, and it
                # inherits from Exception rather than from either of the other
                # two. Without it a negative in a config file escaped as a
                # traceback instead of the warning every other bad value gets.
                print(
                    f"Warning: config file key '{key}' should be "
                    f"{getattr(action.type, '__name__', action.type)}, "
                    f"found {value!r}; ignoring it",
                    file=sys.stderr,
                )
                continue

        setattr(args, key, value)


def main():
    args, parser = parse_args()

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
        apply_config_file(config_file, args, parser)


    repo_root = os.path.abspath(args.repo)
    if args.list_resumable:
        from modules.run_state import list_resumable
        candidates = list_resumable(args.log_dir)
        if not candidates:
            print("No resumable runs found.")
        for run_id in candidates:
            print(run_id)
        sys.exit(0)

    if args.clean:
        from modules.housekeeping import clean, render_clean_summary

        if not args.repo:
            print("ERROR: --repo is required with --clean.", file=sys.stderr)
            sys.exit(2)

        results = clean(
            os.path.abspath(args.repo),
            log_dir=args.log_dir,
            backup_dir=args.backup_dir,
            older_than_days=args.clean_older_than,
            dry_run=args.dry_run,
        )
        print(render_clean_summary(results, dry_run=args.dry_run))
        sys.exit(0)

    if args.list_runs:
        from modules.undo import describe, find_runs

        runs = find_runs(os.path.join(repo_root, args.log_dir))
        if not runs:
            print("No runs found.")
        for run in runs[:20]:
            print(describe(run))
            print()
        sys.exit(0)

    if args.undo:
        from modules.dry_run import ask_confirmation
        from modules.git_integration import GitIntegration
        from modules.undo import UndoError, describe, find_run, undo_run

        try:
            run = find_run(
                os.path.join(repo_root, args.log_dir),
                None if args.undo == "__last__" else args.undo,
            )
        except UndoError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            sys.exit(1)

        print("\nThis will undo:\n")
        print(describe(run))
        print()

        # Asked before removing anything. --yes skips it, matching every other
        # confirmation, so CI is never left waiting on stdin.
        # Computed here rather than read from the later yes_flag, which is
        # defined below this block -- the run-configuration path has not been
        # reached yet at this point.
        skip_prompt = args.yes or os.environ.get("CI") == "true"
        if not skip_prompt and not ask_confirmation("Delete this branch?"):
            print("Left alone.")
            sys.exit(0)

        try:
            done = undo_run(
                GitIntegration(repo_root), run,
                base_branch=args.base_branch,
                branch_prefix=args.branch_prefix,
            )
        except UndoError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            sys.exit(1)

        print("Undone: " + "; ".join(done))
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
            _report_expected_error(exc)
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
        step=args.step,

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
        describe_pr=args.describe_pr,

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
        use_cache=args.use_cache,
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
    except AgentError as e:
        # A condition the agent anticipated -- a missing runner, a budget
        # reached, an unreadable prompt file. The user needs the message and a
        # remedy, not a traceback into code they did not write.
        _report_expected_error(e)
    except Exception as e:
        # Anything else means this tool has a bug. The traceback stays, because
        # that is the part worth putting in a report.
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
