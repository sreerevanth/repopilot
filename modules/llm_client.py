"""
Module 3: LLM Interaction Layer
Clean prompt engineering, structured I/O, token budget handling,
iterative refinement support. Supports Anthropic, OpenAI, Gemini, and Ollama.
"""

import json
import logging
import os
import re
import ssl
import sys
import urllib.error
import urllib.request
import time
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Any
from modules.errors import ConfigurationError, ProviderError

_LOG = logging.getLogger("agent.llm_client")

# ─────────────────────────────────────────────
# Provider availability checks
# ─────────────────────────────────────────────

try:
    import anthropic
    _ANTHROPIC_AVAILABLE = True
except ImportError:
    _ANTHROPIC_AVAILABLE = False

DEFAULT_MODEL = "claude-sonnet-4-20250514"

# Kept as an alias: the constant was renamed to DEFAULT_MODEL but three call
# sites below still reference MODEL. Aliasing is safer than renaming them,
# since anything importing MODEL from this module keeps working.
MODEL = DEFAULT_MODEL
MAX_TOKENS = 8192

# Price per million tokens (input_price, output_price)
MODEL_PRICING = {
    "claude-sonnet-4-20250514": (3.00, 15.00),
    "claude-3-5-sonnet-20241022": (3.00, 15.00),
    "gpt-4o": (5.00, 15.00),
    "gemini-1.5-pro": (1.25, 5.00),
    "ollama": (0.00, 0.00),
}


# ─────────────────────────────────────────────
# Prompt Templates
# ─────────────────────────────────────────────

# ─────────────────────────────────────────────
# Prompt loading
# ─────────────────────────────────────────────
#
# Prompts live in prompts/*.txt so they can be edited, diffed and swapped
# without touching Python. The values below are the fallback: if a file is
# missing or unreadable the built-in text is used, so a bad checkout degrades
# to today's behaviour instead of leaving the agent with no prompt at all.
#
# Point REPOPILOT_PROMPT_DIR at another directory to try an alternative set.

PROMPT_DIR_ENV_VAR = "REPOPILOT_PROMPT_DIR"
DEFAULT_PROMPT_DIR = Path(__file__).resolve().parents[1] / "prompts"


def prompt_dir() -> Path:
    override = os.environ.get(PROMPT_DIR_ENV_VAR)
    return Path(override) if override else DEFAULT_PROMPT_DIR


def load_prompt(name: str, fallback: str) -> str:
    """Read prompts/<name>.txt, falling back to the built-in text."""
    path = prompt_dir() / f"{name}.txt"
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        _LOG.debug("using built-in '%s' prompt (%s)", name, exc)
        return fallback

    if not text.strip():
        _LOG.warning("%s is empty; using the built-in prompt instead", path)
        return fallback
    return text


_BUILTIN_SYSTEM_PROMPT = """You are an expert software engineer embedded in an autonomous code modification pipeline.
You receive:
1. A task description
2. Relevant source files from the repository
3. (Optionally) error output from a previous execution attempt

Your job is to produce ONLY the file changes required to complete the task.

OUTPUT FORMAT (STRICT — machine-parsed):
Return a JSON object with this exact schema:

{
  "analysis": "<brief explanation of what you understood and what changes are needed>",
  "changes": [
    {
      "path": "<relative file path from repo root>",
      "action": "modify" | "create" | "delete" | "rename",
      "content": "<full new file content (for modify/create)>",
      "new_path": "<destination path, ONLY for rename>",
      "explanation": "<why this change>"
    }
  ],
  "confidence": <0.0-1.0 float>,
  "lookups": ["<https doc URL you need before answering>", ...],
  "done": <true if you believe the task is complete, false if more iterations needed>
}

RULES:
- For "modify" and "create": always provide the COMPLETE file content, not diffs or snippets.
- For "patch": content should be a valid unified diff format.
- For "delete": omit "content".
- To move or rename a file, use "rename" with "new_path" — do NOT emit a "create"
  at the new path plus a "delete" at the old one. Include "content" with a rename
  only if the file's contents also change; omit it to move the file unchanged.
- Do NOT include markdown fences, explanation text, or anything outside the JSON object.
- Paths must be relative (e.g., "src/utils.py"), never absolute.
- If you cannot determine a fix, set confidence < 0.3 and done=false with a clear analysis.
- Preserve existing code style, indentation, and conventions.
- If a task depends on a third-party API you are unsure of, put documentation
  URLs in "lookups" and set done=false rather than guessing at method names.
  Only https URLs on documentation hosts are fetched; anything else is refused
  and the reason is returned to you.
"""

_BUILTIN_TASK_PROMPT = """\
## Task
{task}

## Repository Context
{context}

## Instructions
Analyze the code and produce the minimal changes needed to complete the task.
Return ONLY the JSON object specified in the system prompt.
"""

# The planning pass deliberately forbids code. Asking for a plan and changes in
# one response reliably produces changes with a plan-shaped preamble -- the
# model commits to an approach and then justifies it, which is the opposite of
# planning. A separate call with no code in the schema keeps the two apart.
_BUILTIN_PLAN_PROMPT = """\
## Task
{task}

## Repository Context
{context}

## Instructions
Do NOT write any code yet. Work out how you would approach this task.

Return ONLY a JSON object with this schema:

{{
  "plan": ["<step>", "<step>", ...],
  "files_to_change": ["<relative path>", ...],
  "risks": ["<what could go wrong or is unclear>", ...],
  "confidence": <0.0-1.0 float>
}}

Keep the plan to at most 6 steps. If the task is unclear or the context is
insufficient, say so in "risks" and set confidence below 0.4.
"""

# The planning pass deliberately forbids code. Asking for a plan and changes in
# one response produces changes with a plan-shaped preamble -- the model commits
# to an approach and then justifies it, which is the opposite of planning.
_BUILTIN_PLAN_PROMPT = """\
## Task
{task}

## Repository Context
{context}

## Instructions
Do NOT write any code yet. Work out how you would approach this task.

Return ONLY a JSON object with this schema:

{{
  "plan": ["<step>", "<step>", ...],
  "files_to_change": ["<relative path>", ...],
  "risks": ["<what could go wrong or is unclear>", ...],
  "confidence": <0.0-1.0 float>
}}

Keep the plan to at most 6 steps. If the task is unclear or the context is
insufficient, say so in "risks" and set confidence below 0.4.
"""

_BUILTIN_RETRY_PROMPT = """\
## Task
{task}

## Previous Changes Applied
The following files were modified in the previous iteration:
{previous_changes_summary}

## Execution Result (FAILED)
Exit code: {exit_code}

### stdout:
{stdout}

### stderr:
{stderr}

## Current Repository Context
{context}

## Instructions
The previous attempt failed. Analyze the error output carefully and produce corrected changes.
Focus on the root cause of the failure. Return ONLY the JSON object.
"""


# ─────────────────────────────────────────────
# Response types
# ─────────────────────────────────────────────

@dataclass
class FileChange:
    path: str
    action: str      # "modify" | "create" | "delete" | "rename"
    content: str     # full file content
    explanation: str
    new_path: Optional[str] = None   # destination, for "rename"


@dataclass
class Plan:
    raw: str
    steps: list[str]
    files_to_change: list[str]
    risks: list[str]
    confidence: float
    parse_error: Optional[str] = None

    @property
    def usable(self) -> bool:
        """A plan with no steps is not worth putting in the next prompt."""
        return bool(self.steps) and not self.parse_error

    def render(self) -> str:
        """The plan as it appears in the execution prompt."""
        lines = ["## Your Plan", ""]
        lines += [f"{i}. {step}" for i, step in enumerate(self.steps, 1)]
        if self.files_to_change:
            joined = ", ".join(self.files_to_change)
            lines += ["", f"Files you expect to change: {joined}"]
        if self.risks:
            lines += ["", "Risks you identified:"]
            lines += [f"- {risk}" for risk in self.risks]
        return "\n".join(lines)


@dataclass
class LLMResponse:
    raw: str
    analysis: str
    changes: list[FileChange]
    confidence: float
    done: bool
    parse_error: Optional[str] = None
    input_tokens: int = 0
    output_tokens: int = 0
    estimated_cost: float = 0.0
    # _parse_response populates this (the doc-lookup feature). The field was
    # lost in a merge while the parser kept passing it, so every call raised
    # TypeError at runtime -- past the point an import check would notice.
    lookups: list = field(default_factory=list)


SYSTEM_PROMPT = load_prompt("system", _BUILTIN_SYSTEM_PROMPT)
TASK_PROMPT_TEMPLATE = load_prompt("initial", _BUILTIN_TASK_PROMPT)
# How much of the diff to send. Enough to describe the change, bounded because
# this is an extra paid call on top of the run that produced the diff.
PR_DIFF_CHARS = 12_000

_BUILTIN_PR_PROMPT = """\
A code change has been made to satisfy this task:

{task}

Here is the diff:

{diff}

Write a pull request title and description for it.

Return ONLY a JSON object:

{{
  "title": "<one line, imperative, under 72 characters>",
  "body": "<markdown; what changed and why, not a restatement of the task>"
}}

Describe what the diff actually does. If it does less than the task asked for,
say so -- a description that oversells the change is worse than a plain one.
"""


RETRY_PROMPT_TEMPLATE = load_prompt("retry", _BUILTIN_RETRY_PROMPT)
PR_PROMPT_TEMPLATE = load_prompt("pr", _BUILTIN_PR_PROMPT)
PLAN_PROMPT_TEMPLATE = load_prompt("plan", _BUILTIN_PLAN_PROMPT)
PLAN_PROMPT_TEMPLATE = load_prompt("plan", _BUILTIN_PLAN_PROMPT)


# ─────────────────────────────────────────────
# Clients
# ─────────────────────────────────────────────

def _emit_progress(chars: int) -> None:
    """Overwrite one line on the terminal. No-op when stderr is redirected."""
    if not sys.stderr.isatty():
        return
    sys.stderr.write(f"\r  receiving response... {chars:,} chars")
    sys.stderr.flush()


def _end_progress(chars: int) -> None:
    if not sys.stderr.isatty():
        return
    sys.stderr.write(f"\r  response received ({chars:,} chars)\n")
    sys.stderr.flush()


def _redact_secrets(text: str) -> str:
    """
    Mask anything matching a known secret pattern before it is printed.

    --verbose dumps the full prompt, which is repository file contents. If a
    key is committed somewhere in the repo it would otherwise be echoed to the
    terminal and into whatever captures that output. Reuses the patterns from
    secret_scanner so the two cannot drift apart.
    """
    from modules.secret_scanner import SECRET_PATTERNS

    for pattern in SECRET_PATTERNS:
        text = re.sub(
            pattern["regex"],
            lambda m: m.group(0)[:4] + "[REDACTED]",
            text,
            flags=re.IGNORECASE,
        )
    return text


def _dump_payload(label: str, body: str, redact: bool = True) -> None:
    """Print one labelled payload to stderr, so stdout stays parseable."""
    if redact:
        body = _redact_secrets(body)
    rule = "=" * 72
    print(f"\n{rule}\n[verbose] {label} ({len(body):,} chars)\n{rule}",
          file=sys.stderr)
    print(body, file=sys.stderr)
    print(rule, file=sys.stderr, flush=True)


REQUIRED_SCHEMA_KEYS = ("analysis", "changes", "confidence", "done")


class SystemPromptError(ConfigurationError, ValueError):
    """Raised when a --system-prompt file cannot be used."""


def load_system_prompt(path: str) -> str:
    """
    Read a replacement system prompt from disk.

    Warns rather than refuses when the schema keys are missing: the flag exists
    to let people experiment with the persona, and a hard rejection would block
    a legitimate rewrite that phrases the contract differently. The warning is
    there because the failure it predicts is otherwise baffling -- every
    iteration fails to parse and nothing says why.
    """
    try:
        text = Path(path).read_text(encoding="utf-8")
    except OSError as exc:
        raise SystemPromptError(f"Could not read system prompt {path}: {exc}") from exc

    if not text.strip():
        raise SystemPromptError(f"System prompt {path} is empty.")

    missing = [k for k in REQUIRED_SCHEMA_KEYS if k not in text]
    if missing:
        _LOG.warning(
            "Custom system prompt does not mention %s. The parser expects a JSON "
            "object with these keys; without them every iteration will fail to "
            "parse.", ", ".join(missing),
        )
    return text



def _parse_pr_description(raw: str) -> tuple:
    """Pull the title and body out of a response, or (None, None)."""
    text = (raw or "").strip()
    start, end = text.find("{"), text.rfind("}") + 1
    if start == -1 or end == 0:
        return None, None
    try:
        data = json.loads(text[start:end])
    except json.JSONDecodeError:
        return None, None

    title = str(data.get("title") or "").strip()
    body = str(data.get("body") or "").strip()
    if not title or not body:
        return None, None
    return title[:72], body


class BudgetExceededError(ProviderError, RuntimeError):
    """
    Raised when accumulated spend reaches the configured --max-cost.

    agent_loop imports this and catches it before its generic handler, so a
    deliberate stop is reported as "budget_exceeded" rather than "the API
    broke". The class was lost in a merge while its import survived.
    """


# Where each provider is reached when --api-base-url is not given. Ollama is
# the one that matters in practice -- it is self-hosted, so the address is a
# per-user setting rather than a constant.
DEFAULT_BASE_URLS = {
    "ollama": "http://localhost:11434",
    "openai": "https://api.openai.com/v1",
}


class BaseLLMClient:
    """
    Shared behaviour for every provider client.

    The `class BaseLLMClient:` header was lost in a merge, which left this
    body absorbed into a duplicate `class LLMClient:` and every subclass
    inheriting from an undefined name. Restored here along with the two
    methods that went with it.
    """

    def __init__(
        self,
        model: str = MODEL,
        verbose: bool = False,
        max_cost: Optional[float] = None,
        system_prompt: Optional[str] = None,
        cache=None,
    ):
        self.model = model
        self.verbose = verbose
        self.max_cost = max_cost
        # Per-instance so a run can override it; falls back to the module
        # constant, which is itself loaded from prompts/system.txt.
        self.system_prompt = system_prompt or SYSTEM_PROMPT
        # Optional and injected, so this module need not know where a cache
        # lives or whether one exists.
        self.cache = cache
        self.input_tokens_used = 0
        self.output_tokens_used = 0
        self.total_cost = 0.0

    def _check_budget(self) -> None:
        """
        Stop before the next call once --max-cost has been reached.

        Checked before spending rather than after, so the limit is a ceiling on
        what gets spent rather than a report of what already was. Cost is only
        known once a response arrives, so a run can overshoot by at most one
        call -- there is no way to price a request before making it.
        """
        if self.max_cost is None:
            return
        if self.total_cost >= self.max_cost:
            raise BudgetExceededError(
                f"Spent ${self.total_cost:.4f}, which reaches the --max-cost "
                f"limit of ${self.max_cost:.2f}. Stopping before the next call."
            )

    def usage_summary(self) -> str:
        """
        One line describing what this client has spent.

        The budget-exceeded message called `self.llm.usage.summary()`, and no
        `usage` attribute has ever existed -- so a run that correctly stopped at
        its --max-cost limit then crashed while explaining why. The figures were
        all already tracked; nothing was aggregating them.
        """
        return (
            f"{self.input_tokens_used:,} input and {self.output_tokens_used:,} "
            f"output tokens, ${self.total_cost:.4f}"
        )

    def _record_usage(self, input_tok: int, output_tok: int) -> None:
        """Add one call's tokens and cost to the running totals."""
        self.input_tokens_used += input_tok
        self.output_tokens_used += output_tok
        self.total_cost += self._calculate_cost(input_tok, output_tok)

    def _accounted_call(self, prompt: str) -> tuple:
        """
        The only way a paid call should be made.

        Every request goes through here so the budget is checked and the cost
        recorded exactly once. plan_request previously called _call directly,
        which meant --plan-first spent money the budget never saw: the planning
        call was not checked against the limit and its cost never reached
        total_cost, so the next call could exceed the limit by more than one
        request. Funnelling every path through one method is what stops a new
        request type reintroducing that.
        """
        input_tok = self._estimate_tokens(prompt + self.system_prompt)

        # Checked before the budget: a cache hit costs nothing, so a run that
        # has reached its limit can still be served rather than stopping on a
        # request it is not going to pay for.
        key = None
        if self.cache is not None:
            from modules.response_cache import cache_key

            key = cache_key(self.model, self.system_prompt, prompt)
            cached = self.cache.get(key)
            if cached is not None:
                if self.verbose:
                    _dump_payload("response (cached)", cached)
                return cached, input_tok

        self._check_budget()
        if self.verbose:
            _dump_payload("system prompt", self.system_prompt)
            _dump_payload("request", prompt)

        raw = self._call(prompt)

        if self.verbose:
            _dump_payload("response", raw)
        self._record_usage(input_tok, self._estimate_tokens(raw))
        if key is not None:
            self.cache.put(key, raw)
        return raw, input_tok

    def _estimate_tokens(self, text: str) -> int:
        """
        Rough token count from character length.

        Deliberately an estimate: it feeds cost reporting, not a hard budget,
        and a real tokenizer would mean a dependency per provider.
        """
        return max(1, len(text or "") // 4)

    def _calculate_cost(self, input_tok: int, output_tok: int) -> float:
        pricing = MODEL_PRICING.get(self.model, (0.0, 0.0))
        cost = (input_tok / 1_000_000 * pricing[0]) + (output_tok / 1_000_000 * pricing[1])
        return cost

    def _call(self, prompt: str) -> str:
        raise NotImplementedError("Subclasses must implement _call")

    def _parse_response(self, raw: str, input_tok: int) -> LLMResponse:
        """Extract and parse JSON from LLM output."""
        text = raw.strip()
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.MULTILINE)
        text = re.sub(r"```\s*$", "", text, flags=re.MULTILINE)
        text = text.strip()

        start = text.find("{")
        end = text.rfind("}") + 1
        
        # Usage is recorded by _accounted_call, which every request path goes
        # through. Adding it again here would double-count every call.
        output_tok = self._estimate_tokens(raw)
        cost = self._calculate_cost(input_tok, output_tok)

        if start == -1 or end == 0:
            return LLMResponse(
                raw=raw, analysis="", changes=[], confidence=0.0, done=False,
                parse_error=f"No JSON object found in response: {raw[:300]}",
                input_tokens=input_tok, output_tokens=output_tok, estimated_cost=cost
            )

        try:
            data = json.loads(text[start:end])
        except json.JSONDecodeError as e:
            return LLMResponse(
                raw=raw, analysis="", changes=[], confidence=0.0, done=False,
                parse_error=f"JSON parse error: {e}\nText: {text[start:end][:500]}",
                input_tokens=input_tok, output_tokens=output_tok, estimated_cost=cost
            )

        changes = []
        for c in data.get("changes", []):
            changes.append(FileChange(
                path=c.get("path", ""),
                action=c.get("action", "modify"),
                content=c.get("content", ""),
                explanation=c.get("explanation", ""),
                new_path=c.get("new_path") or None,
            ))

        return LLMResponse(
            raw=raw,
            analysis=data.get("analysis", ""),
            changes=changes,
            confidence=float(data.get("confidence", 0.5)),
            lookups=[
                str(u) for u in (data.get("lookups") or [])
                if isinstance(data.get("lookups"), list)
            ],
            done=bool(data.get("done", False)),
            input_tokens=input_tok,
            output_tokens=output_tok,
            estimated_cost=cost
        )

    def _parse_plan(self, raw: str) -> Plan:
        text = raw.strip()
        start, end = text.find("{"), text.rfind("}") + 1
        if start == -1 or end == 0:
            return Plan(raw, [], [], [], 0.0, f"No JSON object in plan: {raw[:200]}")
        try:
            data = json.loads(text[start:end])
        except json.JSONDecodeError as exc:
            return Plan(raw, [], [], [], 0.0, f"Plan JSON parse error: {exc}")

        def as_list(key):
            value = data.get(key) or []
            return [str(v) for v in value] if isinstance(value, list) else []

        return Plan(
            raw=raw,
            steps=as_list("plan"),
            files_to_change=as_list("files_to_change"),
            risks=as_list("risks"),
            confidence=float(data.get("confidence", 0.5)),
        )

    def pr_description_request(self, task: str, diff: str) -> tuple:
        """
        Ask for a pull request title and body describing what changed.

        Returns (title, body). Falls back to the templated pair on any failure:
        a PR that opens with a plain description is better than a run that
        succeeded and then died writing prose about itself.

        Routed through _accounted_call, so this extra request is budget-checked
        and its cost recorded like any other -- --pr with --max-cost should not
        be able to overshoot because of the description.
        """
        prompt = PR_PROMPT_TEMPLATE.format(
            task=task, diff=diff[:PR_DIFF_CHARS]
        )
        try:
            raw, _ = self._accounted_call(prompt)
        except BudgetExceededError:
            raise
        except Exception as exc:
            _LOG.warning(
                "Could not generate a PR description (%s); using the template.", exc
            )
            return None, None

        return _parse_pr_description(raw)

    def plan_request(self, task: str, context_str: str) -> Plan:
        """
        Ask for an approach before any code is written.

        Routed through _accounted_call: a planning pass is a paid request like
        any other, and previously it was neither budget-checked nor counted.
        """
        prompt = PLAN_PROMPT_TEMPLATE.format(task=task, context=context_str)
        raw, _ = self._accounted_call(prompt)
        return self._parse_plan(raw)

    def initial_request(
        self, task: str, context_str: str, plan: Optional["Plan"] = None
    ) -> LLMResponse:
        prompt = TASK_PROMPT_TEMPLATE.format(task=task, context=context_str)
        if plan is not None and plan.usable:
            # Appended rather than templated in, so the prompt is byte-identical
            # when planning is off or the planning pass came back unusable.
            prompt = f"{prompt}\n{plan.render()}\n"
        raw, input_tok = self._accounted_call(prompt)
        return self._parse_response(raw, input_tok)

    def retry_request(
        self,
        task: str,
        context_str: str,
        previous_changes: list[FileChange],
        stdout: str,
        stderr: str,
        exit_code: int,
    ) -> LLMResponse:
        prev_summary = "\n".join(
            f"  - [{c.action}] {c.path}: {c.explanation}"
            for c in previous_changes
        )

        prompt = RETRY_PROMPT_TEMPLATE.format(
            task=task,
            previous_changes_summary=prev_summary or "  (none)",
            exit_code=exit_code,
            stdout=stdout[:4000] if stdout else "(empty)",
            stderr=stderr[:4000] if stderr else "(empty)",
            context=context_str,
        )
        raw, input_tok = self._accounted_call(prompt)
        return self._parse_response(raw, input_tok)


class AnthropicClient(BaseLLMClient):
    def __init__(self, api_key: Optional[str] = None, model: str = MODEL):
        super().__init__(model)
        if not _ANTHROPIC_AVAILABLE:
            raise RuntimeError("anthropic package not installed. Run: pip install anthropic")
        key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise ValueError("ANTHROPIC_API_KEY not set")
        self.client = anthropic.Anthropic(api_key=key)

    def _call(self, prompt: str) -> str:
        # Stream output in console for Issue #23
        print("  [Streaming LLM Response]: ", end="", flush=True)
        response_chunks = []
        try:
            with self.client.messages.stream(
                model=self.model,
                max_tokens=MAX_TOKENS,
                system=self.system_prompt,
                messages=[{"role": "user", "content": prompt}],
            ) as stream:
                for text in stream.text_stream:
                    print(text, end="", flush=True)
                    response_chunks.append(text)
            print("\n")
            return "".join(response_chunks)
        except Exception:
            # Fallback to standard non-stream call on error
            response = self.client.messages.create(
                model=self.model,
                max_tokens=MAX_TOKENS,
                system=self.system_prompt,
                messages=[{"role": "user", "content": prompt}],
            )
            print(response.content[0].text)
            print("\n")
            return response.content[0].text


class OpenAIClient(BaseLLMClient):
    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4o"):
        super().__init__(model)
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY not set")

    def _call(self, prompt: str) -> str:
        print("  [Streaming LLM Response (OpenAI)]: ", end="", flush=True)
        # Using zero-dependency urllib implementation
        url = "https://api.openai.com/v1/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        data = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt}
            ],
            "max_tokens": MAX_TOKENS,
            "temperature": 0.2
        }
        
        req = urllib.request.Request(url, data=json.dumps(data).encode("utf-8"), headers=headers)
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE

        try:
            with urllib.request.urlopen(req, context=ctx) as response:
                res = json.loads(response.read().decode("utf-8"))
                text = res["choices"][0]["message"]["content"]
                print(text)
                print("\n")
                return text
        except Exception as e:
            print(f"Error calling OpenAI API: {e}")
            raise


class GeminiClient(BaseLLMClient):
    def __init__(self, api_key: Optional[str] = None, model: str = "gemini-1.5-pro"):
        super().__init__(model)
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY not set")

    def _call(self, prompt: str) -> str:
        print("  [Streaming LLM Response (Gemini)]: ", end="", flush=True)
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent?key={self.api_key}"
        headers = {
            "Content-Type": "application/json"
        }
        data = {
            "contents": [
                {
                    "parts": [
                        {"text": self.system_prompt + "\n\n" + prompt}
                    ]
                }
            ],
            "generationConfig": {
                "maxOutputTokens": MAX_TOKENS,
                "temperature": 0.2
            }
        }

        req = urllib.request.Request(url, data=json.dumps(data).encode("utf-8"), headers=headers)
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE

        try:
            with urllib.request.urlopen(req, context=ctx) as response:
                res = json.loads(response.read().decode("utf-8"))
                text = res["candidates"][0]["content"]["parts"][0]["text"]
                print(text)
                print("\n")
                return text
        except Exception as e:
            print(f"Error calling Gemini API: {e}")
            raise


class OllamaClient(BaseLLMClient):
    def __init__(self, model: str = "llama3", api_base_url: Optional[str] = None):
        super().__init__(model)
        # Self-hosted, so the address is a per-user setting. --api-base-url was
        # registered but never read, leaving this hardcoded to localhost.
        base = api_base_url or DEFAULT_BASE_URLS["ollama"]
        self.api_base_url = base.rstrip("/")

    def _call(self, prompt: str) -> str:
        print("  [Streaming LLM Response (Ollama)]: ", end="", flush=True)
        url = f"{self.api_base_url}/api/chat"
        headers = {
            "Content-Type": "application/json"
        }
        data = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt}
            ],
            "options": {
                "temperature": 0.2
            },
            "stream": False
        }

        req = urllib.request.Request(url, data=json.dumps(data).encode("utf-8"), headers=headers)
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE

        try:
            with urllib.request.urlopen(req, context=ctx) as response:
                res = json.loads(response.read().decode("utf-8"))
                text = res["message"]["content"]
                print(text)
                print("\n")
                return text
        except Exception as e:
            print(f"Error calling Ollama API (is the Ollama server running at {self.api_base_url}?): {e}")
            raise


class LLMClient(BaseLLMClient):
    """Facade class maintaining backward compatibility while wrapping dynamic clients."""
    
    def __init__(self, api_key: Optional[str] = None, model: str = MODEL,
                 provider: str = "anthropic", verbose: bool = False,
                 max_cost: Optional[float] = None,
                 api_base_url: Optional[str] = None,
                 system_prompt: Optional[str] = None,
                 cache=None):
        super().__init__(model, verbose, max_cost, system_prompt, cache)
        self.provider = provider.lower()
        if self.provider == "openai":
            self.underlying_client = OpenAIClient(api_key, model)
        elif self.provider == "gemini":
            self.underlying_client = GeminiClient(api_key, model)
        elif self.provider == "ollama":
            self.underlying_client = OllamaClient(model, api_base_url)
        else:
            self.underlying_client = AnthropicClient(api_key, model)

        # Set once after the chain rather than in each branch: the flag applies
        # to whichever provider was chosen, and one line cannot drift.
        self.underlying_client.verbose = verbose
        self.underlying_client.max_cost = max_cost
        self.underlying_client.system_prompt = self.system_prompt
        self.underlying_client.cache = cache

    def _call(self, prompt: str) -> str:
        return self.underlying_client._call(prompt)

    def initial_request(
        self, task: str, context_str: str, plan: Optional["Plan"] = None
    ) -> LLMResponse:
        res = self.underlying_client.initial_request(task, context_str, plan)
        self.input_tokens_used = self.underlying_client.input_tokens_used
        self.output_tokens_used = self.underlying_client.output_tokens_used
        self.total_cost = self.underlying_client.total_cost
        return res

    def retry_request(
        self,
        task: str,
        context_str: str,
        previous_changes: list[FileChange],
        stdout: str,
        stderr: str,
        exit_code: int,
    ) -> LLMResponse:
        res = self.underlying_client.retry_request(task, context_str, previous_changes, stdout, stderr, exit_code)
        self.input_tokens_used = self.underlying_client.input_tokens_used
        self.output_tokens_used = self.underlying_client.output_tokens_used
        self.total_cost = self.underlying_client.total_cost
        return res

