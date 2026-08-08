"""
Module 3: LLM Interaction Layer
Clean prompt engineering, structured I/O, token budget handling,
iterative refinement support.

Supports multiple LLM providers:
  - Anthropic Claude (default)
  - OpenAI (GPT-4, etc.)
  - Google Gemini
  - Ollama (local, OpenAI-compatible API)
"""

import json
import os
import re
import time
from dataclasses import dataclass
from typing import Optional

# ─────────────────────────────────────────────
# Provider availability checks
# ─────────────────────────────────────────────

try:
    import anthropic
    _ANTHROPIC_AVAILABLE = True
except ImportError:
    _ANTHROPIC_AVAILABLE = False

try:
    import openai
    _OPENAI_AVAILABLE = True
except ImportError:
    _OPENAI_AVAILABLE = False

try:
    import google.generativeai as genai
    _GEMINI_AVAILABLE = True
except ImportError:
    _GEMINI_AVAILABLE = False


# ─────────────────────────────────────────────
# Defaults
# ─────────────────────────────────────────────

DEFAULT_MODELS = {
    "anthropic": "claude-sonnet-4-20250514",
    "openai": "gpt-4o",
    "gemini": "gemini-2.0-flash",
    "ollama": "llama3",
}

MAX_TOKENS = 8192

# ─────────────────────────────────────────────
# Prompt Templates
# ─────────────────────────────────────────────

SYSTEM_PROMPT = """You are an expert software engineer embedded in an autonomous code modification pipeline.
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
      "action": "modify" | "create" | "delete",
      "content": "<full new file content (for modify/create)>",
      "explanation": "<why this change>"
    }
  ],
  "confidence": <0.0-1.0 float>,
  "done": <true if you believe the task is complete, false if more iterations needed>
}

RULES:
- For "modify" and "create": always provide the COMPLETE file content, not diffs or snippets.
- For "delete": omit "content".
- Do NOT include markdown fences, explanation text, or anything outside the JSON object.
- Paths must be relative (e.g., "src/utils.py"), never absolute.
- If you cannot determine a fix, set confidence < 0.3 and done=false with a clear analysis.
- Preserve existing code style, indentation, and conventions.
"""

TASK_PROMPT_TEMPLATE = """\
## Task
{task}

## Repository Context
{context}

## Instructions
Analyze the code and produce the minimal changes needed to complete the task.
Return ONLY the JSON object specified in the system prompt.
"""

RETRY_PROMPT_TEMPLATE = """\
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
    action: str      # "modify" | "create" | "delete"
    content: str     # full file content
    explanation: str


@dataclass
class LLMResponse:
    raw: str
    analysis: str
    changes: list[FileChange]
    confidence: float
    done: bool
    parse_error: Optional[str] = None


# ─────────────────────────────────────────────
# Base Client
# ─────────────────────────────────────────────

class BaseLLMClient:
    """Abstract base for LLM clients across providers."""

    def __init__(self, model: Optional[str] = None, provider: str = "anthropic"):
        self.provider = provider
        self.model = model or os.environ.get("AGENT_MODEL") or DEFAULT_MODELS.get(provider, "")

    def _call(self, prompt: str, retries: int = 3) -> str:
        """Raw API call with retry. Subclasses must implement."""
        raise NotImplementedError

    def _parse_response(self, raw: str) -> LLMResponse:
        """Extract and parse JSON from LLM output."""
        # Strip any accidental markdown fences
        text = raw.strip()
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.MULTILINE)
        text = re.sub(r"```\s*$", "", text, flags=re.MULTILINE)
        text = text.strip()

        # Find the outermost JSON object
        start = text.find("{")
        end = text.rfind("}") + 1
        if start == -1 or end == 0:
            return LLMResponse(
                raw=raw, analysis="", changes=[], confidence=0.0, done=False,
                parse_error=f"No JSON object found in response: {raw[:300]}"
            )

        try:
            data = json.loads(text[start:end])
        except json.JSONDecodeError as e:
            return LLMResponse(
                raw=raw, analysis="", changes=[], confidence=0.0, done=False,
                parse_error=f"JSON parse error: {e}\nText: {text[start:end][:500]}"
            )

        changes = []
        for c in data.get("changes", []):
            changes.append(FileChange(
                path=c.get("path", ""),
                action=c.get("action", "modify"),
                content=c.get("content", ""),
                explanation=c.get("explanation", ""),
            ))

        return LLMResponse(
            raw=raw,
            analysis=data.get("analysis", ""),
            changes=changes,
            confidence=float(data.get("confidence", 0.5)),
            done=bool(data.get("done", False)),
        )

    def initial_request(self, task: str, context_str: str) -> LLMResponse:
        """First-pass: analyze task and produce code changes."""
        prompt = TASK_PROMPT_TEMPLATE.format(task=task, context=context_str)
        raw = self._call(prompt)
        return self._parse_response(raw)

    def retry_request(
        self,
        task: str,
        context_str: str,
        previous_changes: list[FileChange],
        stdout: str,
        stderr: str,
        exit_code: int,
    ) -> LLMResponse:
        """Retry after a failed execution — feed error output back."""
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
        raw = self._call(prompt)
        return self._parse_response(raw)


# ─────────────────────────────────────────────
# Anthropic Client
# ─────────────────────────────────────────────

class AnthropicLLMClient(BaseLLMClient):
    """LLM client using Anthropic Claude API."""

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        super().__init__(model=model, provider="anthropic")
        if not _ANTHROPIC_AVAILABLE:
            raise RuntimeError(
                "anthropic package not installed. Run: pip install anthropic"
            )
        key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise ValueError("ANTHROPIC_API_KEY not set")
        self.client = anthropic.Anthropic(api_key=key)

    def _call(self, prompt: str, retries: int = 3) -> str:
        for attempt in range(retries):
            try:
                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=MAX_TOKENS,
                    system=SYSTEM_PROMPT,
                    messages=[{"role": "user", "content": prompt}],
                )
                return response.content[0].text
            except Exception as e:
                if attempt == retries - 1:
                    raise
                wait = 2 ** attempt
                time.sleep(wait)
        raise RuntimeError("LLM call failed after retries")


# ─────────────────────────────────────────────
# OpenAI Client
# ─────────────────────────────────────────────

class OpenAILLMClient(BaseLLMClient):
    """LLM client using OpenAI-compatible API (GPT-4, etc.)."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
    ):
        super().__init__(model=model, provider="openai")
        if not _OPENAI_AVAILABLE:
            raise RuntimeError(
                "openai package not installed. Run: pip install openai"
            )
        key = api_key or os.environ.get("OPENAI_API_KEY")
        if not key:
            raise ValueError("OPENAI_API_KEY not set")
        kwargs = {"api_key": key}
        if base_url:
            kwargs["base_url"] = base_url
        self.client = openai.OpenAI(**kwargs)

    def _call(self, prompt: str, retries: int = 3) -> str:
        for attempt in range(retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    max_tokens=MAX_TOKENS,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": prompt},
                    ],
                )
                return response.choices[0].message.content
            except Exception as e:
                if attempt == retries - 1:
                    raise
                wait = 2 ** attempt
                time.sleep(wait)
        raise RuntimeError("LLM call failed after retries")


# ─────────────────────────────────────────────
# Gemini Client
# ─────────────────────────────────────────────

class GeminiLLMClient(BaseLLMClient):
    """LLM client using Google Gemini API."""

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        super().__init__(model=model, provider="gemini")
        if not _GEMINI_AVAILABLE:
            raise RuntimeError(
                "google-generativeai package not installed. Run: pip install google-generativeai"
            )
        key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not key:
            raise ValueError("GEMINI_API_KEY or GOOGLE_API_KEY not set")
        genai.configure(api_key=key)
        self.genai_model = genai.GenerativeModel(
            model_name=self.model,
            system_instruction=SYSTEM_PROMPT,
        )

    def _call(self, prompt: str, retries: int = 3) -> str:
        for attempt in range(retries):
            try:
                response = self.genai_model.generate_content(
                    prompt,
                    generation_config=genai.types.GenerationConfig(
                        max_output_tokens=MAX_TOKENS,
                    ),
                )
                return response.text
            except Exception as e:
                if attempt == retries - 1:
                    raise
                wait = 2 ** attempt
                time.sleep(wait)
        raise RuntimeError("LLM call failed after retries")


# ─────────────────────────────────────────────
# Ollama Client (OpenAI-compatible)
# ─────────────────────────────────────────────

class OllamaLLMClient(BaseLLMClient):
    """LLM client for Ollama using its OpenAI-compatible API."""

    def __init__(
        self,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
    ):
        super().__init__(model=model, provider="ollama")
        if not _OPENAI_AVAILABLE:
            raise RuntimeError(
                "openai package not installed (used for Ollama compatibility). "
                "Run: pip install openai"
            )
        self.client = openai.OpenAI(
            api_key="ollama",  # Ollama doesn't require a real key
            base_url=base_url or os.environ.get("OLLAMA_BASE_URL") or "http://localhost:11434/v1",
        )

    def _call(self, prompt: str, retries: int = 3) -> str:
        for attempt in range(retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": prompt},
                    ],
                )
                return response.choices[0].message.content
            except Exception as e:
                if attempt == retries - 1:
                    raise
                wait = 2 ** attempt
                time.sleep(wait)
        raise RuntimeError("LLM call failed after retries")


# ─────────────────────────────────────────────
# Factory
# ─────────────────────────────────────────────

def create_llm_client(
    provider: str = "anthropic",
    api_key: Optional[str] = None,
    model: Optional[str] = None,
    base_url: Optional[str] = None,
) -> BaseLLMClient:
    """
    Create an LLM client for the specified provider.

    Args:
        provider: One of 'anthropic', 'openai', 'gemini', 'ollama'.
        api_key: API key (or read from env vars).
        model: Model name (or use provider defaults).
        base_url: Custom API base URL (for Ollama or self-hosted).

    Returns:
        A configured BaseLLMClient subclass instance.
    """
    provider = provider.lower().strip()

    if provider == "anthropic":
        return AnthropicLLMClient(api_key=api_key, model=model)
    elif provider == "openai":
        return OpenAILLMClient(api_key=api_key, model=model, base_url=base_url)
    elif provider == "gemini":
        return GeminiLLMClient(api_key=api_key, model=model)
    elif provider == "ollama":
        return OllamaLLMClient(model=model, base_url=base_url)
    else:
        raise ValueError(
            f"Unknown provider: '{provider}'. "
            f"Supported: anthropic, openai, gemini, ollama"
        )


# ─────────────────────────────────────────────
# Backward-compatible alias
# ─────────────────────────────────────────────

class LLMClient(AnthropicLLMClient):
    """Backward-compatible alias for AnthropicLLMClient."""
    pass
