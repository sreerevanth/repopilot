"""
Module: Errors.

One base class for every failure this tool raises deliberately.

The distinction it buys is between two things that currently look identical.
`except Exception` in the agent loop catches "the Docker daemon is not running"
and "there is a bug in the loop" the same way, and reports both as an
unrecoverable error. The first is worth telling the user plainly; the second is
worth a traceback and a bug report.

Every existing exception keeps its original base as well, so this changes
nothing about what already catches them. `LookupRefused` is still a
`ValueError`, `BudgetExceededError` is still a `RuntimeError`, and the
`except ValueError` blocks in `code_modifier.py` and the `except RuntimeError`
in `agent_loop.py` behave exactly as before. Adding a base rather than
replacing one is what makes this safe to land in a single change.
"""


class AgentError(Exception):
    """
    A failure the agent anticipated.

    Raised for a situation the code knows how to describe: a missing runner, a
    budget reached, a refused lookup. Distinct from an unexpected exception,
    which means this tool has a bug.

    Catch this to report a problem to the user. Let anything else propagate --
    swallowing a TypeError as though it were a user-facing condition is how a
    bug becomes a mystery.
    """

    #: Shown to the user instead of a traceback. Subclasses may override with
    #: something more specific.
    remedy: str = ""

    def user_message(self) -> str:
        """The message plus its remedy, when one is known."""
        text = str(self)
        return f"{text} {self.remedy}".strip() if self.remedy else text


class ConfigurationError(AgentError):
    """Something about how the run was set up is wrong."""


class ProviderError(AgentError):
    """The model provider could not be used or did not cooperate."""


class ExecutionError(AgentError):
    """Running the repository's own code or tools failed."""


class StateError(AgentError):
    """Saved run state could not be read, written or resumed."""
