"""LLM Brain — delegates decisions to a language model via LangChain.

Supports multiple providers: Ollama (local), OpenAI, Anthropic (Claude),
Google (Gemini), and DeepSeek. The brain formats game observations into prompts,
sends them to the LLM, and parses the response into a valid game action.

Supports batch decisions: when `decide_batch()` is called with multiple
observations, the brain makes a single LLM call and returns structured
JSON output with decisions for all opponents at once.

When an ``output_schema`` (a Pydantic model) is provided, the brain uses
LangChain's ``.with_structured_output()`` so the LLM is constrained to
return schema-conformant data — no manual JSON parsing needed.
"""

from __future__ import annotations

import contextlib
import json
import logging
import re
import threading
import time
from collections.abc import Callable, Generator
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# SDK retry log interceptor
# ---------------------------------------------------------------------------
# Provider SDKs (google_genai, openai, anthropic) retry rate-limit errors
# internally and log them.  We intercept those log messages so we can
# forward them to the frontend via the status callback in real-time.

# Logger names that emit retry messages for each provider SDK.
_RETRY_LOGGER_NAMES = (
    "google_genai._api_client",
    "openai._base_client",
    "anthropic._base_client",
    "httpx",
)

# Patterns that identify a retry/rate-limit log line.
_RETRY_PATTERNS = re.compile(
    r"retr(y|ying)|rate.?limit|429|resource.?exhausted|quota|too many requests",
    re.IGNORECASE,
)


class _RetryLogHandler(logging.Handler):
    """Temporary handler that fires a callback when an SDK logs a retry."""

    def __init__(
        self,
        agent_label: str,
        status_callback: Callable[[str, dict[str, Any]], None],
        error_ref: list[dict[str, str] | None],
    ):
        super().__init__(level=logging.INFO)
        self._agent_label = agent_label
        self._callback = status_callback
        self._error_ref = error_ref  # mutable ref so caller can read

    def emit(self, record: logging.LogRecord) -> None:
        msg = record.getMessage()
        if not _RETRY_PATTERNS.search(msg):
            return
        summary = _summarize_error(msg)
        self._error_ref[0] = {"type": "ProviderRetry", "message": summary}
        with contextlib.suppress(Exception):
            self._callback(
                "retrying",
                {"agent_label": self._agent_label, "reason": summary},
            )


@contextlib.contextmanager
def _intercept_sdk_retries(
    agent_label: str,
    status_callback: Callable[[str, dict[str, Any]], None] | None,
    error_ref: list[dict[str, str] | None],
) -> Generator[None, None, None]:
    """Context manager that temporarily hooks into SDK loggers to catch retries."""
    if status_callback is None:
        yield
        return
    handler = _RetryLogHandler(agent_label, status_callback, error_ref)
    loggers = [logging.getLogger(name) for name in _RETRY_LOGGER_NAMES]
    for lg in loggers:
        lg.addHandler(handler)
    try:
        yield
    finally:
        for lg in loggers:
            lg.removeHandler(handler)


try:
    from langchain_core.language_models.chat_models import BaseChatModel
    from langchain_core.messages import (
        AIMessage,
        BaseMessage,
        HumanMessage,
        SystemMessage,
    )
except ImportError as _exc:
    raise ImportError(
        "LLM dependencies not installed. Install with: pip install policy-arena[llm]"
    ) from _exc

from pydantic import BaseModel

from policy_arena.brains.base import Brain
from policy_arena.llm.provider import create_chat_model


class LLMBrain(Brain):
    """Brain powered by a language model.

    Parameters
    ----------
    llm : BaseChatModel or None
        A LangChain chat model. If None, creates one using the provider
        and model parameters.
    provider : str
        LLM provider: "ollama", "openai", "anthropic", "gemini", or "deepseek".
    model : str
        Model name (used only when llm is None).
    persona : str
        System prompt describing the agent's personality and strategy.
    output_schema : type[BaseModel] or None
        Pydantic model for structured output. When set, the LLM is
        constrained to return data matching this schema via
        ``.with_structured_output()``.
    batch_observation_formatter : callable
        Maps a list of observations to a single prompt for batch decisions.
    batch_action_extractor : callable
        Converts the LLM response into a list of actions. When
        ``output_schema`` is set, receives a Pydantic model instance;
        otherwise receives a raw string.
    result_formatter : callable or None
        Maps a game result to a string message for conversation history.
        When None, uses a generic formatter that extracts payoff/opponent_action.
    fallback_action_factory : callable or None
        Called with (n,) to produce fallback actions when the LLM fails.
        When None, defaults to [Action.COOPERATE] * n.
    temperature : float
        LLM sampling temperature.
    max_history : int
        Maximum number of message pairs to keep in memory.
        Older messages are trimmed when exceeded.
    brain_name : str
        Human-readable name for this brain instance.
    """

    def __init__(
        self,
        *,
        llm: BaseChatModel | None = None,
        provider: str = "ollama",
        model: str = "llama3",
        api_key: str | None = None,
        base_url: str | None = None,
        persona: str = "You are a rational game theory agent.",
        output_schema: type[BaseModel] | None = None,
        batch_observation_formatter: Callable[[list[Any]], str] | None = None,
        batch_action_extractor: Callable[[Any, int], list[Any]] | None = None,
        result_formatter: Callable[[Any], str] | None = None,
        fallback_action_factory: Callable[[int], list[Any]] | None = None,
        temperature: float = 0.7,
        max_history: int = 20,
        brain_name: str | None = None,
    ):
        if llm is not None:
            self._llm = llm
        else:
            self._llm = create_chat_model(
                provider, model, temperature, api_key=api_key, base_url=base_url
            )

        self._output_schema = output_schema
        self._structured_llm: Any = None
        # DeepSeek supports json_object mode but not OpenAI's json_schema;
        # use json_mode and inject the schema into the prompt instead.
        self._needs_json_hint = provider == "deepseek" and output_schema is not None
        if output_schema is not None:
            so_kwargs: dict[str, Any] = {}
            if provider == "deepseek":
                so_kwargs["method"] = "json_mode"
            self._structured_llm = self._llm.with_structured_output(
                output_schema, **so_kwargs
            )

        self._model_name = model
        self._persona = persona
        self._format_batch = batch_observation_formatter or _default_batch_formatter
        self._extract_batch = batch_action_extractor or _default_batch_extractor
        self._format_result = result_formatter or _default_result_formatter
        self._fallback_actions = fallback_action_factory or _default_fallback_actions
        self._max_history = max_history
        self._brain_name = brain_name or f"llm({model})"
        self._history: list[BaseMessage] = []
        self._provider = provider
        self._label: str = self._brain_name
        self._tracer: Any = None  # SimulationTracer, set externally
        self._status_callback: Callable[[str, dict[str, Any]], None] | None = None
        self._last_response_text: str | None = None
        self._last_error: dict[str, str] | None = None  # {"type": ..., "message": ...}
        self._semaphore: threading.Semaphore | None = None

    @property
    def name(self) -> str:
        return self._brain_name

    @property
    def last_response_text(self) -> str | None:
        """The full LLM response from the most recent decision."""
        return self._last_response_text

    @property
    def last_error(self) -> dict[str, str] | None:
        """Error from the most recent decision, if any (type + message)."""
        return self._last_error

    def set_tracer(self, tracer: Any, label: str) -> None:
        """Attach a SimulationTracer and agent label for observability."""
        self._tracer = tracer
        self._label = label

    def set_status_callback(
        self, callback: Callable[[str, dict[str, Any]], None]
    ) -> None:
        """Attach a callback for LLM status events (thinking, retrying, done)."""
        self._status_callback = callback

    def set_concurrency_semaphore(self, semaphore: threading.Semaphore) -> None:
        """Attach a shared semaphore to limit concurrent LLM calls for this provider."""
        self._semaphore = semaphore

    def decide(self, observation: Any) -> Any:
        """Single-opponent decision. Wraps decide_batch with one observation."""
        return self.decide_batch([observation])[0]

    def decide_batch(self, observations: list[Any]) -> list[Any]:
        """Decide for all opponents in a single LLM call.

        Returns a list of actions in the same order as observations.
        """
        n = len(observations)
        self._last_error = None  # reset per-round
        obs_text = self._format_batch(observations)
        self._history.append(HumanMessage(content=obs_text))

        persona = self._persona
        if self._needs_json_hint:
            schema_hint = json.dumps(self._output_schema.model_json_schema(), indent=2)
            persona += (
                "\n\nYou MUST respond with valid JSON matching this exact schema "
                "(no markdown, no explanation):\n" + schema_hint
            )
        messages = [SystemMessage(content=persona)] + self._history

        # Start tracing
        gen_handle = None
        if self._tracer is not None:
            gen_handle = self._tracer.start_generation(
                self._label,
                model=self._model_name,
                input_messages=[
                    {"role": "system", "content": self._persona},
                    {"role": "user", "content": obs_text},
                ],
                metadata={
                    "brain_name": self._brain_name,
                    "history_length": len(self._history),
                    "n_opponents": n,
                    "structured_output": self._structured_llm is not None,
                },
            )

        logger.debug(
            "%s  calling LLM  model=%s  history=%d msgs  structured=%s",
            self._label,
            self._model_name,
            len(self._history),
            self._structured_llm is not None,
        )

        if self._status_callback:
            self._status_callback(
                "thinking", {"agent_label": self._label, "model": self._model_name}
            )

        t0 = time.perf_counter()
        # Mutable ref so the SDK retry interceptor can store errors it sees.
        sdk_error_ref: list[dict[str, str] | None] = [None]

        if self._semaphore is not None:
            self._semaphore.acquire()
        try:
            with _intercept_sdk_retries(
                self._label, self._status_callback, sdk_error_ref
            ):
                if self._structured_llm is not None:
                    actions, response_text, retried = self._decide_structured(
                        messages, n
                    )
                else:
                    actions, response_text, retried = self._decide_unstructured(
                        messages, n
                    )
        finally:
            if self._semaphore is not None:
                self._semaphore.release()

        # If SDK retried internally but eventually succeeded, still record it.
        if sdk_error_ref[0] and self._last_error is None:
            self._last_error = sdk_error_ref[0]

        elapsed = time.perf_counter() - t0
        self._last_response_text = response_text

        if self._status_callback:
            self._status_callback(
                "done", {"agent_label": self._label, "elapsed_ms": int(elapsed * 1000)}
            )
        action_strs_log = [a.value if hasattr(a, "value") else str(a) for a in actions]
        logger.info(
            "%s  LLM responded  %.2fs  actions=%s%s",
            self._label,
            elapsed,
            action_strs_log,
            "  (retried)" if retried else "",
        )

        # End tracing
        if gen_handle is not None and gen_handle.is_active:
            action_strs = [a.value if hasattr(a, "value") else str(a) for a in actions]
            gen_handle.end(
                output=response_text,
                metadata={
                    "parsed_actions": action_strs,
                    "retried": retried,
                },
            )

        self._history.append(AIMessage(content=response_text))
        self._trim_history()
        return actions

    @staticmethod
    def _is_auth_error(exc: Exception) -> bool:
        """Check if an exception is an authentication/authorization error."""
        exc_str = str(exc).lower()
        return any(
            kw in exc_str
            for kw in (
                "api_key_invalid",
                "api key not valid",
                "invalid api key",
                "401",
                "403",
                "unauthorized",
                "forbidden",
                "permission denied",
                "authentication",
            )
        )

    def _decide_structured(
        self, messages: list[BaseMessage], n: int
    ) -> tuple[list[Any], str, bool]:
        """Use LangChain structured output for schema-constrained responses.

        If structured output fails, retries with an unstructured call that
        explicitly asks for JSON matching the schema, before falling back.
        Auth errors are raised immediately without retry.
        """
        try:
            parsed = self._structured_llm.invoke(messages)
            response_text = parsed.model_dump_json()
            actions = self._extract_batch(parsed, n)
            return actions, response_text, False
        except Exception as exc:
            if self._is_auth_error(exc):
                logger.error(
                    "%s  LLM auth error — check your API key: %s",
                    self._label,
                    exc,
                )
                raise
            logger.warning(
                "%s  structured output failed (%s: %s), retrying unstructured",
                self._label,
                type(exc).__name__,
                exc,
            )
            self._last_error = {
                "type": type(exc).__name__,
                "message": _summarize_error(str(exc)),
            }
            if self._status_callback:
                self._status_callback(
                    "retrying",
                    {
                        "agent_label": self._label,
                        "reason": f"{type(exc).__name__}: {exc}",
                    },
                )

        # Retry: ask the raw LLM for JSON matching the schema
        try:
            schema_hint = self._output_schema.model_json_schema()
            retry_msg = HumanMessage(
                content=(
                    "Respond with ONLY a valid JSON object matching this schema "
                    "(no markdown, no explanation):\n"
                    f"{json.dumps(schema_hint, indent=2)}"
                )
            )
            response = self._llm.invoke(messages + [retry_msg])
            response_text = response.content
            data = _parse_json_from_response(response_text)
            parsed = self._output_schema.model_validate(data)
            actions = self._extract_batch(parsed, n)
            self._last_error = None  # retry succeeded
            return actions, parsed.model_dump_json(), True
        except Exception as exc2:
            if self._is_auth_error(exc2):
                logger.error(
                    "%s  LLM auth error — check your API key: %s",
                    self._label,
                    exc2,
                )
                raise
            logger.warning(
                "%s  unstructured retry also failed (%s: %s), using fallback",
                self._label,
                type(exc2).__name__,
                exc2,
            )
            self._last_error = {
                "type": type(exc2).__name__,
                "message": _summarize_error(str(exc2)),
            }
            return self._fallback_actions(n), "{}", True

    def _decide_unstructured(
        self, messages: list[BaseMessage], n: int
    ) -> tuple[list[Any], str, bool]:
        """Legacy path: parse raw text response as JSON."""
        response = self._llm.invoke(messages)
        response_text = response.content
        retried = False

        try:
            actions = self._extract_batch(response_text, n)
        except (ValueError, KeyError, json.JSONDecodeError):
            retried = True
            retry_msg = HumanMessage(
                content=(
                    "Your response was not valid JSON or had wrong format. "
                    "Respond with ONLY a JSON object like: "
                    '{"decisions": [{"opponent": "name", "rationale": "...", "action": 0}, ...]}'
                    " where action is 0 (DEFECT) or 1 (COOPERATE)."
                )
            )
            messages.append(AIMessage(content=response_text))
            messages.append(retry_msg)
            response = self._llm.invoke(messages)
            response_text = response.content
            try:
                actions = self._extract_batch(response_text, n)
            except (ValueError, KeyError, json.JSONDecodeError):
                actions = self._fallback_actions(n)

        return actions, response_text, retried

    def update(self, result: Any) -> None:
        outcome = self._format_result(result)
        if outcome:
            self._history.append(HumanMessage(content=outcome))
            self._trim_history()

    def update_round_summary(self, summary: str) -> None:
        """Append a single consolidated round summary to history."""
        if summary:
            self._history.append(HumanMessage(content=summary))
            self._trim_history()

    def reset(self) -> None:
        self._history.clear()

    def _trim_history(self) -> None:
        """Keep only the last max_history message pairs."""
        max_messages = self._max_history * 2
        if len(self._history) > max_messages:
            self._history = self._history[-max_messages:]


# ---------------------------------------------------------------------------
# Default batch formatters (simple fallbacks)
# ---------------------------------------------------------------------------


def _default_result_formatter(result: Any) -> str:
    """Generic result formatter — works for any game with a payoff."""
    if not hasattr(result, "payoff"):
        return ""
    outcome = f"[Result: your payoff was {result.payoff}]"
    if hasattr(result, "opponent_action"):
        opp = result.opponent_action
        opp_str = opp.value if hasattr(opp, "value") else str(opp)
        outcome = (
            f"[Result: opponent played {opp_str}, your payoff was {result.payoff}]"
        )
    return outcome


def _default_fallback_actions(n: int) -> list[Any]:
    """Default fallback: cooperate with everyone."""
    from policy_arena.core.types import Action

    return [Action.COOPERATE] * n


def _default_batch_formatter(observations: list[Any]) -> str:
    """Simple batch formatter that lists all opponents."""
    parts = []
    for i, obs in enumerate(observations):
        label = getattr(obs, "extra", {}).get("opponent_label", f"opponent_{i}")
        parts.append(f"Opponent {i + 1}: {label}")
    parts.append(
        'Respond with JSON: {"decisions": [{"opponent": "name", "rationale": "...", "action": 0 or 1}]}'
    )
    return "\n".join(parts)


def _default_batch_extractor(response: str, n: int) -> list[Any]:
    """Extract actions from JSON response."""
    from policy_arena.core.types import Action

    data = _parse_json_from_response(response)
    decisions = data["decisions"]
    actions = []
    for d in decisions[:n]:
        act = int(d["action"])
        actions.append(Action.COOPERATE if act == 1 else Action.DEFECT)
    # Pad if LLM returned fewer decisions than expected
    while len(actions) < n:
        actions.append(Action.COOPERATE)
    return actions


def _summarize_error(error_str: str) -> str:
    """Extract a short, user-friendly message from a verbose LLM error."""
    lower = error_str.lower()
    if (
        "rate" in lower
        or "quota" in lower
        or "resource_exhausted" in lower
        or "429" in lower
    ):
        return "Rate limit exceeded — too many requests to the LLM provider."
    if (
        "401" in lower
        or "unauthorized" in lower
        or "api_key" in lower
        or "invalid key" in lower
    ):
        return "Authentication error — check your API key."
    if "timeout" in lower or "timed out" in lower:
        return "Request timed out — the LLM provider took too long to respond."
    if "connection" in lower or "network" in lower:
        return "Connection error — could not reach the LLM provider."
    if "500" in lower or "internal server error" in lower:
        return "LLM provider internal server error."
    # Truncate to keep it readable
    if len(error_str) > 200:
        return error_str[:200] + "…"
    return error_str


def _parse_json_from_response(text: str) -> dict:
    """Extract JSON object from LLM response, handling markdown fences."""
    # Try to find JSON in markdown code block
    json_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if json_match:
        return json.loads(json_match.group(1))

    # Try to find raw JSON object
    brace_match = re.search(r"\{.*\}", text, re.DOTALL)
    if brace_match:
        return json.loads(brace_match.group(0))

    raise ValueError(f"No JSON found in response: {text!r}")
