"""Base class for dual-role LLM brains (e.g. proposer/responder, investor/trustee).

Games where each agent plays two distinct roles per round share identical
routing logic. This module extracts that boilerplate so game-specific
adapters only need to supply role configurations.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from pydantic import BaseModel

from policy_arena.brains.llm.llm_brain import LLMBrain
from policy_arena.brains.llm.shared import DEFAULT_PERSONA, _build_persona


@dataclass(frozen=True)
class RoleBrainConfig:
    """Configuration for one role's sub-brain."""

    role_name: str  # e.g. "proposer", "investor"
    tag_prefix: str  # e.g. "[PROPOSER", "[INVESTOR"
    persona: str  # fully formatted system prompt
    output_schema: type[BaseModel]
    batch_observation_formatter: Callable[[list[Any]], str]
    batch_action_extractor: Callable[[Any, int], list[Any]]
    result_formatter: Callable[[Any], str]
    fallback_action_factory: Callable[[int], list[Any]]


class DualRoleLLMBrain(LLMBrain):
    """LLM brain that routes decisions between two role-specific sub-brains.

    Subclasses only need to build two ``RoleBrainConfig`` objects and pass
    them to ``super().__init__()``.  All routing, batching, status-callback
    forwarding, and round-summary dispatch are handled here.
    """

    def __init__(
        self,
        *,
        provider: str = "ollama",
        model: str = "llama3",
        temperature: float = 0.7,
        max_history: int = 20,
        api_key: str | None = None,
        base_url: str | None = None,
        role_configs: tuple[RoleBrainConfig, RoleBrainConfig],
    ):
        # Skip LLMBrain.__init__() — we delegate to sub-brains.
        self._brain_name = f"llm({provider}/{model})"
        self._label: str = self._brain_name
        self._tracer: Any = None
        self._last_response_text: str | None = None
        self._last_error: dict[str, str] | None = None

        cfg_a, cfg_b = role_configs
        self._role_a_name = cfg_a.role_name
        self._role_b_name = cfg_b.role_name
        self._tag_a = cfg_a.tag_prefix
        self._tag_b = cfg_b.tag_prefix

        common = dict(
            provider=provider,
            model=model,
            api_key=api_key,
            base_url=base_url,
            temperature=temperature,
            max_history=max_history,
            brain_name=self._brain_name,
        )

        self._brain_a = LLMBrain(
            persona=cfg_a.persona,
            output_schema=cfg_a.output_schema,
            batch_observation_formatter=cfg_a.batch_observation_formatter,
            batch_action_extractor=cfg_a.batch_action_extractor,
            result_formatter=cfg_a.result_formatter,
            fallback_action_factory=cfg_a.fallback_action_factory,
            **common,
        )
        self._brain_b = LLMBrain(
            persona=cfg_b.persona,
            output_schema=cfg_b.output_schema,
            batch_observation_formatter=cfg_b.batch_observation_formatter,
            batch_action_extractor=cfg_b.batch_action_extractor,
            result_formatter=cfg_b.result_formatter,
            fallback_action_factory=cfg_b.fallback_action_factory,
            **common,
        )

    # ------------------------------------------------------------------
    # Role routing helper
    # ------------------------------------------------------------------

    def _brain_for_role(self, role: str) -> LLMBrain:
        if role == self._role_a_name:
            return self._brain_a
        return self._brain_b

    # ------------------------------------------------------------------
    # Brain interface
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        return self._brain_name

    @property
    def last_response_text(self) -> str | None:
        return self._brain_a.last_response_text or self._brain_b.last_response_text

    @property
    def last_error(self) -> dict[str, str] | None:
        return self._brain_a.last_error or self._brain_b.last_error

    def set_tracer(self, tracer: Any, label: str) -> None:
        self._tracer = tracer
        self._label = label
        self._brain_a.set_tracer(tracer, f"{label}_{self._role_a_name}")
        self._brain_b.set_tracer(tracer, f"{label}_{self._role_b_name}")

    def set_status_callback(self, callback: Any) -> None:
        self._brain_a.set_status_callback(callback)
        self._brain_b.set_status_callback(callback)

    def set_concurrency_semaphore(self, semaphore: Any) -> None:
        self._brain_a.set_concurrency_semaphore(semaphore)
        self._brain_b.set_concurrency_semaphore(semaphore)

    def decide(self, observation: Any) -> Any:
        brain = self._brain_for_role(observation.role)
        result = brain.decide(observation)
        self._last_response_text = brain.last_response_text
        return result

    def decide_batch(self, observations: list[Any]) -> list[Any]:
        if not observations:
            return []
        first_role = observations[0].role
        if all(o.role == first_role for o in observations):
            brain = self._brain_for_role(first_role)
            results = brain.decide_batch(observations)
            self._last_response_text = brain.last_response_text
            return results
        # Mixed roles — fall back to sequential.
        return [self.decide(obs) for obs in observations]

    def update(self, result: Any) -> None:
        self._brain_for_role(result.role).update(result)

    def update_round_summary(self, summary: str) -> None:
        if not summary:
            return
        if summary.startswith(self._tag_a):
            self._brain_a.update_round_summary(summary)
        elif summary.startswith(self._tag_b):
            self._brain_b.update_round_summary(summary)

    def reset(self) -> None:
        self._brain_a.reset()
        self._brain_b.reset()
