"""Ultimatum Game LLM adapter."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from policy_arena.brains.llm.dual_role_brain import DualRoleLLMBrain, RoleBrainConfig
from policy_arena.brains.llm.llm_brain import LLMBrain
from policy_arena.brains.llm.shared import DEFAULT_PERSONA, _build_persona, _fmt_num


class UGProposerDecision(BaseModel):
    """A single proposer decision for one opponent."""

    opponent: str = Field(description="Name of the opponent")
    rationale: str = Field(description="1-2 sentence reasoning")
    offer: float = Field(description="Amount to offer the responder (0 to stake)")


class UGProposerDecisionList(BaseModel):
    """Batch of proposer decisions, one per opponent."""

    decisions: list[UGProposerDecision]


class UGResponderDecision(BaseModel):
    """A single responder decision for one offer."""

    opponent: str = Field(description="Name of the opponent/proposer")
    rationale: str = Field(description="1-2 sentence reasoning")
    accept: bool = Field(description="true = accept the offer, false = reject")


class UGResponderDecisionList(BaseModel):
    """Batch of responder decisions, one per offer."""

    decisions: list[UGResponderDecision]


UG_PROPOSER_PROMPT = """\
You are playing an Iterated Ultimatum Game as a PROPOSER.

Each round you propose offers to multiple opponents simultaneously.
You must split a stake of {stake} with each responder. You propose an offer
(the amount the responder receives). If they accept, you keep stake - offer
and they get the offer. If they reject, you BOTH get 0.

Nash Equilibrium: offer the minimum, responder accepts any positive amount.
But in practice, "unfair" offers (below ~30%) are often rejected.

Your goal is to maximize your total payoff across all opponents and rounds.

{persona}

Provide one decision per opponent, in the same order presented.
Choose an offer amount between 0 and {stake} for each."""

UG_RESPONDER_PROMPT = """\
You are playing an Iterated Ultimatum Game as a RESPONDER.

Each round you receive offers from multiple proposers simultaneously.
The proposer splits a stake of {stake}. You see their offer (the amount
you would receive) and choose to ACCEPT or REJECT. If you accept, you get
the offer and they keep the rest. If you reject, you BOTH get 0.

Nash Equilibrium: accept any positive offer (getting something > nothing).
But rejecting "unfair" offers can discipline greedy proposers over time.

Your goal is to maximize your total payoff across all opponents and rounds.

{persona}

Provide one decision per offer, in the same order presented.
Decide whether to accept (true) or reject (false) each offer."""


def _ug_proposer_observation_formatter(observations: list[Any]) -> str:
    if not observations:
        return ""

    round_num = observations[0].round_number + 1
    stake = observations[0].stake
    parts = [
        f"=== Round {round_num} — PROPOSER — {len(observations)} opponent(s) ===\n"
    ]
    parts.append(f"Stake: {_fmt_num(stake)}")

    for i, obs in enumerate(observations):
        opp_parts = [f"\n--- Opp {i + 1} ---"]

        if obs.opponent_past_offers:
            opp_recent = obs.opponent_past_offers[-5:]
            opp_str = ", ".join(f"{o:.0f}" for o in opp_recent)
            opp_parts.append(f"Their past offers to you: [{opp_str}]")

        if not obs.opponent_past_offers:
            opp_parts.append("No history yet with this opponent.")

        parts.append("\n".join(opp_parts))

    # Global summary across all opponents
    first = observations[0]
    if first.my_past_offers_made and first.my_past_responses:
        below_30: list[bool] = []
        between_30_50: list[bool] = []
        above_50: list[bool] = []
        for offer, accepted in zip(
            first.my_past_offers_made, first.my_past_responses, strict=False
        ):
            pct = offer / stake
            if pct < 0.3:
                below_30.append(accepted)
            elif pct <= 0.5:
                between_30_50.append(accepted)
            else:
                above_50.append(accepted)

        rate_parts = []
        if below_30:
            acc = sum(below_30)
            rate_parts.append(f"<30%: {acc}/{len(below_30)} accepted")
        if between_30_50:
            acc = sum(between_30_50)
            rate_parts.append(f"30-50%: {acc}/{len(between_30_50)} accepted")
        if above_50:
            acc = sum(above_50)
            rate_parts.append(f">50%: {acc}/{len(above_50)} accepted")
        if rate_parts:
            parts.append(f"\nOverall acceptance rates: {', '.join(rate_parts)}")

    parts.append(f"\nChoose an offer (0 to {_fmt_num(stake)}) for each opponent.")
    return "\n\n".join(parts)


def _ug_responder_observation_formatter(observations: list[Any]) -> str:
    if not observations:
        return ""

    round_num = observations[0].round_number + 1
    stake = observations[0].stake
    parts = [f"=== Round {round_num} — RESPONDER — {len(observations)} offer(s) ===\n"]
    parts.append(f"Stake: {_fmt_num(stake)}")

    for i, obs in enumerate(observations):
        pct = obs.offer / stake * 100
        opp_parts = [f"\n--- Opp {i + 1} — offer: {obs.offer:.0f} ({pct:.0f}%) ---"]

        if obs.opponent_past_offers:
            opp_recent = obs.opponent_past_offers[-5:]
            opp_str = ", ".join(f"{o:.0f}" for o in opp_recent)
            opp_parts.append(f"Their past offers: [{opp_str}]")

        parts.append("\n".join(opp_parts))

    # Global summary
    first = observations[0]
    if first.my_past_responses:
        accepted_count = sum(first.my_past_responses)
        total = len(first.my_past_responses)
        parts.append(f"\nYour accept rate so far: {accepted_count}/{total}")

    parts.append("\nAccept or reject each offer?")
    return "\n\n".join(parts)


def _ug_proposer_action_extractor(
    response: UGProposerDecisionList, n: int
) -> list[float]:
    actions: list[float] = []
    for d in response.decisions[:n]:
        actions.append(d.offer)
    while len(actions) < n:
        actions.append(50.0)
    return actions


def _ug_responder_action_extractor(
    response: UGResponderDecisionList, n: int
) -> list[bool]:
    actions: list[bool] = []
    for d in response.decisions[:n]:
        actions.append(d.accept)
    while len(actions) < n:
        actions.append(True)
    return actions


def _ug_proposer_result_formatter(result: Any) -> str:
    return ""


def _ug_responder_result_formatter(result: Any) -> str:
    return ""


def _ug_proposer_fallback(n: int) -> list[float]:
    return [50.0] * n  # fair split of default 100 stake


def _ug_responder_fallback(n: int) -> list[bool]:
    return [True] * n  # accept


class UGLLMBrain(DualRoleLLMBrain):
    """Combined LLM brain for the Ultimatum Game.

    Routes proposer and responder observations to separate underlying
    LLM brains, each with its own schema, system prompt, and history.
    """

    def __init__(
        self,
        *,
        provider: str = "ollama",
        model: str = "llama3",
        temperature: float = 0.7,
        max_history: int = 20,
        persona: str | None = None,
        characteristics: dict[str, Any] | None = None,
        stake: float = 100.0,
        api_key: str | None = None,
        base_url: str | None = None,
    ):
        persona_text = (
            persona
            if persona is not None
            else (_build_persona(characteristics) or DEFAULT_PERSONA)
        )

        super().__init__(
            provider=provider,
            model=model,
            temperature=temperature,
            max_history=max_history,
            api_key=api_key,
            base_url=base_url,
            role_configs=(
                RoleBrainConfig(
                    role_name="proposer",
                    tag_prefix="[PROPOSER",
                    persona=UG_PROPOSER_PROMPT.format(
                        persona=persona_text,
                        stake=_fmt_num(stake),
                    ),
                    output_schema=UGProposerDecisionList,
                    batch_observation_formatter=_ug_proposer_observation_formatter,
                    batch_action_extractor=_ug_proposer_action_extractor,
                    result_formatter=_ug_proposer_result_formatter,
                    fallback_action_factory=_ug_proposer_fallback,
                ),
                RoleBrainConfig(
                    role_name="responder",
                    tag_prefix="[RESPONDER",
                    persona=UG_RESPONDER_PROMPT.format(
                        persona=persona_text,
                        stake=_fmt_num(stake),
                    ),
                    output_schema=UGResponderDecisionList,
                    batch_observation_formatter=_ug_responder_observation_formatter,
                    batch_action_extractor=_ug_responder_action_extractor,
                    result_formatter=_ug_responder_result_formatter,
                    fallback_action_factory=_ug_responder_fallback,
                ),
            ),
        )

    @property
    def _proposer(self) -> LLMBrain:
        return self._brain_a

    @property
    def _responder(self) -> LLMBrain:
        return self._brain_b


def ug_llm_combined(
    provider: str = "ollama",
    model: str = "llama3",
    temperature: float = 0.7,
    max_history: int = 20,
    persona: str | None = None,
    characteristics: dict[str, Any] | None = None,
    stake: float = 100.0,
    api_key: str | None = None,
    base_url: str | None = None,
) -> UGLLMBrain:
    """Create a combined LLM brain for the Ultimatum Game.

    The returned brain handles both proposer and responder roles,
    routing each to a role-specific sub-brain with its own schema.
    """
    return UGLLMBrain(
        provider=provider,
        model=model,
        temperature=temperature,
        max_history=max_history,
        persona=persona,
        characteristics=characteristics,
        stake=stake,
        api_key=api_key,
        base_url=base_url,
    )
