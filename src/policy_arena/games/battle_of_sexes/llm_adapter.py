"""Battle of the Sexes LLM adapter."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from policy_arena.brains.llm.llm_brain import LLMBrain
from policy_arena.brains.llm.shared import DEFAULT_PERSONA, _build_persona, _fmt_num
from policy_arena.core.types import Action


class BoSDecision(BaseModel):
    """A single decision for one opponent in the Battle of the Sexes."""

    opponent: str = Field(description="Name of the opponent")
    rationale: str = Field(description="1-2 sentence reasoning for the decision")
    action: Literal[0, 1] = Field(description="0 = Option B, 1 = Option A")


class BoSDecisionList(BaseModel):
    """Batch of decisions, one per opponent."""

    decisions: list[BoSDecision]


BOS_SYSTEM_PROMPT = """\
You are playing an Iterated Battle of the Sexes game against multiple opponents.

Each round you play against every opponent simultaneously. For each opponent,
choose Option A (1) or Option B (0).

Payoffs (per matchup):
- Both Option A: you get {aa_1}, they get {aa_2}
- You Option A, they Option B: you get {ab_1}, they get {ab_2} (miscoordination)
- You Option B, they Option A: you get {ba_1}, they get {ba_2} (miscoordination)
- Both Option B: you get {bb_1}, they get {bb_2}

The key tension: you both prefer to COORDINATE (pick the same option), but you
prefer coordinating on A while your opponent prefers coordinating on B.
Miscoordination gives both players 0.

Your goal is to maximize your total payoff across all opponents and rounds.

{persona}

Provide one decision per opponent, in the same order presented.
Where 0 = Option B, 1 = Option A."""


def _bos_observation_formatter(observations: list[Any]) -> str:
    if not observations:
        return ""

    round_num = observations[0].round_number + 1
    parts = [f"=== Round {round_num} — {len(observations)} opponent(s) ===\n"]

    for i, obs in enumerate(observations):
        opp_name = f"Player {i + 1}"
        section = [f"--- Opponent: {opp_name} ---"]

        if obs.opponent_history:
            recent_opp = obs.opponent_history[-10:]
            opp_str = ", ".join(
                "A" if a == Action.COOPERATE else "B" for a in recent_opp
            )
            section.append(f"Their recent: [{opp_str}]")

            recent_mine = obs.my_history[-10:]
            my_str = ", ".join(
                "A" if a == Action.COOPERATE else "B" for a in recent_mine
            )
            section.append(f"Your recent:  [{my_str}]")

            total = len(obs.opponent_history)
            opp_a = sum(1 for a in obs.opponent_history if a == Action.COOPERATE)
            my_a = sum(1 for a in obs.my_history if a == Action.COOPERATE)
            section.append(
                f"Stats: they chose A {opp_a}/{total} ({opp_a / total:.0%}), "
                f"you chose A {my_a}/{total} ({my_a / total:.0%})"
            )

            coordinated = sum(
                1
                for m, o in zip(obs.my_history, obs.opponent_history, strict=False)
                if m == o
            )
            section.append(
                f"Coordination: matched {coordinated}/{total} rounds "
                f"({coordinated / total:.0%})"
            )
        else:
            section.append("No history yet — first encounter.")

        parts.append("\n".join(section))

    parts.append("\nRespond with your decisions for each opponent.")
    return "\n\n".join(parts)


def _bos_action_extractor(response: BoSDecisionList, n: int) -> list[Action]:
    actions: list[Action] = []
    for d in response.decisions[:n]:
        actions.append(Action.COOPERATE if d.action == 1 else Action.DEFECT)
    while len(actions) < n:
        actions.append(Action.COOPERATE)
    return actions


def _bos_result_formatter(result: Any) -> str:
    # Individual results disabled — consolidated round summary is sent
    # via update_round_summary() in BoSAgent.end_round() instead.
    return ""


def _bos_fallback(n: int) -> list[Action]:
    return [Action.COOPERATE] * n


def bos_llm(
    provider: str = "ollama",
    model: str = "llama3",
    temperature: float = 0.7,
    max_history: int = 20,
    persona: str | None = None,
    characteristics: dict[str, Any] | None = None,
    payoff_matrix: dict[str, float] | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
) -> LLMBrain:
    """Create an LLM brain configured for the Battle of the Sexes."""
    persona_text = (
        persona
        if persona is not None
        else (_build_persona(characteristics) or DEFAULT_PERSONA)
    )

    pm = {
        "aa_1": 3.0,
        "aa_2": 2.0,
        "ab_1": 0.0,
        "ab_2": 0.0,
        "ba_1": 0.0,
        "ba_2": 0.0,
        "bb_1": 2.0,
        "bb_2": 3.0,
    }
    if payoff_matrix:
        key_map = {
            "cc_1": "aa_1",
            "cc_2": "aa_2",
            "cd_1": "ab_1",
            "cd_2": "ab_2",
            "dc_1": "ba_1",
            "dc_2": "ba_2",
            "dd_1": "bb_1",
            "dd_2": "bb_2",
        }
        for old_key, new_key in key_map.items():
            if old_key in payoff_matrix:
                pm[new_key] = payoff_matrix[old_key]

    fmt = {k: _fmt_num(v) for k, v in pm.items()}
    system_prompt = BOS_SYSTEM_PROMPT.format(persona=persona_text, **fmt)

    return LLMBrain(
        provider=provider,
        model=model,
        api_key=api_key,
        base_url=base_url,
        persona=system_prompt,
        output_schema=BoSDecisionList,
        batch_observation_formatter=_bos_observation_formatter,
        batch_action_extractor=_bos_action_extractor,
        result_formatter=_bos_result_formatter,
        fallback_action_factory=_bos_fallback,
        temperature=temperature,
        max_history=max_history,
        brain_name=f"llm({provider}/{model})",
    )
