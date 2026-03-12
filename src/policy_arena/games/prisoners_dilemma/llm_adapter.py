"""Prisoner's Dilemma LLM adapter."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from policy_arena.brains.llm.llm_brain import LLMBrain
from policy_arena.brains.llm.shared import DEFAULT_PERSONA, _build_persona, _fmt_num
from policy_arena.core.types import Action


class PDDecision(BaseModel):
    """A single decision for one opponent in the Prisoner's Dilemma."""
    opponent: str = Field(description="Name of the opponent")
    rationale: str = Field(description="1-2 sentence reasoning for the decision")
    action: Literal[0, 1] = Field(description="0 = DEFECT, 1 = COOPERATE")


class PDDecisionList(BaseModel):
    """Batch of decisions, one per opponent."""
    decisions: list[PDDecision]


PD_SYSTEM_PROMPT = """\
You are playing an Iterated Prisoner's Dilemma tournament against multiple opponents.

Each round you play against every opponent simultaneously. For each opponent,
choose to COOPERATE (1) or DEFECT (0).

Payoffs (per matchup):
- Both COOPERATE: you get {cc_1}, they get {cc_2}
- You COOPERATE, they DEFECT: you get {cd_1}, they get {cd_2}
- You DEFECT, they COOPERATE: you get {dc_1}, they get {dc_2}
- Both DEFECT: you get {dd_1}, they get {dd_2}

Your goal is to maximize your total payoff across all opponents and rounds.

{persona}

Provide one decision per opponent, in the same order presented.
Where 0 = DEFECT, 1 = COOPERATE."""


def _pd_observation_formatter(observations: list[Any]) -> str:
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
                "C" if a == Action.COOPERATE else "D" for a in recent_opp
            )
            section.append(f"Their recent: [{opp_str}]")

            recent_mine = obs.my_history[-10:]
            my_str = ", ".join(
                "C" if a == Action.COOPERATE else "D" for a in recent_mine
            )
            section.append(f"Your recent:  [{my_str}]")

            total = len(obs.opponent_history)
            opp_coops = sum(1 for a in obs.opponent_history if a == Action.COOPERATE)
            my_coops = sum(1 for a in obs.my_history if a == Action.COOPERATE)
            section.append(
                f"Stats: they cooperated {opp_coops}/{total} ({opp_coops / total:.0%}), "
                f"you cooperated {my_coops}/{total} ({my_coops / total:.0%})"
            )

            # Mutual outcome patterns (full history)
            mutual_cc = sum(
                1
                for m, o in zip(obs.my_history, obs.opponent_history, strict=False)
                if m == Action.COOPERATE and o == Action.COOPERATE
            )
            mutual_dd = sum(
                1
                for m, o in zip(obs.my_history, obs.opponent_history, strict=False)
                if m == Action.DEFECT and o == Action.DEFECT
            )
            you_c_they_d = sum(
                1
                for m, o in zip(obs.my_history, obs.opponent_history, strict=False)
                if m == Action.COOPERATE and o == Action.DEFECT
            )
            you_d_they_c = sum(
                1
                for m, o in zip(obs.my_history, obs.opponent_history, strict=False)
                if m == Action.DEFECT and o == Action.COOPERATE
            )
            section.append(
                f"Outcomes: mutual C={mutual_cc}, mutual D={mutual_dd}, "
                f"you C/they D={you_c_they_d}, you D/they C={you_d_they_c}"
            )

            # Streak detection for opponent
            streak_action = obs.opponent_history[-1]
            streak_len = 0
            for a in reversed(obs.opponent_history):
                if a == streak_action:
                    streak_len += 1
                else:
                    break
            streak_label = (
                "cooperated" if streak_action == Action.COOPERATE else "defected"
            )
            if streak_len >= 2:
                section.append(
                    f"Streak: opponent has {streak_label} last {streak_len} rounds in a row"
                )
        else:
            section.append("No history yet — first encounter.")

        parts.append("\n".join(section))

    parts.append("\nRespond with your decisions for each opponent.")
    return "\n\n".join(parts)


def _pd_action_extractor(response: PDDecisionList, n: int) -> list[Action]:
    actions: list[Action] = []
    for d in response.decisions[:n]:
        actions.append(Action.COOPERATE if d.action == 1 else Action.DEFECT)
    while len(actions) < n:
        actions.append(Action.COOPERATE)
    return actions


def _pd_result_formatter(result: Any) -> str:
    # Individual results disabled — consolidated round summary is sent
    # via update_round_summary() in PDAgent.end_round() instead.
    return ""


def _pd_fallback(n: int) -> list[Action]:
    return [Action.COOPERATE] * n


def pd_llm(
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
    """Create an LLM brain configured for the Prisoner's Dilemma."""
    persona_text = (
        persona
        if persona is not None
        else (_build_persona(characteristics) or DEFAULT_PERSONA)
    )

    pm = {
        "cc_1": 3.0,
        "cc_2": 3.0,
        "cd_1": 0.0,
        "cd_2": 5.0,
        "dc_1": 5.0,
        "dc_2": 0.0,
        "dd_1": 1.0,
        "dd_2": 1.0,
    }
    if payoff_matrix:
        pm.update(payoff_matrix)

    fmt = {k: _fmt_num(v) for k, v in pm.items()}
    system_prompt = PD_SYSTEM_PROMPT.format(persona=persona_text, **fmt)

    return LLMBrain(
        provider=provider,
        model=model,
        api_key=api_key,
        base_url=base_url,
        persona=system_prompt,
        output_schema=PDDecisionList,
        batch_observation_formatter=_pd_observation_formatter,
        batch_action_extractor=_pd_action_extractor,
        result_formatter=_pd_result_formatter,
        fallback_action_factory=_pd_fallback,
        temperature=temperature,
        max_history=max_history,
        brain_name=f"llm({provider}/{model})",
    )
