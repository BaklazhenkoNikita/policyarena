"""Hawk-Dove LLM adapter."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from policy_arena.brains.llm.llm_brain import LLMBrain
from policy_arena.brains.llm.shared import DEFAULT_PERSONA, _build_persona, _fmt_num
from policy_arena.core.types import Action


class HDDecision(BaseModel):
    opponent: str = Field(description="Name of the opponent")
    rationale: str = Field(description="1-2 sentence reasoning")
    action: Literal[0, 1] = Field(description="0 = HAWK (fight), 1 = DOVE (share)")


class HDDecisionList(BaseModel):
    decisions: list[HDDecision]


HD_SYSTEM_PROMPT = """\
You are playing an Iterated Hawk-Dove game against multiple opponents.

Each round you play against every opponent simultaneously. For each opponent,
choose DOVE (1) or HAWK (0).

Payoffs (per matchup):
- Both DOVE: you get {dd_1}, they get {dd_2} (share resource peacefully)
- You DOVE, they HAWK: you get {dh_1}, they get {dh_2} (you retreat)
- You HAWK, they DOVE: you get {hd_1}, they get {hd_2} (you take the resource)
- Both HAWK: you get {hh_1}, they get {hh_2} (fight — both injured!)

The key: being a hawk pays off against doves, but mutual hawk fights are costly.

Your goal is to maximize your total payoff across all opponents and rounds.

{persona}

Provide one decision per opponent, in the same order presented.
Where 0 = HAWK, 1 = DOVE."""


def _hd_observation_formatter(observations: list[Any]) -> str:
    if not observations:
        return ""
    round_num = observations[0].round_number + 1
    parts = [f"=== Round {round_num} — {len(observations)} opponent(s) ===\n"]
    for i, obs in enumerate(observations):
        section = [f"--- Opponent: Player {i + 1} ---"]
        if obs.opponent_history:
            recent_opp = obs.opponent_history[-10:]
            opp_str = ", ".join(
                "D" if a == Action.COOPERATE else "H" for a in recent_opp
            )
            section.append(f"Their recent: [{opp_str}]")
            recent_mine = obs.my_history[-10:]
            my_str = ", ".join(
                "D" if a == Action.COOPERATE else "H" for a in recent_mine
            )
            section.append(f"Your recent:  [{my_str}]")
            total = len(obs.opponent_history)
            opp_hawks = sum(1 for a in obs.opponent_history if a == Action.DEFECT)
            section.append(
                f"Stats: they played hawk {opp_hawks}/{total} ({opp_hawks / total:.0%})"
            )
        else:
            section.append("No history yet — first encounter.")
        parts.append("\n".join(section))
    parts.append("\nRespond with your decisions for each opponent.")
    return "\n\n".join(parts)


def _hd_action_extractor(response: HDDecisionList, n: int) -> list[Action]:
    actions = [
        Action.COOPERATE if d.action == 1 else Action.DEFECT
        for d in response.decisions[:n]
    ]
    while len(actions) < n:
        actions.append(Action.COOPERATE)
    return actions


def _hd_result_formatter(result: Any) -> str:
    # Individual results disabled — consolidated round summary is sent
    # via update_round_summary() in HDAgent.end_round() instead.
    return ""


def _hd_fallback(n: int) -> list[Action]:
    return [Action.COOPERATE] * n


def hd_llm(
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
    persona_text = (
        persona
        if persona is not None
        else (_build_persona(characteristics) or DEFAULT_PERSONA)
    )
    pm = {
        "dd_1": 2.0,
        "dd_2": 2.0,
        "dh_1": 0.0,
        "dh_2": 4.0,
        "hd_1": 4.0,
        "hd_2": 0.0,
        "hh_1": -1.0,
        "hh_2": -1.0,
    }
    if payoff_matrix:
        key_map = {
            "cc_1": "dd_1",
            "cc_2": "dd_2",
            "cd_1": "dh_1",
            "cd_2": "dh_2",
            "dc_1": "hd_1",
            "dc_2": "hd_2",
            "dd_1": "hh_1",
            "dd_2": "hh_2",
        }
        for old_key, new_key in key_map.items():
            if old_key in payoff_matrix:
                pm[new_key] = payoff_matrix[old_key]
    fmt = {k: _fmt_num(v) for k, v in pm.items()}
    system_prompt = HD_SYSTEM_PROMPT.format(persona=persona_text, **fmt)
    return LLMBrain(
        provider=provider,
        model=model,
        api_key=api_key,
        base_url=base_url,
        persona=system_prompt,
        output_schema=HDDecisionList,
        batch_observation_formatter=_hd_observation_formatter,
        batch_action_extractor=_hd_action_extractor,
        result_formatter=_hd_result_formatter,
        fallback_action_factory=_hd_fallback,
        temperature=temperature,
        max_history=max_history,
        brain_name=f"llm({provider}/{model})",
    )
