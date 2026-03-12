"""Chicken Game LLM adapter."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from policy_arena.brains.llm.llm_brain import LLMBrain
from policy_arena.brains.llm.shared import DEFAULT_PERSONA, _build_persona, _fmt_num
from policy_arena.core.types import Action


class CKDecision(BaseModel):
    opponent: str = Field(description="Name of the opponent")
    rationale: str = Field(description="1-2 sentence reasoning")
    action: Literal[0, 1] = Field(description="0 = STRAIGHT (dare), 1 = SWERVE (safe)")


class CKDecisionList(BaseModel):
    decisions: list[CKDecision]


CK_SYSTEM_PROMPT = """\
You are playing an Iterated Game of Chicken against multiple opponents.

Each round you play against every opponent simultaneously. For each opponent,
choose SWERVE (1) or STRAIGHT (0).

Payoffs (per matchup):
- Both SWERVE: you get {ss_1}, they get {ss_2} (both safe, tie)
- You SWERVE, they STRAIGHT: you get {sst_1}, they get {sst_2} (you're "chicken")
- You STRAIGHT, they SWERVE: you get {sts_1}, they get {sts_2} (you win!)
- Both STRAIGHT: you get {stst_1}, they get {stst_2} (CRASH! catastrophic)

The key: going straight wins big against swervers, but mutual straight is
catastrophic. Unlike PD, the worst outcome is mutual aggression, not being exploited.

Your goal is to maximize your total payoff across all opponents and rounds.

{persona}

Provide one decision per opponent, in the same order presented.
Where 0 = STRAIGHT, 1 = SWERVE."""


def _ck_observation_formatter(observations: list[Any]) -> str:
    if not observations:
        return ""
    round_num = observations[0].round_number + 1
    parts = [f"=== Round {round_num} — {len(observations)} opponent(s) ===\n"]
    for i, obs in enumerate(observations):
        section = [f"--- Opponent: Player {i + 1} ---"]
        if obs.opponent_history:
            recent_opp = obs.opponent_history[-10:]
            opp_str = ", ".join(
                "Sw" if a == Action.COOPERATE else "St" for a in recent_opp
            )
            section.append(f"Their recent: [{opp_str}]")
            recent_mine = obs.my_history[-10:]
            my_str = ", ".join(
                "Sw" if a == Action.COOPERATE else "St" for a in recent_mine
            )
            section.append(f"Your recent:  [{my_str}]")
            total = len(obs.opponent_history)
            opp_straight = sum(1 for a in obs.opponent_history if a == Action.DEFECT)
            crashes = sum(
                1
                for m, o in zip(obs.my_history, obs.opponent_history, strict=False)
                if m == Action.DEFECT and o == Action.DEFECT
            )
            section.append(
                f"Stats: they went straight {opp_straight}/{total} ({opp_straight / total:.0%}), "
                f"crashes: {crashes}"
            )
        else:
            section.append("No history yet — first encounter.")
        parts.append("\n".join(section))
    parts.append("\nRespond with your decisions for each opponent.")
    return "\n\n".join(parts)


def _ck_action_extractor(response: CKDecisionList, n: int) -> list[Action]:
    actions = [
        Action.COOPERATE if d.action == 1 else Action.DEFECT
        for d in response.decisions[:n]
    ]
    while len(actions) < n:
        actions.append(Action.COOPERATE)
    return actions


def _ck_result_formatter(result: Any) -> str:
    # Individual results disabled — consolidated round summary is sent
    # via update_round_summary() in ChickenAgent.end_round() instead.
    return ""


def _ck_fallback(n: int) -> list[Action]:
    return [Action.COOPERATE] * n


def ck_llm(
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
        "ss_1": 3.0,
        "ss_2": 3.0,
        "sst_1": 1.0,
        "sst_2": 5.0,
        "sts_1": 5.0,
        "sts_2": 1.0,
        "stst_1": -5.0,
        "stst_2": -5.0,
    }
    if payoff_matrix:
        key_map = {
            "cc_1": "ss_1",
            "cc_2": "ss_2",
            "cd_1": "sst_1",
            "cd_2": "sst_2",
            "dc_1": "sts_1",
            "dc_2": "sts_2",
            "dd_1": "stst_1",
            "dd_2": "stst_2",
        }
        for old_key, new_key in key_map.items():
            if old_key in payoff_matrix:
                pm[new_key] = payoff_matrix[old_key]
    fmt = {k: _fmt_num(v) for k, v in pm.items()}
    system_prompt = CK_SYSTEM_PROMPT.format(persona=persona_text, **fmt)
    return LLMBrain(
        provider=provider,
        model=model,
        api_key=api_key,
        base_url=base_url,
        persona=system_prompt,
        output_schema=CKDecisionList,
        batch_observation_formatter=_ck_observation_formatter,
        batch_action_extractor=_ck_action_extractor,
        result_formatter=_ck_result_formatter,
        fallback_action_factory=_ck_fallback,
        temperature=temperature,
        max_history=max_history,
        brain_name=f"llm({provider}/{model})",
    )
