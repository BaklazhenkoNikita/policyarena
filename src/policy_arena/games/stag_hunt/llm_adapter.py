"""Stag Hunt LLM adapter."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from policy_arena.brains.llm.llm_brain import LLMBrain
from policy_arena.brains.llm.shared import DEFAULT_PERSONA, _build_persona, _fmt_num
from policy_arena.core.types import Action


class SHDecision(BaseModel):
    """A single decision for one opponent in the Stag Hunt."""

    opponent: str = Field(description="Name of the opponent")
    rationale: str = Field(description="1-2 sentence reasoning for the decision")
    action: Literal[0, 1] = Field(
        description="0 = HARE (safe), 1 = STAG (risky cooperation)"
    )


class SHDecisionList(BaseModel):
    """Batch of decisions, one per opponent."""

    decisions: list[SHDecision]


SH_SYSTEM_PROMPT = """\
You are playing an Iterated Stag Hunt game against multiple opponents.

Each round you play against every opponent simultaneously. For each opponent,
choose to hunt STAG (1) or HARE (0).

Payoffs (per matchup):
- Both STAG: you get {ss_1}, they get {ss_2} (best collective outcome)
- You STAG, they HARE: you get {sh_1}, they get {sh_2} (stag fails alone)
- You HARE, they STAG: you get {hs_1}, they get {hs_2} (safe individual gain)
- Both HARE: you get {hh_1}, they get {hh_2} (safe but suboptimal)

Both (Stag, Stag) and (Hare, Hare) are Nash Equilibria — the question is
whether you can trust your opponent to coordinate on the better one.

Your goal is to maximize your total payoff across all opponents and rounds.

{persona}

Provide one decision per opponent, in the same order presented.
Where 0 = HARE, 1 = STAG."""


def _sh_observation_formatter(observations: list[Any]) -> str:
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
                "S" if a == Action.COOPERATE else "H" for a in recent_opp
            )
            section.append(f"Their recent: [{opp_str}]")

            recent_mine = obs.my_history[-10:]
            my_str = ", ".join(
                "S" if a == Action.COOPERATE else "H" for a in recent_mine
            )
            section.append(f"Your recent:  [{my_str}]")

            total = len(obs.opponent_history)
            opp_stags = sum(1 for a in obs.opponent_history if a == Action.COOPERATE)
            my_stags = sum(1 for a in obs.my_history if a == Action.COOPERATE)
            section.append(
                f"Stats: they chose stag {opp_stags}/{total} ({opp_stags / total:.0%}), "
                f"you chose stag {my_stags}/{total} ({my_stags / total:.0%})"
            )

            mutual_ss = sum(
                1
                for m, o in zip(obs.my_history, obs.opponent_history, strict=False)
                if m == Action.COOPERATE and o == Action.COOPERATE
            )
            mutual_hh = sum(
                1
                for m, o in zip(obs.my_history, obs.opponent_history, strict=False)
                if m == Action.DEFECT and o == Action.DEFECT
            )
            section.append(
                f"Coordination: mutual stag={mutual_ss}, mutual hare={mutual_hh}, "
                f"mismatches={total - mutual_ss - mutual_hh}"
            )
        else:
            section.append("No history yet — first encounter.")

        parts.append("\n".join(section))

    parts.append("\nRespond with your decisions for each opponent.")
    return "\n\n".join(parts)


def _sh_action_extractor(response: SHDecisionList, n: int) -> list[Action]:
    actions: list[Action] = []
    for d in response.decisions[:n]:
        actions.append(Action.COOPERATE if d.action == 1 else Action.DEFECT)
    while len(actions) < n:
        actions.append(Action.COOPERATE)
    return actions


def _sh_result_formatter(result: Any) -> str:
    # Individual results disabled — consolidated round summary is sent
    # via update_round_summary() in SHAgent.end_round() instead.
    return ""


def _sh_fallback(n: int) -> list[Action]:
    return [Action.COOPERATE] * n


def sh_llm(
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
    """Create an LLM brain configured for the Stag Hunt."""
    persona_text = (
        persona
        if persona is not None
        else (_build_persona(characteristics) or DEFAULT_PERSONA)
    )

    pm = {
        "ss_1": 4.0,
        "ss_2": 4.0,
        "sh_1": 0.0,
        "sh_2": 3.0,
        "hs_1": 3.0,
        "hs_2": 0.0,
        "hh_1": 2.0,
        "hh_2": 2.0,
    }
    if payoff_matrix:
        # Map from PD-style keys to SH keys
        key_map = {
            "cc_1": "ss_1",
            "cc_2": "ss_2",
            "cd_1": "sh_1",
            "cd_2": "sh_2",
            "dc_1": "hs_1",
            "dc_2": "hs_2",
            "dd_1": "hh_1",
            "dd_2": "hh_2",
        }
        for old_key, new_key in key_map.items():
            if old_key in payoff_matrix:
                pm[new_key] = payoff_matrix[old_key]

    fmt = {k: _fmt_num(v) for k, v in pm.items()}
    system_prompt = SH_SYSTEM_PROMPT.format(persona=persona_text, **fmt)

    return LLMBrain(
        provider=provider,
        model=model,
        api_key=api_key,
        base_url=base_url,
        persona=system_prompt,
        output_schema=SHDecisionList,
        batch_observation_formatter=_sh_observation_formatter,
        batch_action_extractor=_sh_action_extractor,
        result_formatter=_sh_result_formatter,
        fallback_action_factory=_sh_fallback,
        temperature=temperature,
        max_history=max_history,
        brain_name=f"llm({provider}/{model})",
    )
