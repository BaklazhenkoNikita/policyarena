"""Minority Game LLM adapter."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from policy_arena.brains.llm.llm_brain import LLMBrain
from policy_arena.brains.llm.shared import DEFAULT_PERSONA, _build_persona, _fmt_num


class MGDecision(BaseModel):
    """Decision for the Minority Game."""
    rationale: str = Field(description="1-2 sentence reasoning")
    choice: Literal["A", "B"] = Field(description="Choose A or B")


MG_SYSTEM_PROMPT = """\
You are playing an Iterated Minority Game with {n_agents} players.

Each round, every player simultaneously chooses A or B.
The minority side wins (+{win_payoff}), the majority side loses ({lose_payoff}).
If exactly tied, everyone gets {tie_payoff}.

Key insight: there is NO pure strategy equilibrium. If everyone picks the same
side, everyone loses. Unpredictability and pattern recognition are essential.

Your goal is to maximize your total payoff across all rounds.

{persona}

Choose A or B."""


def _mg_observation_formatter(observations: list[Any]) -> str:
    obs = observations[0]
    round_num = obs.round_number + 1
    parts = [f"=== Round {round_num} ===\n"]
    parts.append(f"Players: {obs.n_agents}")

    if obs.past_winning_sides:
        recent = obs.past_winning_sides[-15:]
        parts.append(f"Recent winning sides: {' '.join(recent)}")

        a_wins = sum(1 for s in obs.past_winning_sides if s == "A")
        b_wins = sum(1 for s in obs.past_winning_sides if s == "B")
        ties = sum(1 for s in obs.past_winning_sides if s == "tie")
        parts.append(f"Overall: A won {a_wins}x, B won {b_wins}x, ties {ties}x")

    if obs.past_a_counts:
        recent_ac = obs.past_a_counts[-10:]
        ac_str = ", ".join(str(a) for a in recent_ac)
        parts.append(f"Recent A-count: [{ac_str}] (of {obs.n_agents})")

    if obs.my_past_choices:
        recent_choices = obs.my_past_choices[-10:]
        choice_str = " ".join("A" if c else "B" for c in recent_choices)
        parts.append(f"Your recent choices: {choice_str}")

        wins = sum(1 for p in obs.my_past_payoffs if p > 0)
        total = len(obs.my_past_payoffs)
        cumulative = sum(obs.my_past_payoffs)
        parts.append(f"Your win rate: {wins}/{total} ({wins / total * 100:.0f}%)")
        parts.append(f"Your cumulative payoff: {cumulative:.1f}")

    # Per-agent choices (last 5 rounds)
    if obs.all_agent_choices:
        recent_rounds = obs.all_agent_choices[-5:]
        parts.append("--- Per-Agent Choices (recent rounds) ---")
        agents = list(recent_rounds[0].keys())
        header = "Round  " + "  ".join(f"{a[:10]:>10}" for a in agents)
        parts.append(header)
        start_round = max(1, len(obs.all_agent_choices) - 4)
        for i, rd in enumerate(recent_rounds):
            rnum = start_round + i
            vals = "  ".join(f"{rd.get(a, '?'):>10}" for a in agents)
            parts.append(f"  {rnum:<5}{vals}")

    parts.append("\nChoose A or B.")
    return "\n\n".join(parts)


def _mg_action_extractor(response: MGDecision, n: int) -> list[bool]:
    return [response.choice == "A"]


def _mg_result_formatter(result: Any) -> str:
    side = "A" if result.choice else "B"
    outcome = "WON (minority)" if result.won else "LOST (majority)"
    return (
        f"[Result: you chose {side}, A-count={result.a_count}, B-count={result.b_count}, "
        f"{outcome}, payoff: {result.payoff:+.1f}]"
    )


def _mg_fallback(n: int) -> list[bool]:
    return [True] * n  # default to A


def mg_llm(
    provider: str = "ollama",
    model: str = "llama3",
    temperature: float = 0.7,
    max_history: int = 20,
    persona: str | None = None,
    characteristics: dict[str, Any] | None = None,
    n_agents: int = 11,
    win_payoff: float = 1.0,
    lose_payoff: float = -1.0,
    tie_payoff: float = 0.0,
    api_key: str | None = None,
    base_url: str | None = None,
) -> LLMBrain:
    """Create an LLM brain configured for the Minority Game."""
    persona_text = (
        persona
        if persona is not None
        else (_build_persona(characteristics) or DEFAULT_PERSONA)
    )

    system_prompt = MG_SYSTEM_PROMPT.format(
        persona=persona_text,
        n_agents=n_agents,
        win_payoff=_fmt_num(win_payoff),
        lose_payoff=_fmt_num(lose_payoff),
        tie_payoff=_fmt_num(tie_payoff),
    )

    return LLMBrain(
        provider=provider,
        model=model,
        api_key=api_key,
        base_url=base_url,
        persona=system_prompt,
        output_schema=MGDecision,
        batch_observation_formatter=_mg_observation_formatter,
        batch_action_extractor=_mg_action_extractor,
        result_formatter=_mg_result_formatter,
        fallback_action_factory=_mg_fallback,
        temperature=temperature,
        max_history=max_history,
        brain_name=f"llm({provider}/{model})",
    )
