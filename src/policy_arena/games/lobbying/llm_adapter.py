"""Lobbying / Rent-Seeking Contest LLM adapter."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from policy_arena.brains.llm.llm_brain import LLMBrain
from policy_arena.brains.llm.shared import DEFAULT_PERSONA, _build_persona, _fmt_num


class LobbyingDecision(BaseModel):
    """Decision for the Lobbying / Rent-Seeking Contest."""

    rationale: str = Field(description="1-2 sentence reasoning")
    spend: float = Field(description="Amount to spend on lobbying (0 to budget)")


LOBBYING_SYSTEM_PROMPT = """\
You are playing an Iterated Lobbying Contest (Tullock Rent-Seeking Contest).

Each round, {n_players} players compete for a prize worth {prize_value} by
spending resources on lobbying. Your budget is {budget} per round.

Your probability of winning = (your_spend^{sensitivity}) / sum(all_spend^{sensitivity}).
If you win: payoff = {prize_value} - your_spend.
If you lose: payoff = -your_spend.

Nash Equilibrium (symmetric, r=1): each player spends {ne_spend}.
At NE, total spending equals {ne_total_spend} out of {prize_value} prize
(rent dissipation = {ne_dissipation:.0%}).

Spending more increases your win probability but costs more if you lose.
Spending nothing guarantees zero payoff (no loss, no gain).
Total lobbying spending is socially wasteful — it transfers value but
creates no new wealth.

Your goal is to maximize your total payoff across all rounds.

{persona}

Choose a spend amount between 0 and {budget}."""


def _lobbying_observation_formatter(observations: list[Any]) -> str:
    obs = observations[0]
    round_num = obs.round_number + 1
    parts = [f"=== Round {round_num} ===\n"]
    parts.append(
        f"Prize: {_fmt_num(obs.prize_value)}, Budget: {_fmt_num(obs.budget)}, "
        f"Sensitivity (r): {obs.contest_sensitivity}, Players: {obs.n_players}"
    )

    if obs.past_total_spends:
        recent_total = obs.past_total_spends[-10:]
        total_str = ", ".join(f"{t:.1f}" for t in recent_total)
        parts.append(f"Total spends:   [{total_str}]")

        recent_winner = obs.past_winner_spends[-10:]
        winner_str = ", ".join(f"{w:.1f}" for w in recent_winner)
        parts.append(f"Winner spends:  [{winner_str}]")

        # Dissipation trend
        if len(recent_total) >= 4:
            mid = len(recent_total) // 2
            first_half = sum(recent_total[:mid]) / mid
            second_half = sum(recent_total[mid:]) / (len(recent_total) - mid)
            diff = second_half - first_half
            if diff > 2:
                trend = "rising"
            elif diff < -2:
                trend = "falling"
            else:
                trend = "stable"
            parts.append(
                f"Spending trend: {trend} (early avg {first_half:.1f} -> recent avg {second_half:.1f})"
            )

        if obs.my_past_spends:
            my_recent = obs.my_past_spends[-10:]
            my_str = ", ".join(f"{s:.1f}" for s in my_recent)
            parts.append(f"Your spends:    [{my_str}]")

        if obs.my_past_wins:
            recent_wins = obs.my_past_wins[-10:]
            win_rate = sum(recent_wins) / len(recent_wins)
            total_wins = sum(obs.my_past_wins)
            parts.append(
                f"Your recent win rate: {win_rate:.0%} ({sum(recent_wins)}/{len(recent_wins)}), "
                f"total wins: {total_wins}/{len(obs.my_past_wins)}"
            )

        if obs.my_past_payoffs:
            pay_recent = obs.my_past_payoffs[-10:]
            total = sum(pay_recent)
            avg_pay = total / len(pay_recent)
            cumulative = sum(obs.my_past_payoffs)
            parts.append(f"Your recent payoffs: total={total:.1f}, avg={avg_pay:.2f}")
            parts.append(f"Your cumulative payoff: {cumulative:.1f}")

            # Last round breakdown
            if obs.my_past_spends:
                last_spend = obs.my_past_spends[-1]
                last_payoff = obs.my_past_payoffs[-1]
                last_won = obs.my_past_wins[-1] if obs.my_past_wins else False
                outcome = "WON" if last_won else "LOST"
                parts.append(
                    f"Last round: spent {last_spend:.1f}, {outcome}, payoff {last_payoff:.1f}"
                )

        # Per-agent spends (last 5 rounds)
        if obs.all_agent_spends:
            recent_rounds = obs.all_agent_spends[-5:]
            parts.append("--- Per-Agent Spends (recent rounds) ---")
            agents = list(recent_rounds[0].keys())
            header = "Round  " + "  ".join(f"{a[:12]:>12}" for a in agents)
            parts.append(header)
            start_round = max(1, len(obs.all_agent_spends) - 4)
            for i, rd in enumerate(recent_rounds):
                rnum = start_round + i
                vals = "  ".join(f"{rd.get(a, 0):>12.1f}" for a in agents)
                parts.append(f"  {rnum:<5}{vals}")

    else:
        parts.append("No history yet — first round.")

    parts.append(f"\nChoose your lobbying spend (0 to {_fmt_num(obs.budget)}).")
    return "\n\n".join(parts)


def _lobbying_action_extractor(response: LobbyingDecision, n: int) -> list[float]:
    return [response.spend]


def _lobbying_result_formatter(result: Any) -> str:
    outcome = "WON" if result.won else "LOST"
    return (
        f"[Result: you spent {result.my_spend:.1f}, "
        f"total spend was {result.total_spend:.1f}, "
        f"you {outcome}, payoff: {result.payoff:.1f}]"
    )


def _lobbying_fallback(n: int) -> list[float]:
    return [10.0] * n  # moderate spend as fallback


def lobbying_llm(
    provider: str = "ollama",
    model: str = "llama3",
    temperature: float = 0.7,
    max_history: int = 20,
    persona: str | None = None,
    characteristics: dict[str, Any] | None = None,
    prize_value: float = 100.0,
    budget: float = 50.0,
    contest_sensitivity: float = 1.0,
    n_players: int = 5,
    api_key: str | None = None,
    base_url: str | None = None,
) -> LLMBrain:
    """Create an LLM brain configured for the Lobbying Contest."""
    persona_text = (
        persona
        if persona is not None
        else (_build_persona(characteristics) or DEFAULT_PERSONA)
    )

    n = n_players
    ne_spend = (n - 1) / (n**2) * prize_value if n > 1 else 0
    ne_total_spend = ne_spend * n
    ne_dissipation = ne_total_spend / prize_value if prize_value > 0 else 0

    system_prompt = LOBBYING_SYSTEM_PROMPT.format(
        persona=persona_text,
        prize_value=_fmt_num(prize_value),
        budget=_fmt_num(budget),
        sensitivity=contest_sensitivity,
        n_players=n_players,
        ne_spend=_fmt_num(ne_spend),
        ne_total_spend=_fmt_num(ne_total_spend),
        ne_dissipation=ne_dissipation,
    )

    return LLMBrain(
        provider=provider,
        model=model,
        api_key=api_key,
        base_url=base_url,
        persona=system_prompt,
        output_schema=LobbyingDecision,
        batch_observation_formatter=_lobbying_observation_formatter,
        batch_action_extractor=_lobbying_action_extractor,
        result_formatter=_lobbying_result_formatter,
        fallback_action_factory=_lobbying_fallback,
        temperature=temperature,
        max_history=max_history,
        brain_name=f"llm({provider}/{model})",
    )
