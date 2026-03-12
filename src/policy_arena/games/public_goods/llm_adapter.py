"""Public Goods Game LLM adapter."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from policy_arena.brains.llm.llm_brain import LLMBrain
from policy_arena.brains.llm.shared import DEFAULT_PERSONA, _build_persona, _fmt_num


class PGDecision(BaseModel):
    """Decision for the Public Goods Game."""

    rationale: str = Field(description="1-2 sentence reasoning")
    contribution: float = Field(description="Amount to contribute (0 to endowment)")


PG_SYSTEM_PROMPT = """\
You are playing an Iterated Public Goods Game.

Each round, you choose how much of your endowment ({endowment}) to contribute
to a shared pool. The pool is multiplied by {multiplier}x and split equally
among all {n_players} players.

Payoff = (endowment - your_contribution) + (total_contributions × {multiplier}) / {n_players}

Nash Equilibrium: contribute 0 (free-ride). But if everyone free-rides,
everyone gets only {endowment}. If everyone contributes fully, everyone
gets {endowment} × {multiplier} = {full_coop_payoff}.

Your goal is to maximize your total payoff across all rounds.

{persona}

Choose a contribution amount between 0 and {endowment}."""


def _pg_observation_formatter(observations: list[Any]) -> str:
    obs = observations[0]
    round_num = obs.round_number + 1
    parts = [f"=== Round {round_num} ===\n"]
    parts.append(
        f"Endowment: {_fmt_num(obs.endowment)}, Multiplier: {obs.multiplier}x, Players: {obs.n_players}"
    )

    if obs.group_past_averages:
        recent_avg = obs.group_past_averages[-10:]
        avg_str = ", ".join(f"{a:.1f}" for a in recent_avg)
        parts.append(f"Group avg contributions: [{avg_str}]")

        # Contribution trend (compare first half vs second half of recent window)
        if len(recent_avg) >= 4:
            mid = len(recent_avg) // 2
            first_half = sum(recent_avg[:mid]) / mid
            second_half = sum(recent_avg[mid:]) / (len(recent_avg) - mid)
            diff = second_half - first_half
            if diff > 0.5:
                trend = "rising"
            elif diff < -0.5:
                trend = "falling"
            else:
                trend = "stable"
            parts.append(
                f"Group contribution trend: {trend} (early avg {first_half:.1f} -> recent avg {second_half:.1f})"
            )

        if obs.my_past_contributions:
            my_recent = obs.my_past_contributions[-10:]
            my_str = ", ".join(f"{c:.1f}" for c in my_recent)
            parts.append(f"Your contributions:     [{my_str}]")

        if obs.my_past_payoffs:
            pay_recent = obs.my_past_payoffs[-10:]
            total = sum(pay_recent)
            avg_pay = total / len(pay_recent)
            cumulative = sum(obs.my_past_payoffs)
            parts.append(f"Your recent payoffs: total={total:.1f}, avg={avg_pay:.2f}")
            parts.append(f"Your cumulative payoff: {cumulative:.1f}")

            # Payoff breakdown for last round
            if obs.my_past_contributions:
                last_contrib = obs.my_past_contributions[-1]
                last_payoff = obs.my_past_payoffs[-1]
                kept = obs.endowment - last_contrib
                received_share = last_payoff - kept
                parts.append(
                    f"Last round breakdown: kept {kept:.1f} + received share {received_share:.1f} = payoff {last_payoff:.1f}"
                )

            # Personal efficiency: avg payoff when contributing high vs low
            if obs.my_past_contributions and len(obs.my_past_contributions) >= 3:
                median_contrib = sorted(obs.my_past_contributions)[
                    len(obs.my_past_contributions) // 2
                ]
                high_payoffs = [
                    p
                    for c, p in zip(
                        obs.my_past_contributions, obs.my_past_payoffs, strict=False
                    )
                    if c >= median_contrib
                ]
                low_payoffs = [
                    p
                    for c, p in zip(
                        obs.my_past_contributions, obs.my_past_payoffs, strict=False
                    )
                    if c < median_contrib
                ]
                if high_payoffs and low_payoffs:
                    avg_high = sum(high_payoffs) / len(high_payoffs)
                    avg_low = sum(low_payoffs) / len(low_payoffs)
                    parts.append(
                        f"Efficiency: avg payoff when contributing >= median ({median_contrib:.1f}): {avg_high:.2f}, "
                        f"below median: {avg_low:.2f}"
                    )
        # Per-agent contributions (last 5 rounds)
        if obs.all_agent_contributions:
            recent_rounds = obs.all_agent_contributions[-5:]
            parts.append("--- Per-Agent Contributions (recent rounds) ---")
            agents = list(recent_rounds[0].keys())
            header = "Round  " + "  ".join(f"{a[:12]:>12}" for a in agents)
            parts.append(header)
            start_round = max(1, len(obs.all_agent_contributions) - 4)
            for i, rd in enumerate(recent_rounds):
                rnum = start_round + i
                vals = "  ".join(f"{rd.get(a, 0):>12.1f}" for a in agents)
                parts.append(f"  {rnum:<5}{vals}")

    else:
        parts.append("No history yet — first round.")

    parts.append(f"\nChoose your contribution (0 to {_fmt_num(obs.endowment)}).")
    return "\n\n".join(parts)


def _pg_action_extractor(response: PGDecision, n: int) -> list[float]:
    return [response.contribution]


def _pg_result_formatter(result: Any) -> str:
    return (
        f"[Result: you contributed {result.contribution:.1f}, "
        f"group total was {result.group_total_contribution:.1f} "
        f"(avg {result.group_average_contribution:.1f}), "
        f"pool after multiplier: {result.pool_after_multiplier:.1f}, "
        f"your payoff: {result.payoff:.1f}]"
    )


def _pg_fallback(n: int) -> list[float]:
    return [10.0] * n  # contribute half of default endowment


def pg_llm(
    provider: str = "ollama",
    model: str = "llama3",
    temperature: float = 0.7,
    max_history: int = 20,
    persona: str | None = None,
    characteristics: dict[str, Any] | None = None,
    endowment: float = 20.0,
    multiplier: float = 1.6,
    n_players: int = 5,
    api_key: str | None = None,
    base_url: str | None = None,
) -> LLMBrain:
    """Create an LLM brain configured for the Public Goods Game."""
    persona_text = (
        persona
        if persona is not None
        else (_build_persona(characteristics) or DEFAULT_PERSONA)
    )

    full_coop = endowment * multiplier
    system_prompt = PG_SYSTEM_PROMPT.format(
        persona=persona_text,
        endowment=_fmt_num(endowment),
        multiplier=multiplier,
        n_players=n_players,
        full_coop_payoff=_fmt_num(full_coop),
    )

    return LLMBrain(
        provider=provider,
        model=model,
        api_key=api_key,
        base_url=base_url,
        persona=system_prompt,
        output_schema=PGDecision,
        batch_observation_formatter=_pg_observation_formatter,
        batch_action_extractor=_pg_action_extractor,
        result_formatter=_pg_result_formatter,
        fallback_action_factory=_pg_fallback,
        temperature=temperature,
        max_history=max_history,
        brain_name=f"llm({provider}/{model})",
    )
