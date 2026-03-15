"""Cournot Oligopoly LLM adapter."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from policy_arena.brains.llm.llm_brain import LLMBrain
from policy_arena.brains.llm.shared import DEFAULT_PERSONA, _build_persona, _fmt_num


class CournotDecision(BaseModel):
    """Decision for the Cournot Oligopoly."""

    rationale: str = Field(description="1-2 sentence reasoning")
    quantity: float = Field(description="Production quantity (0 to max_quantity)")


COURNOT_SYSTEM_PROMPT = """\
You are a firm competing in a Cournot Oligopoly (quantity competition).

Each round, you choose how much to produce (0 to {max_quantity}).
All firms choose simultaneously. The market price depends on total output:

  Price = max(0, {max_price} - total_quantity_all_firms)
  Your Profit = Price × your_quantity - {marginal_cost} × your_quantity

There are {n_players} firms in total.

Nash Equilibrium quantity per firm: {ne_quantity:.1f} (each firm profits {ne_profit:.1f}).
If all firms collude at monopoly share ({monopoly_q:.1f} each), each profits {monopoly_share_profit:.1f}.
But any firm can deviate from collusion to grab more profit short-term.

Your goal is to maximize your total profit across all rounds.

{persona}

Choose a production quantity between 0 and {max_quantity}."""


def _cournot_observation_formatter(observations: list[Any]) -> str:
    obs = observations[0]
    round_num = obs.round_number + 1
    parts = [f"=== Round {round_num} ===\n"]
    parts.append(
        f"Max price: {_fmt_num(obs.max_price)}, Marginal cost: {_fmt_num(obs.marginal_cost)}, "
        f"Max quantity: {_fmt_num(obs.max_quantity)}, Firms: {obs.n_players}"
    )

    if obs.market_past_prices:
        recent_prices = obs.market_past_prices[-10:]
        price_str = ", ".join(f"{p:.1f}" for p in recent_prices)
        parts.append(f"Market prices:       [{price_str}]")

        recent_totals = obs.market_past_total_quantities[-10:]
        total_str = ", ".join(f"{t:.1f}" for t in recent_totals)
        parts.append(f"Total market output:  [{total_str}]")

        if len(recent_prices) >= 4:
            mid = len(recent_prices) // 2
            first_half = sum(recent_prices[:mid]) / mid
            second_half = sum(recent_prices[mid:]) / (len(recent_prices) - mid)
            diff = second_half - first_half
            if diff > 1.0:
                trend = "rising"
            elif diff < -1.0:
                trend = "falling"
            else:
                trend = "stable"
            parts.append(
                f"Price trend: {trend} (early avg {first_half:.1f} -> recent avg {second_half:.1f})"
            )

        if obs.my_past_quantities:
            my_recent = obs.my_past_quantities[-10:]
            my_str = ", ".join(f"{q:.1f}" for q in my_recent)
            parts.append(f"Your quantities:     [{my_str}]")

        if obs.my_past_profits:
            profit_recent = obs.my_past_profits[-10:]
            total = sum(profit_recent)
            avg_profit = total / len(profit_recent)
            cumulative = sum(obs.my_past_profits)
            parts.append(f"Your recent profits: total={total:.1f}, avg={avg_profit:.2f}")
            parts.append(f"Your cumulative profit: {cumulative:.1f}")

            if obs.my_past_quantities:
                last_q = obs.my_past_quantities[-1]
                last_profit = obs.my_past_profits[-1]
                last_price = obs.market_past_prices[-1]
                revenue = last_price * last_q
                cost = obs.marginal_cost * last_q
                parts.append(
                    f"Last round: produced {last_q:.1f}, price was {last_price:.1f}, "
                    f"revenue {revenue:.1f} - cost {cost:.1f} = profit {last_profit:.1f}"
                )

        # Per-agent quantities (last 5 rounds)
        if obs.all_agent_quantities:
            recent_rounds = obs.all_agent_quantities[-5:]
            parts.append("--- Per-Firm Quantities (recent rounds) ---")
            agents = list(recent_rounds[0].keys())
            header = "Round  " + "  ".join(f"{a[:12]:>12}" for a in agents)
            parts.append(header)
            start_round = max(1, len(obs.all_agent_quantities) - 4)
            for i, rd in enumerate(recent_rounds):
                rnum = start_round + i
                vals = "  ".join(f"{rd.get(a, 0):>12.1f}" for a in agents)
                parts.append(f"  {rnum:<5}{vals}")

    else:
        parts.append("No history yet — first round.")

    parts.append(f"\nChoose your production quantity (0 to {_fmt_num(obs.max_quantity)}).")
    return "\n\n".join(parts)


def _cournot_action_extractor(response: CournotDecision, n: int) -> list[float]:
    return [response.quantity]


def _cournot_result_formatter(result: Any) -> str:
    return (
        f"[Result: you produced {result.quantity:.1f}, "
        f"total market output was {result.market_total_quantity:.1f}, "
        f"price was {result.market_price:.1f}, "
        f"revenue {result.revenue:.1f} - cost {result.cost:.1f} = "
        f"profit {result.profit:.1f}]"
    )


def _cournot_fallback(n: int) -> list[float]:
    return [15.0] * n  # ~NE for default params with 5 players


def cournot_llm(
    provider: str = "ollama",
    model: str = "llama3",
    temperature: float = 0.7,
    max_history: int = 20,
    persona: str | None = None,
    characteristics: dict[str, Any] | None = None,
    max_price: float = 100.0,
    marginal_cost: float = 10.0,
    max_quantity: float = 50.0,
    n_players: int = 5,
    api_key: str | None = None,
    base_url: str | None = None,
) -> LLMBrain:
    """Create an LLM brain configured for the Cournot Oligopoly."""
    persona_text = (
        persona
        if persona is not None
        else (_build_persona(characteristics) or DEFAULT_PERSONA)
    )

    n = n_players
    ne_q = (max_price - marginal_cost) / (n + 1)
    ne_price = max_price - n * ne_q
    ne_profit = (ne_price - marginal_cost) * ne_q

    monopoly_q = (max_price - marginal_cost) / (2 * n)
    monopoly_price = max_price - n * monopoly_q
    monopoly_share_profit = (monopoly_price - marginal_cost) * monopoly_q

    system_prompt = COURNOT_SYSTEM_PROMPT.format(
        persona=persona_text,
        max_price=_fmt_num(max_price),
        marginal_cost=_fmt_num(marginal_cost),
        max_quantity=_fmt_num(max_quantity),
        n_players=n,
        ne_quantity=ne_q,
        ne_profit=ne_profit,
        monopoly_q=monopoly_q,
        monopoly_share_profit=monopoly_share_profit,
    )

    return LLMBrain(
        provider=provider,
        model=model,
        api_key=api_key,
        base_url=base_url,
        persona=system_prompt,
        output_schema=CournotDecision,
        batch_observation_formatter=_cournot_observation_formatter,
        batch_action_extractor=_cournot_action_extractor,
        result_formatter=_cournot_result_formatter,
        fallback_action_factory=_cournot_fallback,
        temperature=temperature,
        max_history=max_history,
        brain_name=f"llm({provider}/{model})",
    )
