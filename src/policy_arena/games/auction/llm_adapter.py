"""Sealed-Bid Auction LLM adapter."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from policy_arena.brains.llm.llm_brain import LLMBrain
from policy_arena.brains.llm.shared import DEFAULT_PERSONA, _build_persona, _fmt_num


class AuctionDecision(BaseModel):
    """Decision for the Sealed-Bid Auction."""

    rationale: str = Field(description="1-2 sentence reasoning")
    bid: float = Field(description="Your bid amount (0 to max_bid)")


AUCTION_SYSTEM_PROMPT = """\
You are participating in a Sealed-Bid Auction ({auction_type}).

Each round, you receive a private value drawn uniformly from [{value_min}, {value_max}].
You submit a sealed bid (0 to {max_bid}). The highest bidder wins.

{price_rule}

Key strategic insights:
- In a second-price auction, bidding your true value is a dominant strategy.
- In a first-price auction, you should shade your bid below your value.
  The theoretical optimal bid with {n_players} players and uniform values is
  (N-1)/N * value = {bne_fraction:.0%} of your value.
- Bidding above your value risks the "winner's curse" — winning but losing money.

Your goal is to maximize your total payoff across all rounds.

{persona}

Submit a bid between 0 and {max_bid}."""


def _auction_observation_formatter(observations: list[Any]) -> str:
    obs = observations[0]
    round_num = obs.round_number + 1
    parts = [f"=== Round {round_num} ===\n"]
    parts.append(f"Auction type: {obs.auction_type}, Players: {obs.n_players}")
    parts.append(f"Your private value this round: {_fmt_num(obs.my_value)}")
    parts.append(
        f"Value range: [{_fmt_num(obs.value_min)}, {_fmt_num(obs.value_max)}], "
        f"Max bid: {_fmt_num(obs.max_bid)}"
    )

    if obs.my_past_bids:
        recent_bids = obs.my_past_bids[-10:]
        recent_values = obs.my_past_values[-10:]
        recent_payoffs = obs.my_past_payoffs[-10:]

        bids_str = ", ".join(f"{b:.1f}" for b in recent_bids)
        vals_str = ", ".join(f"{v:.1f}" for v in recent_values)
        parts.append(f"Your recent bids:    [{bids_str}]")
        parts.append(f"Your recent values:  [{vals_str}]")

        if recent_payoffs:
            pay_str = ", ".join(f"{p:.1f}" for p in recent_payoffs)
            parts.append(f"Your recent payoffs: [{pay_str}]")
            cumulative = sum(obs.my_past_payoffs)
            avg_pay = sum(recent_payoffs) / len(recent_payoffs)
            parts.append(
                f"Recent avg payoff: {avg_pay:.2f}, Cumulative: {cumulative:.1f}"
            )

        if obs.past_winning_bids:
            recent_wins = obs.past_winning_bids[-10:]
            wins_str = ", ".join(f"{w:.1f}" for w in recent_wins)
            parts.append(f"Recent winning bids: [{wins_str}]")

        if obs.past_prices_paid:
            recent_prices = obs.past_prices_paid[-10:]
            prices_str = ", ".join(f"{p:.1f}" for p in recent_prices)
            parts.append(f"Recent prices paid:  [{prices_str}]")

        # Bid-to-value ratio analysis
        if len(recent_bids) >= 3:
            ratios = [
                b / v if v > 0 else 0.0
                for b, v in zip(recent_bids, recent_values, strict=False)
            ]
            avg_ratio = sum(ratios) / len(ratios)
            parts.append(f"Your avg bid/value ratio: {avg_ratio:.2%}")

        # Win rate
        wins = sum(1 for p in obs.my_past_payoffs if p > 0)
        total = len(obs.my_past_payoffs)
        parts.append(f"Your win rate: {wins}/{total} ({wins / total:.0%})")

    else:
        parts.append("No history yet — first round.")

    parts.append(
        f"\nYour value is {_fmt_num(obs.my_value)}. Choose your bid (0 to {_fmt_num(obs.max_bid)})."
    )
    return "\n\n".join(parts)


def _auction_action_extractor(response: AuctionDecision, n: int) -> list[float]:
    return [response.bid]


def _auction_result_formatter(result: Any) -> str:
    won_str = "WON" if result.won else "LOST"
    return (
        f"[Result: you bid {result.my_bid:.1f} (value was {result.my_value:.1f}), "
        f"{won_str}, winning bid: {result.winning_bid:.1f}, "
        f"price paid: {result.price_paid:.1f}, your payoff: {result.payoff:.1f}]"
    )


def _auction_fallback(n: int) -> list[float]:
    return [50.0] * n  # bid half of default value_max


def auction_llm(
    provider: str = "ollama",
    model: str = "llama3",
    temperature: float = 0.7,
    max_history: int = 20,
    persona: str | None = None,
    characteristics: dict[str, Any] | None = None,
    auction_type: str = "first_price",
    value_min: float = 0.0,
    value_max: float = 100.0,
    max_bid: float = 150.0,
    n_players: int = 5,
    api_key: str | None = None,
    base_url: str | None = None,
) -> LLMBrain:
    """Create an LLM brain configured for the Sealed-Bid Auction."""
    persona_text = (
        persona
        if persona is not None
        else (_build_persona(characteristics) or DEFAULT_PERSONA)
    )

    if auction_type == "second_price":
        price_rule = (
            "Payment rule: SECOND-PRICE (Vickrey). The winner pays the "
            "second-highest bid, not their own bid."
        )
    else:
        price_rule = "Payment rule: FIRST-PRICE. The winner pays their own bid."

    bne_fraction = (n_players - 1) / n_players if n_players > 1 else 1.0

    system_prompt = AUCTION_SYSTEM_PROMPT.format(
        persona=persona_text,
        auction_type=auction_type,
        value_min=_fmt_num(value_min),
        value_max=_fmt_num(value_max),
        max_bid=_fmt_num(max_bid),
        n_players=n_players,
        price_rule=price_rule,
        bne_fraction=bne_fraction,
    )

    return LLMBrain(
        provider=provider,
        model=model,
        api_key=api_key,
        base_url=base_url,
        persona=system_prompt,
        output_schema=AuctionDecision,
        batch_observation_formatter=_auction_observation_formatter,
        batch_action_extractor=_auction_action_extractor,
        result_formatter=_auction_result_formatter,
        fallback_action_factory=_auction_fallback,
        temperature=temperature,
        max_history=max_history,
        brain_name=f"llm({provider}/{model})",
    )
