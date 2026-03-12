"""Shared utilities for LLM brain adapters."""

from __future__ import annotations

from typing import Any

DEFAULT_PERSONA = (
    "You are a rational game theory agent. "
    "Analyze patterns and choose actions that maximize your long-term payoff."
)


def _build_persona(characteristics: dict[str, Any] | None) -> str:
    """Build a persona description from a characteristics dict.

    Reusable across all games. Returns empty string if no characteristics.
    """
    if not characteristics:
        return ""

    parts = []

    if "personality" in characteristics:
        parts.append(f"Personality: {characteristics['personality']}")

    if "cooperation_bias" in characteristics:
        bias = characteristics["cooperation_bias"]
        if bias > 0.3:
            parts.append("You have a strong inclination toward cooperation.")
        elif bias > 0:
            parts.append("You lean slightly toward cooperation.")
        elif bias < -0.3:
            parts.append("You have a strong inclination toward defection.")
        elif bias < 0:
            parts.append("You lean slightly toward defection.")

    if "risk_tolerance" in characteristics:
        risk = characteristics["risk_tolerance"]
        if risk > 0.7:
            parts.append("You are a risk-taker, willing to gamble for higher payoffs.")
        elif risk < 0.3:
            parts.append("You are risk-averse, preferring safe, predictable outcomes.")

    if "reasoning_style" in characteristics:
        parts.append(f"Reasoning style: {characteristics['reasoning_style']}")

    if "background_story" in characteristics:
        parts.append(f"Background: {characteristics['background_story']}")

    return "\n".join(parts)


def _fmt_num(v: float) -> str:
    """Format a number: integer if whole, else 1 decimal."""
    return str(int(v)) if v == int(v) else f"{v:.1f}"
