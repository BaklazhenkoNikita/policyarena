"""Tragedy of the Commons LLM adapter."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from policy_arena.brains.llm.llm_brain import LLMBrain
from policy_arena.brains.llm.shared import DEFAULT_PERSONA, _build_persona, _fmt_num


class TCDecision(BaseModel):
    """Decision for the Tragedy of the Commons."""

    rationale: str = Field(description="1-2 sentence reasoning")
    harvest: float = Field(
        description="Amount to harvest (0 to your harvest cap shown in the observation)"
    )


TC_SYSTEM_PROMPT = """\
You are playing an Iterated Tragedy of the Commons.

You share a renewable resource (max capacity: {max_resource}) with {n_agents} other agents.
Each round, everyone simultaneously chooses how much to harvest.

Resource dynamics:
- After all harvests are deducted, the remaining resource regenerates by {growth_rate}x
  (capped at {max_resource}).
- If total harvest requests exceed the resource, harvests are proportionally scaled down.
- If the resource is depleted to 0, no one can harvest anything.

Per-agent harvest cap: {harvest_cap_pct}% of current resource ({harvest_cap_amount:.1f} at full capacity).
The sustainable total harvest per round is: {sustainable_total:.1f}
(your fair sustainable share: {sustainable_per_agent:.1f})

Your payoff each round equals the amount you actually harvest.
Your goal is to maximize your total payoff across all rounds.

{persona}

Choose a harvest amount between 0 and your harvest cap."""


def _tc_observation_formatter(observations: list[Any]) -> str:
    obs = observations[0]
    round_num = obs.round_number + 1
    harvest_cap_frac = getattr(obs, "harvest_cap", 0.15)
    max_harvest = obs.resource_level * harvest_cap_frac
    parts = [f"=== Round {round_num} ===\n"]
    parts.append(
        f"Current resource: {obs.resource_level:.1f} / {obs.max_resource:.1f} ({obs.resource_level / obs.max_resource * 100:.0f}% full)"
    )
    parts.append(
        f"Your harvest cap this round: {max_harvest:.1f} ({harvest_cap_frac * 100:.0f}% of current resource)"
    )
    parts.append(f"Growth rate: {obs.growth_rate}x, Agents: {obs.n_agents}")

    sustainable_total = obs.resource_level * (obs.growth_rate - 1)
    sustainable_share = sustainable_total / obs.n_agents if obs.n_agents > 0 else 0
    parts.append(
        f"Sustainable harvest this round: {sustainable_total:.1f} total, {sustainable_share:.1f} per agent"
    )

    if obs.resource_history:
        recent = obs.resource_history[-10:]
        res_str = ", ".join(f"{r:.1f}" for r in recent)
        parts.append(f"Resource history: [{res_str}]")

        if len(recent) >= 2:
            trend = recent[-1] - recent[0]
            parts.append(
                f"Resource trend: {'rising' if trend > 1 else 'falling' if trend < -1 else 'stable'}"
            )

    if obs.group_past_total_harvests:
        recent_h = obs.group_past_total_harvests[-10:]
        h_str = ", ".join(f"{h:.1f}" for h in recent_h)
        parts.append(f"Group total harvests: [{h_str}]")

    if obs.my_past_harvests:
        my_h = obs.my_past_harvests[-10:]
        parts.append(f"Your harvests: [{', '.join(f'{h:.1f}' for h in my_h)}]")

    if obs.my_past_payoffs:
        cumulative = sum(obs.my_past_payoffs)
        parts.append(f"Your cumulative payoff: {cumulative:.1f}")

    # Per-agent harvests (last 5 rounds)
    if obs.all_agent_harvests:
        recent_rounds = obs.all_agent_harvests[-5:]
        parts.append("--- Per-Agent Harvests (recent rounds) ---")
        agents = list(recent_rounds[0].keys())
        header = "Round  " + "  ".join(f"{a[:12]:>12}" for a in agents)
        parts.append(header)
        start_round = max(1, len(obs.all_agent_harvests) - 4)
        for i, rd in enumerate(recent_rounds):
            rnum = start_round + i
            vals = "  ".join(f"{rd.get(a, 0):>12.1f}" for a in agents)
            parts.append(f"  {rnum:<5}{vals}")

    parts.append("\nChoose your harvest amount.")
    return "\n\n".join(parts)


def _tc_action_extractor(response: TCDecision, n: int) -> list[float]:
    return [response.harvest]


def _tc_result_formatter(result: Any) -> str:
    return (
        f"[Result: you requested {result.harvest_requested:.1f}, "
        f"received {result.harvest_actual:.1f}, "
        f"group total harvest: {result.group_total_harvest:.1f}, "
        f"resource: {result.resource_before:.1f} → {result.resource_after:.1f}]"
    )


def _tc_fallback(n: int) -> list[float]:
    return [5.0] * n


def tc_llm(
    provider: str = "ollama",
    model: str = "llama3",
    temperature: float = 0.7,
    max_history: int = 20,
    persona: str | None = None,
    characteristics: dict[str, Any] | None = None,
    max_resource: float = 100.0,
    growth_rate: float = 1.5,
    harvest_cap: float = 0.15,
    n_agents: int = 8,
    api_key: str | None = None,
    base_url: str | None = None,
) -> LLMBrain:
    """Create an LLM brain configured for the Tragedy of the Commons."""
    persona_text = (
        persona
        if persona is not None
        else (_build_persona(characteristics) or DEFAULT_PERSONA)
    )

    sustainable_total = max_resource * (growth_rate - 1)
    sustainable_per_agent = sustainable_total / n_agents if n_agents > 0 else 0
    harvest_cap_amount = max_resource * harvest_cap

    system_prompt = TC_SYSTEM_PROMPT.format(
        persona=persona_text,
        max_resource=_fmt_num(max_resource),
        growth_rate=growth_rate,
        n_agents=n_agents,
        sustainable_total=sustainable_total,
        sustainable_per_agent=sustainable_per_agent,
        harvest_cap_pct=f"{harvest_cap * 100:.0f}",
        harvest_cap_amount=harvest_cap_amount,
    )

    return LLMBrain(
        provider=provider,
        model=model,
        api_key=api_key,
        base_url=base_url,
        persona=system_prompt,
        output_schema=TCDecision,
        batch_observation_formatter=_tc_observation_formatter,
        batch_action_extractor=_tc_action_extractor,
        result_formatter=_tc_result_formatter,
        fallback_action_factory=_tc_fallback,
        temperature=temperature,
        max_history=max_history,
        brain_name=f"llm({provider}/{model})",
    )
