"""Network Formation Game LLM adapter."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from policy_arena.brains.llm.llm_brain import LLMBrain
from policy_arena.brains.llm.shared import DEFAULT_PERSONA, _build_persona, _fmt_num


class NetworkDecision(BaseModel):
    """Decision for the Network Formation Game."""

    rationale: str = Field(description="1-2 sentence reasoning")
    proposed_links: list[int] = Field(description="List of agent indices to connect to")


NF_SYSTEM_PROMPT = """\
You are playing an Iterated Network Formation Game.

Each round, you choose which other agents to propose links to.
A link forms if EITHER side proposes it (unilateral link formation).

There are {n_players} agents (indexed 0 to {max_index}).
You are agent {my_index}.

Costs and benefits:
- Each direct connection costs you {link_cost} per round
- Each direct connection gives you {direct_benefit} benefit
- Each indirect connection (distance 2) gives you {indirect_benefit} benefit
- Net benefit per direct link: {net_per_link}
- You can maintain at most {max_links} links

Your goal is to maximize your total payoff across all rounds.
Payoff = (direct_benefit x direct_neighbors) + (indirect_benefit x dist-2_neighbors) - (link_cost x your_degree)

{persona}

Choose which agents to connect to (list of agent indices)."""


def _nf_observation_formatter(observations: list[Any]) -> str:
    obs = observations[0]
    round_num = obs.round_number + 1
    parts = [f"=== Round {round_num} ===\n"]
    parts.append(
        f"Players: {obs.n_players}, Link cost: {_fmt_num(obs.link_cost)}, "
        f"Direct benefit: {_fmt_num(obs.direct_benefit)}, "
        f"Decay: {obs.decay_factor}, Max links: {obs.max_links}"
    )
    parts.append(f"You are agent {obs.my_agent_index}.")

    if obs.my_current_links:
        links_str = ", ".join(str(l) for l in obs.my_current_links)
        parts.append(f"Your current links: [{links_str}] (degree: {len(obs.my_current_links)})")
    else:
        parts.append("You currently have no links.")

    if obs.network_adjacency:
        parts.append("--- Current Network ---")
        for idx in sorted(obs.network_adjacency.keys()):
            neighbors = obs.network_adjacency[idx]
            marker = " (you)" if idx == obs.my_agent_index else ""
            if neighbors:
                n_str = ", ".join(str(n) for n in neighbors)
                parts.append(f"  Agent {idx}{marker}: [{n_str}]")
            else:
                parts.append(f"  Agent {idx}{marker}: []")

    if obs.my_past_payoffs:
        recent = obs.my_past_payoffs[-10:]
        pay_str = ", ".join(f"{p:.1f}" for p in recent)
        cumulative = sum(obs.my_past_payoffs)
        avg_pay = sum(recent) / len(recent)
        parts.append(f"Your recent payoffs: [{pay_str}]")
        parts.append(f"Your avg recent payoff: {avg_pay:.2f}, cumulative: {cumulative:.1f}")

    if obs.my_past_links:
        recent_links = obs.my_past_links[-5:]
        parts.append("Your recent link choices:")
        start = max(1, len(obs.my_past_links) - 4)
        for i, links in enumerate(recent_links):
            rnum = start + i
            l_str = ", ".join(str(l) for l in links) if links else "none"
            parts.append(f"  Round {rnum}: [{l_str}]")

    if obs.network_density_history:
        recent_density = obs.network_density_history[-10:]
        d_str = ", ".join(f"{d:.3f}" for d in recent_density)
        parts.append(f"Network density history: [{d_str}]")

    parts.append(
        f"\nChoose agent indices to connect to (0 to {obs.n_players - 1}, excluding yourself {obs.my_agent_index})."
    )
    return "\n\n".join(parts)


def _nf_action_extractor(response: NetworkDecision, n: int) -> list[list[int]]:
    return [response.proposed_links]


def _nf_result_formatter(result: Any) -> str:
    links_str = ", ".join(str(l) for l in result.my_links) if result.my_links else "none"
    return (
        f"[Result: your links [{links_str}], "
        f"degree {result.my_degree}, "
        f"network density {result.network_density:.3f}, "
        f"your payoff: {result.my_payoff:.1f}]"
    )


def _nf_fallback(n: int) -> list[list[int]]:
    return [[]] * n  # propose no links on failure


def nf_llm(
    provider: str = "ollama",
    model: str = "llama3",
    temperature: float = 0.7,
    max_history: int = 20,
    persona: str | None = None,
    characteristics: dict[str, Any] | None = None,
    n_players: int = 6,
    link_cost: float = 5.0,
    direct_benefit: float = 10.0,
    decay_factor: float = 0.3,
    max_links: int | None = None,
    my_index: int = 0,
    api_key: str | None = None,
    base_url: str | None = None,
) -> LLMBrain:
    """Create an LLM brain configured for the Network Formation Game."""
    persona_text = (
        persona
        if persona is not None
        else (_build_persona(characteristics) or DEFAULT_PERSONA)
    )

    effective_max_links = max_links if max_links is not None else n_players - 1
    indirect_benefit = direct_benefit * decay_factor
    net_per_link = direct_benefit - link_cost

    system_prompt = NF_SYSTEM_PROMPT.format(
        persona=persona_text,
        n_players=n_players,
        max_index=n_players - 1,
        my_index=my_index,
        link_cost=_fmt_num(link_cost),
        direct_benefit=_fmt_num(direct_benefit),
        indirect_benefit=_fmt_num(indirect_benefit),
        net_per_link=_fmt_num(net_per_link),
        max_links=effective_max_links,
    )

    return LLMBrain(
        provider=provider,
        model=model,
        api_key=api_key,
        base_url=base_url,
        persona=system_prompt,
        output_schema=NetworkDecision,
        batch_observation_formatter=_nf_observation_formatter,
        batch_action_extractor=_nf_action_extractor,
        result_formatter=_nf_result_formatter,
        fallback_action_factory=_nf_fallback,
        temperature=temperature,
        max_history=max_history,
        brain_name=f"llm({provider}/{model})",
    )
