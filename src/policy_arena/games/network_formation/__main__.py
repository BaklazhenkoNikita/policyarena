"""Run a Network Formation Game and print results.

Usage: python -m policy_arena.games.network_formation
"""

from __future__ import annotations

from policy_arena.core.engine import Engine
from policy_arena.core.scenario import Scenario
from policy_arena.games.network_formation.brains import (
    BestResponseLinker,
    FullyConnected,
    Isolationist,
    PopularityBased,
    RandomLinker,
    StarSeeker,
)
from policy_arena.games.network_formation.model import NetworkModel


def main() -> None:
    brains = [
        FullyConnected(),
        Isolationist(),
        RandomLinker(k=2),
        PopularityBased(k=2),
        BestResponseLinker(),
        StarSeeker(),
    ]
    labels = [
        "FullyConnected",
        "Isolationist",
        "RandomLinker(k=2)",
        "PopularityBased(k=2)",
        "BestResponse",
        "StarSeeker",
    ]

    n_rounds = 50
    link_cost = 5.0
    direct_benefit = 10.0
    decay_factor = 0.3

    scenario = Scenario(
        world_class=NetworkModel,
        world_params={
            "brains": brains,
            "n_rounds": n_rounds,
            "link_cost": link_cost,
            "direct_benefit": direct_benefit,
            "decay_factor": decay_factor,
            "labels": labels,
        },
        steps=n_rounds,
        seed=42,
    )

    engine = Engine()
    results = engine.run(scenario)
    model: NetworkModel = results.extra["model"]

    print("=" * 80)
    print("NETWORK FORMATION GAME")
    print(
        f"  Players: {len(labels)}  |  Rounds: {n_rounds}  |  "
        f"Link Cost: {link_cost}  |  Direct Benefit: {direct_benefit}  |  "
        f"Decay: {decay_factor}"
    )
    print("=" * 80)

    print("\n--- Model-Level Metrics (final round) ---")
    model_df = results.model_metrics
    last = model_df.iloc[-1]
    print(f"  Network Density:       {last['network_density']:.3f}")
    print(f"  Avg Degree:            {last['avg_degree']:.2f}")
    print(f"  Clustering Coeff:      {last['clustering_coefficient']:.3f}")
    print(f"  Avg Payoff:            {last['avg_payoff']:.2f}")
    print(f"  Num Components:        {last['num_components']}")
    print(f"  Strategy Entropy:      {last['strategy_entropy']:.3f}")
    print(f"  Social Welfare:        {last['social_welfare']:.3f}")

    print("\n--- Time Series (every 5 rounds) ---")
    print(
        f"  {'Round':>6}  {'Density':>8}  {'AvgDeg':>7}  "
        f"{'Cluster':>8}  {'AvgPay':>7}  {'Comps':>6}  {'Entropy':>8}"
    )
    for step_idx in range(0, len(model_df), 5):
        row = model_df.iloc[step_idx]
        print(
            f"  {step_idx + 1:>6}  {row['network_density']:>8.3f}  "
            f"{row['avg_degree']:>7.2f}  {row['clustering_coefficient']:>8.3f}  "
            f"{row['avg_payoff']:>7.2f}  {int(row['num_components']):>6}  "
            f"{row['strategy_entropy']:>8.3f}"
        )

    print("\n--- Per-Agent Results ---")
    agents = list(model.agents)
    print(f"  {'Agent':<25} {'Brain':<25} {'Total Payoff':>13} {'Avg Degree':>11}")
    print("  " + "-" * 78)
    for agent in sorted(agents, key=lambda a: a.cumulative_payoff, reverse=True):
        avg_deg = (
            sum(len(links) for links in agent._past_links) / len(agent._past_links)
            if agent._past_links
            else 0
        )
        print(
            f"  {agent.label:<25} {agent.brain_name:<25} "
            f"{agent.cumulative_payoff:>13.1f} {avg_deg:>11.2f}"
        )

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
