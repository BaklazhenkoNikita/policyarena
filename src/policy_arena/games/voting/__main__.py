"""Run a Voting & Election Game and print results.

Usage: python -m policy_arena.games.voting
"""

from __future__ import annotations

from policy_arena.core.engine import Engine
from policy_arena.core.scenario import Scenario
from policy_arena.games.voting.brains import (
    ContrarianVoter,
    RandomVoter,
    SincereVoter,
    StrategicVoter,
)
from policy_arena.games.voting.model import VotingModel


def main() -> None:
    brains = [
        SincereVoter(),
        SincereVoter(),
        StrategicVoter(),
        StrategicVoter(),
        RandomVoter(),
        ContrarianVoter(),
    ]
    labels = [
        "Sincere_1",
        "Sincere_2",
        "Strategic_1",
        "Strategic_2",
        "Random",
        "Contrarian",
    ]

    n_rounds = 50
    n_candidates = 4

    scenario = Scenario(
        world_class=VotingModel,
        world_params={
            "brains": brains,
            "n_rounds": n_rounds,
            "n_candidates": n_candidates,
            "voting_rule": "plurality",
            "labels": labels,
        },
        steps=n_rounds,
        seed=42,
    )

    engine = Engine()
    results = engine.run(scenario)
    model: VotingModel = results.extra["model"]

    print("=" * 80)
    print("VOTING & ELECTION GAME")
    print(
        f"  Voters: {len(labels)}  |  Rounds: {n_rounds}  |  "
        f"Candidates: {n_candidates}  |  Rule: plurality"
    )
    print(
        "  Candidate positions: "
        + ", ".join(f"C{i}={p:.1f}" for i, p in enumerate(model.candidate_positions))
    )
    print("=" * 80)

    print("\n--- Model-Level Metrics (final round) ---")
    model_df = results.model_metrics
    last = model_df.iloc[-1]
    print(f"  Winner Position:             {last['winner_position']:.1f}")
    print(f"  Sincere Voting Rate:         {last['sincere_voting_rate']:.3f}")
    print(
        f"  Effective # of Candidates:   {last['effective_number_of_candidates']:.3f}"
    )
    print(f"  Social Welfare:              {last['social_welfare']:.3f}")
    print(f"  Strategy Entropy:            {last['strategy_entropy']:.3f}")

    print("\n--- Time Series (every 5 rounds) ---")
    print(
        f"  {'Round':>6}  {'Winner':>7}  {'WinPos':>7}  "
        f"{'Sincere':>8}  {'EffCand':>8}  {'Welfare':>8}  {'Entropy':>8}"
    )
    for step_idx in range(0, len(model_df), 5):
        row = model_df.iloc[step_idx]
        winner_id = (
            model.winner_history[step_idx]
            if step_idx < len(model.winner_history)
            else -1
        )
        print(
            f"  {step_idx + 1:>6}  C{winner_id:>5}  {row['winner_position']:>7.1f}  "
            f"{row['sincere_voting_rate']:>8.3f}  {row['effective_number_of_candidates']:>8.3f}  "
            f"{row['social_welfare']:>8.3f}  {row['strategy_entropy']:>8.3f}"
        )

    print("\n--- Per-Agent Results ---")
    agents = list(model.agents)
    print(
        f"  {'Agent':<20} {'Brain':<20} {'Ideal Pt':>9} "
        f"{'Total Payoff':>13} {'Avg Payoff':>11}"
    )
    print("  " + "-" * 77)
    for agent in sorted(agents, key=lambda a: a.cumulative_payoff, reverse=True):
        avg_p = (
            sum(agent._past_payoffs) / len(agent._past_payoffs)
            if agent._past_payoffs
            else 0
        )
        print(
            f"  {agent.label:<20} {agent.brain_name:<20} {agent.ideal_point:>9.1f} "
            f"{agent.cumulative_payoff:>13.1f} {avg_p:>11.2f}"
        )

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
