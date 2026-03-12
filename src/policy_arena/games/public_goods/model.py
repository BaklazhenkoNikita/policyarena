"""Public Goods Game model.

N players simultaneously choose how much of their endowment to contribute
to a shared pool. The pool is multiplied and split equally.

Classic social dilemma: NE is contribute 0, but max welfare is contribute all.
"""

from __future__ import annotations

import mesa

from policy_arena.brains.base import Brain
from policy_arena.games.public_goods.agents import PGAgent
from policy_arena.games.public_goods.types import PGRoundResult
from policy_arena.metrics.entropy import normalized_shannon_entropy
from policy_arena.metrics.social_welfare import compute_social_welfare

CONTRIBUTION_BINS = 5


def _bin_contribution(c: float, endowment: float) -> str:
    """Discretize a contribution into bins for entropy computation."""
    if endowment == 0:
        return "0%"
    frac = c / endowment
    bin_idx = min(int(frac * CONTRIBUTION_BINS), CONTRIBUTION_BINS - 1)
    labels = ["0%", "25%", "50%", "75%", "100%"]
    return labels[bin_idx]


class PublicGoodsModel(mesa.Model):
    """Public Goods Game.

    Each step: all agents simultaneously contribute, pool is multiplied
    and redistributed equally.

    Payoff_i = endowment - contribution_i + (sum(contributions) * multiplier) / N
    """

    def __init__(
        self,
        brains: list[Brain],
        n_rounds: int = 100,
        endowment: float = 20.0,
        multiplier: float = 1.6,
        labels: list[str] | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.n_rounds = n_rounds
        self.endowment = endowment
        self.multiplier = multiplier

        self.group_avg_history: list[float] = []
        self.agent_contribution_history: list[dict[str, float]] = []

        self._round_total_payoff: float = 0.0
        self._round_max_payoff: float = 0.0
        self._round_contributions: list[float] = []

        for i, brain in enumerate(brains):
            label = labels[i] if labels else None
            PGAgent(self, brain=brain, label=label)

        n = len(brains)
        self._max_welfare_per_round = n * endowment * multiplier

        self.datacollector = mesa.DataCollector(
            model_reporters={
                "cooperation_rate": lambda m: m._metric_cooperation_rate(),
                "nash_eq_distance": lambda m: m._metric_nash_distance(),
                "social_welfare": lambda m: compute_social_welfare(m),
                "strategy_entropy": lambda m: m._metric_strategy_entropy(),
                "avg_contribution": lambda m: m._metric_avg_contribution(),
            },
            agent_reporters={
                "cumulative_payoff": "cumulative_payoff",
                "round_payoff": "round_payoff",
                "last_contribution": "last_contribution",
                "brain_name": "brain_name",
                "label": "label",
            },
        )

    def _metric_cooperation_rate(self) -> float:
        """Fraction of endowment contributed on average (0 = all free-ride, 1 = all contribute fully)."""
        if not self._round_contributions or self.endowment == 0:
            return 0.0
        return sum(self._round_contributions) / (
            len(self._round_contributions) * self.endowment
        )

    def _metric_nash_distance(self) -> float:
        """NE = contribute 0. Distance = fraction of agents contributing > 0."""
        if not self._round_contributions:
            return 0.0
        return sum(1 for c in self._round_contributions if c > 0) / len(
            self._round_contributions
        )

    def _metric_strategy_entropy(self) -> float:
        """Shannon entropy over discretized contribution levels."""
        if not self._round_contributions:
            return 0.0
        bins = [_bin_contribution(c, self.endowment) for c in self._round_contributions]
        return normalized_shannon_entropy(bins, n_categories=CONTRIBUTION_BINS)

    def _metric_avg_contribution(self) -> float:
        if not self._round_contributions:
            return 0.0
        return sum(self._round_contributions) / len(self._round_contributions)

    def step(self) -> None:
        agents = list(self.agents)
        n = len(agents)

        from policy_arena.games.parallel import gather_decisions

        max_w = getattr(self, "max_concurrent_llm", 1)
        contributions = gather_decisions(agents, lambda a: a.decide(), max_w)

        total_contributed = sum(contributions.values())
        pool = total_contributed * self.multiplier
        share = pool / n if n > 0 else 0.0
        avg_contribution = total_contributed / n if n > 0 else 0.0

        self._round_contributions = list(contributions.values())
        self._round_total_payoff = 0.0
        self._round_max_payoff = self._max_welfare_per_round

        for agent in agents:
            c = contributions[agent.unique_id]
            payoff = (self.endowment - c) + share

            result = PGRoundResult(
                contribution=c,
                group_total_contribution=total_contributed,
                group_average_contribution=avg_contribution,
                pool_after_multiplier=pool,
                payoff=payoff,
                round_number=self.steps,
            )
            agent.record_result(result)
            self._round_total_payoff += payoff

        self.agent_contribution_history.append(
            {f"Agent {i + 1}": contributions[a.unique_id] for i, a in enumerate(agents)}
        )
        self.group_avg_history.append(avg_contribution)
        self.datacollector.collect(self)

        if self.steps >= self.n_rounds:
            self.running = False
