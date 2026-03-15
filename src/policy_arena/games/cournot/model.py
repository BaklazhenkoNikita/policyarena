"""Cournot Oligopoly model.

N firms simultaneously choose production quantities.
Market price = max(0, A - total_quantity).
Profit_i = price * quantity_i - marginal_cost * quantity_i.

Nash Equilibrium (symmetric): q* = (A - c) / (N + 1) for each firm.
"""

from __future__ import annotations

import mesa

from policy_arena.brains.base import Brain
from policy_arena.games.cournot.agents import CournotAgent
from policy_arena.games.cournot.types import CournotRoundResult
from policy_arena.metrics.entropy import normalized_shannon_entropy
from policy_arena.metrics.social_welfare import compute_social_welfare

QUANTITY_BINS = 5


def _bin_quantity(q: float, max_quantity: float) -> str:
    """Discretize a quantity into bins for entropy computation."""
    if max_quantity == 0:
        return "0%"
    frac = q / max_quantity
    bin_idx = min(int(frac * QUANTITY_BINS), QUANTITY_BINS - 1)
    labels = ["0%", "25%", "50%", "75%", "100%"]
    return labels[bin_idx]


class CournotModel(mesa.Model):
    """Cournot Oligopoly.

    Each step: all firms simultaneously choose quantities, market price clears,
    and profits are computed.

    Price = max(0, max_price - total_quantity)
    Profit_i = price * q_i - marginal_cost * q_i
    """

    def __init__(
        self,
        brains: list[Brain],
        n_rounds: int = 100,
        max_price: float = 100.0,
        marginal_cost: float = 10.0,
        max_quantity: float = 50.0,
        labels: list[str] | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.n_rounds = n_rounds
        self.max_price = max_price
        self.marginal_cost = marginal_cost
        self.max_quantity = max_quantity

        self.price_history: list[float] = []
        self.total_quantity_history: list[float] = []
        self.agent_quantity_history: list[dict[str, float]] = []

        self._round_total_payoff: float = 0.0
        self._round_max_payoff: float = 0.0
        self._round_quantities: list[float] = []

        n = len(brains)
        for i, brain in enumerate(brains):
            label = labels[i] if labels else None
            CournotAgent(self, brain=brain, label=label)

        # Monopoly profit as theoretical max welfare
        # Monopoly q = (A - c) / 2, price = (A + c) / 2, profit = (A - c)^2 / 4
        self._monopoly_profit = (max_price - marginal_cost) ** 2 / 4.0
        # NE quantity per firm
        self._ne_quantity = (max_price - marginal_cost) / (n + 1)

        self.datacollector = mesa.DataCollector(
            model_reporters={
                "market_price": lambda m: m._metric_market_price(),
                "total_quantity": lambda m: m._metric_total_quantity(),
                "avg_profit": lambda m: m._metric_avg_profit(),
                "nash_eq_distance": lambda m: m._metric_nash_distance(),
                "social_welfare": lambda m: compute_social_welfare(m),
                "strategy_entropy": lambda m: m._metric_strategy_entropy(),
                "competition_intensity": lambda m: m._metric_competition_intensity(),
            },
            agent_reporters={
                "cumulative_payoff": "cumulative_payoff",
                "round_payoff": "round_payoff",
                "last_quantity": "last_quantity",
                "brain_name": "brain_name",
                "label": "label",
            },
        )

    def _metric_market_price(self) -> float:
        if not self.price_history:
            return self.max_price
        return self.price_history[-1]

    def _metric_total_quantity(self) -> float:
        if not self.total_quantity_history:
            return 0.0
        return self.total_quantity_history[-1]

    def _metric_avg_profit(self) -> float:
        if not self._round_quantities:
            return 0.0
        n = len(self._round_quantities)
        return self._round_total_payoff / n if n > 0 else 0.0

    def _metric_nash_distance(self) -> float:
        """Average absolute deviation of quantities from NE quantity."""
        if not self._round_quantities:
            return 0.0
        ne_q = self._ne_quantity
        deviations = [abs(q - ne_q) for q in self._round_quantities]
        max_dev = max(self.max_quantity, ne_q)
        if max_dev == 0:
            return 0.0
        return sum(deviations) / (len(deviations) * max_dev)

    def _metric_strategy_entropy(self) -> float:
        """Shannon entropy over discretized quantity levels."""
        if not self._round_quantities:
            return 0.0
        bins = [_bin_quantity(q, self.max_quantity) for q in self._round_quantities]
        return normalized_shannon_entropy(bins, n_categories=QUANTITY_BINS)

    def _metric_competition_intensity(self) -> float:
        """How close total output is to perfectly competitive level.

        Perfect competition: price = marginal_cost, so Q_total = max_price - marginal_cost.
        Monopoly: Q = (max_price - marginal_cost) / 2.
        Returns 0 at monopoly level, 1 at perfect competition level.
        """
        if not self.total_quantity_history:
            return 0.0
        q_total = self.total_quantity_history[-1]
        q_competitive = self.max_price - self.marginal_cost
        q_monopoly = q_competitive / 2.0
        if q_competitive == q_monopoly:
            return 0.0
        return max(0.0, min(1.0, (q_total - q_monopoly) / (q_competitive - q_monopoly)))

    def step(self) -> None:
        agents = list(self.agents)

        from policy_arena.games.parallel import gather_decisions

        max_w = getattr(self, "max_concurrent_llm", 1)
        quantities = gather_decisions(agents, lambda a: a.decide(), max_w)

        total_quantity = sum(quantities.values())
        price = max(0.0, self.max_price - total_quantity)

        self._round_quantities = list(quantities.values())
        self._round_total_payoff = 0.0
        # Max welfare = monopoly profit (collusive outcome)
        self._round_max_payoff = self._monopoly_profit

        for agent in agents:
            q = quantities[agent.unique_id]
            revenue = price * q
            cost = self.marginal_cost * q
            profit = revenue - cost

            result = CournotRoundResult(
                quantity=q,
                market_total_quantity=total_quantity,
                market_price=price,
                revenue=revenue,
                cost=cost,
                profit=profit,
                round_number=self.steps,
            )
            agent.record_result(result)
            self._round_total_payoff += profit

        self.agent_quantity_history.append(
            {f"Agent {i + 1}": quantities[a.unique_id] for i, a in enumerate(agents)}
        )
        self.price_history.append(price)
        self.total_quantity_history.append(total_quantity)
        self.datacollector.collect(self)

        if self.steps >= self.n_rounds:
            self.running = False
