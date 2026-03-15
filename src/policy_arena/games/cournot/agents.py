"""Cournot Oligopoly agent."""

from __future__ import annotations

import mesa

from policy_arena.brains.base import Brain
from policy_arena.games.cournot.types import CournotObservation, CournotRoundResult


class CournotAgent(mesa.Agent):
    """Agent in a Cournot Oligopoly.

    Each round: choose a production quantity. Market price depends on
    total quantity produced by all firms.
    """

    def __init__(self, model: mesa.Model, brain: Brain, label: str | None = None):
        super().__init__(model)
        self.brain = brain
        self.label = label or f"{brain.name}_{self.unique_id}"
        self.cumulative_profit: float = 0.0
        self.round_profit: float = 0.0
        self.last_quantity: float = 0.0

        self._past_quantities: list[float] = []
        self._past_profits: list[float] = []

    @property
    def brain_name(self) -> str:
        return self.brain.name

    # Aliases so the datacollector can use the same names across games
    @property
    def cumulative_payoff(self) -> float:
        return self.cumulative_profit

    @property
    def round_payoff(self) -> float:
        return self.round_profit

    def get_observation(self) -> CournotObservation:
        return CournotObservation(
            round_number=self.model.steps,
            max_price=self.model.max_price,
            marginal_cost=self.model.marginal_cost,
            max_quantity=self.model.max_quantity,
            n_players=len(list(self.model.agents)),
            my_past_quantities=list(self._past_quantities),
            my_past_profits=list(self._past_profits),
            market_past_prices=list(self.model.price_history),
            market_past_total_quantities=list(self.model.total_quantity_history),
            all_agent_quantities=list(self.model.agent_quantity_history),
        )

    def decide(self) -> float:
        obs = self.get_observation()
        raw = self.brain.decide(obs)
        return max(0.0, min(self.model.max_quantity, float(raw)))

    def record_result(self, result: CournotRoundResult) -> None:
        self._past_quantities.append(result.quantity)
        self._past_profits.append(result.profit)
        self.cumulative_profit += result.profit
        self.round_profit = result.profit
        self.last_quantity = result.quantity
        self.brain.update(result)
