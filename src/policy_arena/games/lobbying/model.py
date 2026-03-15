"""Lobbying / Rent-Seeking Contest model (Tullock Contest).

N agents compete for a prize by spending resources on lobbying.
Probability of winning = (own_spend^r) / sum(all_spend^r) where r is
contest sensitivity. Winner gets prize_value - spend. Losers get -spend.
Total spending is socially wasteful (rent dissipation).
"""

from __future__ import annotations

import mesa

from policy_arena.brains.base import Brain
from policy_arena.games.lobbying.agents import LobbyingAgent
from policy_arena.games.lobbying.types import LobbyingRoundResult
from policy_arena.metrics.entropy import normalized_shannon_entropy
from policy_arena.metrics.social_welfare import compute_social_welfare

SPEND_BINS = 5


def _bin_spend(s: float, budget: float) -> str:
    """Discretize a spend into bins for entropy computation."""
    if budget == 0:
        return "0%"
    frac = s / budget
    bin_idx = min(int(frac * SPEND_BINS), SPEND_BINS - 1)
    labels = ["0%", "25%", "50%", "75%", "100%"]
    return labels[bin_idx]


class LobbyingModel(mesa.Model):
    """Lobbying / Rent-Seeking Contest (Tullock Contest).

    Each step: all agents simultaneously choose how much to spend on lobbying.
    Winner is drawn probabilistically via the Tullock contest success function.

    Payoff_winner = prize_value - spend
    Payoff_loser  = -spend
    """

    def __init__(
        self,
        brains: list[Brain],
        n_rounds: int = 100,
        prize_value: float = 100.0,
        budget: float = 50.0,
        contest_sensitivity: float = 1.0,
        labels: list[str] | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.n_rounds = n_rounds
        self.prize_value = prize_value
        self.budget = budget
        self.contest_sensitivity = contest_sensitivity

        self.total_spend_history: list[float] = []
        self.winner_spend_history: list[float] = []
        self.agent_spend_history: list[dict[str, float]] = []

        self._round_total_payoff: float = 0.0
        self._round_max_payoff: float = 0.0
        self._round_spends: list[float] = []
        self._round_total_spend: float = 0.0
        self._round_winner_spend: float = 0.0

        for i, brain in enumerate(brains):
            label = labels[i] if labels else None
            LobbyingAgent(self, brain=brain, label=label)

        self.datacollector = mesa.DataCollector(
            model_reporters={
                "total_dissipation": lambda m: m._metric_total_dissipation(),
                "avg_spend": lambda m: m._metric_avg_spend(),
                "winner_spend": lambda m: m._round_winner_spend,
                "rent_dissipation_rate": lambda m: m._metric_rent_dissipation_rate(),
                "social_welfare": lambda m: compute_social_welfare(m),
                "strategy_entropy": lambda m: m._metric_strategy_entropy(),
            },
            agent_reporters={
                "cumulative_payoff": "cumulative_payoff",
                "round_payoff": "round_payoff",
                "last_spend": "last_spend",
                "brain_name": "brain_name",
                "label": "label",
            },
        )

    def _metric_total_dissipation(self) -> float:
        """Total spend as fraction of prize value."""
        if self.prize_value == 0:
            return 0.0
        return self._round_total_spend / self.prize_value

    def _metric_avg_spend(self) -> float:
        if not self._round_spends:
            return 0.0
        return sum(self._round_spends) / len(self._round_spends)

    def _metric_rent_dissipation_rate(self) -> float:
        """Total spend divided by prize value."""
        if self.prize_value == 0:
            return 0.0
        return self._round_total_spend / self.prize_value

    def _metric_strategy_entropy(self) -> float:
        """Shannon entropy over discretized spend levels."""
        if not self._round_spends:
            return 0.0
        bins = [_bin_spend(s, self.budget) for s in self._round_spends]
        return normalized_shannon_entropy(bins, n_categories=SPEND_BINS)

    def step(self) -> None:
        agents = list(self.agents)
        n = len(agents)

        from policy_arena.games.parallel import gather_decisions

        max_w = getattr(self, "max_concurrent_llm", 1)
        spends = gather_decisions(agents, lambda a: a.decide(), max_w)

        total_spend = sum(spends.values())
        r = self.contest_sensitivity

        # Compute win probabilities using Tullock contest success function
        if total_spend == 0 or all(s == 0 for s in spends.values()):
            # If nobody spends, equal probability
            win_probs = {uid: 1.0 / n for uid in spends}
        else:
            powered = {}
            total_powered = 0.0
            for uid, s in spends.items():
                p = s**r if s > 0 else 0.0
                powered[uid] = p
                total_powered += p
            if total_powered == 0:
                win_probs = {uid: 1.0 / n for uid in spends}
            else:
                win_probs = {uid: p / total_powered for uid, p in powered.items()}

        # Draw winner based on probabilities
        agent_ids = list(win_probs.keys())
        probs = [win_probs[uid] for uid in agent_ids]
        winner_id = self.random.choices(agent_ids, weights=probs, k=1)[0]

        winner_spend = spends[winner_id]

        self._round_spends = list(spends.values())
        self._round_total_spend = total_spend
        self._round_winner_spend = winner_spend
        self._round_total_payoff = 0.0
        # Best case: one player wins spending 0 -> payoff = prize_value
        self._round_max_payoff = self.prize_value

        for agent in agents:
            s = spends[agent.unique_id]
            won = agent.unique_id == winner_id
            payoff = (self.prize_value - s) if won else -s

            result = LobbyingRoundResult(
                my_spend=s,
                total_spend=total_spend,
                won=won,
                prize_value=self.prize_value,
                payoff=payoff,
                round_number=self.steps,
            )
            agent.record_result(result)
            self._round_total_payoff += payoff

        self.agent_spend_history.append(
            {f"Agent {i + 1}": spends[a.unique_id] for i, a in enumerate(agents)}
        )
        self.total_spend_history.append(total_spend)
        self.winner_spend_history.append(winner_spend)
        self.datacollector.collect(self)

        if self.steps >= self.n_rounds:
            self.running = False
