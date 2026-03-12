"""Trust Game model.

Each round: agents are paired. In each pair, one is the investor, the other
the trustee. The investor sends some amount (0 to endowment), which is
multiplied by a factor. The trustee then returns some amount (0 to received).

Investor payoff: endowment - investment + returned
Trustee payoff: investment * multiplier - returned

In round-robin mode, each ordered pair plays once per round (A invests in B,
B invests in A — both directions).
"""

from __future__ import annotations

from collections import defaultdict
from itertools import permutations

import mesa

from policy_arena.brains.base import Brain
from policy_arena.games.parallel import gather_decisions
from policy_arena.games.trust_game.agents import TGAgent
from policy_arena.games.trust_game.types import TGRoundResult
from policy_arena.metrics.entropy import shannon_entropy

INVESTMENT_BINS = ["0%", "1-20%", "20-40%", "40-60%", "60-80%", "80-100%"]


def _bin_investment(investment: float, endowment: float) -> str:
    if endowment == 0:
        return INVESTMENT_BINS[0]
    frac = investment / endowment
    if frac <= 0:
        return INVESTMENT_BINS[0]
    elif frac < 0.2:
        return INVESTMENT_BINS[1]
    elif frac < 0.4:
        return INVESTMENT_BINS[2]
    elif frac < 0.6:
        return INVESTMENT_BINS[3]
    elif frac < 0.8:
        return INVESTMENT_BINS[4]
    else:
        return INVESTMENT_BINS[5]


class TrustGameModel(mesa.Model):
    """Iterated Trust Game with round-robin matching.

    Each step: every ordered pair (A,B) plays once — A invests, B is trustee.
    NE: investor sends 0, trustee returns 0 (backward induction).
    """

    def __init__(
        self,
        brains: list[Brain],
        n_rounds: int = 100,
        endowment: float = 10.0,
        multiplier: float = 3.0,
        labels: list[str] | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.n_rounds = n_rounds
        self.endowment = endowment
        self.multiplier = multiplier

        self._round_investments: list[float] = []
        self._round_returns: list[float] = []
        self._round_amounts_received: list[float] = []
        self._round_total_payoff: float = 0.0
        self._round_max_payoff: float = 0.0

        for i, brain in enumerate(brains):
            label = labels[i] if labels else None
            TGAgent(self, brain=brain, label=label)

        self.datacollector = mesa.DataCollector(
            model_reporters={
                "cooperation_rate": lambda m: m._metric_avg_investment_pct(),
                "avg_return_rate": lambda m: m._metric_avg_return_rate(),
                "nash_eq_distance": lambda m: m._metric_nash_distance(),
                "social_welfare": lambda m: m._metric_social_welfare(),
                "strategy_entropy": lambda m: m._metric_strategy_entropy(),
            },
            agent_reporters={
                "cumulative_payoff": "cumulative_payoff",
                "round_payoff": "round_payoff",
                "brain_name": "brain_name",
                "label": "label",
            },
        )

    def _metric_avg_investment_pct(self) -> float:
        """Average fraction of endowment invested."""
        if not self._round_investments or self.endowment == 0:
            return 0.0
        return sum(s / self.endowment for s in self._round_investments) / len(
            self._round_investments
        )

    def _metric_avg_return_rate(self) -> float:
        """Average fraction of received amount returned."""
        if not self._round_returns:
            return 0.0
        rates = []
        for ret, recv in zip(
            self._round_returns, self._round_amounts_received, strict=False
        ):
            if recv > 0:
                rates.append(ret / recv)
        return sum(rates) / len(rates) if rates else 0.0

    def _metric_nash_distance(self) -> float:
        """NE = invest 0, return 0. Distance = avg fraction invested."""
        if not self._round_investments or self.endowment == 0:
            return 0.0
        return sum(s / self.endowment for s in self._round_investments) / len(
            self._round_investments
        )

    def _metric_social_welfare(self) -> float:
        if self._round_max_payoff == 0:
            return 0.0
        return self._round_total_payoff / self._round_max_payoff

    def _metric_strategy_entropy(self) -> float:
        """Entropy over binned investment amounts."""
        if not self._round_investments:
            return 0.0
        bins = [_bin_investment(s, self.endowment) for s in self._round_investments]
        return shannon_entropy(bins)

    def step(self) -> None:
        agents = list(self.agents)
        agent_map = {a.unique_id: a for a in agents}
        pairs = list(permutations(agents, 2))

        self._round_investments = []
        self._round_returns = []
        self._round_amounts_received = []
        self._round_total_payoff = 0.0
        self._round_max_payoff = self.endowment * self.multiplier * len(pairs)

        for agent in agents:
            agent.begin_round()

        # --- Phase 1: Batch all investor decisions ---
        investor_targets: dict[int, list[int]] = defaultdict(list)
        for investor, trustee in pairs:
            investor_targets[investor.unique_id].append(trustee.unique_id)

        investors_with_targets = [agent_map[iid] for iid in investor_targets]

        def _invest(agent):
            targets = investor_targets[agent.unique_id]
            return agent.make_investments_batch(targets)

        max_w = getattr(self, "max_concurrent_llm", 1)
        all_investments = gather_decisions(investors_with_targets, _invest, max_w)

        # --- Phase 2: Batch all trustee decisions ---
        trustee_received: dict[int, list[tuple[int, float]]] = defaultdict(list)
        for investor, trustee in pairs:
            investment = all_investments[investor.unique_id][trustee.unique_id]
            amount_received = investment * self.multiplier
            trustee_received[trustee.unique_id].append(
                (investor.unique_id, amount_received)
            )

        trustees_with_received = [agent_map[tid] for tid in trustee_received]

        def _return(agent):
            received = trustee_received[agent.unique_id]
            return agent.decide_returns_batch(received)

        all_returns = gather_decisions(trustees_with_received, _return, max_w)

        # --- Phase 3: Resolve payoffs ---
        for investor, trustee in pairs:
            investment = all_investments[investor.unique_id][trustee.unique_id]
            amount_received = investment * self.multiplier
            returned = all_returns[trustee.unique_id][investor.unique_id]

            investor_payoff = self.endowment - investment + returned
            trustee_payoff = amount_received - returned

            investor.record_result(
                TGRoundResult(
                    role="investor",
                    investment=investment,
                    amount_received=amount_received,
                    amount_returned=returned,
                    payoff=investor_payoff,
                    opponent_payoff=trustee_payoff,
                    round_number=self.steps,
                ),
                opponent_id=trustee.unique_id,
            )
            trustee.record_result(
                TGRoundResult(
                    role="trustee",
                    investment=investment,
                    amount_received=amount_received,
                    amount_returned=returned,
                    payoff=trustee_payoff,
                    opponent_payoff=investor_payoff,
                    round_number=self.steps,
                ),
                opponent_id=investor.unique_id,
            )

            self._round_investments.append(investment)
            self._round_returns.append(returned)
            self._round_amounts_received.append(amount_received)
            self._round_total_payoff += investor_payoff + trustee_payoff

        for agent in agents:
            agent.end_round()

        self.datacollector.collect(self)

        if self.steps >= self.n_rounds:
            self.running = False
