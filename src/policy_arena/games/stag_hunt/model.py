"""Iterated Stag Hunt model with round-robin pairwise matching.

The Stag Hunt differs from the Prisoner's Dilemma in that mutual cooperation
(Stag, Stag) is a Nash Equilibrium — but so is mutual defection (Hare, Hare).
The key tension is between payoff dominance (Stag) and risk dominance (Hare).
"""

from __future__ import annotations

from itertools import combinations
from typing import Any

import mesa

from policy_arena.brains.base import Brain
from policy_arena.core.types import Action, RoundResult
from policy_arena.games.stag_hunt.agents import SHAgent
from policy_arena.metrics import (
    shannon_entropy,
)

# Classic Stag Hunt payoffs:
# Stag+Stag = (4,4), Stag+Hare = (0,3), Hare+Stag = (3,0), Hare+Hare = (2,2)
# COOPERATE = Stag, DEFECT = Hare
DEFAULT_PAYOFF_MATRIX: dict[tuple[Action, Action], tuple[float, float]] = {
    (Action.COOPERATE, Action.COOPERATE): (4.0, 4.0),  # Both hunt stag
    (Action.COOPERATE, Action.DEFECT): (0.0, 3.0),  # Stag fails alone
    (Action.DEFECT, Action.COOPERATE): (3.0, 0.0),  # Hare is safe
    (Action.DEFECT, Action.DEFECT): (2.0, 2.0),  # Both hunt hare
}


class StagHuntModel(mesa.Model):
    """Iterated Stag Hunt with round-robin pairwise matching.

    Each step = one round. Every agent plays every other agent once per round.
    Actions are collected simultaneously, then payoffs resolved centrally.
    """

    def __init__(
        self,
        brains: list[Brain],
        n_rounds: int = 100,
        payoff_matrix: dict[tuple[Action, Action], tuple[float, float]] | None = None,
        labels: list[str] | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.n_rounds = n_rounds
        self.payoff_matrix = payoff_matrix or DEFAULT_PAYOFF_MATRIX

        self._round_interactions: list[tuple[int, int, Action, Action]] = []
        self._round_all_actions: list[Action] = []
        self._round_total_payoff: float = 0.0
        ss = self.payoff_matrix[(Action.COOPERATE, Action.COOPERATE)]
        self._max_welfare_per_pair: float = ss[0] + ss[1]
        self._round_max_payoff: float = 0.0

        for i, brain in enumerate(brains):
            label = labels[i] if labels else None
            SHAgent(self, brain=brain, label=label)

        self.datacollector = mesa.DataCollector(
            model_reporters={
                "cooperation_rate": lambda m: self._metric_stag_rate(),
                "nash_eq_distance": lambda m: self._metric_nash_distance(),
                "social_welfare": lambda m: self._metric_social_welfare(),
                "strategy_entropy": lambda m: self._metric_strategy_entropy(),
            },
            agent_reporters={
                "cumulative_payoff": "cumulative_payoff",
                "round_payoff": "round_payoff",
                "cooperation_rate": "cooperated",
                "brain_name": "brain_name",
                "label": "label",
            },
        )

    def _metric_stag_rate(self) -> float:
        """Fraction of agents choosing Stag (COOPERATE)."""
        if not self._round_all_actions:
            return 1.0
        stag = sum(1 for a in self._round_all_actions if a == Action.COOPERATE)
        return stag / len(self._round_all_actions)

    def _metric_nash_distance(self) -> float:
        """Distance from Nash Equilibrium.

        Stag Hunt has TWO pure NE: all-Stag and all-Hare.
        We measure distance from the nearest NE.
        """
        if not self._round_all_actions:
            return 0.0
        n = len(self._round_all_actions)
        stag_count = sum(1 for a in self._round_all_actions if a == Action.COOPERATE)
        stag_rate = stag_count / n
        # Distance from all-Stag NE vs all-Hare NE — take the minimum
        dist_stag_ne = abs(stag_rate - 1.0)
        dist_hare_ne = abs(stag_rate - 0.0)
        return min(dist_stag_ne, dist_hare_ne)

    def _metric_social_welfare(self) -> float:
        if self._round_max_payoff == 0:
            return 1.0
        return self._round_total_payoff / self._round_max_payoff

    def _metric_strategy_entropy(self) -> float:
        if not self._round_all_actions:
            return 0.0
        return shannon_entropy([a.value for a in self._round_all_actions])

    def step(self) -> None:
        agents = list(self.agents)
        pairs = list(combinations(agents, 2))

        self._round_interactions = []
        self._round_all_actions = []
        self._round_total_payoff = 0.0
        self._round_max_payoff = self._max_welfare_per_pair * len(pairs)

        for agent in agents:
            agent.begin_round()

        # Phase 1: Collect observations for all matchups
        agent_obs: dict[int, list[tuple[int, Any]]] = {}
        for a_i, a_j in pairs:
            obs_i = a_i.get_observation(a_j.unique_id)
            obs_j = a_j.get_observation(a_i.unique_id)
            agent_obs.setdefault(a_i.unique_id, []).append((a_j.unique_id, obs_i))
            agent_obs.setdefault(a_j.unique_id, []).append((a_i.unique_id, obs_j))

        # Phase 2: Decide — batch for brains that support it
        from policy_arena.games.parallel import gather_decisions

        agent_map = {a.unique_id: a for a in agents}
        agents_with_obs = [agent_map[aid] for aid in agent_obs]

        def _decide(agent):
            obs_list = agent_obs[agent.unique_id]
            opponent_ids = [opp_id for opp_id, _ in obs_list]
            observations = [obs for _, obs in obs_list]
            acts = agent.brain.decide_batch(observations)
            return dict(zip(opponent_ids, acts, strict=False))

        max_w = getattr(self, "max_concurrent_llm", 1)
        agent_decisions = gather_decisions(agents_with_obs, _decide, max_w)

        # Phase 3: Resolve payoffs
        actions: dict[tuple[int, int], tuple[Action, Action]] = {}
        for a_i, a_j in pairs:
            act_i = agent_decisions[a_i.unique_id][a_j.unique_id]
            act_j = agent_decisions[a_j.unique_id][a_i.unique_id]
            actions[(a_i.unique_id, a_j.unique_id)] = (act_i, act_j)

        for a_i, a_j in pairs:
            act_i, act_j = actions[(a_i.unique_id, a_j.unique_id)]
            pay_i, pay_j = self.payoff_matrix[(act_i, act_j)]

            a_i.record_result(
                RoundResult(
                    action=act_i,
                    opponent_action=act_j,
                    payoff=pay_i,
                    round_number=self.steps,
                ),
                opponent_id=a_j.unique_id,
            )
            a_j.record_result(
                RoundResult(
                    action=act_j,
                    opponent_action=act_i,
                    payoff=pay_j,
                    round_number=self.steps,
                ),
                opponent_id=a_i.unique_id,
            )

            self._round_interactions.append(
                (a_i.unique_id, a_j.unique_id, act_i, act_j)
            )
            self._round_all_actions.extend([act_i, act_j])
            self._round_total_payoff += pay_i + pay_j

        for agent in agents:
            agent.end_round()

        self.datacollector.collect(self)

        if self.steps >= self.n_rounds:
            self.running = False
