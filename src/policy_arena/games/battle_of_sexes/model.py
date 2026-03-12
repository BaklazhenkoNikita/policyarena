"""Iterated Battle of the Sexes model with round-robin pairwise matching.

Two players independently choose between Option A and Option B.
Both prefer to coordinate, but each side of the matchup prefers a different
coordination point:
  - Row player prefers (A, A)
  - Column player prefers (B, B)
  - Miscoordination yields 0 for both

The key tension: coordination is better than miscoordination, but WHO gets
their preferred option? This models negotiation, compromise, and conventions.
"""

from __future__ import annotations

from itertools import combinations
from typing import Any

import mesa

from policy_arena.brains.base import Brain
from policy_arena.core.types import Action, RoundResult
from policy_arena.games.battle_of_sexes.agents import BoSAgent
from policy_arena.metrics import (
    shannon_entropy,
)

# Classic BoS payoffs:
# COOPERATE = Option A, DEFECT = Option B
# (A,A) = (3,2), (A,B) = (0,0), (B,A) = (0,0), (B,B) = (2,3)
DEFAULT_PAYOFF_MATRIX: dict[tuple[Action, Action], tuple[float, float]] = {
    (Action.COOPERATE, Action.COOPERATE): (3.0, 2.0),  # Both pick A (row prefers)
    (Action.COOPERATE, Action.DEFECT): (0.0, 0.0),  # Miscoordination
    (Action.DEFECT, Action.COOPERATE): (0.0, 0.0),  # Miscoordination
    (Action.DEFECT, Action.DEFECT): (2.0, 3.0),  # Both pick B (col prefers)
}


class BattleOfSexesModel(mesa.Model):
    """Iterated Battle of the Sexes with round-robin pairwise matching."""

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
        # Max welfare = max of both coordination payoffs
        aa = self.payoff_matrix[(Action.COOPERATE, Action.COOPERATE)]
        bb = self.payoff_matrix[(Action.DEFECT, Action.DEFECT)]
        self._max_welfare_per_pair: float = max(aa[0] + aa[1], bb[0] + bb[1])
        self._round_max_payoff: float = 0.0

        for i, brain in enumerate(brains):
            label = labels[i] if labels else None
            BoSAgent(self, brain=brain, label=label)

        self.datacollector = mesa.DataCollector(
            model_reporters={
                "cooperation_rate": lambda m: self._metric_option_a_rate(),
                "coordination_rate": lambda m: self._metric_coordination_rate(),
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

    def _metric_option_a_rate(self) -> float:
        """Fraction of agents choosing Option A (COOPERATE)."""
        if not self._round_all_actions:
            return 0.5
        a_count = sum(1 for a in self._round_all_actions if a == Action.COOPERATE)
        return a_count / len(self._round_all_actions)

    def _metric_coordination_rate(self) -> float:
        """Fraction of matchups where both players chose the same option."""
        if not self._round_interactions:
            return 0.0
        coordinated = sum(1 for _, _, a1, a2 in self._round_interactions if a1 == a2)
        return coordinated / len(self._round_interactions)

    def _metric_nash_distance(self) -> float:
        """Distance from Nash Equilibrium.

        BoS has TWO pure NE: all-A and all-B, plus a mixed NE.
        We measure distance from the nearest pure NE.
        """
        if not self._round_all_actions:
            return 0.0
        n = len(self._round_all_actions)
        a_count = sum(1 for a in self._round_all_actions if a == Action.COOPERATE)
        a_rate = a_count / n
        dist_a_ne = abs(a_rate - 1.0)
        dist_b_ne = abs(a_rate - 0.0)
        return min(dist_a_ne, dist_b_ne)

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

        # Phase 1: Collect observations
        agent_obs: dict[int, list[tuple[int, Any]]] = {}
        for a_i, a_j in pairs:
            obs_i = a_i.get_observation(a_j.unique_id)
            obs_j = a_j.get_observation(a_i.unique_id)
            agent_obs.setdefault(a_i.unique_id, []).append((a_j.unique_id, obs_i))
            agent_obs.setdefault(a_j.unique_id, []).append((a_i.unique_id, obs_j))

        # Phase 2: Decide
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
