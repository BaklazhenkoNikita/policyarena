"""Iterated Hawk-Dove model with round-robin pairwise matching.

Two players compete over a resource (value V=4). Each independently chooses
to be a Dove (share peacefully) or a Hawk (fight for it):
  - (Dove, Dove) = (V/2, V/2) = (2, 2) — share the resource
  - (Dove, Hawk) = (0, V) = (0, 4) — dove retreats, hawk takes all
  - (Hawk, Dove) = (V, 0) = (4, 0) — hawk takes all
  - (Hawk, Hawk) = ((V-C)/2, (V-C)/2) = (-1, -1) — both injured

The key tension: hawks exploit doves, but mutual hawkishness is the worst
outcome. Doves are safe but risk exploitation. The game models aggression,
resource competition, and the evolution of animal conflict strategies.
"""

from __future__ import annotations

from itertools import combinations
from typing import Any

import mesa

from policy_arena.brains.base import Brain
from policy_arena.core.types import Action, RoundResult
from policy_arena.games.hawk_dove.agents import HDAgent
from policy_arena.metrics import (
    shannon_entropy,
)

# Hawk-Dove payoffs (V=4, C=6):
# COOPERATE = Dove, DEFECT = Hawk
# (Dove,Dove) = (2,2), (Dove,Hawk) = (0,4), (Hawk,Dove) = (4,0), (Hawk,Hawk) = (-1,-1)
DEFAULT_PAYOFF_MATRIX: dict[tuple[Action, Action], tuple[float, float]] = {
    (Action.COOPERATE, Action.COOPERATE): (2.0, 2.0),  # Both dove — share
    (Action.COOPERATE, Action.DEFECT): (0.0, 4.0),  # Dove retreats, hawk takes all
    (Action.DEFECT, Action.COOPERATE): (4.0, 0.0),  # Hawk takes all
    (Action.DEFECT, Action.DEFECT): (-1.0, -1.0),  # Both hawk — injured
}


class HawkDoveModel(mesa.Model):
    """Iterated Hawk-Dove with round-robin pairwise matching."""

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
        # Max welfare = best symmetric outcome (Dove, Dove) = 2+2 = 4
        dd = self.payoff_matrix[(Action.COOPERATE, Action.COOPERATE)]
        hh = self.payoff_matrix[(Action.DEFECT, Action.DEFECT)]
        self._max_welfare_per_pair: float = max(dd[0] + dd[1], hh[0] + hh[1])
        self._round_max_payoff: float = 0.0

        for i, brain in enumerate(brains):
            label = labels[i] if labels else None
            HDAgent(self, brain=brain, label=label)

        self.datacollector = mesa.DataCollector(
            model_reporters={
                "cooperation_rate": lambda m: self._metric_dove_rate(),
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

    def _metric_dove_rate(self) -> float:
        """Fraction of agents choosing Dove (COOPERATE)."""
        if not self._round_all_actions:
            return 0.5
        a_count = sum(1 for a in self._round_all_actions if a == Action.COOPERATE)
        return a_count / len(self._round_all_actions)

    def _metric_coordination_rate(self) -> float:
        """Fraction of matchups where both players chose the same action."""
        if not self._round_interactions:
            return 0.0
        coordinated = sum(1 for _, _, a1, a2 in self._round_interactions if a1 == a2)
        return coordinated / len(self._round_interactions)

    def _metric_nash_distance(self) -> float:
        """Distance from Nash Equilibrium.

        Hawk-Dove has TWO pure NE: (Hawk, Dove) and (Dove, Hawk), plus a
        mixed NE. We measure distance from the mixed NE dove-rate of V/C
        (= 4/6 = 2/3 for default params).
        """
        if not self._round_all_actions:
            return 0.0
        n = len(self._round_all_actions)
        a_count = sum(1 for a in self._round_all_actions if a == Action.COOPERATE)
        dove_rate = a_count / n
        # Mixed NE: play Dove with probability V/C = 4/6 = 2/3
        mixed_ne_dove_rate = 2.0 / 3.0
        return abs(dove_rate - mixed_ne_dove_rate)

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
