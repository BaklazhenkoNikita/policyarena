"""Schelling Segregation model on a 2D grid.

Agents of any number of types are placed on a grid. Each step, unhappy agents
(those whose neighborhood composition exceeds their tolerance) may relocate
to a random empty cell. The brain decides whether to move.

Agents can have per-group tolerances: different maximum fractions of each
other type they are willing to tolerate as neighbors. If not set, the global
tolerance (minimum fraction same-type) is used as fallback.

Simulation stops early when happiness_rate reaches 1.0 (all agents happy).

Key metrics:
  - segregation_index: average fraction of same-type neighbors
  - happiness_rate: fraction of agents that are happy
  - move_rate: fraction of agents that moved this round
  - n_islands: number of connected clusters of same-type agents
  - largest_island: size of the largest cluster
  - avg_island_size: average cluster size
"""

from __future__ import annotations

from collections import deque

import mesa

from policy_arena.brains.base import Brain
from policy_arena.games.schelling.agents import SchellingAgent
from policy_arena.games.schelling.types import SchellingRoundResult


class SchellingModel(mesa.Model):
    """Schelling Segregation on a toroidal grid."""

    def __init__(
        self,
        brains: list[Brain],
        n_rounds: int = 100,
        width: int = 20,
        height: int = 20,
        density: float = 0.8,
        tolerance: float = 0.375,
        agent_types: list[int] | None = None,
        labels: list[str] | None = None,
        group_tolerances_list: list[dict[int, float]] | None = None,
        **kwargs,
    ):
        # Remove legacy payoff params if passed
        kwargs.pop("happy_bonus", None)
        kwargs.pop("move_cost", None)
        super().__init__(**kwargs)
        self.n_rounds = n_rounds
        self.width = width
        self.height = height
        self.density = density
        self.tolerance = tolerance

        self.grid = mesa.space.SingleGrid(width, height, torus=True)

        # Place agents on the grid
        n_agents = len(brains)
        # Get random empty cells
        all_cells = list(self.grid.coord_iter())
        self.random.shuffle(all_cells)

        for i, brain in enumerate(brains):
            if i >= len(all_cells):
                break
            if agent_types is not None:
                agent_type = agent_types[i]
            else:
                agent_type = 0 if i < n_agents // 2 else 1
            label = labels[i] if labels else None
            gt = group_tolerances_list[i] if group_tolerances_list else None
            agent = SchellingAgent(
                self,
                brain=brain,
                agent_type=agent_type,
                label=label,
                group_tolerances=gt,
            )
            _, pos = all_cells[i]
            self.grid.place_agent(agent, pos)

        self._round_move_count: int = 0
        self._round_happy_count: int = 0
        self._cached_islands_step: int = -1
        self._cached_islands_result: tuple[int, int, float] = (0, 0, 0.0)

        self.datacollector = mesa.DataCollector(
            model_reporters={
                "segregation_index": lambda m: m._metric_segregation_index(),
                "happiness_rate": lambda m: m._metric_happiness_rate(),
                "move_rate": lambda m: m._metric_move_rate(),
                "n_islands": lambda m: m._metric_islands()[0],
                "largest_island": lambda m: m._metric_islands()[1],
                "avg_island_size": lambda m: m._metric_islands()[2],
            },
            agent_reporters={
                "agent_type": "agent_type",
                "happy": "happy",
                "moved": "moved",
                "brain_name": "brain_name",
                "label": "label",
            },
        )

    def _metric_segregation_index(self) -> float:
        """Average fraction of same-type neighbors across all agents."""
        agents = list(self.agents)
        if not agents:
            return 0.0
        total = 0.0
        for agent in agents:
            frac_same, _, _, _, _ = agent.get_neighborhood_info()
            total += frac_same
        return total / len(agents)

    def _metric_happiness_rate(self) -> float:
        agents = list(self.agents)
        if not agents:
            return 0.0
        return self._round_happy_count / len(agents)

    def _metric_move_rate(self) -> float:
        agents = list(self.agents)
        if not agents:
            return 0.0
        return self._round_move_count / len(agents)

    def _metric_islands(self) -> tuple[int, int, float]:
        """Find connected clusters of same-type agents using flood-fill on the grid.

        Returns (n_islands, largest_island_size, avg_island_size).
        Two agents are in the same island if they share the same type and are
        adjacent (Moore neighborhood) on the toroidal grid.
        """
        if self._cached_islands_step == self.steps:
            return self._cached_islands_result

        visited: set[tuple[int, int]] = set()
        island_sizes: list[int] = []

        for agent in self.agents:
            pos = agent.pos
            if pos in visited:
                continue
            # BFS flood-fill
            queue = deque([pos])
            visited.add(pos)
            size = 0
            agent_type = agent.agent_type
            while queue:
                cur = queue.popleft()
                size += 1
                for neighbor_pos in self.grid.get_neighborhood(
                    cur, moore=True, include_center=False
                ):
                    if neighbor_pos in visited:
                        continue
                    cell_contents = self.grid.get_cell_list_contents([neighbor_pos])
                    if cell_contents and cell_contents[0].agent_type == agent_type:
                        visited.add(neighbor_pos)
                        queue.append(neighbor_pos)
            island_sizes.append(size)

        n_islands = len(island_sizes)
        largest = max(island_sizes) if island_sizes else 0
        avg = sum(island_sizes) / n_islands if n_islands > 0 else 0.0

        self._cached_islands_step = self.steps
        self._cached_islands_result = (n_islands, largest, avg)
        return self._cached_islands_result

    def step(self) -> None:
        agents = list(self.agents)
        self.random.shuffle(agents)

        self._round_move_count = 0
        self._round_happy_count = 0

        for agent in agents:
            wants_to_move = agent.decide()

            moved = False
            if wants_to_move:
                empty_cells = list(self.grid.empties)
                if empty_cells:
                    new_pos = self.random.choice(empty_cells)
                    self.grid.move_agent(agent, new_pos)
                    moved = True

            # Re-check happiness after potential move
            frac_same_after, _, _, _, fpt_after = agent.get_neighborhood_info()
            happy_after = agent.check_happy(fpt_after, frac_same_after)

            agent.record_result(
                SchellingRoundResult(
                    moved=moved,
                    was_happy=happy_after,
                    fraction_same=frac_same_after,
                    payoff=0.0,
                    round_number=self.steps,
                )
            )

            if happy_after:
                self._round_happy_count += 1
            if moved:
                self._round_move_count += 1

        self.datacollector.collect(self)

        # Stop early if all agents are happy or no one moved (converged)
        if (
            self._round_happy_count == len(agents)
            or self._round_move_count == 0
            or self.steps >= self.n_rounds
        ):
            self.running = False
