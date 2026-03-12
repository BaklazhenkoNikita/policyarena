"""Tragedy of the Commons model.

N agents simultaneously decide how much to harvest from a shared
renewable resource. The resource regenerates each round but can be
depleted by over-harvesting.

Classic CPR dilemma: individually rational to harvest maximally, but
collective restraint maximizes long-term welfare.
"""

from __future__ import annotations

import mesa

from policy_arena.brains.base import Brain
from policy_arena.games.commons.agents import TCAgent
from policy_arena.games.commons.types import TCRoundResult
from policy_arena.metrics.entropy import normalized_shannon_entropy

HARVEST_BINS = 5


def _bin_harvest(h: float, max_share: float) -> str:
    """Discretize a harvest into bins for entropy computation."""
    if max_share <= 0:
        return "0%"
    frac = min(h / max_share, 1.0)
    bin_idx = min(int(frac * HARVEST_BINS), HARVEST_BINS - 1)
    labels = ["0%", "25%", "50%", "75%", "100%"]
    return labels[bin_idx]


class CommonsModel(mesa.Model):
    """Tragedy of the Commons.

    Each step:
    1. Agents simultaneously request a harvest amount.
    2. If total requests <= resource, each gets what they asked.
       If total > resource, proportional allocation.
    3. Resource regenerates: resource = min(max_resource, resource * growth_rate).

    Payoff = actual harvest received.
    """

    def __init__(
        self,
        brains: list[Brain],
        n_rounds: int = 100,
        max_resource: float = 100.0,
        growth_rate: float = 1.5,
        harvest_cap: float = 15,
        depletion_threshold: float = 20,
        labels: list[str] | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.n_rounds = n_rounds
        self.max_resource = max_resource
        self.growth_rate = growth_rate
        self.harvest_cap = harvest_cap / 100.0
        self.depletion_threshold = depletion_threshold / 100.0
        self.resource_level = max_resource

        self.resource_history: list[float] = []
        self.harvest_history: list[float] = []
        self.agent_harvest_history: list[dict[str, float]] = []

        self._round_harvests: list[float] = []
        self._round_total_payoff: float = 0.0
        self._cumulative_total_harvest: float = 0.0

        for i, brain in enumerate(brains):
            label = labels[i] if labels else None
            TCAgent(self, brain=brain, label=label)

        # Sustainable harvest = resource * (growth_rate - 1), since after
        # harvesting that amount the resource regenerates back to same level.
        self._sustainable_total = max_resource * (growth_rate - 1)

        self.datacollector = mesa.DataCollector(
            model_reporters={
                "resource_level": lambda m: m.resource_level / m.max_resource,
                "cooperation_rate": lambda m: m._metric_cooperation_rate(),
                "sustainability": lambda m: m._metric_sustainability(),
                "total_harvest": lambda m: m._metric_total_harvest(),
                "strategy_entropy": lambda m: m._metric_strategy_entropy(),
            },
            agent_reporters={
                "cumulative_payoff": "cumulative_payoff",
                "round_payoff": "round_payoff",
                "last_harvest": "last_harvest",
                "brain_name": "brain_name",
                "label": "label",
            },
        )

    def _metric_cooperation_rate(self) -> float:
        """How restrained agents are. 1 = everyone harvests sustainably, 0 = max greed."""
        if not self._round_harvests:
            return 1.0
        n = len(self._round_harvests)
        # Use the resource level before THIS round's harvest, not max_resource
        resource_before = (
            self.resource_history[-1] if self.resource_history else self.max_resource
        )
        sustainable_total = resource_before * (self.growth_rate - 1)
        sustainable_per_agent = sustainable_total / n if n > 0 else 0
        if sustainable_per_agent <= 0:
            return 1.0
        # Average ratio of (sustainable - excess) / sustainable, clamped [0, 1]
        ratios = []
        for h in self._round_harvests:
            if h <= sustainable_per_agent:
                ratios.append(1.0)
            else:
                ratios.append(
                    max(0.0, 1.0 - (h - sustainable_per_agent) / sustainable_per_agent)
                )
        return sum(ratios) / len(ratios)

    def _metric_sustainability(self) -> float:
        """Resource level as fraction of max (1 = full, 0 = depleted)."""
        return self.resource_level / self.max_resource if self.max_resource > 0 else 0.0

    def _metric_total_harvest(self) -> float:
        """Cumulative total harvest across all agents over all rounds."""
        return self._cumulative_total_harvest

    def _metric_strategy_entropy(self) -> float:
        """Shannon entropy over discretized harvest levels."""
        if not self._round_harvests:
            return 0.0
        n = len(self._round_harvests)
        max_share = self.resource_level / n if n > 0 else 1.0
        bins = [_bin_harvest(h, max_share) for h in self._round_harvests]
        return normalized_shannon_entropy(bins, n_categories=HARVEST_BINS)

    def step(self) -> None:
        agents = list(self.agents)

        # Record resource level before harvesting
        resource_before = self.resource_level
        self.resource_history.append(resource_before)

        # 1. Agents decide harvest amounts
        from policy_arena.games.parallel import gather_decisions

        max_w = getattr(self, "max_concurrent_llm", 1)
        requests = gather_decisions(agents, lambda a: a.decide(), max_w)

        # 2. Enforce per-agent harvest cap
        max_per_agent = resource_before * self.harvest_cap
        capped: dict[int, float] = {}
        for uid, req in requests.items():
            capped[uid] = min(req, max_per_agent)

        total_requested = sum(capped.values())

        # 3. Allocate: if total <= resource, give as requested; else proportional
        actual_harvests: dict[int, float] = {}
        if total_requested <= resource_before or total_requested <= 0:
            actual_harvests = dict(capped)
        else:
            # Proportional allocation
            for uid, req in capped.items():
                actual_harvests[uid] = req * (resource_before / total_requested)

        total_harvested = sum(actual_harvests.values())

        # 4. Deplete and regenerate resource
        self.resource_level = resource_before - total_harvested
        # Degrade regeneration below the depletion threshold
        effective_growth = self.growth_rate
        if self.max_resource > 0 and self.depletion_threshold > 0:
            fullness = self.resource_level / self.max_resource
            if fullness < self.depletion_threshold:
                # Linear degradation: at 0 resource, growth = 1.0 (no regen)
                t = fullness / self.depletion_threshold
                effective_growth = 1.0 + (self.growth_rate - 1.0) * t
        self.resource_level = min(
            self.max_resource, self.resource_level * effective_growth
        )
        self.resource_level = max(0.0, self.resource_level)

        # Track metrics
        self._round_harvests = list(actual_harvests.values())
        self._round_total_payoff = total_harvested
        self._cumulative_total_harvest += total_harvested
        self.harvest_history.append(total_harvested)
        self.agent_harvest_history.append(
            {
                f"Agent {i + 1}": actual_harvests[a.unique_id]
                for i, a in enumerate(agents)
            }
        )

        # 4. Record results for each agent
        for agent in agents:
            actual = actual_harvests[agent.unique_id]
            result = TCRoundResult(
                harvest_requested=requests[agent.unique_id],
                harvest_actual=actual,
                group_total_harvest=total_harvested,
                resource_before=resource_before,
                resource_after=self.resource_level,
                payoff=actual,
                round_number=self.steps,
            )
            agent.record_result(result)

        self.datacollector.collect(self)

        if self.steps >= self.n_rounds or self.resource_level < 0.01:
            self.running = False
