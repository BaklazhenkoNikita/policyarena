"""SIR Disease Spread agent — delegates isolation decision to a Brain."""

from __future__ import annotations

import mesa

from policy_arena.brains.base import Brain
from policy_arena.games.sir.types import HealthState, SIRObservation, SIRRoundResult


class SIRAgent(mesa.Agent):
    """Agent on a network that can self-isolate to reduce infection spread."""

    def __init__(
        self,
        model: mesa.Model,
        brain: Brain,
        initial_state: HealthState = HealthState.SUSCEPTIBLE,
        label: str | None = None,
    ):
        super().__init__(model)
        self.brain = brain
        self.health_state = initial_state
        self.label = label or f"{brain.name}_{self.unique_id}"
        self.happiness: float = 80.0
        self.happiness_change: float = 0.0
        self.isolated: bool = False
        self.days_infected: int = 0

        # Immunity & vaccination
        self.immunity: float = 0.0
        self.vaccinated: bool = False
        self.vaccine_isolation_remaining: int = (
            0  # forced isolation rounds after vaccination
        )

        self._past_isolations: list[bool] = []

    @property
    def brain_name(self) -> str:
        return self.brain.name

    def get_neighbors(self) -> list[SIRAgent]:
        """Get network neighbors."""
        neighbor_ids = self.model.network.neighbors(self._network_node)
        return [
            self.model._agent_map[nid]
            for nid in neighbor_ids
            if nid in self.model._agent_map
        ]

    def get_observation(self) -> SIRObservation:
        neighbors = self.get_neighbors()
        n_infected = sum(1 for n in neighbors if n.health_state == HealthState.INFECTED)
        total_agents = len(list(self.model.agents))
        infected_count = sum(
            1 for a in self.model.agents if a.health_state == HealthState.INFECTED
        )
        vaccine_available = (
            hasattr(self.model, "vaccine_round")
            and self.model.steps >= self.model.vaccine_round
        )
        return SIRObservation(
            round_number=self.model.steps,
            health_state=self.health_state,
            n_infected_neighbors=n_infected,
            n_total_neighbors=len(neighbors),
            infection_rate=infected_count / total_agents if total_agents > 0 else 0.0,
            my_past_isolations=list(self._past_isolations),
            days_infected=self.days_infected,
            immunity=self.immunity,
            vaccinated=self.vaccinated,
            vaccine_available=vaccine_available,
        )

    def decide(self) -> bool:
        """Return True if agent wants to self-isolate."""
        # Forced isolation after vaccination overrides brain decision
        if self.vaccine_isolation_remaining > 0:
            return True
        obs = self.get_observation()
        return bool(self.brain.decide(obs))

    def record_result(self, result: SIRRoundResult) -> None:
        self._past_isolations.append(result.isolated)
        self.happiness_change = result.happiness_change
        self.happiness = max(0.0, min(100.0, self.happiness + result.happiness_change))
        self.isolated = result.isolated
        self.health_state = result.health_state
        self.immunity = result.immunity
        if result.vaccinated and not self.vaccinated:
            self.vaccinated = True
        if result.got_infected:
            self.days_infected = 1
        elif self.health_state == HealthState.INFECTED:
            self.days_infected += 1
        elif result.recovered:
            self.days_infected = 0
        if self.vaccine_isolation_remaining > 0:
            self.vaccine_isolation_remaining -= 1
        self.brain.update(result)
