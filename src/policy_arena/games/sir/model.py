"""SIR Disease Spread model on a network.

Agents live on a network graph. Each step:
1. Agents decide whether to self-isolate (reduces infection probability).
2. Infected agents can spread disease to susceptible neighbors.
3. Infected agents recover with probability gamma.
4. Immunity (natural + vaccine) wanes over time; recovered agents can become
   susceptible again.
5. Vaccines are administered starting at vaccine_round based on a chosen strategy.

Happiness system (0-100, starting at 80):
  - Healthy & not isolating: +healthy_happiness (default +1, normal life)
  - Healthy & isolating: +isolation_happiness (default -1, safe but costly)
  - Infected: +infected_happiness * (1 + severity * days_infected)
    → default -5 base, worsens with duration
  - Recovered: +recovered_happiness (default +2, immune, normal life)
  - Vaccination cost: vaccine_happiness_cost (one-time hit, default -3)

Key metrics:
  - susceptible_pct, infected_pct, recovered_pct: SIR compartment sizes
  - isolation_rate: fraction currently self-isolating
  - peak_infection: maximum infection rate seen so far
  - vaccinated_pct: fraction that have been vaccinated
  - gini: inequality of happiness scores
"""

from __future__ import annotations

import mesa
import networkx as nx

from policy_arena.brains.base import Brain
from policy_arena.games.sir.agents import SIRAgent
from policy_arena.games.sir.types import HealthState, SIRRoundResult
from policy_arena.metrics.gini import gini_coefficient


class SIRModel(mesa.Model):
    """SIR Disease Spread on a network with vaccination and waning immunity."""

    def __init__(
        self,
        brains: list[Brain],
        n_rounds: int = 200,
        beta: float = 0.3,
        gamma: float = 0.1,
        isolation_effectiveness: float = 0.8,
        initial_infected: int = 3,
        network_type: str = "small_world",
        network_k: int = 4,
        network_p: float = 0.1,
        healthy_happiness: float = 1.0,
        isolation_happiness: float = -1.0,
        infected_happiness: float = -5.0,
        recovered_happiness: float = 2.0,
        infection_severity: float = 0.1,
        # Vaccine parameters
        vaccine_round: int = 1,
        vaccine_efficacy: float = 0.9,
        vaccine_strategy: str = "random",
        vaccine_coverage: float = 0.005,
        vaccine_happiness_cost: float = -3.0,
        vaccine_isolation_rounds: int = 1,
        # Immunity waning parameters
        natural_immunity_decay: float = 0.01,
        vaccine_immunity_decay: float = 0.003,
        labels: list[str] | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.n_rounds = n_rounds
        self.beta = beta
        self.gamma = gamma
        self.isolation_effectiveness = isolation_effectiveness
        self.healthy_happiness = healthy_happiness
        self.isolation_happiness = isolation_happiness
        self.infected_happiness = infected_happiness
        self.recovered_happiness = recovered_happiness
        self.infection_severity = infection_severity

        # Vaccine
        self.vaccine_round = vaccine_round
        self.vaccine_efficacy = vaccine_efficacy
        self.vaccine_strategy = vaccine_strategy
        self.vaccine_coverage = vaccine_coverage
        self.vaccine_happiness_cost = vaccine_happiness_cost
        self.vaccine_isolation_rounds = vaccine_isolation_rounds

        # Immunity waning
        self.natural_immunity_decay = natural_immunity_decay
        self.vaccine_immunity_decay = vaccine_immunity_decay

        n = len(brains)
        rng_seed = self.random.randint(0, 2**31)
        # Build network
        if network_type == "small_world":
            k = min(network_k, n - 1)
            if k % 2 != 0:
                k = max(2, k - 1)
            self.network = nx.watts_strogatz_graph(n, k, network_p, seed=rng_seed)
        elif network_type == "scale_free":
            m = min(network_k // 2, n - 1)
            m = max(1, m)
            self.network = nx.barabasi_albert_graph(n, m, seed=rng_seed)
        elif network_type == "random":
            p = network_p if network_p > 0.01 else (network_k / max(n - 1, 1))
            self.network = nx.erdos_renyi_graph(n, p, seed=rng_seed)
        elif network_type == "complete":
            self.network = nx.complete_graph(n)
        elif network_type == "ring":
            k = min(network_k, n - 1)
            if k % 2 != 0:
                k = max(2, k - 1)
            self.network = nx.watts_strogatz_graph(n, k, 0.0, seed=rng_seed)
        elif network_type == "powerlaw_cluster":
            m = min(network_k // 2, n - 1)
            m = max(1, m)
            p = max(0.0, min(1.0, network_p))
            self.network = nx.powerlaw_cluster_graph(n, m, p, seed=rng_seed)
        elif network_type == "caveman":
            clique_size = max(3, network_k)
            n_cliques = max(2, n // clique_size)
            self.network = nx.connected_caveman_graph(n_cliques, clique_size)
            actual_n = self.network.number_of_nodes()
            if actual_n < n:
                for extra in range(actual_n, n):
                    self.network.add_node(extra)
                    target = self.random.randint(0, actual_n - 1)
                    self.network.add_edge(extra, target)
        else:
            k = min(network_k, n - 1)
            if k % 2 != 0:
                k = max(2, k - 1)
            self.network = nx.watts_strogatz_graph(n, k, network_p, seed=rng_seed)

        # Create agents and map them to network nodes
        self._agent_map: dict[int, SIRAgent] = {}
        infected_indices = set(self.random.sample(range(n), min(initial_infected, n)))

        for i, brain in enumerate(brains):
            state = (
                HealthState.INFECTED
                if i in infected_indices
                else HealthState.SUSCEPTIBLE
            )
            label = labels[i] if labels else None
            agent = SIRAgent(self, brain=brain, initial_state=state, label=label)
            self._agent_map[i] = agent
            agent._network_node = i

        # Pre-compute layout positions for visualization
        self._node_positions: dict[int, tuple[float, float]] = nx.spring_layout(
            self.network, seed=42, scale=1.0
        )

        self._peak_infection: float = 0.0
        self._sir_history: list[dict[str, float]] = []
        self._vaccinated_this_round: set[int] = set()
        self._vaccine_dose_accumulator: float = 0.0

        self.datacollector = mesa.DataCollector(
            model_reporters={
                "susceptible_pct": lambda m: m._metric_compartment_pct(
                    HealthState.SUSCEPTIBLE
                ),
                "infected_pct": lambda m: m._metric_compartment_pct(
                    HealthState.INFECTED
                ),
                "recovered_pct": lambda m: m._metric_compartment_pct(
                    HealthState.RECOVERED
                ),
                "isolation_rate": lambda m: m._metric_isolation_rate(),
                "peak_infection": lambda m: m._peak_infection,
                "vaccinated_pct": lambda m: m._metric_vaccinated_pct(),
                "gini": lambda m: m._metric_gini(),
            },
            agent_reporters={
                "happiness": "happiness",
                "happiness_change": "happiness_change",
                "health_state": lambda a: a.health_state.value,
                "isolated": "isolated",
                "brain_name": "brain_name",
                "label": "label",
            },
        )

    def _metric_compartment_pct(self, state: HealthState) -> float:
        agents = list(self.agents)
        if not agents:
            return 0.0
        return sum(1 for a in agents if a.health_state == state) / len(agents)

    def _metric_isolation_rate(self) -> float:
        agents = list(self.agents)
        if not agents:
            return 0.0
        return sum(1 for a in agents if a.isolated) / len(agents)

    def _metric_vaccinated_pct(self) -> float:
        agents = list(self.agents)
        if not agents:
            return 0.0
        return sum(1 for a in agents if a.vaccinated) / len(agents)

    def _metric_gini(self) -> float:
        scores = [a.happiness for a in self.agents]
        return gini_coefficient(scores)

    def _administer_vaccines(self, agents: list[SIRAgent]) -> None:
        """Administer vaccines to eligible agents based on strategy."""
        self._vaccinated_this_round = set()

        if self.steps < self.vaccine_round:
            return
        if self.vaccine_coverage <= 0:
            return

        # Eligible: susceptible or recovered agents who haven't been vaccinated yet
        eligible = [
            a
            for a in agents
            if not a.vaccinated and a.health_state != HealthState.INFECTED
        ]
        if not eligible:
            return

        # Accumulate fractional doses so small coverage rates work correctly
        self._vaccine_dose_accumulator += len(agents) * self.vaccine_coverage
        n_to_vaccinate = int(self._vaccine_dose_accumulator)
        if n_to_vaccinate <= 0:
            return
        self._vaccine_dose_accumulator -= n_to_vaccinate

        if self.vaccine_strategy == "most_connected":
            # Prioritize agents with most network connections (highest degree)
            eligible.sort(
                key=lambda a: self.network.degree(a._network_node), reverse=True
            )
            selected = eligible[:n_to_vaccinate]
        elif self.vaccine_strategy == "highest_risk":
            # Prioritize agents with most infected neighbors
            def infected_neighbor_count(a: SIRAgent) -> int:
                return sum(
                    1
                    for nb in a.get_neighbors()
                    if nb.health_state == HealthState.INFECTED
                )

            eligible.sort(key=infected_neighbor_count, reverse=True)
            selected = eligible[:n_to_vaccinate]
        else:
            # "random" — shuffle and pick
            self.random.shuffle(eligible)
            selected = eligible[:n_to_vaccinate]

        for agent in selected:
            agent.vaccinated = True
            agent.immunity = min(1.0, agent.immunity + self.vaccine_efficacy)
            agent.vaccine_isolation_remaining = self.vaccine_isolation_rounds
            self._vaccinated_this_round.add(agent.unique_id)

    def _wane_immunity(self, agents: list[SIRAgent]) -> None:
        """Decay immunity each round. Immunity directly reduces infection probability."""
        for agent in agents:
            if agent.immunity <= 0:
                continue

            # Decay rate depends on source of immunity
            if agent.vaccinated:
                decay = self.vaccine_immunity_decay
            else:
                decay = self.natural_immunity_decay

            agent.immunity = max(0.0, agent.immunity - decay)

    def step(self) -> None:
        agents = list(self.agents)

        # 0. Wane immunity
        self._wane_immunity(agents)

        # 1. Administer vaccines (once available)
        self._administer_vaccines(agents)

        # 2. Agents decide whether to isolate
        from policy_arena.games.parallel import gather_decisions

        max_w = getattr(self, "max_concurrent_llm", 1)
        isolation_decisions = gather_decisions(agents, lambda a: a.decide(), max_w)

        # 3. Disease dynamics
        new_states: dict[int, HealthState] = {}
        got_infected: dict[int, bool] = {}
        recovered: dict[int, bool] = {}

        for agent in agents:
            got_infected[agent.unique_id] = False
            recovered[agent.unique_id] = False

            if agent.health_state in (HealthState.SUSCEPTIBLE, HealthState.RECOVERED):
                # Both susceptible and recovered can get (re)infected.
                # Immunity directly reduces infection probability:
                #   e.g. 90% immunity → infection is 10x less likely.
                neighbors = agent.get_neighbors()
                infected_neighbors = [
                    nb
                    for nb in neighbors
                    if nb.health_state == HealthState.INFECTED
                    and not isolation_decisions.get(nb.unique_id, False)
                ]
                effective_beta = self.beta

                # Isolation reduces exposure
                if isolation_decisions[agent.unique_id]:
                    effective_beta *= 1.0 - self.isolation_effectiveness

                # Immunity reduces infection probability
                if agent.immunity > 0:
                    effective_beta *= 1.0 - agent.immunity

                # Each infected neighbor has a chance to transmit
                infected = False
                for _ in infected_neighbors:
                    if self.random.random() < effective_beta:
                        infected = True
                        break

                if infected:
                    new_states[agent.unique_id] = HealthState.INFECTED
                    got_infected[agent.unique_id] = True
                    agent.immunity = 0.0
                else:
                    new_states[agent.unique_id] = agent.health_state

            elif agent.health_state == HealthState.INFECTED:
                if self.random.random() < self.gamma:
                    new_states[agent.unique_id] = HealthState.RECOVERED
                    recovered[agent.unique_id] = True
                    # Natural immunity on recovery
                    agent.immunity = 1.0
                else:
                    new_states[agent.unique_id] = HealthState.INFECTED

        # 4. Compute happiness changes and record results
        for agent in agents:
            new_state = new_states[agent.unique_id]
            is_isolating = isolation_decisions[agent.unique_id]
            just_vaccinated = agent.unique_id in self._vaccinated_this_round

            if new_state == HealthState.INFECTED:
                days = agent.days_infected
                delta = self.infected_happiness * (1.0 + self.infection_severity * days)
            elif new_state == HealthState.RECOVERED:
                delta = self.recovered_happiness
            elif is_isolating:
                delta = self.isolation_happiness
            else:
                delta = self.healthy_happiness

            # One-time vaccination happiness cost
            if just_vaccinated:
                delta += self.vaccine_happiness_cost

            agent.record_result(
                SIRRoundResult(
                    isolated=is_isolating,
                    health_state=new_state,
                    got_infected=got_infected[agent.unique_id],
                    recovered=recovered[agent.unique_id],
                    happiness_change=delta,
                    round_number=self.steps,
                    vaccinated=just_vaccinated,
                    immunity=agent.immunity,
                )
            )

        # Track peak infection
        current_infected = sum(
            1 for a in agents if a.health_state == HealthState.INFECTED
        ) / len(agents)
        self._peak_infection = max(self._peak_infection, current_infected)

        self.datacollector.collect(self)

        # Stop early if no infected agents remain (disease died out)
        if (
            not any(a.health_state == HealthState.INFECTED for a in agents)
            or self.steps >= self.n_rounds
        ):
            self.running = False
