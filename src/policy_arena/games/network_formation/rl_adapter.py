"""Network Formation Game RL adapter."""

from __future__ import annotations

import random

from policy_arena.brains.rl.bandit import BanditBrain
from policy_arena.brains.rl.q_learning import QLearningBrain

# Discretize number of links to propose: 0, 1, 2, 3, or max
NF_LINK_COUNT_LEVELS = [0, 1, 2, 3, 4]


def _nf_state_encoder(obs) -> str:
    """State = binned own degree + binned network density."""
    max_deg = obs.n_players - 1 if obs.n_players > 1 else 1
    deg_frac = len(obs.my_current_links) / max_deg

    if deg_frac < 0.25:
        deg_bin = "low_deg"
    elif deg_frac < 0.5:
        deg_bin = "med_deg"
    elif deg_frac < 0.75:
        deg_bin = "high_deg"
    else:
        deg_bin = "full_deg"

    density = obs.network_density_history[-1] if obs.network_density_history else 0.0

    if density < 0.25:
        den_bin = "sparse"
    elif density < 0.5:
        den_bin = "med_dense"
    elif density < 0.75:
        den_bin = "dense"
    else:
        den_bin = "full_dense"

    return f"{deg_bin}_{den_bin}"


def _nf_reward_extractor(result) -> float:
    return result.my_payoff


class _NFQLearningBrain(QLearningBrain):
    """Q-learning for Network Formation — maps link count to random target selection."""

    def __init__(self, seed: int | None = None, **kwargs):
        super().__init__(
            action_space=NF_LINK_COUNT_LEVELS,
            state_encoder=_nf_state_encoder,
            reward_extractor=_nf_reward_extractor,
            **kwargs,
        )
        self._rng = random.Random(seed)
        self._n_players: int = 0
        self._my_index: int = 0

    @property
    def name(self) -> str:
        return f"q_learning(lr={self._lr},e={self._epsilon:.2f})"

    def decide(self, observation) -> list[int]:
        """Return list of agent indices to connect to."""
        self._n_players = observation.n_players
        self._my_index = observation.my_agent_index
        num_links = super().decide(observation)
        num_links = min(num_links, observation.max_links, observation.n_players - 1)
        others = [
            i for i in range(observation.n_players) if i != observation.my_agent_index
        ]
        k = min(num_links, len(others))
        if k <= 0:
            return []
        return self._rng.sample(others, k)


class _NFBanditBrain(BanditBrain):
    """Bandit for Network Formation — maps link count to random target selection."""

    def __init__(self, seed: int | None = None, **kwargs):
        super().__init__(
            action_space=NF_LINK_COUNT_LEVELS,
            reward_extractor=_nf_reward_extractor,
            **kwargs,
        )
        self._rng = random.Random(seed)
        self._n_players: int = 0
        self._my_index: int = 0

    @property
    def name(self) -> str:
        return f"bandit(e={self._epsilon:.2f})"

    def decide(self, observation) -> list[int]:
        """Return list of agent indices to connect to."""
        self._n_players = observation.n_players
        self._my_index = observation.my_agent_index
        num_links = super().decide(observation)
        num_links = min(num_links, observation.max_links, observation.n_players - 1)
        others = [
            i for i in range(observation.n_players) if i != observation.my_agent_index
        ]
        k = min(num_links, len(others))
        if k <= 0:
            return []
        return self._rng.sample(others, k)


def nf_bandit(
    epsilon: float = 0.3,
    epsilon_decay: float = 0.99,
    epsilon_min: float = 0.01,
    seed: int | None = None,
) -> _NFBanditBrain:
    return _NFBanditBrain(
        epsilon=epsilon,
        epsilon_decay=epsilon_decay,
        epsilon_min=epsilon_min,
        seed=seed,
    )


def nf_q_learning(
    learning_rate: float = 0.15,
    epsilon: float = 0.3,
    discount: float = 0.95,
    epsilon_decay: float = 0.99,
    epsilon_min: float = 0.01,
    seed: int | None = None,
) -> _NFQLearningBrain:
    return _NFQLearningBrain(
        learning_rate=learning_rate,
        discount=discount,
        epsilon=epsilon,
        epsilon_decay=epsilon_decay,
        epsilon_min=epsilon_min,
        seed=seed,
    )
