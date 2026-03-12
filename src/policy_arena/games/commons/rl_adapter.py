"""Tragedy of the Commons RL adapter."""

from __future__ import annotations

from policy_arena.brains.rl.bandit import BanditBrain
from policy_arena.brains.rl.q_learning import QLearningBrain

# Discretize harvests as fraction of fair share: 0%, 25%, 50%, 75%, 100%
TC_HARVEST_LEVELS = [0.0, 0.25, 0.5, 0.75, 1.0]


def _tc_state_encoder(obs) -> str:
    """State = binned resource level relative to max."""
    if not obs.resource_history:
        return "start"
    fullness = obs.resource_level / obs.max_resource if obs.max_resource > 0 else 0
    if fullness < 0.1:
        return "depleted"
    elif fullness < 0.3:
        return "low"
    elif fullness < 0.6:
        return "medium"
    elif fullness < 0.8:
        return "high"
    else:
        return "full"


def _tc_reward_extractor(result) -> float:
    return result.payoff


class _TCQLearningBrain(QLearningBrain):
    """Q-learning for Commons — maps fraction actions to harvest amounts."""

    def __init__(self, **kwargs):
        super().__init__(
            action_space=TC_HARVEST_LEVELS,
            state_encoder=_tc_state_encoder,
            reward_extractor=_tc_reward_extractor,
            **kwargs,
        )

    @property
    def name(self) -> str:
        return f"q_learning(lr={self._lr},e={self._epsilon:.2f})"

    def decide(self, observation) -> float:
        """Return harvest as fraction * fair_share."""
        fraction = super().decide(observation)
        n = observation.n_agents if observation.n_agents > 0 else 1
        fair_share = observation.resource_level / n
        return fraction * fair_share


class _TCBanditBrain(BanditBrain):
    """Bandit for Commons — maps fraction actions to harvest amounts."""

    def __init__(self, **kwargs):
        super().__init__(
            action_space=TC_HARVEST_LEVELS,
            reward_extractor=_tc_reward_extractor,
            **kwargs,
        )

    @property
    def name(self) -> str:
        return f"bandit(e={self._epsilon:.2f})"

    def decide(self, observation) -> float:
        fraction = super().decide(observation)
        n = observation.n_agents if observation.n_agents > 0 else 1
        fair_share = observation.resource_level / n
        return fraction * fair_share


def tc_q_learning(
    learning_rate: float = 0.15,
    epsilon: float = 0.3,
    discount: float = 0.95,
    epsilon_decay: float = 0.99,
    epsilon_min: float = 0.01,
    seed: int | None = None,
) -> _TCQLearningBrain:
    return _TCQLearningBrain(
        learning_rate=learning_rate,
        discount=discount,
        epsilon=epsilon,
        epsilon_decay=epsilon_decay,
        epsilon_min=epsilon_min,
        seed=seed,
    )


def tc_bandit(
    epsilon: float = 0.3,
    epsilon_decay: float = 0.99,
    epsilon_min: float = 0.01,
    seed: int | None = None,
) -> _TCBanditBrain:
    return _TCBanditBrain(
        epsilon=epsilon,
        epsilon_decay=epsilon_decay,
        epsilon_min=epsilon_min,
        seed=seed,
    )
