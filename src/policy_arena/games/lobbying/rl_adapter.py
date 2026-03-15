"""Lobbying / Rent-Seeking Contest RL adapter."""

from __future__ import annotations

from policy_arena.brains.rl.bandit import BanditBrain
from policy_arena.brains.rl.q_learning import QLearningBrain

# Discretize spend into 7 levels as fractions of budget
LOBBYING_SPEND_LEVELS = [0.0, 0.1, 0.2, 0.35, 0.5, 0.7, 1.0]


def _lobbying_state_encoder(obs) -> str:
    """State = binned total dissipation rate from last round."""
    if not obs.past_total_spends:
        return "start"
    dissipation = (
        obs.past_total_spends[-1] / obs.prize_value if obs.prize_value > 0 else 0
    )
    if dissipation < 0.2:
        return "low"
    elif dissipation < 0.4:
        return "med_low"
    elif dissipation < 0.6:
        return "med"
    elif dissipation < 0.8:
        return "med_high"
    else:
        return "high"


def _lobbying_reward_extractor(result) -> float:
    return result.payoff


class _LobbyingQLearningBrain(QLearningBrain):
    """Q-learning for Lobbying — maps discrete action indices to spend amounts."""

    def __init__(self, **kwargs):
        super().__init__(
            action_space=LOBBYING_SPEND_LEVELS,
            state_encoder=_lobbying_state_encoder,
            reward_extractor=_lobbying_reward_extractor,
            **kwargs,
        )

    @property
    def name(self) -> str:
        return f"q_learning(lr={self._lr},e={self._epsilon:.2f})"

    def decide(self, observation) -> float:
        """Return spend as fraction * budget."""
        fraction = super().decide(observation)
        return fraction * observation.budget


class _LobbyingBanditBrain(BanditBrain):
    """Bandit for Lobbying — maps fraction actions to spend amounts."""

    def __init__(self, **kwargs):
        super().__init__(
            action_space=LOBBYING_SPEND_LEVELS,
            reward_extractor=_lobbying_reward_extractor,
            **kwargs,
        )

    @property
    def name(self) -> str:
        return f"bandit(e={self._epsilon:.2f})"

    def decide(self, observation) -> float:
        fraction = super().decide(observation)
        return fraction * observation.budget


def lobbying_bandit(
    epsilon: float = 0.3,
    epsilon_decay: float = 0.99,
    epsilon_min: float = 0.01,
    seed: int | None = None,
) -> _LobbyingBanditBrain:
    return _LobbyingBanditBrain(
        epsilon=epsilon,
        epsilon_decay=epsilon_decay,
        epsilon_min=epsilon_min,
        seed=seed,
    )


def lobbying_q_learning(
    learning_rate: float = 0.15,
    epsilon: float = 0.3,
    discount: float = 0.95,
    epsilon_decay: float = 0.99,
    epsilon_min: float = 0.01,
    seed: int | None = None,
) -> _LobbyingQLearningBrain:
    return _LobbyingQLearningBrain(
        learning_rate=learning_rate,
        discount=discount,
        epsilon=epsilon,
        epsilon_decay=epsilon_decay,
        epsilon_min=epsilon_min,
        seed=seed,
    )
