"""Cournot Oligopoly RL adapter."""

from __future__ import annotations

from policy_arena.brains.rl.bandit import BanditBrain
from policy_arena.brains.rl.q_learning import QLearningBrain

# Discretize quantities into fractions of max_quantity
COURNOT_QUANTITY_LEVELS = [0.0, 0.15, 0.3, 0.45, 0.6, 0.75, 0.9]


def _cournot_state_encoder(obs) -> str:
    """State = binned market price relative to max_price from last round."""
    if not obs.market_past_prices:
        return "start"
    price_frac = obs.market_past_prices[-1] / obs.max_price if obs.max_price > 0 else 0
    if price_frac < 0.2:
        return "price_very_low"
    elif price_frac < 0.4:
        return "price_low"
    elif price_frac < 0.6:
        return "price_mid"
    elif price_frac < 0.8:
        return "price_high"
    else:
        return "price_very_high"


def _cournot_reward_extractor(result) -> float:
    return result.profit


class _CournotQLearningBrain(QLearningBrain):
    """Q-learning for Cournot — maps discrete action indices to quantity amounts."""

    def __init__(self, **kwargs):
        super().__init__(
            action_space=COURNOT_QUANTITY_LEVELS,
            state_encoder=_cournot_state_encoder,
            reward_extractor=_cournot_reward_extractor,
            **kwargs,
        )

    @property
    def name(self) -> str:
        return f"q_learning(lr={self._lr},e={self._epsilon:.2f})"

    def decide(self, observation) -> float:
        """Return quantity as fraction * max_quantity."""
        fraction = super().decide(observation)
        return fraction * observation.max_quantity


class _CournotBanditBrain(BanditBrain):
    """Bandit for Cournot — maps fraction actions to quantity amounts."""

    def __init__(self, **kwargs):
        super().__init__(
            action_space=COURNOT_QUANTITY_LEVELS,
            reward_extractor=_cournot_reward_extractor,
            **kwargs,
        )

    @property
    def name(self) -> str:
        return f"bandit(e={self._epsilon:.2f})"

    def decide(self, observation) -> float:
        fraction = super().decide(observation)
        return fraction * observation.max_quantity


def cournot_bandit(
    epsilon: float = 0.3,
    epsilon_decay: float = 0.99,
    epsilon_min: float = 0.01,
    seed: int | None = None,
) -> _CournotBanditBrain:
    return _CournotBanditBrain(
        epsilon=epsilon,
        epsilon_decay=epsilon_decay,
        epsilon_min=epsilon_min,
        seed=seed,
    )


def cournot_q_learning(
    learning_rate: float = 0.15,
    epsilon: float = 0.3,
    discount: float = 0.95,
    epsilon_decay: float = 0.99,
    epsilon_min: float = 0.01,
    seed: int | None = None,
) -> _CournotQLearningBrain:
    return _CournotQLearningBrain(
        learning_rate=learning_rate,
        discount=discount,
        epsilon=epsilon,
        epsilon_decay=epsilon_decay,
        epsilon_min=epsilon_min,
        seed=seed,
    )
