"""Sealed-Bid Auction RL adapter."""

from __future__ import annotations

from policy_arena.brains.rl.bandit import BanditBrain
from policy_arena.brains.rl.q_learning import QLearningBrain

# Bid fractions of own value
AUCTION_BID_FRACTIONS = [0.0, 0.3, 0.5, 0.7, 0.85, 1.0, 1.1]


def _auction_state_encoder(obs) -> str:
    """State = binned own value category."""
    if obs.value_max == obs.value_min:
        return "mid"
    frac = (obs.my_value - obs.value_min) / (obs.value_max - obs.value_min)
    if frac < 0.2:
        return "very_low"
    elif frac < 0.4:
        return "low"
    elif frac < 0.6:
        return "mid"
    elif frac < 0.8:
        return "high"
    else:
        return "very_high"


def _auction_reward_extractor(result) -> float:
    return result.payoff


class _AuctionQLearningBrain(QLearningBrain):
    """Q-learning for Auction — maps discrete action indices to bid fractions."""

    def __init__(self, **kwargs):
        super().__init__(
            action_space=AUCTION_BID_FRACTIONS,
            state_encoder=_auction_state_encoder,
            reward_extractor=_auction_reward_extractor,
            **kwargs,
        )

    @property
    def name(self) -> str:
        return f"q_learning(lr={self._lr},e={self._epsilon:.2f})"

    def decide(self, observation) -> float:
        """Return bid as fraction * value."""
        fraction = super().decide(observation)
        return fraction * observation.my_value


class _AuctionBanditBrain(BanditBrain):
    """Bandit for Auction — maps fraction actions to bid amounts."""

    def __init__(self, **kwargs):
        super().__init__(
            action_space=AUCTION_BID_FRACTIONS,
            reward_extractor=_auction_reward_extractor,
            **kwargs,
        )

    @property
    def name(self) -> str:
        return f"bandit(e={self._epsilon:.2f})"

    def decide(self, observation) -> float:
        fraction = super().decide(observation)
        return fraction * observation.my_value


def auction_bandit(
    epsilon: float = 0.3,
    epsilon_decay: float = 0.99,
    epsilon_min: float = 0.01,
    seed: int | None = None,
) -> _AuctionBanditBrain:
    return _AuctionBanditBrain(
        epsilon=epsilon,
        epsilon_decay=epsilon_decay,
        epsilon_min=epsilon_min,
        seed=seed,
    )


def auction_q_learning(
    learning_rate: float = 0.15,
    epsilon: float = 0.3,
    discount: float = 0.95,
    epsilon_decay: float = 0.99,
    epsilon_min: float = 0.01,
    seed: int | None = None,
) -> _AuctionQLearningBrain:
    return _AuctionQLearningBrain(
        learning_rate=learning_rate,
        discount=discount,
        epsilon=epsilon,
        epsilon_decay=epsilon_decay,
        epsilon_min=epsilon_min,
        seed=seed,
    )
