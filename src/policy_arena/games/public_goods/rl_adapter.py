"""Public Goods Game RL adapter."""

from __future__ import annotations

from policy_arena.brains.rl.bandit import BanditBrain
from policy_arena.brains.rl.q_learning import QLearningBrain

# Discretize contributions into 5 levels: 0%, 25%, 50%, 75%, 100% of endowment
PG_CONTRIBUTION_LEVELS = [0.0, 0.25, 0.5, 0.75, 1.0]


def _pg_state_encoder(obs) -> str:
    """State = binned group average from last round."""
    if not obs.group_past_averages:
        return "start"
    avg_frac = obs.group_past_averages[-1] / obs.endowment if obs.endowment > 0 else 0
    if avg_frac < 0.2:
        return "low"
    elif avg_frac < 0.4:
        return "med_low"
    elif avg_frac < 0.6:
        return "med"
    elif avg_frac < 0.8:
        return "med_high"
    else:
        return "high"


def _pg_reward_extractor(result) -> float:
    return result.payoff


class _PGQLearningBrain(QLearningBrain):
    """Q-learning for Public Goods — maps discrete action indices to contribution amounts."""

    def __init__(self, **kwargs):
        super().__init__(
            action_space=PG_CONTRIBUTION_LEVELS,
            state_encoder=_pg_state_encoder,
            reward_extractor=_pg_reward_extractor,
            **kwargs,
        )

    @property
    def name(self) -> str:
        return f"q_learning(lr={self._lr},e={self._epsilon:.2f})"

    def decide(self, observation) -> float:
        """Return contribution as fraction * endowment."""
        fraction = super().decide(observation)
        return fraction * observation.endowment


class _PGBanditBrain(BanditBrain):
    """Bandit for Public Goods — maps fraction actions to contribution amounts."""

    def __init__(self, **kwargs):
        super().__init__(
            action_space=PG_CONTRIBUTION_LEVELS,
            reward_extractor=_pg_reward_extractor,
            **kwargs,
        )

    @property
    def name(self) -> str:
        return f"bandit(e={self._epsilon:.2f})"

    def decide(self, observation) -> float:
        fraction = super().decide(observation)
        return fraction * observation.endowment


def pg_bandit(
    epsilon: float = 0.3,
    epsilon_decay: float = 0.99,
    epsilon_min: float = 0.01,
    seed: int | None = None,
) -> _PGBanditBrain:
    return _PGBanditBrain(
        epsilon=epsilon,
        epsilon_decay=epsilon_decay,
        epsilon_min=epsilon_min,
        seed=seed,
    )


def pg_q_learning(
    learning_rate: float = 0.15,
    epsilon: float = 0.3,
    discount: float = 0.95,
    epsilon_decay: float = 0.99,
    epsilon_min: float = 0.01,
    seed: int | None = None,
) -> _PGQLearningBrain:
    return _PGQLearningBrain(
        learning_rate=learning_rate,
        discount=discount,
        epsilon=epsilon,
        epsilon_decay=epsilon_decay,
        epsilon_min=epsilon_min,
        seed=seed,
    )
