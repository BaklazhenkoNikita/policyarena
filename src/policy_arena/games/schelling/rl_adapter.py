"""Schelling Segregation RL adapter."""

from __future__ import annotations

from policy_arena.brains.rl.bandit import BanditBrain
from policy_arena.brains.rl.q_learning import QLearningBrain


def _schelling_state_encoder(obs) -> str:
    """State = binned fraction of same-type neighbors."""
    if obs.n_neighbors == 0:
        return "isolated"
    frac = obs.fraction_same
    if frac < 0.25:
        return "very_low"
    elif frac < 0.5:
        return "low"
    elif frac < 0.75:
        return "moderate"
    else:
        return "high"


def _schelling_reward_extractor(result) -> float:
    return result.payoff


def schelling_q_learning(
    learning_rate: float = 0.15,
    epsilon: float = 0.3,
    discount: float = 0.95,
    epsilon_decay: float = 0.99,
    epsilon_min: float = 0.01,
    seed: int | None = None,
) -> QLearningBrain:
    return QLearningBrain(
        action_space=[True, False],  # move or stay
        state_encoder=_schelling_state_encoder,
        reward_extractor=_schelling_reward_extractor,
        learning_rate=learning_rate,
        discount=discount,
        epsilon=epsilon,
        epsilon_decay=epsilon_decay,
        epsilon_min=epsilon_min,
        seed=seed,
    )


def schelling_bandit(
    epsilon: float = 0.3,
    epsilon_decay: float = 0.99,
    epsilon_min: float = 0.01,
    seed: int | None = None,
) -> BanditBrain:
    return BanditBrain(
        action_space=[True, False],  # move or stay
        reward_extractor=_schelling_reward_extractor,
        epsilon=epsilon,
        epsilon_decay=epsilon_decay,
        epsilon_min=epsilon_min,
        seed=seed,
    )
