"""Minority Game RL adapter."""

from __future__ import annotations

from policy_arena.brains.rl.bandit import BanditBrain
from policy_arena.brains.rl.q_learning import QLearningBrain


def _mg_state_encoder(obs) -> str:
    """State = binned A fraction from last round."""
    if not obs.past_a_counts:
        return "start"
    a_frac = obs.past_a_counts[-1] / obs.n_agents if obs.n_agents > 0 else 0.5
    if a_frac < 0.3:
        return "few_a"
    elif a_frac < 0.45:
        return "slight_a"
    elif a_frac < 0.55:
        return "balanced"
    elif a_frac < 0.7:
        return "slight_b"
    else:
        return "few_b"


def _mg_reward_extractor(result) -> float:
    return result.payoff


def mg_q_learning(
    learning_rate: float = 0.15,
    epsilon: float = 0.3,
    discount: float = 0.95,
    epsilon_decay: float = 0.99,
    epsilon_min: float = 0.01,
    seed: int | None = None,
) -> QLearningBrain:
    return QLearningBrain(
        action_space=[True, False],  # A or B
        state_encoder=_mg_state_encoder,
        reward_extractor=_mg_reward_extractor,
        learning_rate=learning_rate,
        discount=discount,
        epsilon=epsilon,
        epsilon_decay=epsilon_decay,
        epsilon_min=epsilon_min,
        seed=seed,
    )


def mg_bandit(
    epsilon: float = 0.3,
    epsilon_decay: float = 0.99,
    epsilon_min: float = 0.01,
    seed: int | None = None,
) -> BanditBrain:
    return BanditBrain(
        action_space=[True, False],  # A or B
        reward_extractor=_mg_reward_extractor,
        epsilon=epsilon,
        epsilon_decay=epsilon_decay,
        epsilon_min=epsilon_min,
        seed=seed,
    )
