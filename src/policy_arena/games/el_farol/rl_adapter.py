"""El Farol Bar Problem RL adapter."""

from __future__ import annotations

from policy_arena.brains.rl.bandit import BanditBrain
from policy_arena.brains.rl.q_learning import QLearningBrain


def _ef_state_encoder(obs) -> str:
    """State = binned last attendance relative to threshold."""
    if not obs.past_attendance:
        return "start"
    ratio = obs.past_attendance[-1] / obs.threshold if obs.threshold > 0 else 0
    if ratio < 0.5:
        return "very_low"
    elif ratio < 0.8:
        return "low"
    elif ratio < 1.0:
        return "near"
    elif ratio < 1.2:
        return "over"
    else:
        return "very_over"


def _ef_reward_extractor(result) -> float:
    return result.payoff


def ef_bandit(
    epsilon: float = 0.3,
    epsilon_decay: float = 0.99,
    epsilon_min: float = 0.01,
    seed: int | None = None,
) -> BanditBrain:
    return BanditBrain(
        action_space=[True, False],
        reward_extractor=_ef_reward_extractor,
        epsilon=epsilon,
        epsilon_decay=epsilon_decay,
        epsilon_min=epsilon_min,
        seed=seed,
    )


def ef_q_learning(
    learning_rate: float = 0.15,
    epsilon: float = 0.3,
    discount: float = 0.95,
    epsilon_decay: float = 0.99,
    epsilon_min: float = 0.01,
    seed: int | None = None,
) -> QLearningBrain:
    return QLearningBrain(
        action_space=[True, False],  # attend or stay
        state_encoder=_ef_state_encoder,
        reward_extractor=_ef_reward_extractor,
        learning_rate=learning_rate,
        discount=discount,
        epsilon=epsilon,
        epsilon_decay=epsilon_decay,
        epsilon_min=epsilon_min,
        seed=seed,
    )
