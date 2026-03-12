"""SIR Disease Spread RL adapter."""

from __future__ import annotations

from policy_arena.brains.rl.bandit import BanditBrain
from policy_arena.brains.rl.q_learning import QLearningBrain


def _sir_state_encoder(obs) -> str:
    """State = health state + neighborhood infection level."""
    state = obs.health_state.value
    if obs.n_total_neighbors == 0:
        return f"{state}_isolated"
    frac = obs.n_infected_neighbors / obs.n_total_neighbors
    if frac == 0:
        return f"{state}_safe"
    elif frac < 0.3:
        return f"{state}_low_risk"
    elif frac < 0.6:
        return f"{state}_med_risk"
    else:
        return f"{state}_high_risk"


def _sir_reward_extractor(result) -> float:
    return result.happiness_change


def sir_q_learning(
    learning_rate: float = 0.15,
    epsilon: float = 0.3,
    discount: float = 0.95,
    epsilon_decay: float = 0.99,
    epsilon_min: float = 0.01,
    seed: int | None = None,
) -> QLearningBrain:
    return QLearningBrain(
        action_space=[True, False],  # isolate or participate
        state_encoder=_sir_state_encoder,
        reward_extractor=_sir_reward_extractor,
        learning_rate=learning_rate,
        discount=discount,
        epsilon=epsilon,
        epsilon_decay=epsilon_decay,
        epsilon_min=epsilon_min,
        seed=seed,
    )


def sir_bandit(
    epsilon: float = 0.3,
    epsilon_decay: float = 0.99,
    epsilon_min: float = 0.01,
    seed: int | None = None,
) -> BanditBrain:
    return BanditBrain(
        action_space=[True, False],  # isolate or participate
        reward_extractor=_sir_reward_extractor,
        epsilon=epsilon,
        epsilon_decay=epsilon_decay,
        epsilon_min=epsilon_min,
        seed=seed,
    )
