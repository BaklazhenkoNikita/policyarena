"""Information Cascade RL adapter."""

from __future__ import annotations

from policy_arena.brains.rl.bandit import BanditBrain
from policy_arena.brains.rl.q_learning import QLearningBrain

CASCADE_ACTIONS = ["A", "B"]


def _cascade_state_encoder(obs) -> str:
    """State = (signal, majority_of_prior_choices).

    Encodes the agent's private signal and the majority direction
    of prior choices into a discrete state string.
    """
    signal = obs.my_signal

    if not obs.prior_choices:
        majority = "none"
    else:
        count_a = obs.prior_choices.count("A")
        count_b = obs.prior_choices.count("B")
        if count_a > count_b:
            majority = "A"
        elif count_b > count_a:
            majority = "B"
        else:
            majority = "tie"

    return f"sig={signal}_maj={majority}"


def _cascade_reward_extractor(result) -> float:
    return result.payoff


class _CascadeQLearningBrain(QLearningBrain):
    """Q-learning for Information Cascade."""

    def __init__(self, **kwargs):
        super().__init__(
            action_space=CASCADE_ACTIONS,
            state_encoder=_cascade_state_encoder,
            reward_extractor=_cascade_reward_extractor,
            **kwargs,
        )

    @property
    def name(self) -> str:
        return f"q_learning(lr={self._lr},e={self._epsilon:.2f})"

    def decide(self, observation) -> str:
        return super().decide(observation)


class _CascadeBanditBrain(BanditBrain):
    """Bandit for Information Cascade."""

    def __init__(self, **kwargs):
        super().__init__(
            action_space=CASCADE_ACTIONS,
            reward_extractor=_cascade_reward_extractor,
            **kwargs,
        )

    @property
    def name(self) -> str:
        return f"bandit(e={self._epsilon:.2f})"

    def decide(self, observation) -> str:
        return super().decide(observation)


def cascade_bandit(
    epsilon: float = 0.3,
    epsilon_decay: float = 0.99,
    epsilon_min: float = 0.01,
    seed: int | None = None,
) -> _CascadeBanditBrain:
    return _CascadeBanditBrain(
        epsilon=epsilon,
        epsilon_decay=epsilon_decay,
        epsilon_min=epsilon_min,
        seed=seed,
    )


def cascade_q_learning(
    learning_rate: float = 0.15,
    epsilon: float = 0.3,
    discount: float = 0.95,
    epsilon_decay: float = 0.99,
    epsilon_min: float = 0.01,
    seed: int | None = None,
) -> _CascadeQLearningBrain:
    return _CascadeQLearningBrain(
        learning_rate=learning_rate,
        discount=discount,
        epsilon=epsilon,
        epsilon_decay=epsilon_decay,
        epsilon_min=epsilon_min,
        seed=seed,
    )
