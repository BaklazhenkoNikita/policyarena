"""Voting & Election Game RL adapter."""

from __future__ import annotations

from policy_arena.brains.rl.bandit import BanditBrain
from policy_arena.brains.rl.q_learning import QLearningBrain


def _voting_state_encoder(obs) -> str:
    """State = binned distance of last winner from voter's ideal point."""
    if not obs.past_winner_positions:
        return "start"
    last_winner_pos = obs.past_winner_positions[-1]
    distance = abs(last_winner_pos - obs.my_ideal_point)
    if distance < 10:
        return "very_close"
    elif distance < 25:
        return "close"
    elif distance < 50:
        return "moderate"
    elif distance < 75:
        return "far"
    else:
        return "very_far"


def _voting_reward_extractor(result) -> float:
    return result.payoff


class _VotingQLearningBrain(QLearningBrain):
    """Q-learning for Voting — action space is candidate indices."""

    def __init__(self, n_candidates: int = 4, **kwargs):
        super().__init__(
            action_space=list(range(n_candidates)),
            state_encoder=_voting_state_encoder,
            reward_extractor=_voting_reward_extractor,
            **kwargs,
        )

    @property
    def name(self) -> str:
        return f"q_learning(lr={self._lr},e={self._epsilon:.2f})"

    def decide(self, observation) -> int:
        """Return candidate index to vote for."""
        return super().decide(observation)


class _VotingBanditBrain(BanditBrain):
    """Bandit for Voting — action space is candidate indices."""

    def __init__(self, n_candidates: int = 4, **kwargs):
        super().__init__(
            action_space=list(range(n_candidates)),
            reward_extractor=_voting_reward_extractor,
            **kwargs,
        )

    @property
    def name(self) -> str:
        return f"bandit(e={self._epsilon:.2f})"

    def decide(self, observation) -> int:
        return super().decide(observation)


def voting_bandit(
    n_candidates: int = 4,
    epsilon: float = 0.3,
    epsilon_decay: float = 0.99,
    epsilon_min: float = 0.01,
    seed: int | None = None,
) -> _VotingBanditBrain:
    return _VotingBanditBrain(
        n_candidates=n_candidates,
        epsilon=epsilon,
        epsilon_decay=epsilon_decay,
        epsilon_min=epsilon_min,
        seed=seed,
    )


def voting_q_learning(
    n_candidates: int = 4,
    learning_rate: float = 0.15,
    epsilon: float = 0.3,
    discount: float = 0.95,
    epsilon_decay: float = 0.99,
    epsilon_min: float = 0.01,
    seed: int | None = None,
) -> _VotingQLearningBrain:
    return _VotingQLearningBrain(
        n_candidates=n_candidates,
        learning_rate=learning_rate,
        discount=discount,
        epsilon=epsilon,
        epsilon_decay=epsilon_decay,
        epsilon_min=epsilon_min,
        seed=seed,
    )
