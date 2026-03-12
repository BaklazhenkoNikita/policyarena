"""Hawk-Dove RL adapter."""

from __future__ import annotations

from policy_arena.brains.rl.bandit import BanditBrain
from policy_arena.brains.rl.best_response import BestResponseBrain
from policy_arena.brains.rl.q_learning import QLearningBrain
from policy_arena.core.types import Action


def _hd_state_encoder(obs) -> str:
    if obs.opponent_history:
        return obs.opponent_history[-1].value
    return "start"


def hd_q_learning(
    learning_rate: float = 0.15,
    epsilon: float = 0.3,
    discount: float = 0.95,
    epsilon_decay: float = 0.99,
    epsilon_min: float = 0.01,
    seed: int | None = None,
) -> QLearningBrain:
    return QLearningBrain(
        action_space=[Action.COOPERATE, Action.DEFECT],
        state_encoder=_hd_state_encoder,
        learning_rate=learning_rate,
        discount=discount,
        epsilon=epsilon,
        epsilon_decay=epsilon_decay,
        epsilon_min=epsilon_min,
        seed=seed,
    )


def _hd_payoff_func(my_action: Action, opp_action: Action) -> float:
    matrix = {
        (Action.COOPERATE, Action.COOPERATE): 2.0,
        (Action.COOPERATE, Action.DEFECT): 0.0,
        (Action.DEFECT, Action.COOPERATE): 4.0,
        (Action.DEFECT, Action.DEFECT): -1.0,
    }
    return matrix[(my_action, opp_action)]


def hd_best_response() -> BestResponseBrain:
    return BestResponseBrain(
        action_space=[Action.COOPERATE, Action.DEFECT],
        payoff_func=_hd_payoff_func,
    )


def hd_bandit(
    epsilon: float = 0.3,
    epsilon_decay: float = 0.99,
    epsilon_min: float = 0.01,
    seed: int | None = None,
) -> BanditBrain:
    return BanditBrain(
        action_space=[Action.COOPERATE, Action.DEFECT],
        epsilon=epsilon,
        epsilon_decay=epsilon_decay,
        epsilon_min=epsilon_min,
        seed=seed,
    )
