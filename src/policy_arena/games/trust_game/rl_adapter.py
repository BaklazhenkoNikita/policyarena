"""Trust Game RL adapter."""

from __future__ import annotations

from policy_arena.brains.rl.bandit import BanditBrain
from policy_arena.brains.rl.q_learning import QLearningBrain

TG_INVEST_LEVELS = [0.0, 0.25, 0.5, 0.75, 1.0]
TG_RETURN_LEVELS = [0.0, 0.1, 0.2, 0.33, 0.5]


def _tg_state_encoder(obs) -> str:
    if obs.role == "investor":
        if not obs.opponent_past_returns:
            return "investor_start"
        # Was opponent generous?
        last = obs.opponent_past_returns[-1]
        recv = obs.endowment * obs.multiplier  # approximate
        rate = last / recv if recv > 0 else 0
        if rate < 0.2:
            return "investor_stingy"
        elif rate < 0.4:
            return "investor_fair"
        else:
            return "investor_generous"
    else:
        if obs.amount_received is None:
            return "trustee_start"
        frac = (
            obs.amount_received / (obs.endowment * obs.multiplier)
            if obs.endowment > 0
            else 0
        )
        if frac < 0.3:
            return "trustee_low_invest"
        elif frac < 0.7:
            return "trustee_mid_invest"
        else:
            return "trustee_high_invest"


def _tg_reward_extractor(result) -> float:
    return result.payoff


class _TGQLearningBrain(QLearningBrain):
    """Q-learning for Trust Game — handles dual role."""

    def __init__(self, **kwargs):
        all_actions = TG_INVEST_LEVELS + TG_RETURN_LEVELS
        super().__init__(
            action_space=all_actions,
            state_encoder=_tg_state_encoder,
            reward_extractor=_tg_reward_extractor,
            **kwargs,
        )
        self._invest_actions = TG_INVEST_LEVELS
        self._return_actions = TG_RETURN_LEVELS

    @property
    def name(self) -> str:
        return f"q_learning(lr={self._lr},e={self._epsilon:.2f})"

    def decide(self, observation) -> float:
        state = self._state_encoder(observation)
        self._last_state = state

        if observation.role == "investor":
            valid = self._invest_actions
        else:
            valid = self._return_actions

        if self._rng.random() < self._epsilon:
            action = self._rng.choice(valid)
        else:
            q_values = self._q[state]
            max_q = max(q_values.get(a, 0.0) for a in valid)
            best = [a for a in valid if q_values.get(a, 0.0) == max_q]
            action = self._rng.choice(best)

        self._last_action = action

        if observation.role == "investor":
            return action * observation.endowment
        else:
            return action * (observation.amount_received or 0.0)


class _TGBanditBrain(BanditBrain):
    """Bandit for Trust Game — handles dual role."""

    def __init__(self, **kwargs):
        all_actions = TG_INVEST_LEVELS + TG_RETURN_LEVELS
        super().__init__(
            action_space=all_actions,
            reward_extractor=_tg_reward_extractor,
            **kwargs,
        )
        self._invest_actions = TG_INVEST_LEVELS
        self._return_actions = TG_RETURN_LEVELS
        self._role_totals: dict[str, dict] = {
            "investor": {a: 0.0 for a in self._invest_actions},
            "trustee": {a: 0.0 for a in self._return_actions},
        }
        self._role_counts: dict[str, dict] = {
            "investor": {a: 0 for a in self._invest_actions},
            "trustee": {a: 0 for a in self._return_actions},
        }
        self._last_role: str | None = None

    @property
    def name(self) -> str:
        return f"bandit(e={self._epsilon:.2f})"

    def decide(self, observation) -> float:
        role = observation.role
        self._last_role = role
        valid = self._invest_actions if role == "investor" else self._return_actions
        totals = self._role_totals[role]
        counts = self._role_counts[role]

        if self._rng.random() < self._epsilon:
            action = self._rng.choice(valid)
        else:
            best_avg = float("-inf")
            best_actions = []
            for a in valid:
                avg = totals[a] / counts[a] if counts[a] > 0 else 0.0
                if avg > best_avg:
                    best_avg = avg
                    best_actions = [a]
                elif avg == best_avg:
                    best_actions.append(a)
            action = self._rng.choice(best_actions)

        self._last_action = action
        if role == "investor":
            return action * observation.endowment
        return action * (observation.amount_received or 0.0)

    def update(self, result) -> None:
        if self._last_action is None or self._last_role is None:
            return
        reward = self._reward_extractor(result)
        self._role_totals[self._last_role][self._last_action] += reward
        self._role_counts[self._last_role][self._last_action] += 1
        self._epsilon = max(self._epsilon_min, self._epsilon * self._epsilon_decay)

    def reset(self) -> None:
        super().reset()
        self._role_totals = {
            "investor": {a: 0.0 for a in self._invest_actions},
            "trustee": {a: 0.0 for a in self._return_actions},
        }
        self._role_counts = {
            "investor": {a: 0 for a in self._invest_actions},
            "trustee": {a: 0 for a in self._return_actions},
        }
        self._last_role = None


def tg_q_learning(
    learning_rate: float = 0.15,
    epsilon: float = 0.3,
    discount: float = 0.95,
    epsilon_decay: float = 0.99,
    epsilon_min: float = 0.01,
    seed: int | None = None,
) -> _TGQLearningBrain:
    return _TGQLearningBrain(
        learning_rate=learning_rate,
        discount=discount,
        epsilon=epsilon,
        epsilon_decay=epsilon_decay,
        epsilon_min=epsilon_min,
        seed=seed,
    )


def tg_bandit(
    epsilon: float = 0.3,
    epsilon_decay: float = 0.99,
    epsilon_min: float = 0.01,
    seed: int | None = None,
) -> _TGBanditBrain:
    return _TGBanditBrain(
        epsilon=epsilon,
        epsilon_decay=epsilon_decay,
        epsilon_min=epsilon_min,
        seed=seed,
    )
