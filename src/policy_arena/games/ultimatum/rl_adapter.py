"""Ultimatum Game RL adapter."""

from __future__ import annotations

from policy_arena.brains.rl.bandit import BanditBrain
from policy_arena.brains.rl.q_learning import QLearningBrain

# Discretize offers as fractions of stake
UG_OFFER_LEVELS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
UG_RESPONSE_ACTIONS = [True, False]  # accept / reject


def _ug_state_encoder(obs) -> str:
    """State = role + context."""
    if obs.role == "proposer":
        if not obs.my_past_offers_made:
            return "proposer_start"
        # State based on last acceptance
        if obs.my_past_responses and obs.my_past_responses[-1]:
            return "proposer_accepted"
        return "proposer_rejected"
    else:
        # Responder: state based on offer level
        if obs.offer is None:
            return "responder_start"
        frac = obs.offer / obs.stake if obs.stake > 0 else 0
        if frac < 0.3:
            return "responder_low_offer"
        elif frac < 0.5:
            return "responder_mid_offer"
        else:
            return "responder_high_offer"


def _ug_reward_extractor(result) -> float:
    return result.payoff


class _UGBanditBrain(BanditBrain):
    """Bandit for Ultimatum — handles dual role with separate action sets."""

    def __init__(self, **kwargs):
        all_actions = UG_OFFER_LEVELS + UG_RESPONSE_ACTIONS
        super().__init__(
            action_space=all_actions,
            reward_extractor=_ug_reward_extractor,
            **kwargs,
        )
        self._proposer_actions = UG_OFFER_LEVELS
        self._responder_actions = UG_RESPONSE_ACTIONS
        # Separate tracking per role
        self._role_totals: dict[str, dict] = {
            "proposer": {a: 0.0 for a in self._proposer_actions},
            "responder": {a: 0.0 for a in self._responder_actions},
        }
        self._role_counts: dict[str, dict] = {
            "proposer": {a: 0 for a in self._proposer_actions},
            "responder": {a: 0 for a in self._responder_actions},
        }
        self._last_role: str | None = None

    @property
    def name(self) -> str:
        return f"bandit(e={self._epsilon:.2f})"

    def decide(self, observation) -> float | bool:
        role = observation.role
        self._last_role = role
        valid_actions = (
            self._proposer_actions if role == "proposer" else self._responder_actions
        )
        totals = self._role_totals[role]
        counts = self._role_counts[role]

        if self._rng.random() < self._epsilon:
            action = self._rng.choice(valid_actions)
        else:
            best_avg = float("-inf")
            best_actions = []
            for a in valid_actions:
                avg = totals[a] / counts[a] if counts[a] > 0 else 0.0
                if avg > best_avg:
                    best_avg = avg
                    best_actions = [a]
                elif avg == best_avg:
                    best_actions.append(a)
            action = self._rng.choice(best_actions)

        self._last_action = action
        if role == "proposer":
            return action * observation.stake
        return action

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
            "proposer": {a: 0.0 for a in self._proposer_actions},
            "responder": {a: 0.0 for a in self._responder_actions},
        }
        self._role_counts = {
            "proposer": {a: 0 for a in self._proposer_actions},
            "responder": {a: 0 for a in self._responder_actions},
        }
        self._last_role = None


class _UGQLearningBrain(QLearningBrain):
    """Q-learning for Ultimatum — handles dual role (proposer/responder).

    Uses two separate Q-tables internally by encoding the role into the state.
    For proposer: action is an offer fraction → converted to amount.
    For responder: action is accept/reject boolean.
    """

    def __init__(self, **kwargs):
        # Action space includes both offer fractions and accept/reject
        # The state encoder distinguishes proposer from responder states
        all_actions = UG_OFFER_LEVELS + UG_RESPONSE_ACTIONS
        super().__init__(
            action_space=all_actions,
            state_encoder=_ug_state_encoder,
            reward_extractor=_ug_reward_extractor,
            **kwargs,
        )
        self._proposer_actions = UG_OFFER_LEVELS
        self._responder_actions = UG_RESPONSE_ACTIONS

    @property
    def name(self) -> str:
        return f"q_learning(lr={self._lr},e={self._epsilon:.2f})"

    def decide(self, observation) -> float | bool:
        state = self._state_encoder(observation)
        self._last_state = state

        if observation.role == "proposer":
            valid_actions = self._proposer_actions
        else:
            valid_actions = self._responder_actions

        if self._rng.random() < self._epsilon:
            action = self._rng.choice(valid_actions)
        else:
            q_values = self._q[state]
            max_q = max(q_values.get(a, 0.0) for a in valid_actions)
            best = [a for a in valid_actions if q_values.get(a, 0.0) == max_q]
            action = self._rng.choice(best)

        self._last_action = action

        if observation.role == "proposer":
            return action * observation.stake
        return action


def ug_bandit(
    epsilon: float = 0.3,
    epsilon_decay: float = 0.99,
    epsilon_min: float = 0.01,
    seed: int | None = None,
) -> _UGBanditBrain:
    return _UGBanditBrain(
        epsilon=epsilon,
        epsilon_decay=epsilon_decay,
        epsilon_min=epsilon_min,
        seed=seed,
    )


def ug_q_learning(
    learning_rate: float = 0.15,
    epsilon: float = 0.3,
    discount: float = 0.95,
    epsilon_decay: float = 0.99,
    epsilon_min: float = 0.01,
    seed: int | None = None,
) -> _UGQLearningBrain:
    return _UGQLearningBrain(
        learning_rate=learning_rate,
        discount=discount,
        epsilon=epsilon,
        epsilon_decay=epsilon_decay,
        epsilon_min=epsilon_min,
        seed=seed,
    )
