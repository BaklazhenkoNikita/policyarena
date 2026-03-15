"""Voting & Election Game -- N-player voting under various rules."""

from policy_arena.registration import GameRegistration

from .brains import (
    ContrarianVoter,
    RandomVoter,
    SincereVoter,
    StrategicVoter,
)
from .model import VotingModel


def _lazy_llm(**kw):
    from .llm_adapter import voting_llm

    return voting_llm(**kw)


from .rl_adapter import voting_bandit, voting_q_learning

REGISTRATION = GameRegistration(
    id="voting",
    model_class=VotingModel,
    brain_factories={
        "sincere_voter": lambda **kw: SincereVoter(
            approval_threshold=kw.get("approval_threshold", 25.0)
        ),
        "strategic_voter": lambda **kw: StrategicVoter(
            approval_threshold=kw.get("approval_threshold", 25.0)
        ),
        "random_voter": lambda **_: RandomVoter(),
        "contrarian_voter": lambda **_: ContrarianVoter(),
        "q_learning": lambda **kw: voting_q_learning(**kw),
        "bandit": lambda **kw: voting_bandit(**kw),
        "llm": _lazy_llm,
    },
    llm_factory=_lazy_llm,
    llm_extra_kwargs=frozenset(
        {"n_candidates", "candidate_positions", "voting_rule", "ideal_point"}
    ),
)
