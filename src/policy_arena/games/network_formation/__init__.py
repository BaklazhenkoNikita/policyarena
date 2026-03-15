"""Network Formation — agents form links to build a network."""

from policy_arena.registration import GameRegistration

from .brains import (
    BestResponseLinker,
    FullyConnected,
    Isolationist,
    PopularityBased,
    RandomLinker,
    StarSeeker,
)
from .model import NetworkModel


def _lazy_llm(**kw):
    from .llm_adapter import nf_llm

    return nf_llm(**kw)


from .rl_adapter import nf_bandit, nf_q_learning

REGISTRATION = GameRegistration(
    id="network_formation",
    model_class=NetworkModel,
    brain_factories={
        "fully_connected": lambda **_: FullyConnected(),
        "isolationist": lambda **_: Isolationist(),
        "random_linker": lambda **kw: RandomLinker(
            k=kw.get("k", 2), seed=kw.get("seed"),
        ),
        "popularity_based": lambda **kw: PopularityBased(k=kw.get("k", 2)),
        "best_response_linker": lambda **kw: BestResponseLinker(seed=kw.get("seed")),
        "star_seeker": lambda **_: StarSeeker(),
        "q_learning": lambda **kw: nf_q_learning(**kw),
        "bandit": lambda **kw: nf_bandit(**kw),
        "llm": _lazy_llm,
    },
    llm_factory=_lazy_llm,
    llm_extra_kwargs=frozenset({"n_players", "link_cost", "direct_benefit", "decay_factor", "max_links"}),
)
