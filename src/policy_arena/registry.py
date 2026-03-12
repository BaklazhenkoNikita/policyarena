"""Central registry for brain factories and model classes.

Both ``config_loader`` and ``simulation_manager`` import from here instead
of duplicating ~200 lines of imports and ~400 lines of factory definitions.
"""

from __future__ import annotations

from typing import Any

# ---------------------------------------------------------------------------
# Rule-based brains (shared across multiple games)
# ---------------------------------------------------------------------------
from policy_arena.brains.rule_based import (
    AlwaysCooperate,
    AlwaysDefect,
    Pavlov,
    RandomBrain,
    TitForTat,
)

# ---------------------------------------------------------------------------
# Game-specific brains
# ---------------------------------------------------------------------------
from policy_arena.games.battle_of_sexes.brains import (
    AdaptiveCompromiser,
    Alternator,
    AlwaysA,
    AlwaysB,
    Compromiser,
    MixedStrategy,
    Stubborn,
)

# ---------------------------------------------------------------------------
# LLM adapters
# ---------------------------------------------------------------------------
from policy_arena.games.battle_of_sexes.llm_adapter import bos_llm

# ---------------------------------------------------------------------------
# Model classes
# ---------------------------------------------------------------------------
from policy_arena.games.battle_of_sexes.model import BattleOfSexesModel

# ---------------------------------------------------------------------------
# RL adapters
# ---------------------------------------------------------------------------
from policy_arena.games.battle_of_sexes.rl_adapter import (
    bos_bandit,
    bos_best_response,
    bos_q_learning,
)
from policy_arena.games.chicken.brains import (
    AdaptiveChicken,
    AlwaysStraight,
    AlwaysSwerve,
    Brinksman,
    Cautious,
    Escalator,
)
from policy_arena.games.chicken.llm_adapter import ck_llm
from policy_arena.games.chicken.model import ChickenModel
from policy_arena.games.chicken.rl_adapter import (
    ck_bandit,
    ck_best_response,
    ck_q_learning,
)
from policy_arena.games.commons.brains import (
    Adaptive as TCAdaptive,
)
from policy_arena.games.commons.brains import (
    FixedHarvest,
    Opportunist,
    Restraint,
    Sustainable,
)
from policy_arena.games.commons.brains import (
    Greedy as TCGreedy,
)
from policy_arena.games.commons.llm_adapter import tc_llm
from policy_arena.games.commons.model import CommonsModel
from policy_arena.games.commons.rl_adapter import tc_bandit, tc_q_learning
from policy_arena.games.el_farol.brains import (
    AlwaysAttend,
    ContrarianBrain,
    LastWeekPredictor,
    MovingAveragePredictor,
    NeverAttend,
    RandomAttend,
    ReinforcedAttendance,
    TrendFollower,
)
from policy_arena.games.el_farol.llm_adapter import ef_llm
from policy_arena.games.el_farol.model import ElFarolModel
from policy_arena.games.el_farol.rl_adapter import ef_bandit, ef_q_learning
from policy_arena.games.hawk_dove.brains import (
    AlwaysDove,
    AlwaysHawk,
    Bully,
    GradualHawk,
    Prober,
)
from policy_arena.games.hawk_dove.brains import (
    Retaliator as HDRetaliator,
)
from policy_arena.games.hawk_dove.llm_adapter import hd_llm
from policy_arena.games.hawk_dove.model import HawkDoveModel
from policy_arena.games.hawk_dove.rl_adapter import (
    hd_bandit,
    hd_best_response,
    hd_q_learning,
)
from policy_arena.games.minority_game.brains import (
    AlwaysA as MGAlwaysA,
)
from policy_arena.games.minority_game.brains import (
    AlwaysB as MGAlwaysB,
)
from policy_arena.games.minority_game.brains import (
    Contrarian as MGContrarian,
)
from policy_arena.games.minority_game.brains import (
    MajorityAvoider,
    PatternMatcher,
    RandomChoice,
    StickOrSwitch,
)
from policy_arena.games.minority_game.brains import (
    Reinforced as MGReinforced,
)
from policy_arena.games.minority_game.llm_adapter import mg_llm
from policy_arena.games.minority_game.model import MinorityGameModel
from policy_arena.games.minority_game.rl_adapter import mg_bandit, mg_q_learning
from policy_arena.games.prisoners_dilemma.llm_adapter import pd_llm
from policy_arena.games.prisoners_dilemma.model import PrisonersDilemmaModel
from policy_arena.games.prisoners_dilemma.rl_adapter import (
    pd_bandit,
    pd_best_response,
    pd_q_learning,
)
from policy_arena.games.public_goods.brains import (
    AverageUp,
    ConditionalCooperator,
    FixedContributor,
    FreeRider,
    FullContributor,
)
from policy_arena.games.public_goods.llm_adapter import pg_llm
from policy_arena.games.public_goods.model import PublicGoodsModel
from policy_arena.games.public_goods.rl_adapter import pg_bandit, pg_q_learning
from policy_arena.games.schelling.brains import (
    AlwaysMove,
    IntolerantBrain,
    ModerateBrain,
    NeverMove,
    TolerantBrain,
)
from policy_arena.games.schelling.model import SchellingModel
from policy_arena.games.schelling.rl_adapter import (
    schelling_bandit,
    schelling_q_learning,
)
from policy_arena.games.sir.brains import (
    AlwaysIsolate,
    FearfulBrain,
    NeverIsolate,
    RandomIsolate,
    SelfAwareBrain,
    ThresholdIsolator,
)
from policy_arena.games.sir.model import SIRModel
from policy_arena.games.sir.rl_adapter import sir_bandit, sir_q_learning
from policy_arena.games.stag_hunt.brains import (
    AlwaysHare,
    AlwaysStag,
    CautiousStag,
    MajorityStag,
    OptimisticHare,
    TrustButVerify,
)
from policy_arena.games.stag_hunt.llm_adapter import sh_llm
from policy_arena.games.stag_hunt.model import StagHuntModel
from policy_arena.games.stag_hunt.rl_adapter import (
    sh_bandit,
    sh_best_response,
    sh_q_learning,
)
from policy_arena.games.trust_game.brains import (
    AdaptiveTrust,
    Exploiter,
    FullTrust,
    GradualTrust,
    NoTrust,
    Reciprocator,
)
from policy_arena.games.trust_game.brains import (
    FairPlayer as TGFairPlayer,
)
from policy_arena.games.trust_game.llm_adapter import tg_llm_combined
from policy_arena.games.trust_game.model import TrustGameModel
from policy_arena.games.trust_game.rl_adapter import tg_bandit, tg_q_learning
from policy_arena.games.ultimatum.brains import (
    AdaptivePlayer,
    FairPlayer,
    GenerousPlayer,
    GreedyPlayer,
    SpitefulPlayer,
)
from policy_arena.games.ultimatum.llm_adapter import ug_llm_combined
from policy_arena.games.ultimatum.model import UltimatumModel
from policy_arena.games.ultimatum.rl_adapter import ug_bandit, ug_q_learning

# ---------------------------------------------------------------------------
# MODEL_CLASSES — shared by config_loader and simulation_manager
# ---------------------------------------------------------------------------

MODEL_CLASSES: dict[str, type] = {
    "prisoners_dilemma": PrisonersDilemmaModel,
    "public_goods": PublicGoodsModel,
    "ultimatum": UltimatumModel,
    "el_farol": ElFarolModel,
    "schelling": SchellingModel,
    "sir": SIRModel,
    "stag_hunt": StagHuntModel,
    "battle_of_sexes": BattleOfSexesModel,
    "hawk_dove": HawkDoveModel,
    "chicken": ChickenModel,
    "trust_game": TrustGameModel,
    "commons": CommonsModel,
    "minority_game": MinorityGameModel,
}

# ---------------------------------------------------------------------------
# BRAIN_FACTORIES — base factories shared by both consumers
#
# config_loader uses this directly.
# simulation_manager wraps it with kwarg filtering and LLM presets via
# ``build_api_brain_factories()``.
# ---------------------------------------------------------------------------

BRAIN_FACTORIES: dict[str, dict[str, Any]] = {
    "prisoners_dilemma": {
        "tit_for_tat": lambda **_: TitForTat(),
        "always_defect": lambda **_: AlwaysDefect(),
        "always_cooperate": lambda **_: AlwaysCooperate(),
        "pavlov": lambda **_: Pavlov(),
        "random": lambda **kw: RandomBrain(
            cooperation_probability=kw.get("cooperation_probability", 0.5),
            seed=kw.get("seed"),
        ),
        "q_learning": lambda **kw: pd_q_learning(**kw),
        "best_response": lambda **_: pd_best_response(),
        "bandit": lambda **kw: pd_bandit(**kw),
        "llm": lambda **kw: pd_llm(**kw),
    },
    "public_goods": {
        "free_rider": lambda **_: FreeRider(),
        "full_contributor": lambda **_: FullContributor(),
        "fixed_contributor": lambda **kw: FixedContributor(
            fraction=kw.get("fraction", 0.5)
        ),
        "conditional_cooperator": lambda **_: ConditionalCooperator(),
        "average_up": lambda **kw: AverageUp(uplift=kw.get("uplift", 2.0)),
        "q_learning": lambda **kw: pg_q_learning(**kw),
        "bandit": lambda **kw: pg_bandit(**kw),
        "llm": lambda **kw: pg_llm(**kw),
    },
    "ultimatum": {
        "fair_player": lambda **_: FairPlayer(),
        "greedy_player": lambda **kw: GreedyPlayer(min_offer=kw.get("min_offer", 1.0)),
        "generous_player": lambda **_: GenerousPlayer(),
        "spiteful_player": lambda **_: SpitefulPlayer(),
        "adaptive_player": lambda **_: AdaptivePlayer(),
        "q_learning": lambda **kw: ug_q_learning(**kw),
        "bandit": lambda **kw: ug_bandit(**kw),
        "llm": lambda **kw: ug_llm_combined(**kw),
    },
    "el_farol": {
        "always_attend": lambda **_: AlwaysAttend(),
        "never_attend": lambda **_: NeverAttend(),
        "random_attend": lambda **kw: RandomAttend(
            probability=kw.get("probability", 0.5),
            seed=kw.get("seed"),
        ),
        "last_week": lambda **_: LastWeekPredictor(),
        "moving_average": lambda **kw: MovingAveragePredictor(
            window=kw.get("window", 4)
        ),
        "contrarian": lambda **_: ContrarianBrain(),
        "trend_follower": lambda **_: TrendFollower(),
        "reinforced": lambda **kw: ReinforcedAttendance(
            delta=kw.get("delta", 0.05),
            seed=kw.get("seed"),
        ),
        "q_learning": lambda **kw: ef_q_learning(**kw),
        "bandit": lambda **kw: ef_bandit(**kw),
        "llm": lambda **kw: ef_llm(**kw),
    },
    "schelling": {
        "moderate": lambda **kw: ModerateBrain(threshold=kw.get("threshold", 0.375)),
        "tolerant": lambda **kw: TolerantBrain(threshold=kw.get("threshold", 0.25)),
        "intolerant": lambda **kw: IntolerantBrain(
            threshold=kw.get("threshold", 0.625)
        ),
        "never_move": lambda **_: NeverMove(),
        "always_move": lambda **_: AlwaysMove(),
        "q_learning": lambda **kw: schelling_q_learning(**kw),
        "bandit": lambda **kw: schelling_bandit(**kw),
    },
    "sir": {
        "never_isolate": lambda **_: NeverIsolate(),
        "always_isolate": lambda **_: AlwaysIsolate(),
        "threshold_isolator": lambda **kw: ThresholdIsolator(
            threshold=kw.get("threshold", 0.3)
        ),
        "fearful": lambda **kw: FearfulBrain(
            fear_threshold=kw.get("fear_threshold", 0.1)
        ),
        "self_aware": lambda **_: SelfAwareBrain(),
        "random_isolate": lambda **kw: RandomIsolate(
            probability=kw.get("probability", 0.3),
            seed=kw.get("seed"),
        ),
        "q_learning": lambda **kw: sir_q_learning(**kw),
        "bandit": lambda **kw: sir_bandit(**kw),
    },
    "stag_hunt": {
        "always_stag": lambda **_: AlwaysStag(),
        "always_hare": lambda **_: AlwaysHare(),
        "trust_but_verify": lambda **_: TrustButVerify(),
        "cautious_stag": lambda **_: CautiousStag(),
        "majority_stag": lambda **_: MajorityStag(),
        "optimistic_hare": lambda **kw: OptimisticHare(
            probe_interval=kw.get("probe_interval", 5)
        ),
        "random": lambda **kw: RandomBrain(
            cooperation_probability=kw.get("cooperation_probability", 0.5),
            seed=kw.get("seed"),
        ),
        "q_learning": lambda **kw: sh_q_learning(**kw),
        "best_response": lambda **_: sh_best_response(),
        "bandit": lambda **kw: sh_bandit(**kw),
        "llm": lambda **kw: sh_llm(**kw),
    },
    "battle_of_sexes": {
        "always_a": lambda **_: AlwaysA(),
        "always_b": lambda **_: AlwaysB(),
        "alternator": lambda **_: Alternator(),
        "compromiser": lambda **_: Compromiser(),
        "stubborn": lambda **_: Stubborn(),
        "adaptive_compromiser": lambda **_: AdaptiveCompromiser(),
        "mixed_strategy": lambda **kw: MixedStrategy(
            p_a=kw.get("p_a", 0.6),
            seed=kw.get("seed"),
        ),
        "random": lambda **kw: RandomBrain(
            cooperation_probability=kw.get("cooperation_probability", 0.5),
            seed=kw.get("seed"),
        ),
        "q_learning": lambda **kw: bos_q_learning(**kw),
        "best_response": lambda **_: bos_best_response(),
        "bandit": lambda **kw: bos_bandit(**kw),
        "llm": lambda **kw: bos_llm(**kw),
    },
    "hawk_dove": {
        "always_dove": lambda **_: AlwaysDove(),
        "always_hawk": lambda **_: AlwaysHawk(),
        "retaliator": lambda **_: HDRetaliator(),
        "bully": lambda **_: Bully(),
        "prober": lambda **kw: Prober(probe_interval=kw.get("probe_interval", 5)),
        "gradual_hawk": lambda **_: GradualHawk(),
        "random": lambda **kw: RandomBrain(
            cooperation_probability=kw.get("cooperation_probability", 0.5),
            seed=kw.get("seed"),
        ),
        "q_learning": lambda **kw: hd_q_learning(**kw),
        "best_response": lambda **_: hd_best_response(),
        "bandit": lambda **kw: hd_bandit(**kw),
        "llm": lambda **kw: hd_llm(**kw),
    },
    "chicken": {
        "always_swerve": lambda **_: AlwaysSwerve(),
        "always_straight": lambda **_: AlwaysStraight(),
        "cautious": lambda **_: Cautious(),
        "brinksman": lambda **_: Brinksman(),
        "escalator": lambda **_: Escalator(),
        "adaptive_chicken": lambda **_: AdaptiveChicken(),
        "random": lambda **kw: RandomBrain(
            cooperation_probability=kw.get("cooperation_probability", 0.5),
            seed=kw.get("seed"),
        ),
        "q_learning": lambda **kw: ck_q_learning(**kw),
        "best_response": lambda **_: ck_best_response(),
        "bandit": lambda **kw: ck_bandit(**kw),
        "llm": lambda **kw: ck_llm(**kw),
    },
    "trust_game": {
        "full_trust": lambda **_: FullTrust(),
        "no_trust": lambda **_: NoTrust(),
        "fair_player": lambda **_: TGFairPlayer(),
        "exploiter": lambda **_: Exploiter(),
        "gradual_trust": lambda **_: GradualTrust(),
        "reciprocator": lambda **_: Reciprocator(),
        "adaptive_trust": lambda **_: AdaptiveTrust(),
        "q_learning": lambda **kw: tg_q_learning(**kw),
        "bandit": lambda **kw: tg_bandit(**kw),
        "llm": lambda **kw: tg_llm_combined(**kw),
    },
    "commons": {
        "greedy": lambda **_: TCGreedy(),
        "sustainable": lambda **_: Sustainable(),
        "fixed_harvest": lambda **kw: FixedHarvest(amount=kw.get("amount", 5.0)),
        "adaptive": lambda **kw: TCAdaptive(base_fraction=kw.get("base_fraction", 0.5)),
        "restraint": lambda **_: Restraint(),
        "opportunist": lambda **kw: Opportunist(uplift=kw.get("uplift", 1.2)),
        "q_learning": lambda **kw: tc_q_learning(**kw),
        "bandit": lambda **kw: tc_bandit(**kw),
        "llm": lambda **kw: tc_llm(**kw),
    },
    "minority_game": {
        "always_a": lambda **_: MGAlwaysA(),
        "always_b": lambda **_: MGAlwaysB(),
        "random_choice": lambda **kw: RandomChoice(
            p_a=kw.get("p_a", 0.5),
            seed=kw.get("seed"),
        ),
        "contrarian": lambda **_: MGContrarian(),
        "majority_avoider": lambda **_: MajorityAvoider(),
        "stick_or_switch": lambda **kw: StickOrSwitch(seed=kw.get("seed")),
        "pattern_matcher": lambda **kw: PatternMatcher(
            memory=kw.get("memory", 3),
            seed=kw.get("seed"),
        ),
        "reinforced": lambda **kw: MGReinforced(
            delta=kw.get("delta", 0.05),
            seed=kw.get("seed"),
        ),
        "q_learning": lambda **kw: mg_q_learning(**kw),
        "bandit": lambda **kw: mg_bandit(**kw),
        "llm": lambda **kw: mg_llm(**kw),
    },
}

# ---------------------------------------------------------------------------
# LLM factory metadata — used by simulation_manager to build API presets
# ---------------------------------------------------------------------------

# Common kwargs accepted by all LLM factories
LLM_COMMON_KWARGS: frozenset[str] = frozenset(
    {"provider", "model", "temperature", "max_history", "persona", "characteristics", "api_key", "base_url"}
)

# Per-game extra kwargs beyond the common set
LLM_GAME_KWARGS: dict[str, frozenset[str]] = {
    "prisoners_dilemma": frozenset({"payoff_matrix"}),
    "stag_hunt": frozenset({"payoff_matrix"}),
    "battle_of_sexes": frozenset({"payoff_matrix"}),
    "hawk_dove": frozenset({"payoff_matrix"}),
    "chicken": frozenset({"payoff_matrix"}),
    "public_goods": frozenset({"endowment", "multiplier", "n_players"}),
    "el_farol": frozenset(
        {"n_agents", "threshold", "attend_payoff", "overcrowded_payoff", "stay_payoff"}
    ),
    "ultimatum": frozenset({"stake"}),
    "trust_game": frozenset({"endowment", "multiplier"}),
    "commons": frozenset({"max_resource", "growth_rate", "harvest_cap", "n_agents"}),
    "minority_game": frozenset({"n_agents", "win_payoff", "lose_payoff", "tie_payoff"}),
}

# LLM factory function per game (the raw callable, NOT the lambda wrapper)
LLM_FACTORY_FN: dict[str, Any] = {
    "prisoners_dilemma": pd_llm,
    "public_goods": pg_llm,
    "ultimatum": ug_llm_combined,
    "el_farol": ef_llm,
    "stag_hunt": sh_llm,
    "battle_of_sexes": bos_llm,
    "hawk_dove": hd_llm,
    "chicken": ck_llm,
    "trust_game": tg_llm_combined,
    "commons": tc_llm,
    "minority_game": mg_llm,
}

# RL kwarg whitelists
RL_Q_KWARGS: frozenset[str] = frozenset(
    {"learning_rate", "epsilon", "epsilon_decay", "epsilon_min", "seed"}
)
RL_BANDIT_KWARGS: frozenset[str] = frozenset(
    {"epsilon", "epsilon_decay", "epsilon_min", "seed"}
)

LLM_PRESET_NAMES: tuple[str, ...] = (
    "llm_exploiter",
    "llm_adaptive",
    "llm_manipulator",
    "llm_custom",
)


# ---------------------------------------------------------------------------
# Helper to build API-layer brain factories with kwarg filtering + LLM presets
# ---------------------------------------------------------------------------


def _filter_kwargs(kw: dict[str, Any], allowed: frozenset[str]) -> dict[str, Any]:
    """Filter a kwargs dict to only include allowed keys with non-None values."""
    return {k: v for k, v in kw.items() if k in allowed and v is not None}


def _make_filtered_factory(
    factory: Any, allowed: frozenset[str]
) -> Any:
    """Create a lambda wrapper that filters kwargs before calling factory."""
    def wrapped(**kw: Any) -> Any:
        return factory(**_filter_kwargs(kw, allowed))
    return wrapped


def build_api_brain_factories() -> dict[str, dict[str, Any]]:
    """Build BRAIN_FACTORIES for the API layer (simulation_manager).

    Takes the base factories and:
    1. Wraps RL entries (q_learning, bandit) with kwarg filtering
    2. Replaces ``"llm"`` entries with four LLM presets (exploiter, adaptive,
       manipulator, custom), each with kwarg filtering
    """
    result: dict[str, dict[str, Any]] = {}

    for game, factories in BRAIN_FACTORIES.items():
        game_factories: dict[str, Any] = {}

        for name, factory in factories.items():
            if name == "q_learning":
                game_factories[name] = _make_filtered_factory(factory, RL_Q_KWARGS)
            elif name == "bandit":
                game_factories[name] = _make_filtered_factory(factory, RL_BANDIT_KWARGS)
            elif name == "llm":
                # Replace single "llm" entry with four presets
                allowed = LLM_COMMON_KWARGS | LLM_GAME_KWARGS.get(game, frozenset())
                llm_fn = LLM_FACTORY_FN[game]
                for preset in LLM_PRESET_NAMES:
                    game_factories[preset] = _make_filtered_factory(llm_fn, allowed)
            else:
                # Rule-based and game-specific brains — pass through as-is
                game_factories[name] = factory

        result[game] = game_factories

    return result
