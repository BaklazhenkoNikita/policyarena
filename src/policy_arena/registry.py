"""Central registry for brain factories and model classes.

Games self-register via ``GameRegistration`` in their ``__init__.py``.
This module provides backward-compatible dicts (``MODEL_CLASSES``,
``BRAIN_FACTORIES``) built from the auto-discovered registrations.
"""

from __future__ import annotations

from typing import Any

from policy_arena.registration import get_registry

# ---------------------------------------------------------------------------
# Backward-compatible accessors — these are lazy properties that resolve
# on first access so that import order doesn't matter.
# ---------------------------------------------------------------------------


def _get_model_classes() -> dict[str, type]:
    return get_registry().model_classes


def _get_brain_factories() -> dict[str, dict[str, Any]]:
    return get_registry().brain_factories


class _LazyDict(dict):
    """Dict that populates itself on first access from a factory function."""

    def __init__(self, factory):
        self._factory = factory
        self._loaded = False

    def _ensure_loaded(self):
        if not self._loaded:
            self._loaded = True
            self.update(self._factory())

    def __getitem__(self, key):
        self._ensure_loaded()
        return super().__getitem__(key)

    def __contains__(self, key):
        self._ensure_loaded()
        return super().__contains__(key)

    def __iter__(self):
        self._ensure_loaded()
        return super().__iter__()

    def __len__(self):
        self._ensure_loaded()
        return super().__len__()

    def keys(self):
        self._ensure_loaded()
        return super().keys()

    def values(self):
        self._ensure_loaded()
        return super().values()

    def items(self):
        self._ensure_loaded()
        return super().items()

    def get(self, key, default=None):
        self._ensure_loaded()
        return super().get(key, default)


MODEL_CLASSES: dict[str, type] = _LazyDict(_get_model_classes)
BRAIN_FACTORIES: dict[str, dict[str, Any]] = _LazyDict(_get_brain_factories)

# ---------------------------------------------------------------------------
# LLM / RL metadata — used by build_api_brain_factories()
# ---------------------------------------------------------------------------

LLM_COMMON_KWARGS: frozenset[str] = frozenset(
    {
        "provider",
        "model",
        "temperature",
        "max_history",
        "persona",
        "characteristics",
        "api_key",
        "base_url",
    }
)

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
    return {k: v for k, v in kw.items() if k in allowed and v is not None}


def _make_filtered_factory(factory: Any, allowed: frozenset[str]) -> Any:
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
    registry = get_registry()
    result: dict[str, dict[str, Any]] = {}

    for game_id, reg in registry.items():
        game_factories: dict[str, Any] = {}

        for name, factory in reg.brain_factories.items():
            if name == "q_learning":
                game_factories[name] = _make_filtered_factory(factory, RL_Q_KWARGS)
            elif name == "bandit":
                game_factories[name] = _make_filtered_factory(factory, RL_BANDIT_KWARGS)
            elif name == "llm" and reg.llm_factory is not None:
                allowed = LLM_COMMON_KWARGS | reg.llm_extra_kwargs
                for preset in LLM_PRESET_NAMES:
                    game_factories[preset] = _make_filtered_factory(
                        reg.llm_factory, allowed
                    )
            else:
                game_factories[name] = factory

        result[game_id] = game_factories

    return result
