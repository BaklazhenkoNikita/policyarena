"""Game registration system for PolicyArena.

Each game defines a ``REGISTRATION`` attribute in its ``__init__.py``.
The registry auto-discovers built-in games by scanning subpackages of
``policy_arena.games``, and third-party games via entry points.
"""

from __future__ import annotations

import importlib
import logging
import pkgutil
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class GameRegistration:
    """Everything the framework needs to know about a game."""

    id: str
    model_class: type
    brain_factories: dict[str, Callable[..., Any]]
    llm_factory: Callable[..., Any] | None = None
    llm_extra_kwargs: frozenset[str] = field(default_factory=frozenset)


class GameRegistry:
    """Central registry that collects game registrations."""

    def __init__(self) -> None:
        self._games: dict[str, GameRegistration] = {}

    def register(self, reg: GameRegistration) -> None:
        if reg.id in self._games:
            logger.warning("Game '%s' already registered, overwriting", reg.id)
        self._games[reg.id] = reg

    def get(self, game_id: str) -> GameRegistration:
        if game_id not in self._games:
            raise KeyError(
                f"Unknown game '{game_id}'. "
                f"Available: {sorted(self._games)}"
            )
        return self._games[game_id]

    def __contains__(self, game_id: str) -> bool:
        return game_id in self._games

    def __iter__(self):
        return iter(self._games)

    def items(self):
        return self._games.items()

    def keys(self):
        return self._games.keys()

    def values(self):
        return self._games.values()

    def __len__(self) -> int:
        return len(self._games)

    @property
    def model_classes(self) -> dict[str, type]:
        return {gid: g.model_class for gid, g in self._games.items()}

    @property
    def brain_factories(self) -> dict[str, dict[str, Callable[..., Any]]]:
        return {gid: dict(g.brain_factories) for gid, g in self._games.items()}


# ---------------------------------------------------------------------------
# Singleton registry
# ---------------------------------------------------------------------------

_registry: GameRegistry | None = None


def get_registry() -> GameRegistry:
    """Return the global game registry, discovering games on first call."""
    global _registry
    if _registry is None:
        _registry = GameRegistry()
        _discover_builtin_games(_registry)
        _discover_entrypoint_games(_registry)
    return _registry


def _discover_builtin_games(registry: GameRegistry) -> None:
    """Import all subpackages of ``policy_arena.games`` and collect registrations."""
    import policy_arena.games as games_pkg

    for importer, modname, ispkg in pkgutil.iter_modules(
        games_pkg.__path__, games_pkg.__name__ + "."
    ):
        if not ispkg:
            continue
        try:
            mod = importlib.import_module(modname)
            reg = getattr(mod, "REGISTRATION", None)
            if isinstance(reg, GameRegistration):
                registry.register(reg)
        except Exception:
            logger.exception("Failed to load game module %s", modname)


def _discover_entrypoint_games(registry: GameRegistry) -> None:
    """Discover third-party games registered via entry points.

    Skips games already found by builtin discovery to avoid duplicate warnings.
    """
    try:
        from importlib.metadata import entry_points

        eps = entry_points(group="policy_arena.games")
        for ep in eps:
            if ep.name in registry:
                continue  # Already discovered as a builtin game
            try:
                mod = ep.load()
                reg = getattr(mod, "REGISTRATION", None)
                if isinstance(reg, GameRegistration):
                    registry.register(reg)
            except Exception:
                logger.exception("Failed to load game entry point %s", ep.name)
    except Exception:
        pass
