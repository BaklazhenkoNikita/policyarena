"""PolicyArena — cross-paradigm agent simulation engine.

Run game-theoretic simulations with rule-based, reinforcement learning,
and LLM-powered agents.

Quick start::

    import policy_arena as pa

    # Run from a YAML config
    results = pa.run("scenarios/pd_rl_vs_rulebased.yaml")

    # Access results
    print(results.model_metrics.tail())
    print(results.agent_metrics.tail())
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from policy_arena.brains.base import Brain
from policy_arena.core.engine import Engine, RunResults
from policy_arena.core.scenario import Scenario
from policy_arena.core.types import Action, Observation, RoundResult
from policy_arena.registration import GameRegistration, get_registry

if TYPE_CHECKING:
    from policy_arena.io.schemas import ScenarioConfig

from importlib.metadata import version as _pkg_version

__version__ = _pkg_version("policy-arena")

__all__ = [
    # Core
    "Action",
    "Brain",
    "Engine",
    "GameRegistration",
    "Observation",
    "RoundResult",
    "RunResults",
    "Scenario",
    # High-level API
    "get_registry",
    "get_scenario_path",
    "list_games",
    "list_scenarios",
    "load_config",
    "run",
]


def list_scenarios() -> list[str]:
    """Return sorted list of built-in scenario names."""
    scenarios_dir = Path(__file__).parent / "scenarios"
    return sorted(p.stem for p in scenarios_dir.glob("*.yaml"))


def get_scenario_path(name: str) -> Path:
    """Return the path to a built-in scenario YAML file.

    Parameters
    ----------
    name
        Scenario name (without .yaml extension).
        Use ``list_scenarios()`` to see available names.

    Raises
    ------
    FileNotFoundError
        If no built-in scenario matches the name.
    """
    scenarios_dir = Path(__file__).parent / "scenarios"
    path = scenarios_dir / f"{name}.yaml"
    if not path.exists():
        available = list_scenarios()
        raise FileNotFoundError(
            f"No built-in scenario '{name}'. "
            f"Available: {available}"
        )
    return path


def load_config(path: str | Path) -> ScenarioConfig:
    """Load and validate a YAML scenario config file."""
    from policy_arena.io.config_loader import load_config as _load

    return _load(path)


def list_games() -> list[str]:
    """Return sorted list of available game IDs."""
    return sorted(get_registry().keys())


def run(
    config: str | Path | ScenarioConfig,
    *,
    seed: int | None = None,
    rounds: int | None = None,
) -> RunResults:
    """Run a simulation and return results.

    Parameters
    ----------
    config
        Path to a YAML config file, or a ``ScenarioConfig`` object.
    seed
        Override the random seed from the config.
    rounds
        Override the number of rounds from the config.

    Returns
    -------
    RunResults
        Contains ``model_metrics`` and ``agent_metrics`` DataFrames.
    """
    from policy_arena.io.config_loader import build_scenario, load_config as _load
    from policy_arena.io.schemas import ScenarioConfig as _SC

    if isinstance(config, (str, Path)):
        cfg = _load(config)
    elif isinstance(config, _SC):
        cfg = config
    else:
        raise TypeError(
            f"config must be a path or ScenarioConfig, got {type(config).__name__}"
        )

    updates: dict = {}
    if seed is not None:
        updates["seed"] = seed
    if rounds is not None:
        updates["rounds"] = rounds
    if updates:
        cfg = cfg.model_copy(update=updates)

    scenario = build_scenario(cfg)
    return Engine().run(scenario)
