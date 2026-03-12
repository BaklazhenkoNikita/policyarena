"""Load scenario configuration from YAML and build Scenario objects."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from policy_arena.brains.base import Brain
from policy_arena.core.scenario import Scenario
from policy_arena.core.types import Action
from policy_arena.io.schemas import AgentConfig, ScenarioConfig
from policy_arena.registry import BRAIN_FACTORIES, MODEL_CLASSES


def _create_brain(game: str, agent_cfg: AgentConfig) -> Brain:
    """Create a single brain instance from agent config."""
    factories = BRAIN_FACTORIES.get(game)
    if factories is None:
        raise ValueError(f"No brain factories for game '{game}'")

    factory = factories.get(agent_cfg.strategy)
    if factory is None:
        available = sorted(factories.keys())
        raise ValueError(
            f"Unknown strategy '{agent_cfg.strategy}' for game '{game}'. "
            f"Available: {available}"
        )

    return factory(**agent_cfg.parameters)


def _build_world_params(
    config: ScenarioConfig,
    brains: list[Brain],
    labels: list[str],
) -> dict[str, Any]:
    """Build world_params dict for the Mesa model constructor."""
    params: dict[str, Any] = {
        "brains": brains,
        "n_rounds": config.rounds,
        "labels": labels,
    }

    game_params = dict(config.game_params)

    # Handle PD payoff matrix
    if config.game in (
        "prisoners_dilemma",
        "stag_hunt",
        "battle_of_sexes",
        "hawk_dove",
        "chicken",
    ):
        payoff_keys = {"cc_1", "cc_2", "cd_1", "cd_2", "dc_1", "dc_2", "dd_1", "dd_2"}
        if any(k in game_params for k in payoff_keys):
            cc = (game_params.pop("cc_1", 3.0), game_params.pop("cc_2", 3.0))
            cd = (game_params.pop("cd_1", 0.0), game_params.pop("cd_2", 5.0))
            dc = (game_params.pop("dc_1", 5.0), game_params.pop("dc_2", 0.0))
            dd = (game_params.pop("dd_1", 1.0), game_params.pop("dd_2", 1.0))
            params["payoff_matrix"] = {
                (Action.COOPERATE, Action.COOPERATE): cc,
                (Action.COOPERATE, Action.DEFECT): cd,
                (Action.DEFECT, Action.COOPERATE): dc,
                (Action.DEFECT, Action.DEFECT): dd,
            }

    # El Farol: threshold=0 means auto
    if config.game == "el_farol" and game_params.get("threshold") == 0:
        game_params.pop("threshold")

    # Pass remaining game params
    params.update(game_params)
    return params


def load_config(path: str | Path) -> ScenarioConfig:
    """Load and validate a YAML config file."""
    path = Path(path)
    with open(path) as f:
        raw = yaml.safe_load(f)

    return ScenarioConfig(**raw)


def load_scenario(path: str | Path) -> Scenario:
    """Load a YAML config and build a ready-to-run Scenario."""
    config = load_config(path)
    return build_scenario(config)


def build_scenario(config: ScenarioConfig) -> Scenario:
    """Build a Scenario from a validated ScenarioConfig."""
    brains: list[Brain] = []
    labels: list[str] = []

    for agent_cfg in config.agents:
        for i in range(agent_cfg.count):
            params = dict(agent_cfg.parameters)
            if "seed" in params and params["seed"] is not None:
                params["seed"] = params["seed"] + i
            agent_cfg_copy = agent_cfg.model_copy(update={"parameters": params})
            brain = _create_brain(config.game, agent_cfg_copy)
            brains.append(brain)
            label = f"{agent_cfg.name}_{i}" if agent_cfg.count > 1 else agent_cfg.name
            labels.append(label)

    world_params = _build_world_params(config, brains, labels)

    return Scenario(
        world_class=MODEL_CLASSES[config.game],
        world_params=world_params,
        steps=config.rounds,
        seed=config.seed,
    )
