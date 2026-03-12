"""CLI for PolicyArena — run simulations from YAML configs."""

from __future__ import annotations

import json
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import typer

app = typer.Typer(
    name="policy-arena",
    help="PolicyArena — run game-theoretic simulations with different agent types.",
    no_args_is_help=True,
)


def _step_model(model: Any, game_id: str, n_rounds: int) -> list[dict[str, Any]]:
    """Step through a model and collect StepPayload-compatible dicts per round."""
    from policy_arena.core.extractors import (
        extract_agent_states,
        extract_game_data,
        extract_model_metrics,
    )

    steps: list[dict[str, Any]] = []
    for step_num in range(1, n_rounds + 1):
        model.step()
        metrics = extract_model_metrics(model, game_id)
        agents = extract_agent_states(model, game_id)
        game_data = extract_game_data(model, game_id)
        steps.append({
            "round": step_num,
            "total_rounds": n_rounds,
            "model_metrics": metrics,
            "agents": agents,
            "game_data": game_data,
        })
        if not model.running:
            break
    return steps


def _config_to_brains(config: Any) -> list[dict[str, Any]]:
    """Convert ScenarioConfig agents to BrainSelection-compatible dicts."""
    brains = []
    for agent_cfg in config.agents:
        brains.append({
            "brain_id": agent_cfg.strategy,
            "count": agent_cfg.count,
            "params": agent_cfg.parameters,
            "label_prefix": agent_cfg.name,
        })
    return brains


def _write_run_json(
    steps: list[dict[str, Any]],
    config: Any,
    out_path: Path,
) -> None:
    """Write a RunSnapshot JSON file compatible with the frontend."""
    snapshot = {
        "version": 1,
        "game_id": config.game,
        "brains": _config_to_brains(config),
        "game_params": config.game_params,
        "seed": config.seed,
        "steps": steps,
        "exported_at": datetime.now(UTC).isoformat(),
    }
    with open(out_path, "w") as f:
        json.dump(snapshot, f, default=str)


def _write_config_yaml(config: Any, out_path: Path) -> None:
    """Write the scenario config as a YAML file."""
    import yaml

    data = config.model_dump()
    with open(out_path, "w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)


@app.command()
def run(
    scenario_path: Path = typer.Argument(
        ...,
        help="Path to a YAML scenario config file.",
        exists=True,
        readable=True,
    ),
    seed: int | None = typer.Option(
        None, "--seed", "-s", help="Override the random seed."
    ),
    output_dir: str | None = typer.Option(
        None,
        "--output-dir",
        "-o",
        help="Output directory for results (default: results/).",
    ),
    no_save: bool = typer.Option(
        False, "--no-save", help="Skip saving results to disk."
    ),
    export_json: bool = typer.Option(
        False, "--export-json", help="Export a full run JSON file (frontend-compatible)."
    ),
    export_yaml: bool = typer.Option(
        False, "--export-yaml", help="Export the scenario config as YAML."
    ),
) -> None:
    """Run a simulation from a YAML scenario config."""
    from policy_arena.io.config_loader import build_scenario, load_config

    config = load_config(scenario_path)

    if seed is not None:
        config = config.model_copy(update={"seed": seed})
    if output_dir is not None:
        config = config.model_copy(update={"output_dir": output_dir})

    typer.echo(f"Running: {config.name}")
    typer.echo(f"  Game: {config.game}")
    typer.echo(f"  Agents: {sum(a.count for a in config.agents)}")
    typer.echo(f"  Rounds: {config.rounds}")
    typer.echo(f"  Seed: {config.seed}")
    typer.echo()

    scenario = build_scenario(config)

    # Build model and step through it to capture per-round data
    model = scenario.world_class(
        **scenario.world_params,
        rng=scenario.seed,
    )

    steps = _step_model(model, config.game, config.rounds)

    # Print summary
    if steps:
        last_metrics = steps[-1]["model_metrics"]
        typer.echo("--- Final Metrics ---")
        for k, v in last_metrics.items():
            typer.echo(f"  {k}: {v:.4f}")
        typer.echo()

    score_attr = (
        "happiness"
        if hasattr(next(iter(model.agents)), "happiness")
        else "cumulative_payoff"
    )
    score_label = "Happiness" if score_attr == "happiness" else "Payoff"
    agents = sorted(
        model.agents, key=lambda a: getattr(a, score_attr, 0.0), reverse=True
    )
    typer.echo("--- Agent Results ---")
    typer.echo(f"  {'Label':<25} {'Brain':<22} {score_label:>10}")
    typer.echo("  " + "-" * 60)
    for agent in agents:
        label = getattr(agent, "label", str(agent.unique_id))
        brain = getattr(agent, "brain_name", "unknown")
        score = getattr(agent, score_attr, 0.0)
        typer.echo(f"  {label:<25} {brain:<22} {score:>10.1f}")
    typer.echo()

    if not no_save or export_json or export_yaml:
        import uuid

        run_id = uuid.uuid4().hex[:12]
        out = Path(config.output_dir or output_dir or "results") / run_id
        out.mkdir(parents=True, exist_ok=True)

        saved: list[str] = []

        if not no_save:
            from policy_arena.core.engine import RunResults
            from policy_arena.io.results_writer import write_results

            model_df = model.datacollector.get_model_vars_dataframe()
            agent_df = model.datacollector.get_agent_vars_dataframe()
            results = RunResults(model_metrics=model_df, agent_metrics=agent_df)
            write_results(results, config=config, output_dir=str(out.parent), run_id=run_id)
            saved.append("rounds.parquet, metrics.parquet")

        if export_json:
            json_path = out / f"{config.game}_run.json"
            _write_run_json(steps, config, json_path)
            saved.append(json_path.name)

        if export_yaml:
            yaml_path = out / f"{config.game}_config.yaml"
            shutil.copy2(scenario_path, yaml_path)
            saved.append(yaml_path.name)

        typer.echo(f"Results saved to: {out}")
        for s in saved:
            typer.echo(f"  {s}")
    else:
        typer.echo("Results not saved (--no-save).")


@app.command()
def games() -> None:
    """List available game templates."""
    from policy_arena.io.config_loader import BRAIN_FACTORIES

    typer.echo("Available games:")
    typer.echo()
    for game_id, factories in sorted(BRAIN_FACTORIES.items()):
        strategies = sorted(factories.keys())
        typer.echo(f"  {game_id}")
        typer.echo(f"    Strategies: {', '.join(strategies)}")
        typer.echo()


if __name__ == "__main__":
    app()
