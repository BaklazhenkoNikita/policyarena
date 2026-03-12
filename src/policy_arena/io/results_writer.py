"""Write simulation results to Parquet files via polars."""

from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import polars as pl

from policy_arena.core.engine import RunResults
from policy_arena.io.schemas import ScenarioConfig


def write_results(
    results: RunResults,
    config: ScenarioConfig | None = None,
    output_dir: str | Path | None = None,
    run_id: str | None = None,
) -> Path:
    """Write simulation results to Parquet files.

    Creates a directory structure:
        {output_dir}/{run_id}/
            rounds.parquet      — per-agent per-round data
            metrics.parquet     — model-level metrics per round
            run_metadata.json   — config snapshot + timing

    Returns the run directory path.
    """
    run_id = run_id or uuid.uuid4().hex[:12]
    output_dir = Path(output_dir or "results")
    run_dir = output_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    _write_rounds(results, run_id, run_dir)
    _write_metrics(results, run_id, run_dir)
    _write_metadata(config, run_id, run_dir)

    return run_dir


def _pandas_to_polars(df) -> pl.DataFrame:
    """Convert pandas DataFrame to polars without pyarrow.

    Converts each column individually via numpy or Python lists,
    handling mixed-type columns gracefully.
    """
    data: dict[str, list] = {}
    for col in df.columns:
        series = df[col]
        try:
            data[str(col)] = series.to_numpy().tolist()
        except (TypeError, ValueError):
            data[str(col)] = series.tolist()
    return pl.DataFrame(data)


def _write_rounds(results: RunResults, run_id: str, run_dir: Path) -> None:
    """Write per-agent per-round data to rounds.parquet."""
    agent_df = results.agent_metrics

    if agent_df.empty:
        return

    # Mesa agent DataFrame has multi-index (Step, AgentID)
    df = agent_df.reset_index()

    rename_map = {}
    if "Step" in df.columns:
        rename_map["Step"] = "round"
    if "AgentID" in df.columns:
        rename_map["AgentID"] = "agent_id"

    if rename_map:
        df = df.rename(columns=rename_map)

    df["run_id"] = run_id

    pl_df = _pandas_to_polars(df)
    pl_df.write_parquet(run_dir / "rounds.parquet")


def _write_metrics(results: RunResults, run_id: str, run_dir: Path) -> None:
    """Write model-level metrics to metrics.parquet (long format)."""
    model_df = results.model_metrics

    if model_df.empty:
        return

    df = model_df.reset_index()

    if "Step" in df.columns:
        df = df.rename(columns={"Step": "round"})

    metric_cols = [c for c in df.columns if c != "round"]

    rows: list[dict[str, Any]] = []
    for _, row in df.iterrows():
        for metric in metric_cols:
            rows.append(
                {
                    "run_id": run_id,
                    "round": int(row.get("round", 0)),
                    "metric_name": metric,
                    "value": float(row[metric]),
                }
            )

    if rows:
        pl_df = pl.DataFrame(rows)
        pl_df.write_parquet(run_dir / "metrics.parquet")


def _write_metadata(
    config: ScenarioConfig | None,
    run_id: str,
    run_dir: Path,
) -> None:
    """Write run metadata as JSON."""
    metadata: dict[str, Any] = {
        "run_id": run_id,
        "completed_at": datetime.now(UTC).isoformat(),
    }

    if config:
        metadata["scenario_config"] = config.model_dump()

    with open(run_dir / "run_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2, default=str)
