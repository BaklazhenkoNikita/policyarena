"""Read simulation results from Parquet files."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import polars as pl


@dataclass
class StoredResults:
    """Results loaded from disk."""

    run_id: str
    rounds: pl.DataFrame | None = None
    metrics: pl.DataFrame | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


def read_results(run_dir: str | Path) -> StoredResults:
    """Read simulation results from a run directory."""
    run_dir = Path(run_dir)

    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")

    run_id = run_dir.name

    rounds = None
    rounds_path = run_dir / "rounds.parquet"
    if rounds_path.exists():
        rounds = pl.read_parquet(rounds_path)

    metrics = None
    metrics_path = run_dir / "metrics.parquet"
    if metrics_path.exists():
        metrics = pl.read_parquet(metrics_path)

    metadata: dict[str, Any] = {}
    meta_path = run_dir / "run_metadata.json"
    if meta_path.exists():
        with open(meta_path) as f:
            metadata = json.load(f)

    return StoredResults(
        run_id=run_id,
        rounds=rounds,
        metrics=metrics,
        metadata=metadata,
    )


def list_runs(results_dir: str | Path = "results") -> list[str]:
    """List all run IDs in the results directory."""
    results_dir = Path(results_dir)
    if not results_dir.exists():
        return []
    return sorted(
        d.name
        for d in results_dir.iterdir()
        if d.is_dir() and (d / "run_metadata.json").exists()
    )
