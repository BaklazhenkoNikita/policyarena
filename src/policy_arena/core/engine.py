"""Engine — thin orchestration layer on top of Mesa's run loop."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from policy_arena.core.scenario import Scenario


@dataclass
class RunResults:
    """Container for simulation output."""

    model_metrics: pd.DataFrame
    agent_metrics: pd.DataFrame
    extra: dict[str, Any] = field(default_factory=dict)


class Engine:
    """Instantiates a scenario, runs it, extracts results."""

    def run(self, scenario: Scenario) -> RunResults:
        model = scenario.world_class(
            **scenario.world_params,
            rng=scenario.seed,
        )
        model.run_model()

        model_df = model.datacollector.get_model_vars_dataframe()
        agent_df = model.datacollector.get_agent_vars_dataframe()

        return RunResults(
            model_metrics=model_df,
            agent_metrics=agent_df,
            extra={"model": model},
        )
