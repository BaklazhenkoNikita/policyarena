# API Reference

Auto-generated from source code docstrings.

- [Core](core.md) — `Engine`, `RunResults`, `Scenario`, `Action`, `Observation`, `RoundResult`
- [Brains](brains.md) — `Brain` ABC, RL brains, rule-based strategies
- [Metrics](metrics.md) — cooperation rate, Nash distance, social welfare, entropy, Gini, regret, reciprocity, adaptation speed
- [IO](io.md) — config loading, Pydantic schemas, Parquet reader/writer

## Top-level API

::: policy_arena
    options:
      show_submodules: false
      members:
        - run
        - list_games
        - list_scenarios
        - get_scenario_path
        - load_config
        - get_registry
        - configure_logging
