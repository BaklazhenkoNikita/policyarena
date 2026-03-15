# PolicyArena

A simulation engine for game-theoretic agent research. Pit rule-based strategies, reinforcement learning, and LLM-powered agents against each other — same game, same seed, same metrics.

<p align="center">
  <img src="images/schelling-segregation.png" alt="Schelling Segregation on policyarena.dev" width="100%">
</p>

All built-in games are playable at [policyarena.dev](https://www.policyarena.dev/). New games added to the repo appear there automatically.

## Quick links

- [Getting Started](getting-started.md) — install, run your first simulation
- [Architecture](architecture.md) — how the engine works under the hood
- [Configuration](configuration.md) — YAML config reference
- [Built-in Games](games.md) — all available games
- [API Reference](reference/index.md) — auto-generated from source

## Install

```bash
pip install policy-arena          # core (rule-based + RL)
pip install policy-arena[llm]     # + LLM agents
pip install policy-arena[all]     # everything
```

## Run

```python
import policy_arena as pa

results = pa.run(pa.get_scenario_path("pd_rl_vs_rulebased"))
print(results.model_metrics.tail())
```

```bash
policy-arena run --example pd_rl_vs_rulebased --no-save
```
