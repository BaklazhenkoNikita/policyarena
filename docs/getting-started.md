# Getting Started

## Installation

```bash
pip install policy-arena
```

This installs the core package (rule-based + RL agents). For LLM-powered agents:

```bash
pip install policy-arena[llm]
```

Or install everything:

```bash
pip install policy-arena[all]
```

With [uv](https://docs.astral.sh/uv/):

```bash
uv add policy-arena            # core only
uv add policy-arena[llm]       # + LLM support
uv add policy-arena[all]       # everything
```

!!! note
    Requires Python 3.12+

## Run a Built-in Example

```bash
# List built-in scenarios
policy-arena examples

# Run one instantly
policy-arena run --example pd_rl_vs_rulebased --no-save
```

## Python API

```python
import policy_arena as pa

# Run a built-in scenario
results = pa.run(pa.get_scenario_path("pd_rl_vs_rulebased"))

# Access results as pandas DataFrames
print(results.model_metrics.tail())
print(results.agent_metrics.tail())

# Override parameters
results = pa.run(pa.get_scenario_path("pd_rl_vs_rulebased"), seed=123, rounds=500)

# List available games
pa.list_games()

# Inspect a game's strategies
registry = pa.get_registry()
reg = registry.get("prisoners_dilemma")
print(sorted(reg.brain_factories.keys()))
```

## Example Output

Running the Prisoner's Dilemma produces two DataFrames:

**Model metrics** (aggregate per round):

```
     cooperation_rate  nash_eq_distance  social_welfare  strategy_entropy
195          0.333333          0.466667        0.600000          0.918296
196          0.366667          0.533333        0.633333          0.948078
197          0.333333          0.466667        0.600000          0.918296
198          0.366667          0.533333        0.633333          0.948078
199          0.333333          0.466667        0.600000          0.918296
```

**Agent metrics** (per agent per round):

```
               cumulative_payoff  round_payoff  cooperation_rate                  brain_name             label
Step  AgentID
200.0 1                   1816.0           9.0               0.4                 tit_for_tat               tft
      2                   2232.0           9.0               0.0               always_defect     always_defect
      3                   1230.0           6.0               1.0            always_cooperate  always_cooperate
      4                   1516.0           8.0               0.6                      pavlov            pavlov
      5                   2190.0           9.0               0.0  q_learning(lr=0.15,e=0.01)         q_learner
      6                   2224.0          13.0               0.0               best_response         best_resp
```

## CLI

```bash
policy-arena games                    # list all games and strategies
policy-arena info prisoners_dilemma   # detailed game info
policy-arena run config.yaml          # run from YAML
policy-arena run config.yaml --seed 42 --no-save
policy-arena validate config.yaml     # validate without running
policy-arena version                  # show version
```

## LLM Setup

!!! note
    Requires `pip install policy-arena[llm]`

Set API keys as environment variables:

```bash
export ANTHROPIC_API_KEY=sk-ant-...
export OPENAI_API_KEY=sk-...
export GOOGLE_API_KEY=...
```

Or use a `.env` file. For local models, run [Ollama](https://ollama.ai/) and use `provider: ollama` in your config.

| Provider | Package | Example Model |
|----------|---------|---------------|
| Anthropic | `langchain-anthropic` | `claude-sonnet-4-6` |
| OpenAI | `langchain-openai` | `gpt-5.4` |
| Google | `langchain-google-genai` | `gemini-3.1-flash` |
| Ollama (local) | `langchain-ollama` | `llama4` |
