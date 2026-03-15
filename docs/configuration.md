# Configuration

PolicyArena experiments are defined in YAML config files, validated with Pydantic at load time.

## YAML Schema

```yaml
name: "Experiment name"           # optional, default: "Unnamed Scenario"
game: prisoners_dilemma           # required — game ID from the registry
rounds: 200                       # optional, default: 100
seed: 42                          # optional — makes the run reproducible
agents:                           # required — at least one agent group
  - name: tft                     # label prefix for this group
    strategy: tit_for_tat         # brain factory key (see `policy-arena info <game>`)
    count: 3                      # optional, default: 1
    parameters:                   # optional — passed to the brain factory
      learning_rate: 0.15
      epsilon: 0.2
game_params:                      # optional — passed to the Mesa model constructor
  payoff_matrix:
    cc: [3, 3]
    cd: [0, 5]
    dc: [5, 0]
    dd: [1, 1]
```

## Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `name` | string | `"Unnamed Scenario"` | Human-readable experiment name |
| `game` | string | required | Game ID from the registry |
| `rounds` | int | `100` | Number of simulation steps |
| `seed` | int \| null | `null` | Random seed for reproducibility |
| `agents` | list | required | Agent group definitions |
| `game_params` | dict | `{}` | Extra parameters passed to the game model |

### Agent Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `name` | string | required | Label prefix for agents in this group |
| `strategy` | string | required | Brain factory key |
| `count` | int | `1` | Number of agents with this strategy |
| `parameters` | dict | `{}` | Parameters passed to the brain factory |

## Example: Prisoner's Dilemma

```yaml
name: "PD — RL vs Rule-Based"
game: prisoners_dilemma
rounds: 200
seed: 42
agents:
  - name: tft
    strategy: tit_for_tat
    count: 3
  - name: always_defect
    strategy: always_defect
    count: 3
  - name: q_learner
    strategy: q_learning
    count: 2
    parameters:
      learning_rate: 0.15
      epsilon: 0.2
game_params:
  payoff_matrix:
    cc: [3, 3]
    cd: [0, 5]
    dc: [5, 0]
    dd: [1, 1]
```

## Example: LLM Agents

```yaml
name: "PD — LLM vs Rule-Based"
game: prisoners_dilemma
rounds: 50
seed: 42
agents:
  - name: claude_greedy
    strategy: llm
    parameters:
      provider: anthropic
      model: claude-sonnet-4-6
      persona: greedy
  - name: tft
    strategy: tit_for_tat
    count: 3
```

## Discovering Available Strategies

```bash
# List all games
policy-arena games

# Show strategies for a specific game
policy-arena info prisoners_dilemma
```

```python
import policy_arena as pa

registry = pa.get_registry()
reg = registry.get("prisoners_dilemma")
print(sorted(reg.brain_factories.keys()))
```

## Overriding at Runtime

Both the Python API and CLI support overriding `seed` and `rounds`:

```python
results = pa.run("config.yaml", seed=123, rounds=500)
```

```bash
policy-arena run config.yaml --seed 123
```

## Validation

Validate a config without running:

```bash
policy-arena validate config.yaml
```

```python
config = pa.load_config("config.yaml")  # raises ConfigValidationError on invalid input
```
