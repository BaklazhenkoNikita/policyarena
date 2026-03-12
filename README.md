# PolicyArena

Cross-paradigm simulation engine for game-theoretic agent research. Run experiments with rule-based, reinforcement learning, and LLM-powered agents in classic game theory scenarios.

**What makes it different:** the [Brain abstraction](src/policy_arena/brains/base.py) lets rule-based, RL, and LLM agents run in the same simulation with the same interface — `decide()`, `update()`, `reset()`. Compare how different paradigms behave on the same game, same seed, same metrics.

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

With uv:

```bash
uv add policy-arena            # core only
uv add policy-arena[llm]       # + LLM support
uv add policy-arena[all]       # everything
```

## Quick Start

### Run a Built-in Example (No Config Needed)

```bash
# List built-in scenarios
policy-arena examples

# Run one instantly
policy-arena run --example pd_rl_vs_rulebased --no-save
```

### Python API

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
# ['battle_of_sexes', 'chicken', 'commons', 'el_farol', 'hawk_dove',
#  'minority_game', 'prisoners_dilemma', 'public_goods', 'schelling',
#  'sir', 'stag_hunt', 'trust_game', 'ultimatum']

# Inspect a game's available strategies
registry = pa.get_registry()
reg = registry.get("prisoners_dilemma")
print(sorted(reg.brain_factories.keys()))
# ['always_cooperate', 'always_defect', 'bandit', 'best_response',
#  'llm', 'pavlov', 'q_learning', 'random', 'tit_for_tat']

# List built-in scenarios
pa.list_scenarios()
# ['battle_of_sexes_coordination', 'chicken_brinkmanship', ...]
```

### CLI

```bash
# List all games and their strategies
policy-arena games

# Show detailed info about a game
policy-arena info prisoners_dilemma

# Run from a YAML config
policy-arena run scenarios/pd_rl_vs_rulebased.yaml

# Run with overrides
policy-arena run scenarios/pd_rl_vs_rulebased.yaml --seed 42 --no-save

# Run a built-in example (no file needed)
policy-arena run --example pd_rl_vs_rulebased

# Validate a config without running
policy-arena validate scenarios/pd_rl_vs_rulebased.yaml

# Export results as JSON and YAML
policy-arena run scenarios/pd_rl_vs_rulebased.yaml --export-json --export-yaml
```

### YAML Config

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
    type: rl
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

13 built-in scenarios are included — one for each game. See them with `policy-arena examples`.

## How It Works

Every agent is controlled by a **Brain** — the same interface regardless of paradigm:

```python
class Brain(ABC):
    def decide(self, observation) -> action   # Choose an action
    def update(self, result) -> None          # Learn from outcome
    def reset(self) -> None                   # Reset for new game
```

A Tit-for-Tat brain is 4 lines. A Q-learning brain maintains a Q-table. An LLM brain makes an API call to Claude/GPT/Gemini. The engine doesn't care — same interface, same metrics, same run loop.

Games are [Mesa 3](https://mesa.readthedocs.io/) models. Each step: agents decide simultaneously, the model resolves outcomes, brains learn. Mesa handles scheduling, topologies, and data collection.

See [ARCHITECTURE.md](ARCHITECTURE.md) for the full design with code examples.

## Games

### Pairwise (Round-Robin)

| Game | Description |
|------|-------------|
| **Prisoner's Dilemma** | Classic cooperation vs defection dilemma |
| **Stag Hunt** | Risky cooperation (stag) vs safe defection (hare) |
| **Hawk-Dove** | Aggression vs sharing over a resource |
| **Chicken** | Anti-coordination — swerve or crash |
| **Battle of the Sexes** | Coordination with conflicting preferences |
| **Trust Game** | Sender sends money (multiplied), receiver returns a share |
| **Ultimatum** | Proposer offers a split, responder accepts or rejects |

### N-Player (Collective)

| Game | Description |
|------|-------------|
| **Public Goods** | Contribute to a shared pool, multiplied and split equally |
| **El Farol Bar** | Attend only if crowd is below threshold |
| **Tragedy of the Commons** | Extract from a shared renewable resource |
| **Minority Game** | Choose between two options — minority wins |

### Spatial / Network

| Game | Description |
|------|-------------|
| **Schelling Segregation** | Agents on a grid relocate based on neighbor similarity |
| **SIR Epidemic** | Disease spread on network with strategic isolation |

All pairwise and collective games support rule-based, RL, and LLM agents. Spatial/network games support rule-based and RL.

## Agent Types

**Rule-based** (`brains/rule_based/`) — Fixed strategies: Tit-for-Tat, Always Cooperate, Always Defect, Pavlov, Random, plus game-specific heuristics. Deterministic, fast, interpretable.

**Reinforcement Learning** (`brains/rl/`) — Tabular Q-learning with epsilon-greedy exploration, best response (tracks opponent frequencies), and multi-armed bandit. Configurable `learning_rate`, `epsilon`, `epsilon_decay`, `discount`, `seed`.

**LLM-powered** (`brains/llm/`) — Language model agents via LangChain. Uses Pydantic schemas with `with_structured_output()` for reliable action parsing. Supports configurable personas, conversation history, batch decisions (one LLM call per round), and fallback actions on failure.

## LLM Setup

> Requires `pip install policy-arena[llm]`

Set API keys as environment variables:

```bash
export ANTHROPIC_API_KEY=sk-ant-...
export OPENAI_API_KEY=sk-...
export GOOGLE_API_KEY=...
```

Or use a `.env` file. For local models, run [Ollama](https://ollama.ai/) and use `provider: ollama` in your config.

Supported: **Anthropic** (Claude), **OpenAI** (GPT), **Google** (Gemini), **Ollama** (local). Optional [Langfuse](https://langfuse.com/) tracing.

## Extending with New Games

Games self-register via the [`GameRegistration`](src/policy_arena/registration.py) system. Create a new package under `policy_arena/games/`:

```python
# policy_arena/games/my_game/__init__.py
from policy_arena.registration import GameRegistration
from .model import MyGameModel
from .brains import StrategyA, StrategyB

REGISTRATION = GameRegistration(
    id="my_game",
    model_class=MyGameModel,
    brain_factories={
        "strategy_a": lambda **_: StrategyA(),
        "strategy_b": lambda **kw: StrategyB(param=kw.get("param", 1.0)),
    },
)
```

The game is auto-discovered on next import — no need to edit any central file. Third-party packages can also register via entry points:

```toml
# In your package's pyproject.toml
[project.entry-points."policy_arena.games"]
my_game = "my_package.games.my_game"
```

See [ARCHITECTURE.md](ARCHITECTURE.md) for the full game package structure and extending guide.

## Built With

- [Mesa 3](https://mesa.readthedocs.io/) — Agent-based modeling (scheduling, topologies, data collection)
- [LangChain](https://python.langchain.com/) — Provider-agnostic LLM integration
- [Pydantic](https://docs.pydantic.dev/) — Config validation and LLM structured output
- [Polars](https://pola.rs/) — Parquet output for results
- [Typer](https://typer.tiangolo.com/) — CLI
- [Langfuse](https://langfuse.com/) — Optional LLM tracing

## Error Handling

All domain errors inherit from `PolicyArenaError` and carry machine-readable `code`, `message`, and `details` fields:

```python
from policy_arena.errors import GameNotFoundError, StrategyNotFoundError

try:
    pa.run(config)
except GameNotFoundError as e:
    print(e.code)     # "GAME_NOT_FOUND"
    print(e.details)  # {"game_id": "...", "available": [...]}
except StrategyNotFoundError as e:
    print(e.code)     # "STRATEGY_NOT_FOUND"
```

Error types: `GameNotFoundError`, `StrategyNotFoundError`, `ConfigValidationError`, `SimulationError`, `LLMProviderError`.

## Development

```bash
git clone https://github.com/BaklazhenkoNikita/policyarena.git
cd policyarena
uv sync --all-extras          # install all optional deps
uv run pre-commit install     # set up ruff check + format hooks
uv run pytest tests/ -x       # 400 tests
uv run ruff check src/ tests/
uv run ruff format --check src/ tests/
```

## Contributing

We use a `develop → main` branching model:

- **`main`** — stable releases only, protected (requires PR from `develop` + approval)
- **`develop`** — integration branch, all feature work merges here first

**Workflow:**

1. Create a feature branch from `develop`: `git checkout -b feat/my-feature develop`
2. Make changes, commit, push
3. Open a PR targeting `develop`
4. After merging to `develop`, a separate PR from `develop → main` is created for releases

Direct PRs to `main` from feature branches are blocked by CI.

## License

MIT
