# PolicyArena

Cross-paradigm simulation engine for game-theoretic agent research. Run experiments with rule-based, reinforcement learning, and LLM-powered agents in classic game theory scenarios.

**What makes it different:** the Brain abstraction layer lets rule-based, RL, and LLM agents run in the same simulation with the same interface. Compare how different paradigms behave on the same game, same seed, same metrics.

## Installation

```bash
pip install policy-arena
```

Or with uv:

```bash
uv add policy-arena
```

## Quick Start

### Python API

```python
import policy_arena as pa

# Run from a YAML config
results = pa.run("scenarios/pd_rl_vs_rulebased.yaml")

# Access results as DataFrames
print(results.model_metrics.tail())
print(results.agent_metrics.tail())

# Override parameters
results = pa.run("scenarios/pd_rl_vs_rulebased.yaml", seed=123, rounds=500)

# List available games
pa.list_games()
# ['battle_of_sexes', 'chicken', 'commons', 'el_farol', 'hawk_dove',
#  'minority_game', 'prisoners_dilemma', 'public_goods', 'schelling',
#  'sir', 'stag_hunt', 'trust_game', 'ultimatum']

# Inspect a game's strategies
registry = pa.get_registry()
reg = registry.get("prisoners_dilemma")
print(sorted(reg.brain_factories.keys()))
# ['always_cooperate', 'always_defect', 'bandit', 'best_response',
#  'llm', 'pavlov', 'q_learning', 'random', 'tit_for_tat']
```

### CLI

```bash
# List available games and strategies
policy-arena games

# Run a simulation
policy-arena run scenarios/pd_rl_vs_rulebased.yaml

# Run with overrides
policy-arena run scenarios/pd_rl_vs_rulebased.yaml --seed 42 --no-save

# Show game details
policy-arena info prisoners_dilemma

# Validate a config without running
policy-arena validate scenarios/pd_rl_vs_rulebased.yaml

# Export results as JSON (frontend-compatible) and YAML
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

## Games

### Pairwise (Round-Robin)

| Game | Description | Strategies |
|------|-------------|------------|
| Prisoner's Dilemma | Classic cooperation vs defection dilemma | TitForTat, AlwaysDefect, AlwaysCooperate, Pavlov, Random, Q-Learning, BestResponse, Bandit, LLM |
| Stag Hunt | Risky cooperation (stag) vs safe defection (hare) | AlwaysStag, AlwaysHare, TitForTat, Pavlov, Random, Q-Learning, Bandit, LLM |
| Hawk-Dove | Aggression vs sharing over a resource | AlwaysHawk, AlwaysDove, TitForTat, Pavlov, Random, Q-Learning, Bandit, LLM |
| Chicken | Anti-coordination — swerve or crash | AlwaysSwerve, AlwaysStay, TitForTat, Pavlov, Random, Q-Learning, Bandit, LLM |
| Battle of the Sexes | Coordination with conflicting preferences | AlwaysOpera, AlwaysFootball, TitForTat, Pavlov, Random, Q-Learning, Bandit, LLM |
| Trust Game | Sender sends money (multiplied), receiver returns a share | Game-specific trust/return strategies, Q-Learning, Bandit, LLM |
| Ultimatum | Proposer offers a split, responder accepts or rejects | FairPlayer, GreedyPlayer, GenerousPlayer, SpitefulPlayer, AdaptivePlayer, Q-Learning, Bandit, LLM |

### N-Player (Collective)

| Game | Description | Strategies |
|------|-------------|------------|
| Public Goods | Contribute to a shared pool, multiplied and split equally | FreeRider, FullContributor, FixedContributor, ConditionalCooperator, Q-Learning, Bandit, LLM |
| El Farol Bar | Attend only if crowd is below threshold | AlwaysAttend, NeverAttend, MovingAverage, Contrarian, TrendFollower, Q-Learning, Bandit, LLM |
| Tragedy of the Commons | Extract from a shared renewable resource | Game-specific extraction strategies, Q-Learning, Bandit, LLM |
| Minority Game | Choose between two options — minority wins | Game-specific strategies, Q-Learning, Bandit, LLM |

### Spatial / Network

| Game | Description | Strategies |
|------|-------------|------------|
| Schelling Segregation | Agents on a grid relocate based on neighbor similarity | Moderate, Tolerant, Intolerant, NeverMove, AlwaysMove, Q-Learning, Bandit |
| SIR Epidemic | Disease spread on network with strategic isolation | NeverIsolate, AlwaysIsolate, ThresholdIsolator, FearfulBrain, SelfAwareBrain, Q-Learning, Bandit |

## Agent Types

**Rule-based** -- Fixed strategies (Tit-for-Tat, Always Defect, etc.) plus game-specific heuristics. Deterministic, fast, interpretable.

**Reinforcement Learning** -- Q-learning, best response, and multi-armed bandit agents. Configurable learning rate, epsilon, discount factor, and decay.

**LLM-powered** -- Language model agents via LangChain. Supports Anthropic (Claude), OpenAI, Google (Gemini), and Ollama (local models). Uses structured output for reliable action parsing, with configurable personas and conversation history.

## LLM Setup

Set API keys as environment variables:

```bash
export ANTHROPIC_API_KEY=sk-...
export OPENAI_API_KEY=sk-...
export GOOGLE_API_KEY=...
```

Or use a `.env` file. For local models, run [Ollama](https://ollama.ai/) and use `provider: ollama` in your config.

## Extending with New Games

Games self-register via the `GameRegistration` system. Create a new package under `policy_arena.games`:

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

Third-party packages can register games via entry points:

```toml
# In your package's pyproject.toml
[project.entry-points."policy_arena.games"]
my_game = "my_package.games.my_game"
```

See [ARCHITECTURE.md](ARCHITECTURE.md) for the full design documentation.

## Development

```bash
git clone <repo-url>
cd policyarena
uv sync
uv run pytest tests/ -x
uv run ruff check src/ tests/
uv run ruff format --check src/ tests/
```

## License

MIT
