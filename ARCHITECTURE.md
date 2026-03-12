# PolicyArena Architecture

Deep dive into the design and internals of PolicyArena.

## Overview

PolicyArena is a simulation engine for running game-theoretic experiments with agents controlled by different paradigms — rule-based strategies, reinforcement learning, and large language models — all within the same simulation. It builds on [Mesa 3](https://mesa.readthedocs.io/) for agent-based modeling primitives and [LangChain](https://python.langchain.com/) for provider-agnostic LLM integration.

## Core Concepts

### Brain (Agent Controller)

The central abstraction. Every agent is controlled by a Brain that implements three methods:

```python
class Brain(ABC):
    def decide(self, observation) -> action    # Choose an action
    def update(self, result) -> None           # Learn from outcome
    def reset(self) -> None                    # Reset for a new game
```

This interface is paradigm-agnostic. The engine doesn't know or care whether a brain is a one-line rule, a Q-table lookup, or an LLM API call.

**Rule-based brains** — Deterministic strategies like Tit-for-Tat, Always Cooperate, Pavlov. Fast, interpretable, the baseline for comparison.

**RL brains** — Q-learning, best response, and multi-armed bandit agents. Configurable learning rate, epsilon, discount factor, and decay. Game adapters map each game's state/action space to the RL interface.

**LLM brains** — Language model agents via LangChain. Support Anthropic, OpenAI, Google, and Ollama. Use Pydantic schemas with `with_structured_output()` for reliable action parsing. Features include conversation history, configurable personas, batch decisions (one LLM call per round), and fallback actions on failure.

### World (Mesa Model)

Each game defines a `mesa.Model` subclass that owns the simulation state. Mesa provides agent management (`self.agents`), seeded RNG (`self.random`), step counting (`self.steps`), and data collection (`self.datacollector`).

```python
class PrisonersDilemmaModel(mesa.Model):
    def step(self):
        actions = self.agents.map("decide")      # Collect all decisions
        self._resolve_payoffs(actions)            # Model resolves outcomes
        self.agents.do("update")                  # Brains learn
        self.datacollector.collect(self)           # Record metrics
```

**Topologies vary by game:**
- No space — abstract games (PD, Public Goods, Ultimatum, El Farol)
- Grid — spatial games (Schelling uses `OrthogonalMooreGrid`)
- Network — graph-based games (SIR uses small-world, scale-free, etc.)

### Entity (Mesa Agent)

Each `mesa.Agent` subclass holds a Brain and game-specific state. The agent delegates decisions to its brain and accumulates results.

### Scenario

A complete experiment specification: game ID, agent configurations (brain type + parameters + count), game parameters, number of rounds, and random seed. Defined as YAML files or constructed programmatically.

### Engine

Thin orchestration layer. Takes a Scenario, builds the Mesa model, steps it, and returns `RunResults` containing model-level and agent-level DataFrames from Mesa's `DataCollector`.

## Game Registration

Games self-register via the `GameRegistration` system. Each game package defines a `REGISTRATION` in its `__init__.py`:

```python
# policy_arena/games/prisoners_dilemma/__init__.py
REGISTRATION = GameRegistration(
    id="prisoners_dilemma",
    model_class=PrisonersDilemmaModel,
    brain_factories={
        "tit_for_tat": lambda **_: TitForTat(),
        "q_learning": lambda **kw: pd_q_learning(**kw),
        "llm": lambda **kw: pd_llm(**kw),
        # ...
    },
    llm_factory=pd_llm,
    llm_extra_kwargs=frozenset({"payoff_matrix"}),
)
```

**Discovery happens automatically** on first access to the registry:

1. **Built-in discovery** — scans all subpackages of `policy_arena.games` via `pkgutil.iter_modules`
2. **Entry point discovery** — finds third-party games registered under the `policy_arena.games` entry point group

The registry is a singleton. `get_registry()` returns the same instance after the initial discovery.

## Game Structure

Every game follows the same package layout:

```
games/my_game/
    __init__.py       # REGISTRATION
    model.py          # Mesa Model subclass
    agents.py         # Mesa Agent subclass
    brains.py         # Game-specific rule-based brains
    types.py          # Observation, RoundResult dataclasses
    llm_adapter.py    # LLM brain factory (if LLM-supported)
    rl_adapter.py     # RL brain factories (if RL-supported)
```

### Games by Category

**Pairwise (round-robin tournaments):**
Prisoner's Dilemma, Stag Hunt, Hawk-Dove, Chicken, Battle of the Sexes, Trust Game, Ultimatum

**N-player (collective action):**
Public Goods, El Farol Bar, Tragedy of the Commons, Minority Game

**Spatial/Network:**
Schelling Segregation (grid), SIR Epidemic (network graph)

## Metrics

Mesa's `DataCollector` records metrics each step.

**Core metrics** (available across games):
- **Cooperation Rate** — % cooperative actions per round
- **Nash Equilibrium Distance** — deviation from stage-game Nash equilibrium
- **Social Welfare** — total payoffs as % of theoretical maximum
- **Strategy Entropy** — Shannon entropy over the action distribution
- **Gini Coefficient** — payoff inequality
- **Individual Regret** — best fixed action in hindsight minus actual payoff
- **Reciprocity** — degree of reciprocal cooperation patterns
- **Adaptation Speed** — rounds until strategy stabilizes

**Game-specific metrics** are defined per game (e.g., `segregation_index` for Schelling, `infected_pct` for SIR, `avg_contribution` for Public Goods).

## Data Output

Results are saved as Parquet files via Polars:

- **`rounds.parquet`** — one row per agent per round (agent name, brain type, action, payoff, cumulative payoff)
- **`metrics.parquet`** — one row per metric per round

The `RunResults` object also exposes these as DataFrames for in-memory analysis.

## Configuration

Scenarios are defined in YAML:

```yaml
name: "PD — RL vs Rule-Based"
game: prisoners_dilemma
rounds: 200
seed: 42
agents:
  - name: tft
    strategy: tit_for_tat
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

`game_params` are passed to the Mesa Model constructor. Each game defines which parameters it accepts.

## LLM Integration

LLM brains use LangChain's `ChatModel` abstraction. Supported providers:

| Provider | Package | Example Model |
|----------|---------|---------------|
| Anthropic | `langchain-anthropic` | `claude-haiku-4-5-20251001` |
| OpenAI | `langchain-openai` | `gpt-4o-mini` |
| Google | `langchain-google-genai` | `gemini-2.5-flash` |
| Ollama (local) | `langchain-ollama` | `llama3.1` |

API keys are read from environment variables (`ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `GOOGLE_API_KEY`). Optional [Langfuse](https://langfuse.com/) integration for tracing LLM calls.

## Extending PolicyArena

### Adding a New Game

1. Create a package under `policy_arena/games/your_game/`
2. Implement `model.py` (Mesa Model), `agents.py` (Mesa Agent), `brains.py` (strategies), `types.py` (Observation/RoundResult)
3. Define `REGISTRATION` in `__init__.py`
4. The game is automatically discovered on next registry access

### Third-Party Games

External packages can register games via entry points in `pyproject.toml`:

```toml
[project.entry-points."policy_arena.games"]
my_game = "my_package.games.my_game"
```

The module must export a `REGISTRATION` attribute of type `GameRegistration`.

### Adding a New Brain Type

Subclass `Brain`, implement `decide`/`update`/`reset`, and add a factory function to the relevant game's `brain_factories` dict.

## Design Decisions

| Decision | Rationale |
|----------|-----------|
| Build on Mesa | Solves scheduling, topologies, data collection — don't reinvent solved problems |
| LangChain for LLM | Provider-agnostic; swap Anthropic/OpenAI/Ollama with one line |
| Brain ABC with `decide`/`update`/`reset` | Engine stays paradigm-agnostic; all three paradigms compose into the same run loop |
| Game self-registration | Games are self-contained packages; no central registry file to maintain |
| Entry points for third-party games | Standard Python packaging mechanism; no framework-specific plugin system |
| Parquet via Polars | Columnar, typed, fast analytical queries |
| Plain dataclasses internally, Pydantic at boundaries | Lightweight core; strict validation only at config/API boundaries |
| Reproducibility by default | Everything is seeded; configs are snapshot-able |
