# PolicyArena Architecture

Deep dive into the design, internals, and code structure of PolicyArena.

## Overview

PolicyArena is a simulation engine for running game-theoretic experiments with agents controlled by different paradigms — rule-based strategies, reinforcement learning, and large language models — all within the same simulation.

It builds on two foundations:
- [Mesa 3](https://mesa.readthedocs.io/) for agent-based modeling (scheduling, topologies, data collection)
- [LangChain](https://python.langchain.com/) for provider-agnostic LLM integration (Anthropic, OpenAI, Google, Ollama)

## How It Works

A simulation flows through four stages:

```
YAML Config  -->  Scenario  -->  Mesa Model  -->  RunResults
(or Python)       (dataclass)    (step loop)      (DataFrames)
```

1. A YAML config (or `ScenarioConfig` object) specifies the game, agents, parameters, and seed
2. `config_loader.py` resolves brain factories from the registry and builds a `Scenario` dataclass
3. `Engine.run()` instantiates the Mesa model, calls `model.run_model()`, and collects DataFrames
4. Results come back as `RunResults` with `model_metrics` and `agent_metrics`

## Core Abstractions

### Brain (`brains/base.py`)

The central abstraction. Every agent is controlled by a Brain — the engine is completely paradigm-agnostic.

```python
# src/policy_arena/brains/base.py

class Brain(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable identifier for this brain/strategy."""

    @abstractmethod
    def decide(self, observation: Any) -> Any:
        """Choose an action given the current observation."""

    def decide_batch(self, observations: list[Any]) -> list[Any]:
        """Decide for multiple opponents at once.
        Default: calls decide() individually.
        LLM brains override this to make a single LLM call."""
        return [self.decide(obs) for obs in observations]

    @abstractmethod
    def update(self, result: Any) -> None:
        """Learn from the outcome of the last round."""

    @abstractmethod
    def reset(self) -> None:
        """Reset internal state for a new game."""
```

The `decide`/`update` signatures use `Any` because observation and result types are game-specific — each game defines its own dataclasses in `types.py`.

### Three Brain Paradigms

**Rule-based** (`brains/rule_based/`) — Deterministic strategies. Example: Tit-for-Tat is 7 lines of logic:

```python
# src/policy_arena/brains/rule_based/tit_for_tat.py

class TitForTat(Brain):
    def decide(self, observation: Observation) -> Action:
        if not observation.opponent_history:
            return Action.COOPERATE
        return observation.opponent_history[-1]

    def update(self, result: RoundResult) -> None:
        pass  # stateless

    def reset(self) -> None:
        pass
```

**RL** (`brains/rl/`) — Tabular Q-learning with epsilon-greedy exploration, best response, and multi-armed bandit. The Q-learning brain is game-agnostic through pluggable `state_encoder` and `reward_extractor` functions:

```python
# src/policy_arena/brains/rl/q_learning.py

class QLearningBrain(Brain):
    def __init__(
        self,
        action_space: Sequence[Any],
        state_encoder: Callable[[Any], Hashable] | None = None,
        reward_extractor: Callable[[Any], float] | None = None,
        learning_rate: float = 0.1,
        discount: float = 0.95,
        epsilon: float = 0.1,
        epsilon_decay: float = 1.0,
        epsilon_min: float = 0.01,
        seed: int | None = None,
    ): ...

    def decide(self, observation: Any) -> Any:
        state = self._state_encoder(observation)
        if self._rng.random() < self._epsilon:
            return self._rng.choice(self._action_space)  # explore
        # exploit: pick action with highest Q-value
        q_values = self._q[state]
        max_q = max(q_values.values())
        best = [a for a, q in q_values.items() if q == max_q]
        return self._rng.choice(best)
```

Each game provides adapter functions (`rl_adapter.py`) that configure the state/action space for Q-learning and bandit brains.

**LLM** (`brains/llm/`) — Language model agents via LangChain's `BaseChatModel`. Key features:
- **Structured output** — Pydantic schemas via `with_structured_output()` for reliable action parsing
- **Batch decisions** — `decide_batch()` makes a single LLM call for all opponents in a round
- **Conversation history** — configurable sliding window of past messages
- **Personas** — system prompts that shape agent behavior (greedy, cooperative, etc.)
- **Fallback actions** — graceful handling when LLM fails or returns invalid output
- **Provider retry interception** — captures and surfaces LangChain's internal retry errors

Each game provides an `llm_adapter.py` with observation formatters, action schemas, and factory functions.

### World (`mesa.Model` subclass)

Each game defines a model that owns the simulation state. Here's the Prisoner's Dilemma as a concrete example:

```python
# src/policy_arena/games/prisoners_dilemma/model.py

class PrisonersDilemmaModel(mesa.Model):
    def __init__(self, brains, n_rounds=100, payoff_matrix=None, labels=None, **kwargs):
        super().__init__(**kwargs)
        self.n_rounds = n_rounds
        self.payoff_matrix = payoff_matrix or DEFAULT_PAYOFF_MATRIX

        for i, brain in enumerate(brains):
            PDAgent(self, brain=brain, label=labels[i] if labels else None)

        self.datacollector = mesa.DataCollector(
            model_reporters={
                "cooperation_rate": lambda m: compute_cooperation_rate(m),
                "nash_eq_distance": lambda m: compute_nash_distance(m),
                "social_welfare": lambda m: compute_social_welfare(m),
                "strategy_entropy": lambda m: compute_strategy_entropy(m),
            },
            agent_reporters={"cumulative_payoff": "cumulative_payoff", ...},
        )

    def step(self):
        # Phase 1: Collect observations for all matchups (round-robin)
        # Phase 2: Batch decide — agents choose actions simultaneously
        # Phase 3: Resolve payoffs centrally
        # Phase 4: Brains learn from results
        self.datacollector.collect(self)
```

**Topologies vary by game:**
- No space — abstract games (PD, Public Goods, Ultimatum, El Farol, Commons, Minority Game)
- Grid — `OrthogonalMooreGrid` for Schelling segregation
- Network — small-world, scale-free, complete graphs for SIR epidemic

### Entity (`mesa.Agent` subclass)

Each agent holds a Brain and game-specific state. The agent delegates decisions to its brain and records results:

```python
# src/policy_arena/games/prisoners_dilemma/agents.py

class PDAgent(mesa.Agent):
    def __init__(self, model, brain: Brain, label=None):
        super().__init__(model)
        self.brain = brain
        self.cumulative_payoff = 0.0
        self._my_history: dict[int, list[Action]] = {}       # per-opponent
        self._opponent_history: dict[int, list[Action]] = {}  # per-opponent

    def get_observation(self, opponent_id) -> Observation:
        return Observation(
            my_history=self._my_history.get(opponent_id, []),
            opponent_history=self._opponent_history.get(opponent_id, []),
            round_number=self.model.steps,
        )

    def record_result(self, result: RoundResult, opponent_id):
        self._my_history.setdefault(opponent_id, []).append(result.action)
        self.cumulative_payoff += result.payoff
        self.brain.update(result)  # brain learns
```

### Scenario (`core/scenario.py`)

A plain dataclass — the fully resolved specification for a run:

```python
@dataclass
class Scenario:
    world_class: type           # e.g. PrisonersDilemmaModel
    world_params: dict[str, Any]  # brains, n_rounds, payoff_matrix, labels, ...
    steps: int = 100
    seed: int | None = None
```

### Engine (`core/engine.py`)

Thin orchestration — 18 lines of actual logic:

```python
class Engine:
    def run(self, scenario: Scenario) -> RunResults:
        model = scenario.world_class(**scenario.world_params, rng=scenario.seed)
        model.run_model()
        return RunResults(
            model_metrics=model.datacollector.get_model_vars_dataframe(),
            agent_metrics=model.datacollector.get_agent_vars_dataframe(),
            extra={"model": model},
        )
```

### Types (`core/types.py`)

Shared types used across games. Plain dataclasses — no Pydantic overhead in the hot path:

```python
class Action(Enum):
    COOPERATE = "cooperate"
    DEFECT = "defect"

@dataclass(frozen=True)
class Observation:
    my_history: list[Action]
    opponent_history: list[Action]
    round_number: int = 0
    extra: dict[str, Any] = field(default_factory=dict)

@dataclass(frozen=True)
class RoundResult:
    action: Action
    opponent_action: Action
    payoff: float
    round_number: int
```

Note: these are the base types for pairwise matrix games. Games with different interaction structures (Public Goods, Schelling, SIR) define their own observation/result types in their `types.py`.

## Game Registration (`registration.py`)

Games self-register. Each game package defines a `REGISTRATION` in its `__init__.py`:

```python
# src/policy_arena/games/prisoners_dilemma/__init__.py

REGISTRATION = GameRegistration(
    id="prisoners_dilemma",
    model_class=PrisonersDilemmaModel,
    brain_factories={
        "tit_for_tat": lambda **_: TitForTat(),
        "always_defect": lambda **_: AlwaysDefect(),
        "q_learning": lambda **kw: pd_q_learning(**kw),
        "llm": lambda **kw: pd_llm(**kw),
        # ...
    },
    llm_factory=pd_llm,
    llm_extra_kwargs=frozenset({"payoff_matrix"}),
)
```

**Discovery happens automatically** on first access to `get_registry()`:

1. **Built-in discovery** — `pkgutil.iter_modules` scans all subpackages of `policy_arena.games`, imports each, and collects any `REGISTRATION` attribute
2. **Entry point discovery** — `importlib.metadata.entry_points(group="policy_arena.games")` finds third-party games

The registry is a singleton — discovery runs once, on first call to `get_registry()`.

## Game Package Structure

Every game follows the same layout:

```
games/my_game/
    __init__.py       # REGISTRATION — ties everything together
    model.py          # mesa.Model subclass — game rules, step logic
    agents.py         # mesa.Agent subclass — holds brain, tracks state
    brains.py         # Game-specific rule-based strategies
    types.py          # Observation, RoundResult dataclasses
    llm_adapter.py    # LLM brain factory, observation formatters, action schemas
    rl_adapter.py     # RL brain factories (Q-learning, bandit, best response)
```

### 13 Implemented Games

**Pairwise (round-robin tournaments):**
Prisoner's Dilemma, Stag Hunt, Hawk-Dove, Chicken, Battle of the Sexes, Trust Game, Ultimatum

**N-player (collective action):**
Public Goods, El Farol Bar, Tragedy of the Commons, Minority Game

**Spatial/Network:**
Schelling Segregation (grid), SIR Epidemic (network graph)

## Configuration (`io/`)

YAML configs are validated with Pydantic (`io/schemas.py`):

```python
class ScenarioConfig(BaseModel):
    name: str = "Unnamed Scenario"
    game: str                          # e.g. "prisoners_dilemma"
    agents: list[AgentConfig]          # brain type + params + count
    rounds: int = 100
    seed: int | None = None
    game_params: dict[str, Any] = {}   # passed to Mesa model constructor

class AgentConfig(BaseModel):
    name: str                          # label prefix
    strategy: str                      # brain factory key
    count: int = 1
    parameters: dict[str, Any] = {}    # brain-specific params
```

The `config_loader.py` resolves brain factories from the registry, instantiates brains, and builds a `Scenario` dataclass.

## Metrics (`metrics/`)

Mesa's `DataCollector` records metrics each step. Core metrics are implemented as standalone functions:

| Metric | Module | Description |
|--------|--------|-------------|
| Cooperation Rate | `cooperation.py` | % cooperative actions per round |
| Nash Equilibrium Distance | `nash_distance.py` | Deviation from stage-game NE |
| Social Welfare | `social_welfare.py` | Total payoffs as % of theoretical max |
| Strategy Entropy | `entropy.py` | Shannon entropy over action distribution |
| Gini Coefficient | `gini.py` | Payoff inequality across agents |
| Individual Regret | `regret.py` | Best fixed action in hindsight minus actual payoff |
| Reciprocity | `reciprocity.py` | Degree of reciprocal cooperation patterns |
| Adaptation Speed | `adaptation_speed.py` | Rounds until strategy stabilizes |

Games define additional metrics in their model's `DataCollector` setup (e.g., `segregation_index` for Schelling, `infected_pct` for SIR, `avg_contribution` for Public Goods).

## Data Output (`io/`)

Results are saved as Parquet files via Polars:

- **`rounds.parquet`** — one row per agent per round (agent name, brain type, action, payoff, cumulative payoff)
- **`metrics.parquet`** — one row per metric per round

The `RunResults` object exposes these as pandas DataFrames for in-memory analysis.

## LLM Integration

LLM brains use LangChain's `BaseChatModel` abstraction (`brains/llm/llm_brain.py`). The `LLMBrain` class handles:

- Message construction (system prompt + conversation history + current observation)
- Structured output parsing via Pydantic schemas
- Rate limiting with configurable semaphore
- Error recovery with fallback actions
- Tracing via optional Langfuse integration

Supported providers (configured in `llm/provider.py`):

| Provider | Package | Example Model |
|----------|---------|---------------|
| Anthropic | `langchain-anthropic` | `claude-haiku-4-5-20251001` |
| OpenAI | `langchain-openai` | `gpt-4o-mini` |
| Google | `langchain-google-genai` | `gemini-2.5-flash` |
| Ollama (local) | `langchain-ollama` | `llama3.1` |

API keys are read from environment variables (`ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `GOOGLE_API_KEY`).

## Project Structure

```
src/policy_arena/
    __init__.py              # Public API: run(), list_games(), get_registry(), ...
    errors.py                # Structured error hierarchy (GameNotFoundError, etc.)
    registration.py          # GameRegistration, GameRegistry, auto-discovery
    registry.py              # Backward-compat MODEL_CLASSES / BRAIN_FACTORIES dicts
    core/
        engine.py            # Engine.run() — thin orchestration
        scenario.py          # Scenario dataclass
        types.py             # Action, Observation, RoundResult
        extractors.py        # Extract agent/model state as plain dicts
    brains/
        base.py              # Brain ABC
        rule_based/          # TitForTat, AlwaysCooperate, AlwaysDefect, Pavlov, Random
        rl/                  # QLearningBrain, BestResponseBrain, BanditBrain + adapters
        llm/                 # LLMBrain, DualRoleBrain + adapters (lazy-loaded)
    games/                   # 13 game packages, each with model/agents/brains/types
    metrics/                 # cooperation, nash_distance, social_welfare, entropy, ...
    io/                      # YAML config loader, Pydantic schemas, Parquet writer/reader
    cli/                     # Typer CLI (run, games, info, validate, examples, version)
    llm/                     # LLM provider factory (lazy-loaded, requires [llm] extra)
    scenarios/               # 13 built-in YAML scenario configs
```

## Error Handling (`errors.py`)

All domain errors inherit from `PolicyArenaError` and carry structured metadata:

```python
class PolicyArenaError(Exception):
    code: str                    # Machine-readable: "GAME_NOT_FOUND", "STRATEGY_NOT_FOUND", etc.
    message: str                 # Human-readable description
    details: dict[str, Any]      # Structured context (game_id, available strategies, etc.)
```

Error hierarchy:

| Error | Code | Raised When |
|-------|------|-------------|
| `GameNotFoundError` | `GAME_NOT_FOUND` | Game ID not in registry |
| `StrategyNotFoundError` | `STRATEGY_NOT_FOUND` | Strategy not registered for a game |
| `ConfigValidationError` | `CONFIG_VALIDATION_ERROR` | Scenario config fails validation |
| `SimulationError` | `SIMULATION_ERROR` | Simulation fails during execution |
| `LLMProviderError` | `LLM_PROVIDER_ERROR` | LLM provider call fails irrecoverably |
| `LLMNotInstalledError` | `LLM_NOT_INSTALLED` | LLM deps missing (`pip install policy-arena[llm]`) |

## Dependency Architecture

The package uses optional dependency groups to keep the core lightweight:

```
policy-arena           → mesa, numpy, networkx, polars, pydantic, pyyaml, typer
policy-arena[llm]      → + langchain-*, langfuse, python-dotenv
policy-arena[api]      → + fastapi, uvicorn, sse-starlette (planned)
policy-arena[all]      → everything
```

LLM modules are lazy-loaded via `__getattr__` in `brains/llm/__init__.py`, `brains/__init__.py`, and `llm/__init__.py`. Game packages use `_lazy_llm()` wrapper functions to defer LLM adapter imports. This means `import policy_arena` works without LangChain installed — the error only surfaces when you actually try to use an LLM brain.

## Extending PolicyArena

### Adding a New Game

1. Create `src/policy_arena/games/your_game/` with the standard file layout
2. Define `REGISTRATION` in `__init__.py` with model class and brain factories
3. The game is automatically discovered — no need to edit any central file

### Third-Party Games

External packages register via entry points in `pyproject.toml`:

```toml
[project.entry-points."policy_arena.games"]
my_game = "my_package.games.my_game"
```

The target module must export a `REGISTRATION` of type `GameRegistration`.

### Adding a New Brain Type

1. Subclass `Brain` from `policy_arena.brains.base`
2. Implement `name`, `decide`, `update`, `reset`
3. Add a factory function to the relevant game's `brain_factories` dict

## Design Decisions

| Decision | Rationale |
|----------|-----------|
| Build on Mesa 3 | Solves scheduling, topologies, data collection — don't reinvent solved problems |
| LangChain for LLM brains | Provider-agnostic `BaseChatModel`; swap providers with one config change |
| Brain ABC with `decide`/`update`/`reset` | Engine stays paradigm-agnostic; all three brain types compose into the same run loop |
| Game self-registration via `REGISTRATION` | Games are self-contained packages; no central registry file to maintain |
| Entry points for third-party games | Standard Python packaging mechanism; no framework-specific plugin system |
| `decide_batch()` on Brain | Enables single LLM call per round (cost optimization) while rule-based/RL brains just loop |
| Parquet via Polars | Columnar, typed, fast analytical queries out of the box |
| Plain dataclasses for types, Pydantic for config | Lightweight core; strict validation only at I/O boundaries |
| Reproducibility by default | Everything is seeded via Mesa's built-in RNG; configs are snapshot-able |
| Game adapters for RL and LLM | Same `QLearningBrain` / `LLMBrain` class reused across all games; adapters map state/action spaces |
| Optional LLM dependencies | Core installs without LangChain; `[llm]` extra adds provider SDKs — keeps installs fast for non-LLM use cases |
| Structured error hierarchy | All domain errors carry `code` + `details` dict; frontends and API layers can handle errors programmatically |
| Lazy LLM imports | LLM modules use `__getattr__` and deferred imports so the package loads without LLM deps installed |
