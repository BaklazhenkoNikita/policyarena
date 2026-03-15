# Contributing to PolicyArena

Thanks for your interest in contributing! Here's how to get started.

## Setup

```bash
git clone https://github.com/BaklazhenkoNikita/policyarena.git
cd policyarena
uv sync --all-extras
uv run pre-commit install
```

## Branching Model

1. Fork the repo
2. Create a feature branch from `main`: `git checkout -b feat/my-feature`
3. Make your changes
4. Run checks: `uv run ruff check src/ tests/ && uv run ruff format --check src/ tests/ && uv run pytest tests/ -x`
5. Push and open a PR targeting `main`

## Code Style

- **Formatter/linter:** [Ruff](https://docs.astral.sh/ruff/) — runs automatically via pre-commit hooks
- **Type checking:** [mypy](https://mypy-lang.org/) with `strict = false`
- **Line length:** 88 characters
- **Quotes:** double
- **Python:** 3.12+

## Tests

```bash
uv run pytest tests/ -x              # run all tests, stop on first failure
uv run pytest tests/ -x --cov=policy_arena --cov-report=term-missing
```

CI enforces 65% coverage minimum.

## Adding a New Game

1. Create `src/policy_arena/games/your_game/` with the standard layout:
   - `__init__.py` — `REGISTRATION` with model class and brain factories
   - `model.py` — Mesa model subclass
   - `agents.py` — Mesa agent subclass
   - `brains.py` — Game-specific rule-based strategies
   - `types.py` — Observation and RoundResult dataclasses
   - `rl_adapter.py` — RL brain factories
   - `llm_adapter.py` — LLM brain factory
2. Add a built-in scenario YAML in `src/policy_arena/scenarios/`
3. Add tests in `tests/test_your_game.py`
4. Add the entry point in `pyproject.toml` under `[project.entry-points."policy_arena.games"]`

The game is auto-discovered — no central registry to update. See the [architecture docs](https://BaklazhenkoNikita.github.io/policyarena/architecture/) for the full design.

## Commit Messages

Keep them short and descriptive. Use imperative mood:

- `add hawk-dove game with RL adapters`
- `fix Q-learning epsilon decay not applied`
- `update LLM provider to support Gemini 3.1`

## Questions?

Open an [issue](https://github.com/BaklazhenkoNikita/policyarena/issues) or start a discussion.
