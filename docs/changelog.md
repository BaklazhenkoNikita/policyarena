# Changelog

All notable changes to PolicyArena will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/), and this project adheres to [Semantic Versioning](https://semver.org/).

## [0.1.3] - 2026-03-15

### Added
- Documentation site via MkDocs + mkdocstrings with auto-generated API reference
- GitHub Actions workflow for docs deployment to GitHub Pages
- Project URLs in pyproject.toml (homepage, repository, issues, docs)
- CONTRIBUTING.md with contributor guidelines
- CHANGELOG.md
- README screenshots from policyarena.dev (Schelling Segregation, Prisoner's Dilemma)
- "Who is this for" and example output sections in README

### Changed
- Simplified branching model: feature branches merge directly to `main`
- ARCHITECTURE.md moved into `docs/` site, root file redirects
- Updated LLM provider examples to latest models (Claude Sonnet 4.6, GPT-5.4, Gemini 3.1 Flash)
- Removed hardcoded game counts throughout docs

## [0.1.2] - 2026-03-13

### Added
- Built-in scenario for every game
- CLI `version` command

### Fixed
- CI: pin Python version, install LLM extras, skip LLM tests when deps missing
- Author name typo in pyproject.toml

## [0.1.1] - 2026-03-12

### Added
- LICENSE, bundled scenario configs, publish workflow
- README and ARCHITECTURE docs with real code examples
- Optional deps, structured error hierarchy
- CI branch guard (PRs to main from develop only)

## [0.1.0] - 2026-03-12

### Added
- Initial release
- Core simulation engine with Brain abstraction (`decide`, `update`, `reset`)
- Rule-based brains: Tit-for-Tat, Always Cooperate, Always Defect, Pavlov, Random
- RL brains: Q-learning, Best Response, Multi-armed Bandit
- LLM brains via LangChain (Anthropic, OpenAI, Google, Ollama)
- Pairwise games: Prisoner's Dilemma, Stag Hunt, Hawk-Dove, Chicken, Battle of the Sexes, Trust Game, Ultimatum
- N-player games: Public Goods, El Farol Bar, Tragedy of the Commons, Minority Game
- Spatial/network games: Schelling Segregation, SIR Epidemic
- Game self-registration and auto-discovery system
- Third-party game support via entry points
- YAML config loader with Pydantic validation
- CLI: `run`, `games`, `info`, `validate`, `examples`
- Parquet output via Polars
- Metrics: cooperation rate, Nash distance, social welfare, entropy, Gini, regret, reciprocity, adaptation speed
- Optional Langfuse tracing for LLM observability
