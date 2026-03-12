"""Tests for IO — config loading, results writing/reading."""

import pytest

from policy_arena.brains.rule_based import AlwaysDefect, TitForTat
from policy_arena.core.engine import Engine
from policy_arena.core.scenario import Scenario
from policy_arena.games.prisoners_dilemma.model import PrisonersDilemmaModel
from policy_arena.io.config_loader import load_config, load_scenario
from policy_arena.io.results_reader import list_runs, read_results
from policy_arena.io.results_writer import write_results
from policy_arena.io.schemas import AgentConfig, ScenarioConfig

SAMPLE_YAML = """\
name: "Test PD"
description: "A test scenario"
game: prisoners_dilemma
rounds: 10
seed: 42

agents:
  - name: tft
    strategy: tit_for_tat

  - name: defector
    strategy: always_defect
"""

SAMPLE_YAML_RL = """\
name: "Test PD with RL"
game: prisoners_dilemma
rounds: 20
seed: 99

agents:
  - name: q_agent
    type: rl
    strategy: q_learning
    parameters:
      learning_rate: 0.2
      epsilon: 0.15

  - name: defector
    strategy: always_defect
"""

SAMPLE_YAML_MULTI = """\
name: "Multi-agent"
game: prisoners_dilemma
rounds: 10
seed: 1

agents:
  - name: coop
    strategy: always_cooperate
    count: 3

  - name: defect
    strategy: always_defect
    count: 2
"""


class TestConfigLoading:
    def test_load_basic_config(self, tmp_path):
        cfg_path = tmp_path / "test.yaml"
        cfg_path.write_text(SAMPLE_YAML)

        config = load_config(cfg_path)
        assert config.name == "Test PD"
        assert config.game == "prisoners_dilemma"
        assert config.rounds == 10
        assert config.seed == 42
        assert len(config.agents) == 2

    def test_load_rl_config(self, tmp_path):
        cfg_path = tmp_path / "test_rl.yaml"
        cfg_path.write_text(SAMPLE_YAML_RL)

        config = load_config(cfg_path)
        assert config.agents[0].strategy == "q_learning"
        assert config.agents[0].parameters["learning_rate"] == 0.2

    def test_load_multi_agent(self, tmp_path):
        cfg_path = tmp_path / "test_multi.yaml"
        cfg_path.write_text(SAMPLE_YAML_MULTI)

        config = load_config(cfg_path)
        assert config.agents[0].count == 3
        assert config.agents[1].count == 2

    def test_build_scenario(self, tmp_path):
        cfg_path = tmp_path / "test.yaml"
        cfg_path.write_text(SAMPLE_YAML)

        scenario = load_scenario(cfg_path)
        assert scenario.world_class == PrisonersDilemmaModel
        assert scenario.steps == 10

    def test_run_loaded_scenario(self, tmp_path):
        cfg_path = tmp_path / "test.yaml"
        cfg_path.write_text(SAMPLE_YAML)

        scenario = load_scenario(cfg_path)
        engine = Engine()
        results = engine.run(scenario)
        assert len(results.model_metrics) == 10

    def test_invalid_game(self, tmp_path):
        bad_yaml = (
            "name: test\ngame: nonexistent\nagents:\n  - name: x\n    strategy: y\n"
        )
        cfg_path = tmp_path / "bad.yaml"
        cfg_path.write_text(bad_yaml)

        with pytest.raises(Exception):
            load_config(cfg_path)

    def test_multi_count_produces_correct_agents(self, tmp_path):
        cfg_path = tmp_path / "test_multi.yaml"
        cfg_path.write_text(SAMPLE_YAML_MULTI)

        scenario = load_scenario(cfg_path)
        engine = Engine()
        results = engine.run(scenario)
        model = results.extra["model"]
        assert len(list(model.agents)) == 5


class TestResultsIO:
    def _run_and_get_results(self):
        brains = [TitForTat(), AlwaysDefect()]
        scenario = Scenario(
            world_class=PrisonersDilemmaModel,
            world_params={"brains": brains, "n_rounds": 10},
            steps=10,
            seed=42,
        )
        engine = Engine()
        return engine.run(scenario)

    def test_write_and_read_results(self, tmp_path):
        results = self._run_and_get_results()
        run_dir = write_results(results, output_dir=tmp_path, run_id="test_run")

        assert (run_dir / "rounds.parquet").exists()
        assert (run_dir / "metrics.parquet").exists()
        assert (run_dir / "run_metadata.json").exists()

    def test_read_results(self, tmp_path):
        results = self._run_and_get_results()
        write_results(results, output_dir=tmp_path, run_id="test_run")

        stored = read_results(tmp_path / "test_run")
        assert stored.run_id == "test_run"
        assert stored.rounds is not None
        assert stored.metrics is not None
        assert len(stored.rounds) > 0
        assert len(stored.metrics) > 0

    def test_write_with_config(self, tmp_path):
        results = self._run_and_get_results()
        config = ScenarioConfig(
            name="Test",
            game="prisoners_dilemma",
            agents=[
                AgentConfig(name="tft", strategy="tit_for_tat"),
                AgentConfig(name="ad", strategy="always_defect"),
            ],
            rounds=10,
            seed=42,
        )
        write_results(results, config=config, output_dir=tmp_path, run_id="cfg_run")

        stored = read_results(tmp_path / "cfg_run")
        assert stored.metadata["scenario_config"]["name"] == "Test"

    def test_list_runs(self, tmp_path):
        results = self._run_and_get_results()
        write_results(results, output_dir=tmp_path, run_id="run_a")
        write_results(results, output_dir=tmp_path, run_id="run_b")

        runs = list_runs(tmp_path)
        assert "run_a" in runs
        assert "run_b" in runs

    def test_read_nonexistent(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            read_results(tmp_path / "nonexistent")
