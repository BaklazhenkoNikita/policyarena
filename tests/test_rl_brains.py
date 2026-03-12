"""Tests for RL brains — Q-Learning, Best Response, and Bandit."""

from policy_arena.brains.rule_based import AlwaysCooperate, AlwaysDefect, TitForTat
from policy_arena.core.engine import Engine
from policy_arena.core.scenario import Scenario
from policy_arena.core.types import Action, Observation, RoundResult
from policy_arena.games.el_farol.rl_adapter import ef_bandit, ef_q_learning
from policy_arena.games.prisoners_dilemma.model import PrisonersDilemmaModel
from policy_arena.games.prisoners_dilemma.rl_adapter import (
    pd_bandit,
    pd_best_response,
    pd_q_learning,
)
from policy_arena.games.public_goods.rl_adapter import pg_bandit, pg_q_learning
from policy_arena.games.ultimatum.rl_adapter import ug_bandit, ug_q_learning


def run_pd(brains, n_rounds=50, seed=42):
    scenario = Scenario(
        world_class=PrisonersDilemmaModel,
        world_params={"brains": brains, "n_rounds": n_rounds},
        steps=n_rounds,
        seed=seed,
    )
    engine = Engine()
    results = engine.run(scenario)
    return results, results.extra["model"]


class TestQLearningBrain:
    def test_implements_brain_interface(self):
        brain = pd_q_learning(seed=1)
        assert hasattr(brain, "name")
        assert hasattr(brain, "decide")
        assert hasattr(brain, "update")
        assert hasattr(brain, "reset")

    def test_name(self):
        brain = pd_q_learning(learning_rate=0.2, epsilon=0.05, seed=1)
        assert "q_learning" in brain.name

    def test_exploration(self):
        """With epsilon=1.0, all actions should be random."""
        brain = pd_q_learning(epsilon=1.0, seed=42)
        obs = Observation(round_number=0)
        actions = [brain.decide(obs) for _ in range(100)]
        assert Action.COOPERATE in actions
        assert Action.DEFECT in actions

    def test_exploitation(self):
        """With epsilon=0.0 and trained Q-values, should pick best action."""
        brain = pd_q_learning(epsilon=0.0, seed=42)
        # Manually set Q-values
        brain._q["start"][Action.DEFECT] = 10.0
        brain._q["start"][Action.COOPERATE] = 1.0

        obs = Observation(round_number=0)
        action = brain.decide(obs)
        assert action == Action.DEFECT

    def test_q_update(self):
        brain = pd_q_learning(learning_rate=0.5, epsilon=0.0, seed=1)
        obs = Observation(round_number=0)
        brain.decide(obs)
        brain.update(
            RoundResult(
                action=Action.COOPERATE,
                opponent_action=Action.COOPERATE,
                payoff=3.0,
                round_number=0,
            )
        )
        # Q-value should have been updated (pending queue drained)
        assert any(v > 0 for v in brain._q["start"].values())

    def test_reset_clears_q_table(self):
        brain = pd_q_learning(seed=1)
        obs = Observation(round_number=0)
        brain.decide(obs)
        brain.update(
            RoundResult(
                action=Action.COOPERATE,
                opponent_action=Action.DEFECT,
                payoff=0.0,
                round_number=0,
            )
        )
        brain.reset()
        assert len(brain._q) == 0


class TestBestResponseBrain:
    def test_implements_brain_interface(self):
        brain = pd_best_response()
        assert hasattr(brain, "name")
        assert hasattr(brain, "decide")
        assert hasattr(brain, "update")
        assert hasattr(brain, "reset")

    def test_name(self):
        brain = pd_best_response()
        assert brain.name == "best_response"

    def test_defaults_to_first_action(self):
        """With no observations, should return first action."""
        brain = pd_best_response()
        obs = Observation(round_number=0)
        action = brain.decide(obs)
        assert action == Action.COOPERATE

    def test_learns_to_defect_against_defector(self):
        """Against AllDefect, best response is Defect (payoff 1 vs 0)."""
        brain = pd_best_response()
        obs = Observation(round_number=0)

        # Feed opponent defection history
        for _ in range(20):
            brain.update(
                RoundResult(
                    action=Action.COOPERATE,
                    opponent_action=Action.DEFECT,
                    payoff=0.0,
                    round_number=0,
                )
            )

        action = brain.decide(obs)
        assert action == Action.DEFECT

    def test_learns_to_cooperate_against_cooperator(self):
        """Against AllCoop, best response is Defect (payoff 5 vs 3). Rational!"""
        brain = pd_best_response()
        obs = Observation(round_number=0)

        for _ in range(20):
            brain.update(
                RoundResult(
                    action=Action.COOPERATE,
                    opponent_action=Action.COOPERATE,
                    payoff=3.0,
                    round_number=0,
                )
            )

        action = brain.decide(obs)
        # Against pure cooperator, defecting gives 5 > cooperating gives 3
        assert action == Action.DEFECT

    def test_reset(self):
        brain = pd_best_response()
        brain.update(
            RoundResult(
                action=Action.COOPERATE,
                opponent_action=Action.DEFECT,
                payoff=0.0,
                round_number=0,
            )
        )
        assert brain._total_observations == 1
        brain.reset()
        assert brain._total_observations == 0


class TestQLearningInPD:
    """Integration: Q-learning in actual PD tournament."""

    def test_runs_without_error(self):
        brains = [pd_q_learning(seed=1), AlwaysDefect()]
        results, model = run_pd(brains, n_rounds=20)
        assert len(results.model_metrics) == 20

    def test_q_learning_vs_always_defect(self):
        """Q-learning should learn to defect against AllDefect over many rounds."""
        brains = [
            pd_q_learning(learning_rate=0.3, epsilon=0.1, seed=42),
            AlwaysDefect(),
        ]
        _, model = run_pd(brains, n_rounds=200)

        q_agent = [a for a in model.agents if "q_learning" in a.brain_name][0]
        ad_agent = [a for a in model.agents if a.brain_name == "always_defect"][0]

        # Q-learner should have learned — check last 50 rounds are mostly defect
        history = q_agent._my_history[ad_agent.unique_id]
        last_50 = history[-50:]
        defect_rate = sum(1 for a in last_50 if a == Action.DEFECT) / len(last_50)
        assert defect_rate > 0.7, (
            f"Expected Q-learner to mostly defect, got defect_rate={defect_rate}"
        )

    def test_q_learning_vs_tft(self):
        """Q-learning should learn to cooperate with TFT (mutual cooperation = best outcome)."""
        brains = [
            pd_q_learning(learning_rate=0.2, epsilon=0.1, epsilon_decay=0.99, seed=42),
            TitForTat(),
        ]
        _, model = run_pd(brains, n_rounds=300)

        q_agent = [a for a in model.agents if "q_learning" in a.brain_name][0]
        tft_agent = [a for a in model.agents if a.brain_name == "tit_for_tat"][0]

        # Against TFT, cooperation is learned because defecting triggers retaliation
        history = q_agent._my_history[tft_agent.unique_id]
        last_50 = history[-50:]
        coop_rate = sum(1 for a in last_50 if a == Action.COOPERATE) / len(last_50)
        assert coop_rate > 0.5, (
            f"Expected Q-learner to mostly cooperate with TFT, got coop_rate={coop_rate}"
        )


class TestBestResponseInPD:
    def test_runs_without_error(self):
        brains = [pd_best_response(), AlwaysCooperate()]
        results, model = run_pd(brains, n_rounds=20)
        assert len(results.model_metrics) == 20

    def test_best_response_vs_always_cooperate(self):
        """Best response to AllCoop should be Defect."""
        brains = [pd_best_response(), AlwaysCooperate()]
        _, model = run_pd(brains, n_rounds=50)

        br_agent = [a for a in model.agents if a.brain_name == "best_response"][0]
        ac_agent = [a for a in model.agents if a.brain_name == "always_cooperate"][0]

        history = br_agent._my_history[ac_agent.unique_id]
        # After first round, should defect every time
        last_40 = history[-40:]
        defect_rate = sum(1 for a in last_40 if a == Action.DEFECT) / len(last_40)
        assert defect_rate == 1.0


class TestRLAdaptersExist:
    """Verify all game-specific RL adapters can be instantiated."""

    def test_pd_q_learning(self):
        brain = pd_q_learning(seed=1)
        assert brain is not None

    def test_pd_best_response(self):
        brain = pd_best_response()
        assert brain is not None

    def test_pg_q_learning(self):
        brain = pg_q_learning(seed=1)
        assert brain is not None

    def test_ef_q_learning(self):
        brain = ef_q_learning(seed=1)
        assert brain is not None

    def test_ug_q_learning(self):
        brain = ug_q_learning(seed=1)
        assert brain is not None

    def test_pd_bandit(self):
        brain = pd_bandit(seed=1)
        assert brain is not None

    def test_pg_bandit(self):
        brain = pg_bandit(seed=1)
        assert brain is not None

    def test_ef_bandit(self):
        brain = ef_bandit(seed=1)
        assert brain is not None

    def test_ug_bandit(self):
        brain = ug_bandit(seed=1)
        assert brain is not None


class TestBanditBrain:
    def test_implements_brain_interface(self):
        brain = pd_bandit(seed=1)
        assert hasattr(brain, "name")
        assert hasattr(brain, "decide")
        assert hasattr(brain, "update")
        assert hasattr(brain, "reset")

    def test_name(self):
        brain = pd_bandit(epsilon=0.15, seed=1)
        assert "bandit" in brain.name

    def test_exploration(self):
        """With epsilon=1.0, all actions should be random."""
        brain = pd_bandit(epsilon=1.0, seed=42)
        obs = Observation(round_number=0)
        actions = [brain.decide(obs) for _ in range(100)]
        assert Action.COOPERATE in actions
        assert Action.DEFECT in actions

    def test_exploitation(self):
        """With epsilon=0.0, should pick the action with highest average reward."""
        brain = pd_bandit(epsilon=0.0, seed=42)
        # Feed some rewards: defect gets higher average
        # Simulate decide+update pairs so pending queue is correct
        obs0 = Observation(round_number=0)
        brain.decide(obs0)  # queues an action
        brain._pending_actions[-1] = Action.DEFECT  # force defect
        brain.update(
            RoundResult(
                action=Action.DEFECT,
                opponent_action=Action.COOPERATE,
                payoff=5.0,
                round_number=0,
            )
        )
        obs1 = Observation(round_number=1)
        brain.decide(obs1)
        brain._pending_actions[-1] = Action.COOPERATE  # force cooperate
        brain.update(
            RoundResult(
                action=Action.COOPERATE,
                opponent_action=Action.COOPERATE,
                payoff=3.0,
                round_number=1,
            )
        )

        obs = Observation(round_number=2)
        action = brain.decide(obs)
        assert action == Action.DEFECT

    def test_update_tracks_counts(self):
        brain = pd_bandit(seed=1)
        obs = Observation(round_number=0)
        brain.decide(obs)
        brain.update(
            RoundResult(
                action=Action.COOPERATE,
                opponent_action=Action.COOPERATE,
                payoff=3.0,
                round_number=0,
            )
        )
        total_counts = sum(brain._counts.values())
        assert total_counts == 1

    def test_reset(self):
        brain = pd_bandit(seed=1)
        obs = Observation(round_number=0)
        brain.decide(obs)
        brain.update(
            RoundResult(
                action=Action.COOPERATE,
                opponent_action=Action.DEFECT,
                payoff=0.0,
                round_number=0,
            )
        )
        brain.reset()
        assert all(c == 0 for c in brain._counts.values())
        assert all(t == 0.0 for t in brain._totals.values())
        assert len(brain._pending_actions) == 0


class TestBanditInPD:
    """Integration: Bandit in actual PD tournament."""

    def test_runs_without_error(self):
        brains = [pd_bandit(seed=1), AlwaysDefect()]
        results, model = run_pd(brains, n_rounds=20)
        assert len(results.model_metrics) == 20

    def test_bandit_vs_always_defect(self):
        """Bandit should learn to defect against AlwaysDefect."""
        brains = [pd_bandit(epsilon=0.1, epsilon_decay=0.99, seed=42), AlwaysDefect()]
        _, model = run_pd(brains, n_rounds=200)

        bandit_agent = [a for a in model.agents if "bandit" in a.brain_name][0]
        ad_agent = [a for a in model.agents if a.brain_name == "always_defect"][0]

        history = bandit_agent._my_history[ad_agent.unique_id]
        last_50 = history[-50:]
        defect_rate = sum(1 for a in last_50 if a == Action.DEFECT) / len(last_50)
        assert defect_rate > 0.7, (
            f"Expected bandit to mostly defect, got defect_rate={defect_rate}"
        )

    def test_bandit_vs_always_cooperate(self):
        """Bandit should learn to defect against AlwaysCooperate (higher payoff)."""
        brains = [
            pd_bandit(epsilon=0.1, epsilon_decay=0.99, seed=42),
            AlwaysCooperate(),
        ]
        _, model = run_pd(brains, n_rounds=200)

        bandit_agent = [a for a in model.agents if "bandit" in a.brain_name][0]
        ac_agent = [a for a in model.agents if a.brain_name == "always_cooperate"][0]

        history = bandit_agent._my_history[ac_agent.unique_id]
        last_50 = history[-50:]
        defect_rate = sum(1 for a in last_50 if a == Action.DEFECT) / len(last_50)
        assert defect_rate > 0.7, (
            f"Expected bandit to mostly defect, got defect_rate={defect_rate}"
        )
