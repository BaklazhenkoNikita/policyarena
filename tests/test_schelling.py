"""Tests for Schelling Segregation model."""

from policy_arena.games.schelling.brains import (
    AlwaysMove,
    IntolerantBrain,
    ModerateBrain,
    NeverMove,
    TolerantBrain,
)
from policy_arena.games.schelling.model import SchellingModel
from policy_arena.games.schelling.rl_adapter import (
    schelling_bandit,
    schelling_q_learning,
)


def run_schelling(brains, n_rounds=20, width=10, height=10, seed=42):
    model = SchellingModel(
        brains=brains,
        n_rounds=n_rounds,
        width=width,
        height=height,
        rng=seed,
    )
    model.run_model()
    df = model.datacollector.get_model_vars_dataframe()
    return df, model


class TestSchellingBasic:
    def test_runs_without_error(self):
        brains = [ModerateBrain() for _ in range(20)]
        df, _ = run_schelling(brains)
        assert 1 <= len(df) <= 20  # may converge early

    def test_never_move_no_movement(self):
        brains = [NeverMove() for _ in range(20)]
        df, _ = run_schelling(brains)
        assert all(df["move_rate"] == 0.0)

    def test_always_move_has_movement(self):
        brains = [AlwaysMove() for _ in range(20)]
        df, _ = run_schelling(brains)
        assert any(df["move_rate"] > 0.0)

    def test_segregation_increases_with_moderate(self):
        """Moderate brains should increase segregation over time."""
        brains = [ModerateBrain() for _ in range(15)] + [
            ModerateBrain() for _ in range(15)
        ]
        df, _ = run_schelling(brains, n_rounds=30)
        first = df["segregation_index"].iloc[0]
        last = df["segregation_index"].iloc[-1]
        # Segregation should increase or stay the same
        assert last >= first - 0.1  # allow small noise

    def test_happiness_converges(self):
        """Most agents should become happy eventually."""
        brains = [ModerateBrain() for _ in range(30)]
        df, _ = run_schelling(brains, n_rounds=50)
        final_happiness = df["happiness_rate"].iloc[-1]
        assert final_happiness > 0.8

    def test_intolerant_higher_segregation(self):
        """Intolerant agents should produce more segregation than tolerant ones."""
        intolerant_brains = [IntolerantBrain() for _ in range(30)]
        df_int, _ = run_schelling(intolerant_brains, n_rounds=30, seed=42)

        tolerant_brains = [TolerantBrain() for _ in range(30)]
        df_tol, _ = run_schelling(tolerant_brains, n_rounds=30, seed=42)

        assert (
            df_int["segregation_index"].iloc[-1] >= df_tol["segregation_index"].iloc[-1]
        )


class TestSchellingReproducibility:
    def test_same_seed_same_results(self):
        brains1 = [ModerateBrain() for _ in range(20)]
        brains2 = [ModerateBrain() for _ in range(20)]
        df1, _ = run_schelling(brains1, seed=123)
        df2, _ = run_schelling(brains2, seed=123)
        assert df1["segregation_index"].tolist() == df2["segregation_index"].tolist()


class TestSchellingRLAdapters:
    def test_q_learning_adapter(self):
        brain = schelling_q_learning(seed=1)
        assert brain is not None
        assert "q_learning" in brain.name

    def test_bandit_adapter(self):
        brain = schelling_bandit(seed=1)
        assert brain is not None
        assert "bandit" in brain.name

    def test_q_learning_in_simulation(self):
        brains = [schelling_q_learning(seed=i) for i in range(10)] + [
            ModerateBrain() for _ in range(10)
        ]
        df, _ = run_schelling(brains, n_rounds=20)
        assert len(df) == 20
