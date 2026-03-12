"""Tests for SIR Disease Spread model."""

from policy_arena.games.sir.brains import (
    AlwaysIsolate,
    FearfulBrain,
    NeverIsolate,
    RandomIsolate,
    SelfAwareBrain,
    ThresholdIsolator,
)
from policy_arena.games.sir.model import SIRModel
from policy_arena.games.sir.rl_adapter import sir_bandit, sir_q_learning


def run_sir(brains, n_rounds=30, seed=42, **kwargs):
    model = SIRModel(
        brains=brains,
        n_rounds=n_rounds,
        rng=seed,
        **kwargs,
    )
    model.run_model()
    df = model.datacollector.get_model_vars_dataframe()
    return df, model


class TestSIRBasic:
    def test_runs_without_error(self):
        brains = [NeverIsolate() for _ in range(30)]
        df, _ = run_sir(brains)
        assert len(df) == 30

    def test_compartments_sum_to_one(self):
        brains = [NeverIsolate() for _ in range(30)]
        df, _ = run_sir(brains)
        for _, row in df.iterrows():
            total = row["susceptible_pct"] + row["infected_pct"] + row["recovered_pct"]
            assert abs(total - 1.0) < 1e-10

    def test_infection_spreads(self):
        """With no isolation, infection should spread."""
        brains = [NeverIsolate() for _ in range(30)]
        df, _ = run_sir(brains, n_rounds=50, beta=0.5, initial_infected=3)
        # Peak infection should be > initial infected fraction
        assert df["peak_infection"].iloc[-1] > 3 / 30

    def test_always_isolate_reduces_spread(self):
        """Always-isolate should result in less infection than never-isolate."""
        brains_no = [NeverIsolate() for _ in range(30)]
        df_no, _ = run_sir(brains_no, n_rounds=50, seed=42, beta=0.3)

        brains_yes = [AlwaysIsolate() for _ in range(30)]
        df_yes, _ = run_sir(brains_yes, n_rounds=50, seed=42, beta=0.3)

        assert df_yes["peak_infection"].iloc[-1] <= df_no["peak_infection"].iloc[-1]

    def test_recovery_happens(self):
        """Some agents should recover over time."""
        brains = [NeverIsolate() for _ in range(30)]
        df, _ = run_sir(brains, n_rounds=80, gamma=0.2)
        assert df["recovered_pct"].iloc[-1] > 0

    def test_isolation_rate_with_always_isolate(self):
        brains = [AlwaysIsolate() for _ in range(20)]
        df, _ = run_sir(brains)
        assert all(df["isolation_rate"] == 1.0)

    def test_isolation_rate_with_never_isolate(self):
        brains = [NeverIsolate() for _ in range(20)]
        df, _ = run_sir(brains, vaccine_coverage=0.0)
        assert all(df["isolation_rate"] == 0.0)


class TestSIRBrains:
    def test_threshold_isolator(self):
        brains = [ThresholdIsolator(threshold=0.3) for _ in range(30)]
        df, _ = run_sir(brains)
        assert len(df) == 30

    def test_fearful(self):
        brains = [FearfulBrain(fear_threshold=0.1) for _ in range(30)]
        df, _ = run_sir(brains)
        assert len(df) == 30

    def test_self_aware(self):
        brains = [SelfAwareBrain() for _ in range(30)]
        df, _ = run_sir(brains)
        assert len(df) == 30

    def test_random_isolate(self):
        brains = [RandomIsolate(probability=0.5, seed=i) for i in range(30)]
        df, _ = run_sir(brains)
        assert len(df) == 30


class TestSIRReproducibility:
    def test_same_seed_same_results(self):
        brains1 = [NeverIsolate() for _ in range(20)]
        brains2 = [NeverIsolate() for _ in range(20)]
        df1, _ = run_sir(brains1, seed=123)
        df2, _ = run_sir(brains2, seed=123)
        assert df1["infected_pct"].tolist() == df2["infected_pct"].tolist()


class TestSIRRLAdapters:
    def test_q_learning_adapter(self):
        brain = sir_q_learning(seed=1)
        assert brain is not None
        assert "q_learning" in brain.name

    def test_bandit_adapter(self):
        brain = sir_bandit(seed=1)
        assert brain is not None
        assert "bandit" in brain.name

    def test_q_learning_in_simulation(self):
        brains = [sir_q_learning(seed=i) for i in range(15)] + [
            NeverIsolate() for _ in range(15)
        ]
        df, _ = run_sir(brains, n_rounds=20)
        assert len(df) == 20


class TestSIRNetworkTypes:
    def test_small_world(self):
        brains = [NeverIsolate() for _ in range(20)]
        df, _ = run_sir(brains, n_rounds=10, network_type="small_world")
        assert len(df) == 10

    def test_scale_free(self):
        brains = [NeverIsolate() for _ in range(20)]
        df, _ = run_sir(brains, n_rounds=10, network_type="scale_free")
        assert len(df) == 10
