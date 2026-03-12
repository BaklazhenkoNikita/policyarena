"""Quick demo: python -m policy_arena.games.sir"""

from policy_arena.games.sir.brains import (
    NeverIsolate,
    SelfAwareBrain,
    ThresholdIsolator,
)
from policy_arena.games.sir.model import SIRModel

brains = (
    [NeverIsolate() for _ in range(15)]
    + [ThresholdIsolator() for _ in range(10)]
    + [SelfAwareBrain() for _ in range(5)]
)
model = SIRModel(brains=brains, n_rounds=50, rng=42)
model.run_model()
df = model.datacollector.get_model_vars_dataframe()
print(df.to_string())
