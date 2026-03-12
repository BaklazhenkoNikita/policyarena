"""Quick demo: python -m policy_arena.games.schelling"""

from policy_arena.games.schelling.brains import (
    IntolerantBrain,
    ModerateBrain,
    TolerantBrain,
)
from policy_arena.games.schelling.model import SchellingModel

brains = (
    [ModerateBrain() for _ in range(15)]
    + [TolerantBrain() for _ in range(10)]
    + [IntolerantBrain() for _ in range(5)]
)
model = SchellingModel(brains=brains, n_rounds=30, width=10, height=10, rng=42)
model.run_model()
df = model.datacollector.get_model_vars_dataframe()
print(df.to_string())
