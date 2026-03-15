# Built-in Games

Every game listed below has a built-in scenario you can run immediately with `policy-arena run --example <name>`. Use `policy-arena examples` to see all available scenarios.

All built-in games are also playable interactively at [policyarena.dev](https://www.policyarena.dev/).

## Pairwise (Round-Robin)

Agents play every other agent each round in a round-robin tournament.

| Game | ID | Description |
|------|----|-------------|
| **Prisoner's Dilemma** | `prisoners_dilemma` | Classic cooperation vs defection dilemma |
| **Stag Hunt** | `stag_hunt` | Risky cooperation (stag) vs safe defection (hare) |
| **Hawk-Dove** | `hawk_dove` | Aggression vs sharing over a resource |
| **Chicken** | `chicken` | Anti-coordination — swerve or crash |
| **Battle of the Sexes** | `battle_of_sexes` | Coordination with conflicting preferences |
| **Trust Game** | `trust_game` | Sender sends money (multiplied), receiver returns a share |
| **Ultimatum** | `ultimatum` | Proposer offers a split, responder accepts or rejects |

## N-Player (Collective)

All agents participate simultaneously in each round.

| Game | ID | Description |
|------|----|-------------|
| **Public Goods** | `public_goods` | Contribute to a shared pool, multiplied and split equally |
| **Cournot Oligopoly** | `cournot` | Firms choose production quantities; market price falls with total output |
| **El Farol Bar** | `el_farol` | Attend only if crowd is below threshold |
| **Tragedy of the Commons** | `commons` | Extract from a shared renewable resource |
| **Minority Game** | `minority_game` | Choose between two options — minority wins |
| **Voting & Election** | `voting` | N voters elect candidates under plurality, approval, or Borda rules |
| **Sealed-Bid Auction** | `auction` | First-price or second-price (Vickrey) sealed-bid auction with private values |
| **Information Cascade** | `info_cascade` | Sequential binary decisions with private signals — herding dynamics |
| **Lobbying Contest** | `lobbying` | Tullock rent-seeking contest — spend to win a prize, highest spender most likely wins |

## Spatial / Network

Agents interact based on spatial proximity or network topology.

| Game | ID | Description |
|------|----|-------------|
| **Schelling Segregation** | `schelling` | Agents on a grid relocate based on neighbor similarity |
| **SIR Epidemic** | `sir` | Disease spread on network with strategic isolation |

## Agent Paradigm Support

| Category | Rule-based | RL | LLM |
|----------|:---:|:---:|:---:|
| Pairwise | yes | yes | yes |
| N-Player | yes | yes | yes |
| Spatial / Network | yes | yes | — |

## Inspecting a Game

```bash
# Show all strategies available for a game
policy-arena info prisoners_dilemma
```

```python
import policy_arena as pa

reg = pa.get_registry().get("prisoners_dilemma")
print(sorted(reg.brain_factories.keys()))
# ['always_cooperate', 'always_defect', 'bandit', 'best_response',
#  'llm', 'pavlov', 'q_learning', 'random', 'tit_for_tat']
```
