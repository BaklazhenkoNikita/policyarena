# Public Goods Game

## Overview

The Public Goods Game (PGG) is the N-player generalisation of the Prisoner's Dilemma. Each player decides how much of their private endowment to contribute to a shared pool. Contributions are multiplied by a factor r > 1 and the enlarged pool is split equally among all players — regardless of who contributed.

The dilemma: any individual earns more by keeping their endowment than by contributing (free-riding dominates), yet the group as a whole would be far better off if everyone contributed fully. It models the tension underlying taxation, climate agreements, open-source software, and any situation where private costs produce public benefits.

---

## Formal Definition

- **N players**, each endowed with M tokens per round
- Each player i simultaneously chooses contribution **g_i ∈ [0, M]**
- The pool is multiplied by **r** (the "multiplier", typically 1.5–2.0) and split equally
- Individual payoff: **π_i = (M − g_i) + r × Σg_j / N**

The **Marginal Per Capita Return (MPCR)** = r/N. For a genuine social dilemma, the constraint must hold:

```
1/N < MPCR < 1
```

- MPCR < 1: contributing one token returns less than one token to the contributor → free-riding is individually optimal
- MPCR > 1/N: the group's total return per token exceeds 1 → full contribution is socially optimal

**This simulation uses:** endowment = 10, multiplier = 1.6, so for 6 players MPCR ≈ 0.27.

---

## Nash Equilibrium

Under standard assumptions of rationality and self-interest, the **unique Nash Equilibrium is zero contribution** (complete free-riding). Contributing nothing strictly dominates contributing any positive amount: since MPCR < 1, each token kept is always worth more to the individual than the return from contributing it, regardless of what others do.

The social optimum is full contribution by everyone. But this creates a dominant-strategy defection problem identical in structure to the Prisoner's Dilemma, extended to N players.

---

## What Actually Happens in Experiments

Real subjects routinely violate the NE prediction:

- **Round 1 contributions:** 40–60% of endowment across a wide range of experimental designs — well above zero.
- **Decay over rounds:** Contributions reliably decline across repeated rounds, typically reaching 10–25% by the final round in no-punishment conditions.
- **End-game collapse:** Contributions drop sharply in the last round, consistent with backward induction once subjects anticipate termination.
- **Restart effect (Andreoni 1988):** When a game ends and restarts with the same subjects, contributions jump back to near-initial levels before decaying again — showing the decay is not irreversible learning about the payoff structure.

The foundational survey is Ledyard (1995), which documented these regularities across the first generation of PGG experiments.

---

## Player Types

Fischbacher, Gächter & Fehr (2001) used the strategy method to identify three stable player archetypes:

| Type | Share | Behaviour |
|---|---|---|
| **Conditional cooperators** | ~50% | Increase contribution proportionally with group average — cooperate if others do |
| **Free riders** | ~30% | Contribute zero regardless of others |
| **Hump-shaped** | ~14% | Cooperate up to a point, then defect at high contribution levels |

The decay of cooperation arises from this mix: free riders pull down the group average, prompting conditional cooperators to reduce contributions, which further reduces the average — a downward spiral.

---

## Mechanisms That Sustain Cooperation

### Costly Punishment (Fehr & Gächter 2000)
After observing contributions, each player can spend personal tokens to reduce another's payoff (at a 1:3 ratio — spend 1 to reduce target by 3). Even in one-shot anonymous interactions with no reputation benefit, players punish free riders at personal cost — "altruistic punishment."

**Effect:** Without punishment, contributions decay to ~20% by the final round. With punishment, contributions stabilise at 70–100%. Punishment is the single most powerful cooperation-sustaining mechanism identified in the lab.

**Caveat (Herrmann et al. 2008):** In societies with weaker civic norms, subjects engage in *antisocial punishment* — punishing high contributors, not free riders. In these settings, punishment can actually reduce cooperation.

### Reward
Players spend tokens to increase others' payoffs. Weaker than punishment on average, but the combination of punishment + reward yields the best outcomes (Sefton, Steinberg & Walker 2007).

### Communication
Non-binding pre-play communication ("cheap talk") substantially raises contributions. Face-to-face communication nearly eliminates free-riding (Isaac & Walker 1988). Communication enables norm-setting and commitment, even without enforcement.

### Conditional Cooperation and Social Norms
Conditional cooperators maintain high contributions as long as the group average is high. Sustaining cooperation requires keeping average contributions high — which requires either a low share of free riders, or mechanisms (punishment, reward, communication) that suppress free-riding.

### Reputation and Repeated Interaction
In partner-matching protocols (same group each round), reputation effects provide incentives absent in stranger protocols. Long-term relationships sustain cooperation without explicit punishment.

---

## How Contribution Levels Evolve Over Rounds

A canonical time series in standard no-punishment experiments:

1. **Round 1:** 40–60% average contribution
2. **Rounds 2–10:** Steady decline as free riders' behaviour becomes visible and conditional cooperators downward-revise
3. **Final round:** Sharp drop — "end-game effect"
4. **With punishment:** Contributions stabilise at 70–100% across all rounds

A 2020 meta-analysis of 237 public goods games (Peysakhovich et al., PMC) found payoff-based learning — agents update toward higher-payoff choices — best explains the rate of decline across studies.

---

## Optional and Threshold Variants

**Optional PGG (Hauert et al. 2002, *Science*):** Players can opt out entirely for a fixed "loner" payoff. This introduces rock-paper-scissors dynamics: defectors dominate cooperators, loners outcompete defectors (small group is more efficient), cooperators can invade loner populations. Results in cyclic oscillations rather than convergence to full defection — cooperation persists.

**Threshold PGG:** The public good is only provided if total contributions exceed a threshold. This adds coordination game elements. Provision-point mechanisms (contributions returned if threshold not met) can boost contributions through focal-point effects.

**Network / Spatial PGG:** Players on social networks interact only with neighbours. Spatial structure allows cooperators to cluster, shielding them from defectors (extending Nowak & May 1992).

---

## Real-World Applications

- **Climate agreements:** Emission reductions are individually costly, benefits are non-excludable. The Paris Agreement is an attempt to convert a one-shot game into a repeated one with reputational stakes — the PGG is the canonical model for why it's hard.
- **Open-source software:** Contributions benefit all users regardless of who contributed. GitHub reputation and community norms are informal punishment/reward mechanisms.
- **Vaccination:** Herd immunity is a public good. Free riders benefit from others' vaccination without incurring the (small) cost and risk. Incentive design for vaccination uptake draws directly on PGG findings.
- **Tax compliance:** Taxes fund public goods; evasion is free-riding. Audit probability and penalties are formalised punishment mechanisms.
- **Fisheries and commons:** Ostrom (1990) documented communities that successfully self-govern shared resources through monitoring, graduated sanctions, and local norms — her design principles directly operationalise the PGG cooperation mechanisms.
- **Team production:** Within organisations, each member's effort contributes to team output shared regardless of individual contribution — a workplace PGG. Peer monitoring and relative performance evaluation address the free-rider problem.

---

## Key Literature

| Reference | Contribution |
|---|---|
| Ledyard, J.O. (1995). In Kagel & Roth (eds.), *Handbook of Experimental Economics*. Princeton UP | Definitive survey of first-generation PGG experiments |
| Isaac, R.M. & Walker, J.M. (1988). *QJE* 103(1), 179–199 | Foundational MPCR and group size experiments |
| Andreoni, J. (1988). *J. Public Economics* 37(3), 291–304 | Restart effect |
| Andreoni, J. (1990). *Economic Journal* 100(401), 464–477 | Warm-glow giving theory |
| Andreoni, J. (1995). *AER* 85(4), 891–904 | Cooperation from altruism, not confusion |
| Fehr, E. & Gächter, S. (2000). *AER* 90(4), 980–994 | Altruistic punishment sustains cooperation |
| Fischbacher, U., Gächter, S. & Fehr, E. (2001). *Economics Letters* 71(3), 397–404 | Conditional cooperator / free rider typology |
| Fehr, E. & Gächter, S. (2002). *Nature* 415, 137–140 | Altruistic punishment in one-shot strangers |
| Herrmann, B., Thöni, C. & Gächter, S. (2008). *Science* 319, 1362–1367 | Cross-cultural variation; antisocial punishment |
| Hauert, C. et al. (2002). *Science* 296, 1129–1132 | Optional PGG; cyclic dynamics |
| Ostrom, E. (1990). *Governing the Commons*. Cambridge UP | Self-governance of commons through norms and sanctions |
| Ostrom, E., Walker, J.M. & Gardner, R. (1992). *APSR* 86(2), 404–417 | Communication and sanctioning in commons experiments |
| Chaudhuri, A. (2011). *Experimental Economics* 14(1), 47–83 | Post-Ledyard survey covering punishment, reward, communication |

---

## Simulation Setup

This implementation runs a **simultaneous contribution game**: all N agents contribute each round, the pool is multiplied and split equally.

**Strategies implemented:** FreeRider, FullContributor, FixedContributor (50%), ConditionalCooperator, AverageUp

**Metrics tracked each round:**
- `cooperation_rate` — average contribution as fraction of endowment
- `avg_contribution` — mean tokens contributed per round
- `social_welfare` — total payoffs as % of maximum (all-contribute world)
- `nash_eq_distance` — fraction of agents contributing above zero (deviating from NE)
- `strategy_entropy` — Shannon entropy over discretised contribution bins
- `individual_regret` — post-run: best fixed contribution level in hindsight minus actual payoff
