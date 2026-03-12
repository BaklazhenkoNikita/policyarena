# Prisoner's Dilemma

## Overview

The Prisoner's Dilemma (PD) is the most studied game in all of game theory. Two players simultaneously choose to either **Cooperate (C)** or **Defect (D)**. The defining tension: defection is always individually rational, yet mutual defection leaves both players worse off than mutual cooperation.

The name comes from the canonical framing: two suspects are interrogated in separate rooms. Each can stay silent (cooperate with each other) or testify against the other (defect). If both stay silent, both get a light sentence. If both testify, both get a heavy sentence. If one testifies and the other stays silent, the testifier goes free and the silent one gets the maximum penalty.

---

## Payoff Matrix

|                     | Opponent Cooperates | Opponent Defects |
|---------------------|---------------------|------------------|
| **Player Cooperates** | R, R                | S, T             |
| **Player Defects**    | T, S                | P, P             |

Where the payoffs satisfy: **T > R > P > S** and **2R > T + S**

- **T** (Temptation): unilateral defection reward
- **R** (Reward): mutual cooperation payoff
- **P** (Punishment): mutual defection payoff
- **S** (Sucker): exploited cooperator payoff

The second condition (2R > T + S) ensures mutual cooperation beats taking turns exploiting each other.

**This simulation uses:** T = 5, R = 3, P = 1, S = 0 — the values from Axelrod's original tournaments.

---

## Nash Equilibrium

**(Defect, Defect)** is the unique Nash Equilibrium in the one-shot game. Defection is a **strictly dominant strategy**: regardless of what the opponent does, a player always gets a higher payoff by defecting (T > R when opponent cooperates; P > S when opponent defects).

The NE is **Pareto-inefficient**: both players would be strictly better off at (Cooperate, Cooperate), which yields R = 3 for each versus only P = 1 at the NE. This is the core tragedy — individual rationality produces collective irrationality.

**Finitely repeated games:** With a known end date, backward induction extends mutual defection to every round. In round N, both players defect (as in the one-shot game). Knowing this, they defect in round N-1, and so on back to round 1.

**Infinitely repeated games (Folk Theorem):** If the game continues indefinitely and players are sufficiently patient, cooperation *can* be a Nash Equilibrium. The condition for Grim Trigger to sustain cooperation is:

```
δ ≥ (T − R) / (T − P)
```

With Axelrod's values: δ ≥ (5−3)/(5−1) = **0.5**. If each player values future payoffs at least half as much as current ones, cooperation is sustainable.

---

## Iterated Prisoner's Dilemma and Axelrod's Tournaments

The iterated variant transforms the game entirely. Players can condition behavior on history, enabling reciprocity: reward cooperation, punish defection.

**Robert Axelrod** (University of Michigan) ran two landmark computer tournaments in 1980. Game theorists submitted strategies as programs; each played every other in a round-robin. The winner both times: **Tit-for-Tat**, submitted by Anatol Rapoport — the simplest entry, just two rules:

1. Cooperate on round 1.
2. Copy whatever the opponent did last round.

This launched one of the most productive research programmes in the social sciences.

---

## Major Strategies

### Tit-for-Tat (TFT)
Copy the opponent's previous move. Cooperate first.

- **Strengths:** Clear, immediately retaliatory, immediately forgiving, never first to defect.
- **Weakness:** Noise-sensitive. A single accidental defection causes mutual retaliation spirals.

### Pavlov / Win-Stay Lose-Shift (WSLS)
Repeat last action if the outcome was good (CC or DC); switch if it was bad (DD or CD).

- **Strengths:** Recovers from mutual defection (DD → both switch to C). Can exploit unconditional cooperators. Outperforms TFT under evolutionary dynamics and noise.
- **Weakness:** Exploitable by persistent defectors.
- **Key paper:** Nowak & Sigmund, *Nature* 364, 56–58 (1993)

### Generous Tit-for-Tat (GTFT)
Cooperate after opponent cooperates; after opponent defects, cooperate with probability ~1/3 anyway.

- **Strengths:** The random forgiveness breaks TFT's retaliation spirals. Most effective strategy in alternating-move PD.
- Suggested by Sugden (1986); formalized by Nowak & Sigmund.

### Grim Trigger
Cooperate until the opponent defects once; then defect forever.

- Theoretically powerful (strong deterrent), but catastrophically fragile to noise — one accident ends cooperation permanently.

### Always Defect (AllD)
Defect every round. Dominant in one-shot games, but performs poorly over many iterated rounds against retaliating strategies.

### Zero-Determinant Strategies (ZD)
Discovered by Press & Dyson (2012): a class of memory-one strategies that can unilaterally enforce a linear relationship between both players' scores.

- **Extortion ZD:** Force the opponent's payoff to be a fixed fraction of yours. Mathematically elegant but **not evolutionarily stable** — extortionate strategies are ultimately outcompeted.
- **Generous ZD:** Favour the opponent; these *are* evolutionarily stable and converge to cooperative outcomes.
- Key papers: Hilbe et al. (2013); Adami & Hintze (2013); Stewart & Plotkin (2013)

---

## What Makes a Strategy Successful

Axelrod distilled four properties from his tournament analysis:

| Property | Description |
|---|---|
| **Nice** | Never be the first to defect. All top strategies satisfied this. |
| **Retaliatory** | Respond quickly to defection — don't let exploitation continue. |
| **Forgiving** | Return to cooperation once the opponent cooperates again. Avoid endless feuds. |
| **Clear** | Be predictable. Opponents can adapt to cooperate with a legible strategy. |

A fifth principle: **don't be envious.** Winning strategies maximize their own absolute score, not their score relative to the opponent. TFT never beats its partner — it matches. But it does extremely well across the whole population.

---

## When Defection Wins vs. When Cooperation Wins

**Defection tends to dominate when:**
- The game is one-shot or finitely repeated with a known endpoint
- The discount factor δ is low (players don't value future interactions)
- Opponents don't retaliate
- The population is mostly defectors (TFT entering an AllD world loses its first interaction)
- Players are anonymous with no shared history

**Cooperation tends to dominate when:**
- The game is indefinitely or infinitely repeated
- Players value the future highly (δ close to 1)
- Players are identifiable and reputation persists
- Matches are long (cumulative R >> one-time T)
- Spatial structure allows cooperators to cluster (Nowak & May, 1992)
- The population already has significant cooperative strategies

---

## Real-World Applications

- **Nuclear deterrence:** Mutually Assured Destruction is a Grim Trigger strategy; arms control treaties (START, SALT) solve the repeated game through verification and reciprocity
- **Trade policy:** Tariff wars are mutual-defection equilibria; the WTO sustains cooperation through repeated play and dispute mechanisms
- **Fisheries:** Overfishing is a multi-player PD; community enforcement and quota systems implement iterated-game solutions
- **Vampire bats:** Reciprocal blood-meal sharing follows TFT-like patterns — bats that receive food are more likely to donate in turn (Wilkinson, 1984)
- **Oligopoly pricing:** Firms in repeated market interactions can sustain tacit collusion above the competitive equilibrium price
- **Open source software:** Contribution to shared codebases is a multi-player PD; reputation systems and community norms sustain cooperation

---

## Key Literature

| Reference | Contribution |
|---|---|
| Axelrod, R. & Hamilton, W.D. (1981). *Science* 211, 1390–1396 | Original tournament paper; four properties of successful strategies |
| Axelrod, R. (1984). *The Evolution of Cooperation*. Basic Books | The landmark book; most cited work in social science on cooperation |
| Trivers, R.L. (1971). *Quarterly Review of Biology* 46(1), 35–57 | Reciprocal altruism as evolutionary basis for cooperation |
| Nowak, M.A. & Sigmund, K. (1993). *Nature* 364, 56–58 | Pavlov/WSLS outperforms TFT under evolutionary dynamics |
| Nowak, M.A. & May, R.M. (1992). *Nature* 359, 826–829 | Spatial structure enables cooperator clusters to resist defectors |
| Press, W.H. & Dyson, F.J. (2012). *PNAS* 109(26), 10409–10413 | Zero-determinant strategies; unilateral payoff control |
| Hilbe, C., Nowak, M.A. & Sigmund, K. (2013). *PNAS* 110(17), 6913–6918 | Extortion ZD strategies not evolutionarily stable |
| Stewart, A.J. & Plotkin, J.B. (2013). *PNAS* 110(38), 15348–15353 | Evolution drives ZD strategies from extortion toward generosity |
| Mathieu, P. & Beaufils, B. (2024). *PLOS Computational Biology* | 195 strategies across thousands of tournaments; adaptability matters most |
| Wilkinson, G.S. (1984). *Nature* 308, 181–184 | Empirical reciprocal altruism in vampire bats |

---

## Simulation Setup

This implementation runs an **N-player round-robin tournament**: every agent plays every other agent once per round. Payoffs are accumulated across all pairings.

**Strategies implemented:** TitForTat, AlwaysDefect, AlwaysCooperate, Pavlov, RandomBrain

**Metrics tracked each round:**
- `cooperation_rate` — fraction of C actions across all pairings
- `social_welfare` — total payoffs as % of maximum (all-C world)
- `nash_eq_distance` — fraction of pairings not at (D, D)
- `strategy_entropy` — Shannon entropy over action distribution
- `individual_regret` — post-run: best fixed action in hindsight minus actual payoff
