# Ultimatum Game

## Overview

The Ultimatum Game (UG) is a two-player sequential bargaining game that exposes the gap between classical economic theory and human behaviour. A **Proposer** is given a sum of money and must offer a share to a **Responder**. The Responder can accept — in which case each player receives their share — or reject, in which case **both receive nothing**.

The "ultimatum" character comes from finality: the Responder has no counter-offer, only accept or reject. The game is a clean test of the rational actor model because backward induction delivers a sharp, falsifiable prediction — and real subjects routinely violate it.

First studied experimentally by Güth, Schmittberger & Schwarze (1982).

---

## Game Structure

- **Stake:** a fixed amount (this simulation uses stake = 100)
- **Proposer** chooses an offer S ∈ [0, stake]
- **Responder** observes S and chooses Accept or Reject
- **If accepted:** Proposer receives (stake − S), Responder receives S
- **If rejected:** Both receive 0

The game is sequential (Proposer moves first) and information is complete (Responder sees the exact offer).

---

## Nash Equilibrium

Backward induction yields the **subgame-perfect Nash equilibrium (SPNE)**:

1. A rational, self-interested Responder prefers any S > 0 to zero — so they accept any positive offer.
2. Anticipating this, a rational Proposer offers the minimum positive amount (effectively 0) and keeps almost everything.

**The SPNE prediction:** Proposer offers epsilon → 0; Responder accepts.

This prediction is massively and robustly violated in practice. It also admits many non-subgame-perfect Nash equilibria — for example, "Proposer offers 50%, Responder rejects anything below 50%" is a Nash equilibrium, but the Responder's threat is not credible under pure self-interest.

---

## What Actually Happens in Experiments

The empirical record since 1982 is strikingly consistent:

| Finding | Detail |
|---|---|
| Modal offer (Western samples) | 40–50% of endowment |
| Mean offer | ~40–45% |
| Rejection threshold | Offers below 20–30% are rejected ~50% of the time |
| SPNE prediction | Proposer offers ~0%, Responder accepts anything |
| Gap between theory and data | Massive and replicates across decades and cultures |

**Key observations:**
- Proposers voluntarily offer substantial shares far above the rational minimum, even in dictator games (where Responders cannot reject) — suggesting genuine other-regarding preferences, not just strategic anticipation of rejection.
- Responders reject positive offers, sacrificing real money to punish perceived unfairness.
- High-stakes experiments in Indonesia (Cameron 1999, up to 3× monthly income) found Responders became somewhat more willing to accept low offers, but Proposer behaviour barely changed. The gap never closed.

---

## Why People Reject Positive Offers

Several mechanisms explain Responder rejections:

**Fairness norms:** Subjects have internalised a norm that equal or near-equal splits are fair. The 50/50 split is a focal point and social benchmark. Violations trigger a punitive response.

**Negative emotions:** Low offers elicit anger and disgust. Neuroimaging shows anterior insula activation (emotional arousal) predicts rejection — the decision is partly emotional, not purely calculated (Sanfey et al. 2003, *Science*).

**Costly punishment:** Rejection signals that unfair behaviour will be punished. Even in one-shot anonymous interactions (no reputation benefit), people punish at personal cost — "altruistic punishment" (Fehr & Gächter 2002).

**Thaler (1988):** The UG results require positing both a preference for higher payoffs *and* a preference for equity — a dual-motivation structure incompatible with standard utility maximization over own payoffs.

---

## Behavioural Models

### Fehr & Schmidt (1999) — Inequity Aversion

The most influential formal model of UG behaviour. Each player i has utility:

```
Uᵢ(x) = xᵢ − αᵢ · max{xⱼ − xᵢ, 0} − βᵢ · max{xᵢ − xⱼ, 0}
```

- **αᵢ:** weight on disadvantageous inequity (suffering when you receive less)
- **βᵢ:** weight on advantageous inequity (discomfort from receiving more); βᵢ < αᵢ

People dislike being behind more than being ahead, but also have some aversion to receiving too much. With realistic α and β, the model predicts: Responders reject low offers (disadvantageous inequity is sufficiently costly), and Proposers offer generously (advantageous inequity aversion). The model explains data from PD, public goods, and market experiments, not just the UG.

---

## Cross-Cultural Variation

Henrich et al. (2001, *AER*) ran the UG across 15 small-scale societies — hunter-gatherers, horticulturalists, pastoralists. Key findings:

| Society | Pattern |
|---|---|
| **Machiguenga** (Peru) | Modal offer ~15%; low rejection rates. Limited market integration, few cooperative activities with strangers. |
| **Lamalera** (Indonesia) | Modal offers >50% ("hyper-fair"). Whale hunters with deep communal norms around meat distribution. |
| **Hadza** (Tanzania) | Low offers but high rejection rates — independent variation in proposer and responder behaviour. |

**Predictors of offer levels across societies:** degree of market integration and payoffs to cooperation in daily economic life. Societies embedded in market exchange and reliant on broad cooperation make higher, more equal offers. Fairness norms are culturally learned, not universal.

---

## Iterated Ultimatum Game

When played repeatedly between the same pair:

- **Reciprocal adaptation:** Responders who hold firm (reject low offers) often receive higher offers in subsequent rounds. Proposers who make low offers learn to raise them.
- **Proposer learning:** Offers tend to rise over early rounds as Proposers learn what Responders will accept.
- **Responder learning:** Responders who accept low offers early may receive lower offers in subsequent rounds.
- **End-game effects:** Self-interested behaviour increases in final rounds — backward induction becomes more salient as the game terminates.
- **Dual-fMRI evidence (Migliore et al. 2018):** Reciprocal behaviour in iterated UG is associated with neural alignment between players, suggesting mutual modelling of counterpart behaviour underlies cooperative dynamics.

---

## Strategies in This Simulation

| Strategy | Proposal | Response threshold |
|---|---|---|
| **FairPlayer** | 50% | Rejects < 40% |
| **GreedyPlayer** | 1% (minimum) | Accepts any positive offer |
| **GenerousPlayer** | 60% | Rejects < 20% |
| **SpitefulPlayer** | 50% | Rejects < 50% (punishes greed) |
| **AdaptivePlayer** | Starts 40%, raises 5% after rejection, lowers 2% after acceptance | Accepts ≥ 30% |

---

## Real-World Applications

- **Wage negotiations:** Employers making "best and final" offers instantiate the UG structure. Gächter & Falk (2002) found wages significantly higher when subjects performed real effort before bargaining — effort raises the fairness benchmark.
- **Labor-management bargaining:** Take-it-or-leave-it offers in collective bargaining that feel deeply unfair trigger costly strikes even when the offer is materially positive.
- **Retail pricing:** Posted prices are UG offers. Price gouging (e.g., post-disaster) triggers consumer backlash even when consumers could materially benefit — UG-type rejection on fairness grounds.
- **Legal settlements:** Settlement offers function as ultimatums. Fairness perceptions strongly influence acceptance, consistent with UG findings — "fair" offers settle; perceived lowball offers drive litigation.
- **Pay transparency:** Revealing within-firm pay differences raises rejection thresholds among lower-paid workers who discover they earn less than comparators.
- **International diplomacy:** Take-it-or-leave-it ultimatums in geopolitical negotiations carry rejection risk because the offended party may prefer a worse outcome to accepting a perceived humiliation.

---

## Key Literature

| Reference | Contribution |
|---|---|
| Güth, W., Schmittberger, R. & Schwarze, B. (1982). *J. Economic Behavior & Organization* 3(4), 367–388 | Original ultimatum game experiment |
| Thaler, R.H. (1988). *J. Economic Perspectives* 2(4), 195–206 | "Anomalies" framing; dual-motivation argument |
| Fehr, E. & Schmidt, K.M. (1999). *QJE* 114(3), 817–868 | Inequity aversion model |
| Henrich, J. et al. (2001). *AER* 91(2), 73–78 | Cross-cultural variation across 15 small-scale societies |
| Cameron, L.A. (1999). *Economic Inquiry* 37(1), 47–59 | High-stakes UG in Indonesia; partial but incomplete convergence |
| Fehr, E. & Gächter, S. (2002). *Nature* 415, 137–140 | Altruistic punishment in one-shot anonymous interactions |
| Sanfey, A.G. et al. (2003). *Science* 300, 1755–1758 | Neural basis of UG decisions; insula activation predicts rejection |
| Migliore, S. et al. (2018). *Scientific Reports* 8, 11196 | Dual-fMRI of iterated UG; neural alignment and reciprocal behaviour |

---

## Simulation Setup

This implementation runs a **round-robin tournament with all ordered pairs**: each agent acts as both Proposer (to every other agent) and Responder (to every other agent) each round.

- **If accepted:** Proposer earns (stake − offer), Responder earns offer
- **If rejected:** Both earn 0

**Metrics tracked each round:**
- `cooperation_rate` — fraction of offers accepted
- `avg_offer_pct` — mean offer as percentage of stake
- `social_welfare` — total payoffs as % of maximum (all accepted at 50/50)
- `nash_eq_distance` — fraction of interactions deviating from SPNE (offer > 0 accepted, or any rejection)
- `strategy_entropy` — Shannon entropy over discretised offer bins
- `individual_regret` — post-run: best fixed offer/threshold in hindsight
