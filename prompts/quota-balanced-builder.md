Contained in this file are my prompt and a response from a very sophisticated large language model. It explains how we can build actual field lineups for pre-contest simulation instead of using our current flawed analytic method. Read it, and come up with a plan to implement the "quota-balanced random builder" into our codebase. When we compute top 1% finish rate, we'll use this instead of our current method.

**My prompt**
I have created a lineup optimizer and contest simulator for Draftkings NFL daily fantasy showdown captain mode contests. I have also built a "contest flashback" simulator: I input the pre-contest player projections and correlations along with the actual contest data -- the actual lineups -- people played. Then I simulate the football player scores a large number of times and compute the ROI and top 1% finish rate of every lineup in the contest. 

My pre-contest simulator generates an arbitrary number of lineups, usually between 200 and 2000, and then ranks them by projected top 1% finish rate in a contest of a given size. I have uncovered a flaw in my approach. It stems from the fact that I do not attempt to generate field lineups before evaluating the top 1% finish rates for my lineups. Instead, I take each player's projected FLEX ownership and CPT ownership and use an analytical method to approximate the field's score distribution as a mixture of player scores. 

What I have uncovered from comparing my pre-contest sims to my flashback sims, and also looking at what the most skillful DFS players play, is that while my system does identify some good lineups, it underrates others, particularly lineups with higher-owned CPT, because put simply the analytical method I use does not accurately compute the scoring distribution of the DFS contest field.

I want to remedy this by generating actual field lineups for my pre-contest sims. Inputs would be player projections (mean projection, standard deviation, correlation to other players), ownership projections (for both CPT and FLEX), and number of entrants in this hypothetical contest. For now, I'm excluding exact payout structure and just concerned with simulating how often each lineup finishes in the top 1% in the contest because I find this is a good proxy for ROI.

Of course I will not be able to perfectly predict the field lineups. Some entrants play sharp lineups and some play poorly built lineups. But my simulated field needs to line up with ownership projections and mostly consist of lineups that "make sense". If the pre-contest ownership projections are fairly accurate, the same lineups should have high top 1% finish rates in the pre-contest sim and the flashback sim.

I have attached two spreadsheets for context. "top1pct_lineups_20251204_195915.xlsx" contains my pre-contest generated lineups in the first tab. Their projected top 1% finish rates are in the last column. "flashback_185654486.xlsx" contains my flashback sim results. The "Simulation" tab contains the actual lineups from the contest, their top 1% finish rates, ROIs, and some other data.

Help me come up with one or multiple methods to generate field lineups so I can improve my pre-contest top 1% predictions.

**LLM Response**
# Quota-Balanced Random Builder for Simulated DFS Fields

This document describes a practical method to generate realistic **field lineups** for DraftKings NFL Showdown captain-mode contests. The goal is to create a simulated contest field whose **player CPT/FLEX ownership matches your projections** and whose lineups are mostly reasonable (stacking, salary use, etc.), while also reflecting a mix of sharp and weak players.

---

## 1. Overview

Instead of approximating the field as a mixture of **independent player scores weighted by ownership**, we explicitly model the field as a distribution over **complete lineups**.

We do this by:

1. **Translating ownership projections into quotas** for each player at CPT and FLEX.
2. **Sampling lineups sequentially**, filling CPT first and then each FLEX slot, while:
   - Respecting **remaining player quotas**,
   - Incorporating **projections, correlations, and lineup rules**,
   - Allowing for different **entrant types** (sharp vs rec).
3. Optionally refining the final set of lineups with **post-processing swaps** to sharpen ownership matching.

This produces a set of N lineups (N = field size) that:

- Matches projected CPT/FLEX ownership reasonably closely,
- Uses realistic lineup construction patterns,
- Naturally generates **duplications** and chalk clustering.

---

## 2. Inputs

For each player \( j \) on the slate, you assume you have:

- `p_cpt[j]`: projected CPT ownership (fraction of lineups using j at CPT).
- `p_flex[j]`: projected FLEX ownership (fraction using j at FLEX).
- `proj[j]`: mean fantasy projection (DK points in FLEX slot).
- `salary[j]`: DK salary in FLEX slot.
- `corr[j, k]`: correlation coefficient between players j and k, or a simpler correlation score.
- Position (`QB`, `RB`, `WR`, `TE`, `K`, `DST`) and team info.

Global inputs:

- `N`: desired number of field lineups (contest size).
- Salary cap and minimum salary usage.
- Any hard constraints / rules (e.g. valid Showdown roster structure, team count rules, etc.).

From projected ownership and field size, compute **quotas**:

- \( T^{CPT}_j = 	ext{round}(p_{cpt}[j] \cdot N) \)
- \( T^{FLEX}_j = 	ext{round}(p_{flex}[j] \cdot N) \)

Initialize **remaining quotas**:

- \( R^{CPT}_j \leftarrow T^{CPT}_j \)
- \( R^{FLEX}_j \leftarrow T^{FLEX}_j \)

These will be decremented as you build lineups.

---

## 3. Entrant Types (Sharp vs Rec)

Real contests have a mix of player skill levels. Model this with **entrant types**:

- Example types:
  - `sharp`
  - `semi_sharp`
  - `fish` (recreational)
- Assign mixture weights:
  - e.g. `sharp = 0.1`, `semi_sharp = 0.3`, `fish = 0.6`.

Each type \( s \) has its own hyperparameters:

- \( lpha_s \): how strongly the type chases projection/value.
- \( eta_s \): how strongly the type follows ownership quotas.
- \( \gamma_s \): how strongly the type cares about correlation/stacking.
- \( \delta_s \): how much the type cares about salary usage / lineups looking "clean".

**Sharps** → high \( lpha_s, \gamma_s, \delta_s \), moderate \( eta_s \).  
**Fish** → lower \( lpha_s, \gamma_s, \delta_s \), higher \( eta_s \) (they follow ownership more blindly).

For each lineup you generate, you first sample an entrant type \( s \) according to the mixture weights.

---

## 4. Sampling a CPT

For a given lineup and chosen entrant type \( s \), you sample a CPT with probabilities that depend on:

- Player projection/value at CPT,
- Remaining CPT quota,
- Any CPT-specific heuristics/rules.

Define a CPT weight for each player \( j \):

\[
w^{CPT}_j \propto
\underbrace{\text{value}^{CPT}_j}^{\text{projection / salary}}
^{\alpha_s}
\cdot
\underbrace{(R^{CPT}_j + \epsilon)}_{\text{remaining quota}}^{\beta_s}
\cdot
\underbrace{\text{soft\_rules\_CPT}(j)}_{\text{pos/team preferences}}^{\gamma_s}
\]

Where:

- `value^CPT_j` could be something like:
  - \( \frac{1.5 \cdot \text{proj}[j]}{1.5 \cdot \text{salary}[j]} = \frac{\text{proj}[j]}{\text{salary}[j]} \), or any CPT-adjusted metric you prefer.
- `(R^{CPT}_j + ε)` downweights players whose quota is nearly exhausted. Set \( \epsilon \) small (e.g. \( 10^{-6} \)) and **skip** players with \( R^{CPT}_j \le 0 \).
- `soft_rules_CPT(j)` encodes heuristics, e.g.:
  - Slight boost for QB/RB/WR CPTs.
  - Slight penalty for K/DST CPTs.
  - Modulation based on game spread/total.

After computing `w^CPT_j` over all eligible j, normalize to a probability distribution and sample one CPT, call it `cpt`.

Update its remaining quota:

- \( R^{CPT}_{cpt} \leftarrow R^{CPT}_{cpt} - 1 \)

---

## 5. Sampling FLEX Spots Sequentially

With CPT chosen, fill the 5 FLEX spots one-by-one, conditioning on:

- Players already in the lineup,
- Remaining quota for each player,
- Remaining salary and open slots,
- Team composition so far,
- Correlations with chosen players.

Let `L` be the current lineup (starting with just the CPT). For each FLEX slot:

1. Determine candidates \( j \) that:
   - Are not already in the lineup (no duplicates),
   - Have \( R^{FLEX}_j > 0 \),
   - Are legal given position and team constraints,
   - Can fit under the salary cap with at least some plausible completions.

2. For each candidate \( j \), compute a FLEX weight:

\[
w^{FLEX}_j \propto
\underbrace{\text{value}^{FLEX}_j}_{\text{projection / salary}}^{\alpha_s}
\cdot
\underbrace{(R^{FLEX}_j + \epsilon)}_{\text{remaining quota}}^{\beta_s}
\cdot
\underbrace{\exp\Big( \gamma_s \sum_{k \in L} \text{corr}(j, k) \Big)}_{\text{correlation factor}}
\cdot
\underbrace{\text{rule\_penalties}(j, L)}_{\text{lineup rule bias}}
\]

Where:

- `value^FLEX_j` is something like `proj[j] / salary[j]` or your preferred value metric.
- `corr(j, k)` is your player-level correlation estimate:
  - Positive for QB–WR, WR–QB, RB–DST, etc.
  - Negative for opposing RB–RB, WR–DST vs opposing O, etc.
- `rule_penalties(j, L)` encodes soft lineup rules, for example:
  - Avoid same-team RB + QB + 3 pass catchers if that’s rarely played.
  - Encourage 4–2 or 5–1 stacks depending on game script tendencies.
  - Penalize extreme salary patterns (e.g. leaving \$7k+ on the table) for certain entrant types.

3. Normalize those weights to probabilities and randomly choose a player for the slot.
4. Add that player to the lineup, decrement quota:
   - \( R^{FLEX}_j \leftarrow R^{FLEX}_j - 1 \)

Repeat until all FLEX spots are filled.

### Handling Dead Ends

Sometimes you may reach a state where no candidate can fill the remaining slots legally while respecting quotas and salary constraints. In that case:

- **Fallback 1:** Temporarily ignore quotas and pick from all legal players with a simpler weight based on projection and correlation. Quotas will be slightly off but the lineup remains reasonable.
- **Fallback 2:** Loosen rule penalties; allow less typical stack patterns or more salary left.
- **Fallback 3:** Abort this lineup and restart it (this is more expensive but clean).

In practice, a combination of Fallback 1 and 2 is usually enough.

---

## 6. Building the Full Field

Repeat the CPT + FLEX sampling process **N times** to create N field lineups. For each lineup:

1. Sample entrant type \( s \sim \{w_s\} \).
2. Sample CPT using type-specific parameters.
3. Sample 5 FLEX spots sequentially as described above.

This yields a field where:

- Player-level CPT/FLEX usage is driven towards your ownership projections via quotas.
- Lineups have realistic stacking and correlation structure.
- Chalk combinations appear frequently and therefore produce realistic **duplication levels**.

---

## 7. Post-Processing Ownership and Lineup Quality

After generating all N lineups:

1. Compute **achieved CPT/FLEX ownership**:
   - \( \hat{p}^{CPT}_j = \frac{1}{N} \sum_{k=1}^N \mathbb{1}\{j \text{ is CPT in lineup } k\} \)
   - \( \hat{p}^{FLEX}_j = \frac{1}{N} \sum_{k=1}^N \mathbb{1}\{j \text{ is FLEX in lineup } k\} \)

2. Compare \( \hat{p}_j \) to target \( p_j \). If the error is larger than you’d like, run **local swap improvements**:

   - Randomly pick two lineups, `L1` and `L2`.
   - Consider swapping one or two FLEX players between them.
   - Accept swaps that:
     - Keep both lineups legal,
     - Move exposures closer to targets,
     - Do not severely break correlation/stacking rules.

3. You can also optionally filter out truly egregious lineups:
   - Very low projection (e.g., more than X standard deviations below the mean of the field),
   - Extreme salary leave (if you believe the field rarely does that),
   - Violations of basic football logic (e.g., DST CPT + 5 opposing offensive players).

If you remove lineups, regenerate replacements using the same procedure or by adjusting quotas slightly.

---

## 8. Using the Generated Field in Top 1% Simulations

Once you have a simulated field of N lineups:

1. For each simulation of game outcomes:
   - Sample player scores from your multivariate distribution (mean projections, standard deviations, correlations).
   - Compute scores for each field lineup and for each of your candidate lineups.
2. Rank all lineups by score for that simulation.
3. For each of your candidate lineups, record whether it finishes in the **top 1%** of this simulated contest.
4. Repeat for many simulations (e.g., 10k+).

The empirical frequency with which each lineup finishes top 1% is your estimate of its **top 1% finish rate** under a realistic field distribution, rather than the previous approximation based on independent player mixtures.

Because the field now consists of lineups that:

- Respect ownership projections,
- Reflect realistic stacking and chalk behavior,
- Include both sharp and recreational constructions,

your pre-contest **top 1% estimates** should line up much better with the **flashback simulations**, especially for lineups built around high-owned CPTs.

---

## 9. Implementation Notes & Extensions

- You can start with a **single entrant type** (one set of \( \alpha, \beta, \gamma, \delta \)) and add more types later once the core logic is working.
- You can calibrate \( \alpha, \beta, \gamma, \delta \) and stack preferences by comparing the distributions of:
  - Stack types (5–1, 4–2, 3–3),
  - Salary usage,
  - Chalkiness (sum of ownership, salary-weighted ownership),
  - CPT positional breakdown (QB vs RB vs WR vs K vs DST),
  between your simulated field and actual historical contests.
- Later, you can combine this quota-balanced builder with more sophisticated methods (e.g., max-entropy reweighting of a large lineup pool) to get even closer matches to historical field behavior.

This quota-balanced random builder is a tractable way to move from **“ownership-only player mixtures”** to **“ownership-aware, correlation-aware field lineups”**, which is crucial for reliable pre-contest top 1% finish rate estimates.
