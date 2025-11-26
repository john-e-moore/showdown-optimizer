File: 02_top1pct_finish_rate.md

Estimating Top 1 Percent Finish Rate for NFL Showdown Lineups
=============================================================

Goal
----
Given:

- Player projections (mean and standard deviation of DK points)
- A player by player correlation matrix
- Ownership projections for CPT and FLEX
- A set of candidate lineups (from the optimizer)
- A contest field size

Estimate, for each lineup, the probability that it finishes in the top 1 percent of the contest.


1. Inputs
---------

1) Player list:

Use the same ordered list of Player objects as in the optimizer, for example:

- players = [p0, p1, ..., p_{P-1}]

You must have:

- dk_mean[i]: mean DK points for player i (from SaberSim)
- dk_std[i]: standard deviation of DK points for player i (from SaberSim or from your simulation)

2) Correlation matrix:

- corr is a P by P matrix
- corr[i, j] is the correlation between DK scores of players i and j
- You already generate this from your simulation code

3) Ownership projections:

For each player i:

- cpt_own[i]: projected CPT ownership (fraction between 0 and 1)
- flex_own[i]: projected FLEX ownership (fraction between 0 and 1 for being in any FLEX spot)

These should satisfy roughly:

- sum over i of cpt_own[i] is approximately 1
- sum over i of flex_own[i] is approximately 5

4) Lineups to evaluate:

Each lineup L is defined by:

- One CPT player
- Five FLEX players

You can reuse the Lineup class from the optimizer.

5) Field size:

- field_size: total number of lineups in the contest, for example 23529
- For top 1 percent, the number of top lineups is num_top = floor(field_size * 0.01)


2. Build covariance matrix
--------------------------

From dk_std and corr, build the covariance matrix cov:

- cov[i, j] = corr[i, j] * dk_std[i] * dk_std[j]

Implementation outline:

- Take dk_std as a length P array
- Compute outer = dk_std outer product dk_std
- Multiply elementwise by corr to get cov

If cov has numerical issues (not quite positive semi definite), add a small epsilon to the diagonal.


3. Simulate correlated player outcomes
--------------------------------------

Treat the P-dimensional vector of DK points X as multivariate normal:

- X ~ Normal(mu, cov)
- mu is dk_mean
- cov is the covariance matrix from the previous step

Simulate S game outcomes, for example S = 20000:

- For s from 1 to S, draw X_s ~ Normal(mu, cov)

Collect them in a matrix:

- X is num_sims by P
- X[s, i] is the DK score of player i in simulation s

You only need to simulate once per slate. You will reuse the same X for all lineups.


4. Approximate the field score distribution using ownership
-----------------------------------------------------------

For each simulation s, you know X_s, the realized scores of all players.

Now approximate the distribution of scores of a random lineup from the field in that simulation using only ownership.

4.1 Ownership as mixture probabilities
--------------------------------------

Interpret projected ownership as follows:

- For the CPT slot:
  - A random field lineup chooses player i as CPT with probability cpt_own[i]

- For FLEX slots:
  - A random field lineup has 5 FLEX slots
  - Let pi[i] be the per-slot probability of choosing player i in any given FLEX slot
  - Approximate pi[i] = flex_own[i] / 5 so that sum pi[i] is approximately 1

Assume:

- CPT choice and each FLEX slot choice are independent, and FLEX slots share the same distribution pi.

This will not exactly match all correlation between roster spots in real lineups, but is usually sufficient to model the distribution of scores across the field.


4.2 CPT score distribution for a fix simulation
-----------------------------------------------

Given X_s:

- CPT score C_s is a discrete random variable:
  - With probability cpt_own[i], CPT is player i and the CPT score is 1.5 * X_s[i]

Compute the conditional mean and variance of C_s:

- mu_C[s] = sum over i of cpt_own[i] * (1.5 * X_s[i])
- second moment mC2[s] = sum over i of cpt_own[i] * (1.5 * X_s[i])^2
- var_C[s] = mC2[s] minus (mu_C[s])^2


4.3 FLEX score distribution for a fix simulation
------------------------------------------------

A single FLEX slot score F_s is a discrete mixture:

- With probability pi[i], F_s = X_s[i]

Compute:

- mu_F[s] = sum over i of pi[i] * X_s[i]
- second moment mF2[s] = sum over i of pi[i] * X_s[i]^2
- var_F[s] = mF2[s] minus (mu_F[s])^2

Total FLEX score is the sum of 5 independent copies of F_s:

- F_total_s = F_s1 + F_s2 + F_s3 + F_s4 + F_s5

So:

- mu_F_total[s] = 5 * mu_F[s]
- var_F_total[s] = 5 * var_F[s]


4.4 Total field lineup score distribution for a simulation
----------------------------------------------------------

Total score of a random field lineup in simulation s is:

- L_field_s = C_s + F_total_s

Assuming CPT selection and FLEX selections are independent:

- mu_field[s] = mu_C[s] + mu_F_total[s]
- var_field[s] = var_C[s] + var_F_total[s]

Approximate L_field_s as normal with these moments:

- L_field_s approximately Normal(mu_field[s], var_field[s])

Then the approximate 99th percentile (top 1 percent threshold) of the field score distribution in simulation s is:

- T_field_0.99[s] = mu_field[s] + z * sqrt(var_field[s])

where z is the 99th percentile of the standard normal, approximately 2.326.


5. Scoring lineups in the simulations
-------------------------------------

Each lineup L can be represented by a weight vector w_L of length P:

- For each player i:
  - w_L[i] = 1.5 if player i is CPT in lineup L
  - w_L[i] = 1.0 if player i is one of the FLEX players in L
  - w_L[i] = 0 otherwise

Assume you construct an index from player_id to position in the players array so you can build w_L easily.

Given X, a matrix of shape num_sims by P:

- The scores of lineup L across simulations are:

  score_L[s] = sum over i of w_L[i] * X[s, i]

If you collect K lineups and build a matrix W of shape K by P, with one row per lineup, you can compute all scores at once:

- scores = X dot transpose of W
- scores is num_sims by K
- scores[s, k] is the score of lineup k in simulation s


6. Estimating top 1 percent finish probability
----------------------------------------------

You have:

- thresholds T_field_0.99, an array of length num_sims with the approximate 99th percentile field score for each simulation
- scores, a num_sims by K matrix of lineup scores

For each lineup k and simulation s:

- Define indicator I_k[s] = 1 if scores[s, k] >= T_field_0.99[s], else 0

Estimate the top 1 percent finish probability for lineup k as:

- p_top1[k] = (1 / num_sims) * sum over s of I_k[s]

This is simply the fraction of simulations where the lineup score is at or above the simulated field 99th percentile threshold.


7. Field size and interpretation
--------------------------------

The method above effectively models a field with a very large number of lineups. The threshold T_field_0.99[s] is the score that a random lineup from this modeled field must reach to be in the top 1 percent in simulation s.

For a real contest with field size field_size:

- The actual top 1 percent cutoff is random, but it is closely related to the 99th percentile of the field score distribution in that game script.
- You can interpret p_top1[k] as the probability that lineup k finishes inside the top floor(field_size * 0.01) lineups, especially when field_size is reasonably large.


8. Pipeline summary
-------------------

For a given slate:

1) Precomputation:

- Load SaberSim projections from data/sabersim/NFL_*.csv
- Build an ordered player list and arrays dk_mean and dk_std
- Load the correlation matrix and align its rows and columns to the player order
- Build the covariance matrix from dk_std and corr
- Simulate X, a num_sims by P matrix of player scores
- Load ownership projections for CPT and FLEX, align them to the player order
- Compute T_field_0.99 using the ownership and X as described above

2) Lineup evaluation:

- Generate candidate lineups using the lineup optimizer (for example the top N by mean projection)
- For each lineup, build a weight vector w_L of length P
- Stack all weight vectors into W (K by P)
- Compute scores = X dot transpose of W
- Compare scores to T_field_0.99 to compute p_top1 for each lineup

3) Usage:

- Use p_top1 as your main metric for lineup quality
- For example, among a pool of 500 or 1000 well projected lineups, select the ones with the highest p_top1 values subject to your exposure and duplication preferences

This gives you a practical way to optimize your entries for top 1 percent finish rate without explicitly building a full fake field of lineups in every simulation.