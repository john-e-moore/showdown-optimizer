---
name: mass-lineup-enum-and-top1pct
overview: Create a standalone scratch script that (1) loads Sabersim projections + correlations, (2) enumerates and prunes a very large set of Showdown lineups, writes them to Parquet with progress/timing, and (3) grades those lineups’ top-1% finish rates using the existing `src/shared/top1pct_core.py` simulation/field-threshold logic (explicit field model) in chunked matrix multiplies.
todos:
  - id: script-skeleton
    content: Create a scratch Python script with argparse, logging/timing helpers, and I/O wiring (Sabersim CSV + corr workbook -> internal arrays -> parquet writer).
    status: pending
  - id: candidate-enum
    content: Implement MITM enumeration (pairs+triples) with pruning score, per-CPT quotas, and batch Parquet writing with progress/ETA.
    status: pending
  - id: explicit-field-thresholds
    content: Construct required DataFrames from Sabersim inputs and reuse `build_quota_balanced_field()` + `np.partition` thresholds (explicit field model).
    status: pending
  - id: chunked-grading
    content: Read candidate Parquet in chunks, build W_chunk, compute scores via matrix multiply against simulated X, and write graded outputs with progress/ETA.
    status: pending
  - id: sanity-perf
    content: Add validation checks (name alignment, ownership sums, salary legality) and expose key performance knobs; document an example invocation.
    status: pending
---

# Mass lineup enumeration + top1% scratch script

## Goal

Build a **separate, one-off script** (not integrated into `src/`) that can:

- Read **Sabersim projections CSV** (with salary/projection/std/ownership) and an existing **correlation workbook** (`Sabersim_Projections` + `Correlation_Matrix`).
- Generate **hundreds of thousands to millions** of candidate Showdown lineups using **cheap pruning** designed to preserve “weird”/high-tail lineups.
- Save candidates to **Parquet** (fast, columnar, chunked writes).
- Grade candidates’ **top 1% finish rate** using your existing simulation approach in `src/shared/top1pct_core.py`, with the **explicit** field model.
- Print **timing + progress** throughout.

## Inputs / outputs

- **Inputs**
- `--sabersim-csv PATH`: projections (FLEX-style rows; script will treat CPT as 1.5x).
- `--corr-xlsx PATH`: correlation workbook with:
- `Correlation_Matrix` (used for `corr_full`)
- `Sabersim_Projections` (optional; can be used to cross-check names)
- Ownership columns in the Sabersim CSV:
- Provide `--cpt-own-col` and `--flex-own-col`.
- Default autodetect: try common names like `cpt_ownership`, `CPT_Ownership`, `CPT Own%`, and likewise for flex.
- **Outputs**
- `--out-parquet PATH`: candidate lineups + metadata.
- `--out-graded-parquet PATH` (or `--out-graded-csv`): same lineups plus `top1_pct_finish_rate`.

## Reuse of existing code (no integration yet)

Use these functions/patterns from `src/shared/top1pct_core.py`:

- `_build_full_correlation(universe_names, corr_df)` to map your correlation matrix onto the player universe.
- `_simulate_player_scores(mu, sigma, corr_full, num_sims, rng)` to draw correlated outcomes.
- `_build_lineup_weights(lineups_df, player_index)` **or** replicate its tiny behavior (recommended for speed) to build W for chunked Parquet lineups.
- For explicit field thresholds, mimic the same logic used in `run_top1pct()`:
- build `field_lineups_df` via `src/shared/field_builder.py: build_quota_balanced_field()`
- compute `scores_field = X @ W_field.T`
- compute per-sim threshold via `np.partition(..., q_index)` (see `top1pct_core.py` around the explicit field branch).

Important: `build_quota_balanced_field()` expects `ownership_df`, `sabersim_proj_df`, `lineups_proj_df`, and `corr_df`. In the scratch script you’ll construct these DataFrames from the Sabersim CSV + corr workbook:

- `ownership_df` columns: `player, cpt_ownership, flex_ownership` (percent units as expected by `top1pct_core`).
- `sabersim_proj_df` columns at least: `Name, My Proj` (+ optional `dk_std`, `Pos`).
- `lineups_proj_df` columns: `Name, Team, Salary, My Proj` (+ optional `dk_std`, `Pos`).

## Candidate lineup generation algorithm (fast + preserves “weird”)

### A) Preprocess player pool

From Sabersim CSV:

- Keep players with `My Proj > 0` (consistent with `src/shared/lineup_optimizer.py` filtering).
- Build arrays by player index:
- `name[i]`, `team[i]`, `salary[i]`, `mu[i]`, `sigma[i] `(use `dk_std` if present else fallback, like `top1pct_core.py` does), `pos[i]`.
- `cpt_own[i]`, `flex_own[i]` from the specified ownership columns.
- Load `corr_df` from the correlation workbook and build `corr_full` over the Sabersim player list using `_build_full_correlation`.

### B) Enumerate with meet-in-the-middle (MITM)

For each CPT candidate `c`:

- Remaining salary cap: `B = 50000 - round(1.5 * salary[c])`.
- Consider FLEX pool: all players excluding `c`.
- MITM approach:
- Precompute all FLEX **pairs** and **triples** once (global), storing:
- bit/tuple of indices, salary sum, mean sum, quick ownership proxy, and (optional) precomputed internal pairwise-corr penalty.
- For CPT `c`, iterate over eligible pairs `p` (excluding `c`) and match with triples `t` such that:
- disjoint indices
- `salary[p] + salary[t] <= B`
- team constraint: at least 1 from each team across 6 (consistent with optimizer base constraint)
- This yields huge counts quickly without recursion.

### C) Cheap pruning criteria (keep recall)

Apply pruning as **filters + a ranking score** (avoid hard “looks normal” rules):

- **Hard filters** (cheap and safe)
- Salary cap and 6 distinct players.
- Team presence: at least 1 from each team.
- Optional: remove lineups with more than `max_neg_pairs_hard` pairs where corr < `neg_corr_hard` (e.g. any corr < -0.40).
- **Soft / ranking-based pruning** (preserve weird)
- Compute lineup mean `mu_L`.
- Compute lineup variance `sigma_L^2` using 6-player covariance from `corr_full` + `sigma` (15 pair terms + diag).
- Compute “extreme negative pairs” count: number of pairs with corr < -0.25 (tunable).
- Compute ownership leverage proxy: `sum(log(max(own, eps)))` using CPT ownership for CPT and per-slot flex ownership for flex.
- Compute a **tail score** for keeping:
- `score = mu_L + k * sigma_L - penalty(neg_pairs) - penalty(chalk)`
- Keep top `K_per_cpt` per CPT (and also per salary bin) to ensure diversity.

### D) Chunked Parquet writing with progress

Write candidates in batches (e.g. 200k–1M rows):

- Columns to write:
- `cpt`, `flex1..flex5` (names)
- numeric meta: `lineup_salary`, `mu_L`, `sigma_L`, `neg_pairs`, `own_logsum`, `score`
- optional: `stack` (team counts like `4|2`)
- Print progress:
- per CPT: candidates scanned, kept, rate lineups/sec
- per batch: rows written, total rows, elapsed, ETA

## Top 1% grading step (explicit field model, chunked)

### A) Simulate player outcomes once

- Draw `X` of shape `(S, P)` via `_simulate_player_scores(mu, sigma, corr_full, num_sims, rng)`.
- Apply the same flooring behavior as `run_top1pct()` if you want parity:
- floor to 0 for non-`DST`/`K` positions (see `top1pct_core.py` behavior).

### B) Build explicit field thresholds

- Build required DataFrames from Sabersim CSV + corr workbook and call:
- `build_quota_balanced_field(field_size=..., ownership_df=..., sabersim_proj_df=..., lineups_proj_df=..., corr_df=...)`
- Build `W_field` and compute `scores_field = X @ W_field.T`.
- For each sim, set threshold using the same `np.partition` approach:
- `q_index = floor(0.99 * K_field)`
- `thresholds = partition(scores_field, q_index)[:, q_index]`

### C) Grade candidate lineups from Parquet in chunks

- Scan candidate Parquet in chunks (row groups or fixed batch size).
- Build a weight matrix for each chunk:
- `W_chunk` shape `(K_chunk, P)` with 1.5 on CPT index and 1.0 on each flex index.
- (For speed, avoid pandas row iteration; build via vectorized index mapping + sparse-ish assembly.)
- Compute `scores = X @ W_chunk.T` producing `(S, K_chunk)`.
- Compute `p_top1 = mean(scores >= thresholds[:, None], axis=0)`.
- Append `top1_pct_finish_rate = 100 * p_top1` to output Parquet/CSV.
- Print progress per chunk: sims/sec, lineups/sec, ETA.

## Timing / logging

Add a small utility `Timer` context manager (in the script) to print elapsed seconds for:

- loading inputs
- building correlations
- precomputing pairs/triples
- per-CPT enumeration
- parquet write batches
- simulation generation
- field building + threshold computation
- per grading chunk

## Script location and invocation (scratch)

- Put the script outside the library code, e.g. `scratch/explore_mass_lineups.py` (or `scripts/`), and run it directly.
- Example CLI:
- `python scratch/explore_mass_lineups.py --sabersim-csv ... --corr-xlsx ... --cpt-own-col ... --flex-own-col ... --out-parquet ... --field-size 250000 --num-sims 20000 --K-per-cpt 50000 --max-lineups 1000000`

## Guardrails / sanity checks

- Validate name alignment:
- Warn if Sabersim player names are missing from `corr_df.index` (those players will be treated as independent if you rely on `_build_full_correlation`).
- Validate ownership sums:
- If CPT ownership sums are far from ~1.0 (or 100%), warn.
- Validate generated lineup legality:
- salary <= 50000 and 6 distinct players.

## Expected performance knobs

- `--K-per-cpt` and optional salary/ownership bin quotas (controls candidate count)
- `--num-sims` and grading chunk size (controls runtime)
- `--field-size` and explicit field lineup count (controls threshold quality)