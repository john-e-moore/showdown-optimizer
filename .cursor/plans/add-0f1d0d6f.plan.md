<!-- 0f1d0d6f-c004-4184-bca2-9461a3c33a39 3bdc23c9-cf79-43fc-bddd-58003003be59 -->
# NBA box-score–based correlation plan

### Goal

Make the NBA correlation pipeline mirror the NFL one: simulate NBA **box-score stats** from Sabersim projections, convert them to **DK fantasy points**, then compute empirical **DK-point correlations** between all players.

### Step 1: Analyze existing NFL flow and NBA inputs

- **Review NFL correlation stack**:
- `src/build_corr_matrix_from_projections.py` (Sabersim → canonical features for ML/historical, may reuse ideas but not ML part).
- `src/simulation_corr.py` (stat-level simulator + DK scoring for NFL).
- `src/main.py` (NFL correlation CLI wiring).
- **Catalog NBA Sabersim columns** from the actual CSV (already seen: `PTS`, `Min`, `RB`, `AST`, `STL`, `BLK`, `TO`, 2PT/3PT-related columns, plus percentile/`dk_points` fields) and decide which will drive the NBA stat model.

### Step 2: Design NBA stat and scoring model

- **Define a minimal NBA stat schema** for simulation (per player, per game):
- Core scoring stats: `pts`, `reb`, `ast`, `stl`, `blk`, `tov`, and optionally `fg3m`.
- Pull per-player *expected* values from Sabersim columns (`PTS`, `RB`, `AST`, etc.).
- **Specify DK scoring function** for NBA Showdown (simplified first pass):
- Implement a deterministic function `compute_dk_points_nba(pts, reb, ast, stl, blk, tov, fg3m)` in a new NBA module.
- Optionally include standard DK bonuses (double-double, triple-double, 3PM bonus) as a second iteration once the core is working.
- **Choose simulation family**:
- For a first version, model each stat as an independent positive random variable per player (e.g., Poisson or Gamma with mean equal to the Sabersim expectation), ignoring intra-team stat sharing.
- Document that this is less structured than NFL but still produces a meaningful joint DK distribution once correlated via the covariance implied by the Sabersim-derived stats.

### Step 3: Implement NBA stat simulator

- **Create `src/nba/stat_sim.py`** (or similar) that:
- Defines a function `simulate_nba_stats_from_projections(df, n_sims, rng)` that:
- Takes FLEX-only Sabersim NBA projections (`Name`, `Team`, `PTS`, `RB`, `AST`, `STL`, `BLK`, `TO`, optionally `3PT`/`2PT` split).
- For each player and each stat, samples `n_sims` draws from a parametric distribution (e.g., Poisson with lambda equal to projected stat, or a scaled Gamma/Normal with truncation at 0).
- Returns arrays of shape `(n_players, n_sims)` for each stat.
- **Add an NBA DK scoring helper** in the same module:
- `simulate_nba_dk_points(df, n_sims, rng)` that:
- Calls the stat simulator.
- Applies the DK scoring function to get a DK-points matrix `(n_players, n_sims)`.

### Step 4: Build NBA correlation matrix from simulated DK points

- **Create `src/nba/corr_from_sim.py`** (or fold into `stat_sim.py`) that:
- Takes the DK-points matrix `(n_players, n_sims)` and player names.
- Drops players with zero variance across sims.
- Computes `np.corrcoef` across players, clamps NaNs/inf, and fills diagonal with 1.0.
- Returns a `pd.DataFrame` indexed/columned by player name.
- **Wire this into the NBA pipeline** by updating `src/nba/simulation_corr.py` to:
- Call the new NBA stat+scoring simulator instead of the generic DK-pool Dirichlet engine.
- Keep the same public API (`simulate_corr_matrix_from_projections(sabersim_df, n_sims, random_seed)`), so `src/nba/main.py` doesn’t change.

### Step 5: Validate NBA correlations and compare to current approach

- **Run the NBA correlation CLI** on the OKC–GSW slate and inspect:
- Distribution of simulated DK means vs `My Proj` per player.
- Sanity of correlation signs/magnitudes (e.g., high-usage teammates slightly positively correlated; bench players with low mins closer to independent or weak correlations).
- **Compare against the old generic DK-only model**:
- For a few players, compare row/column of the old vs new `Correlation_Matrix` to understand the impact of stat modeling.
- Ensure there are no pathological values (NaN, inf, all zeros) and that the Excel output structure matches NFL (same sheet names, etc.).

### Step 6: Optional refinements and documentation

- **Refine NBA stat model** based on behavior:
- Introduce simple team-level or pace-based coupling (e.g., correlate teammates’ minutes or points using shared game-level variance) if necessary.
- Add DK bonuses (3PM, double-double, triple-double) if they materially change tail behavior.
- **Document the NBA pipeline behavior**:
- Update `README.md` with a short “NBA correlations” section explaining that NBA correlations are now built from simulated box-score stats and DK scoring, mirroring the NFL process.
- Briefly note any simplifications vs NFL (e.g., fewer team-level constraints in the stat simulator at first).

### To-dos

- [ ] Review NFL correlation implementation and catalog which NBA Sabersim columns to use for stat simulation.
- [ ] Define the NBA stat schema (PTS, RB, AST, STL, BLK, TOV, FG3M, etc.) and a DK scoring function for Showdown.
- [ ] Implement nba/stat_sim.py to sample per-player stat lines from Sabersim projections and convert them to DK points.
- [ ] Implement an NBA-specific correlation builder that computes corr(DK_i, DK_j) from simulated DK-point matrices and wire it into src/nba/simulation_corr.py.
- [ ] Run the updated NBA pipeline on the OKC–GSW slate, inspect projections vs simulated means and spot-check correlations for sanity.
- [ ] Optionally refine the NBA stat model (bonuses, team coupling) and update README with a short description of the NBA correlation process.