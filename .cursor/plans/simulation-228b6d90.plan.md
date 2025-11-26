<!-- 228b6d90-d2fd-4281-a792-7538dc177e08 0b42896f-be6b-4c5d-8074-5aad290920cc -->
# Simulation-based Correlation Refactor

## Goals

- Replace the current ML correlation model at inference time with a **Monte Carlo simulator** that uses Sabersim box-score projections to generate many joint game outcomes and compute a correlation matrix of DraftKings fantasy points.
- Keep the existing historical/ML code available for research, but make the **default Showdown path** purely simulation-based and consistent with team-level constraints.

## 1. Configuration and interfaces

- **Extend config** in [`src/config.py`](/home/john/showdown-optimizer/src/config.py):
- Add simulation parameters, e.g. `SIM_N_GAMES` (default 5000), `SIM_RANDOM_SEED`, and a few concentration/variance knobs (e.g. `SIM_DIRICHLET_K_YARDS`, `SIM_DIRICHLET_K_RECEPTIONS`, `SIM_DIRICHLET_K_TDS`).
- Optionally add an enum/flag like `CORR_METHOD = "simulation"` or CLI-exposed `--corr-method` in `main.py`, defaulting to `simulation` but leaving room to re-enable the ML model.
- **Define a clear simulator API**, e.g. in a new module `src/simulation_corr.py`:
- `simulate_corr_matrix_from_projections(sabersim_df: pd.DataFrame, n_sims: int) -> pd.DataFrame` that returns a player-by-player correlation matrix of DK points.

## 2. Map Sabersim projections to canonical stat inputs

- In a new helper (either in `simulation_corr.py` or a small `sabersim_loading.py`), build structured inputs from Sabersim:
- Parse per-player projected stats from the CSV (beyond what `build_corr_matrix_from_projections.py` currently uses): `Pass Yds`, `Pass TD`, `Rush Yds`, `Rush TD`, `Rec`, `Rec Yds`, `Rec TD`, and `My Proj`.
- Map these to canonical columns used by the scoring rules: `pass_yards`, `pass_tds`, `rush_yards`, `rush_tds`, `rec_yards`, `rec_tds`, `receptions`.
- For each team, compute **team-level projection totals** for each stat by summing player projections (e.g. `team_rec_yards = sum_i rec_yards_i`, `team_rush_yards`, `team_pass_tds`, etc.).
- Identify the primary **QB(s)** for each team (e.g. players with non-zero `Pass Yds` or `Pos == 'QB'`) and mark them so pass stats can be tied correctly to them.

## 3. Team-level simulation design (simple pools, no play-level model)

- **Per-team, per-simulation game state:**
- For each offensive stat type, build a simple pool model:
- Yards stats (`pass_yards`, `rush_yards`, `rec_yards`): treat team totals as either fixed at the sum of projections or with a modest team-level random multiplier (e.g. lognormal around 1 with config-driven variance).
- TD stats (`pass_tds`, `rush_tds`, `rec_tds`): use Poisson or Negative Binomial draws with means equal to the team projected totals, then treat the realized count as the total TD pool for that stat.
- Receptions: either fixed at projected team total or drawn from a Poisson with that mean.
- **Allocate team totals to players using random shares:**
- For each stat type S and team:
- Compute expected shares `w_i^S = proj_i_S / max(sum_team_S, eps)` for each player on that team.
- Sample a random share vector `p^S` from a Dirichlet distribution with parameters `alpha_i = k_S * w_i^S` (k from config) to introduce variability around projections.
- For continuous stats (yards), set `stat_i = p_i^S * team_total_S`.
- For count stats (TDs, receptions), either:
- Draw team count (e.g. `k_TD ~ Poisson(team_total_TD)`) and allocate integer outcomes via a multinomial with probabilities `p^S`, or
- Approximate with `round(p_i^S * team_total_S)` and adjust to preserve the total.
- **Enforce key football consistency constraints:**
- For each team and simulation:
- Ensure `QB_pass_yards = team_rec_yards` (or very close) by:
- Simulating `team_pass_yards`, allocating `rec_yards` across receivers via Dirichlet, and then setting QB `pass_yards` equal to the sum of receiver `rec_yards`.
- Ensure `team_pass_tds == team_rec_tds` and assign that TD count to the team QB (or split if multiple QBs), while allocating receiving TDs to receivers via a multinomial over their TD shares.
- Rushing stats: draw `team_rush_yards` and `team_rush_tds`, then allocate to RBs/QBs/WRs as a shared pool via Dirichlet, so RB–QB rushing correlations emerge naturally from shared team-level variance.

## 4. Per-simulation fantasy scoring and correlation

- After generating simulated **box-score stats** per player for a single simulated game:
- Assemble a DataFrame with canonical columns (`pass_yards`, `pass_tds`, `interceptions`, `rush_yards`, `rush_tds`, `rec_yards`, `rec_tds`, `receptions`).
- Reuse the existing DK scoring logic by either:
- Calling `fantasy_scoring.compute_dk_points_offense` on this per-simulation DF, or
- Implementing a light, vectorized scoring function inside `simulation_corr.py` that mirrors the same rules.
- Repeat this game simulation **`SIM_N_GAMES` times**:
- Build a 2D array of shape `(n_players, n_sims)` of simulated DK points.
- Compute the empirical correlation matrix across players `corr_mat = np.corrcoef(points, rowvar=True)`.
- Convert to a `pd.DataFrame` with player names as both index and columns, and explicitly set the diagonal to 1.0 for numerical stability.

## 5. Integrate simulator into the main pipeline

- In [`src/main.py`](/home/john/showdown-optimizer/src/main.py):
- Add CLI options such as `--corr-method {simulation,ml}` and `--n-sims` (defaulting to config values), but make `simulation` the default path.
- On the simulation path:
- Load the Sabersim CSV using the existing `load_sabersim_projections` (or a small refactor to share a loader between `build_corr_matrix_from_projections.py` and `simulation_corr.py`).
- Call `simulate_corr_matrix_from_projections` to get the correlation matrix, bypassing the ML model entirely.
- Keep the existing ML-based `build_corr_matrix_from_sabersim` available under a different branch (e.g. `--corr-method ml`) for comparison, without changing the training pipeline.
- Update the Excel output step to use the **simulation-based correlation matrix** when `corr-method=simulation`.

## 6. Diagnostics and sanity checks

- Add a lightweight diagnostics function in `simulation_corr.py` that, for a single run:
- Compares per-player simulated mean DK points vs `My Proj` (scatter or summary stats) to ensure simulations are centered near projections.
- Prints or logs sample correlations for a few key pairs (e.g., QB–WR1, QB–RB1, RB1–WR1, low-usage WR–QB) for a test slate.
- Optionally write JSON summaries (similar to existing diagnostics) into `diagnostics/simulation/` showing:
- Distribution of team totals vs projected totals.
- Distribution of correlation values by position-pair type (QB–WR, QB–RB, WR–WR, RB–RB, etc.).

## 7. Documentation

- Update [`README.md`](/home/john/showdown-optimizer/README.md):
- Document the new simulation-based correlation method, its assumptions (team-level pools, Dirichlet allocation, Poisson TDs), and tunable config knobs.
- Explain how to run the pipeline with simulation (default) vs ML (`--corr-method ml`) and how to adjust `--n-sims` for speed vs stability.
- Call out that 0-projection players can now be handled more naturally (e.g., by giving them tiny but non-zero shares, or explicitly dropping them from the correlation matrix if desired).

### To-dos

- [ ] Add simulation-related configuration parameters and optional correlation method flag in src/config.py (and wire a matching CLI flag in main.py).
- [ ] Implement a helper to map Sabersim projection columns (yards, TDs, receptions) into canonical stat fields and compute team-level projected totals per stat and per team.
- [ ] Implement simulation engine in a new module (e.g., src/simulation_corr.py) that simulates team-level stat pools and allocates them to players with Dirichlet/multinomial sampling, enforcing basic passing and receiving consistency constraints.
- [ ] Integrate fantasy scoring into the simulator, run many simulated games, and compute the empirical player-by-player correlation matrix of DK points.
- [ ] Wire the simulator into main.py so the Showdown correlation matrix is built via simulation by default, with an optional ML fallback path for comparison.
- [ ] Add diagnostics to validate that simulated DK means are close to projections and that key player pairs (e.g., QB–WR, QB–RB) have sensible correlation signs and magnitudes.
- [ ] Update README.md to describe the simulation-based correlation method, configuration, and how to run it.