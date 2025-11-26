<!-- 228b6d90-d2fd-4281-a792-7538dc177e08 785b5ff7-c607-4388-9672-1051f71e2431 -->
## Fixing zero DK simulation outputs and adding richer diagnostics

### Diagnosis (what’s likely wrong)

- **Root symptom**: `dk_sim_mean` and `dk_sim_std` are identically 0 for all players, which implies the `dk_points` matrix in `simulate_corr_matrix_from_projections` is all zeros.
- From the code, `dk_points` only gets non-zero entries via `_simulate_team_offense`, which in turn only produces non-zero stats when team-level totals (`team_tot_rec_yards`, `team_tot_receptions`, `team_tot_rush_yards`, `team_tot_rush_tds`, `team_tot_rec_tds`) are positive and finite.
- The most probable cause is that, for this Sabersim file, the canonical stat columns (`rec_yards`, `receptions`, `rush_yards`, etc.) or their team totals end up as **all zeros or NaNs**, so the simulator never allocates any yards/TDs to players in any simulation.
- To confirm and fix this robustly, we should:
- Snapshot the canonical per-player stat inputs used by the simulator.
- Snapshot the per-team `team_tot_*` values used inside `_simulate_team_offense`.
- Inspect a slice of the raw `dk_points` matrix (first 100 sims) to see if it’s truly all zeros.

### Planned changes

1. **Snapshot canonical stat inputs for the simulator**

- In `_prepare_simulation_players` (in `src/simulation_corr.py`), after building the per-player dataframe `df` and `team_infos`, add a diagnostics snapshot via `diagnostics.write_df_snapshot`:
 - Include columns: `player_name`, `team`, `position`, `dk_proj`, and canonical stat columns `pass_yards`, `pass_tds`, `rush_yards`, `rush_tds`, `rec_yards`, `rec_tds`, `receptions`.
 - Write under step `"simulation"` with a name like `"sim_input_players"` so you can inspect whether projections are flowing correctly from the CSV into the simulator.

2. **Snapshot per-team totals used by the simulator**

- Still in `_prepare_simulation_players`, construct a small dataframe from `team_infos` with one row per team:
 - Columns: `team`, `team_tot_rec_yards`, `team_tot_rec_tds`, `team_tot_receptions`, `team_tot_rush_yards`, `team_tot_rush_tds`, and a count of players per team.
- Write this via `diagnostics.write_df_snapshot(..., name="sim_team_totals", step="simulation")` to verify that each team has the expected non-zero totals.

3. **Write a wide DK-points matrix for the first 100 simulations**

- In `simulate_corr_matrix_from_projections`, after the main simulation loop (once `dk_points` is fully populated) and before computing correlations:
 - Extract the first `n_debug_sims = min(100, n_sims)` columns of `dk_points`.
 - Build a dataframe with one row per player and columns:
 - `player_name`, `team`, `position`.
 - `sim_0`, `sim_1`, ..., `sim_{n_debug_sims-1}` corresponding to DK points in each of the first `n_debug_sims` simulations.
 - Write this to `diagnostics/simulation/dk_points_first_100.csv` (or similar) for easy inspection in Excel.

4. **Adjust simulation logic once diagnostics are inspected**

- After you inspect `sim_input_players`, `sim_team_totals`, and `dk_points_first_100.csv`:
 - If canonical stat columns or team totals are indeed all zeros/NaNs, fix the **stat mapping** in `_prepare_simulation_players` so they are sourced correctly from the Sabersim CSV (or fall back to deriving team totals from `dk_proj` when all stat projections are zero).
 - If the inputs look correct but `dk_points_first_100.csv` is still all zeros, further instrument `_simulate_team_offense` to snapshot one simulated game’s box-score stats per team and confirm that yards/TDs are being allocated; then adjust any conditional logic (`> 0` checks, handling of NaNs) that is preventing non-zero outputs.

5. **Keep existing DK vs projection summary**

- Preserve the existing `dk_sim_vs_proj` snapshot and CSV so you can compare `dk_sim_mean` and `dk_sim_std` before and after any fixes.

Once these diagnostics are in place and show non-zero simulated stats, we can iterate on tuning (e.g., Dirichlet concentration, team-level variance) separately from this zero-output bug.

### To-dos

- [ ] Add a diagnostics snapshot in _prepare_simulation_players capturing per-player canonical stat inputs (team, position, dk_proj, pass/rush/rec stats).
- [ ] Add a diagnostics snapshot summarizing per-team stat totals used by the simulator (team_tot_rec_yards, team_tot_receptions, team_tot_rush_yards, etc.).
- [ ] In simulate_corr_matrix_from_projections, write a CSV with the first 100 simulated DK outcomes per player (player_name, team, position, sim_0..sim_99).
- [ ] After inspecting the new diagnostics, adjust stat mapping and/or simulator conditionals so that simulated DK points reflect the Sabersim stat projections and dk_sim_mean/dk_sim_std are non-zero where expected.