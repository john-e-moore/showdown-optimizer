---
name: add-k-and-dst-correlation
overview: Extend the NFL Showdown correlation pipeline to include kicker and DST players with simple but realistic DK point simulations, and integrate them cleanly into the top1% pipeline.
todos:
  - id: include-dst-in-input
    content: Adjust NFL Sabersim projections loading so K and DST rows are preserved and flow into the correlation simulator, and confirm consistency with the optimizer’s Sabersim handling.
    status: completed
  - id: implement-k-simulation
    content: Extend the NFL correlation simulator to assign simulated DK points to kickers based on their Sabersim projections (and optionally team offense) so they have realistic means and variance.
    status: completed
    dependencies:
      - include-dst-in-input
  - id: implement-dst-simulation
    content: Extend the NFL correlation simulator to assign DST DK points per simulation as a noisy, negatively-related function of opponent offensive DK output, calibrated to match Sabersim projections on average.
    status: completed
    dependencies:
      - include-dst-in-input
  - id: validate-top1pct-integration
    content: Regenerate correlations for a test slate, run the full pipeline including top1% estimation, and validate that K and DST appear in the correlation matrix and downstream outputs with sensible diagnostics.
    status: completed
    dependencies:
      - implement-k-simulation
      - implement-dst-simulation
---

# Add K and DST to NFL correlation matrix

## Goal

Extend the NFL Showdown correlation pipeline so that both kickers (K) and defenses (DST) are included in the simulated correlation matrix, using:

- A simple kicker DK model tied to their Sabersim projection (and optionally team offense).
- A simple DST DK model as a noisy, negatively-related function of opponent offensive performance.
All changes should remain compatible with existing optimizer and top1% code paths.

## Step 1: Ensure K and DST flow into the correlation pipeline

- **Review position filtering** in [`src/nfl/build_corr_matrix_from_projections.py`](src/nfl/build_corr_matrix_from_projections.py) and [`src/nfl/config.py`](src/nfl/config.py):
- Confirm how `OFFENSIVE_POSITIONS` is used to filter Sabersim rows before simulation.
- **Adjust filters to include DST**:
- Either extend `OFFENSIVE_POSITIONS` to include `"DST"` or relax the filter in the projections loader so that K and DST rows reach the simulator.
- **Verify optimizer consistency** by checking [`src/shared/lineup_optimizer.py`](src/shared/lineup_optimizer.py) and [`src/nfl/showdown_optimizer_main.py`](src/nfl/showdown_optimizer_main.py) to ensure K/DST naming and positions match what the correlation side will see (e.g., `Pos == "K"` / `"DST"`).

## Step 2: Add kicker DK simulation to `simulation_corr`

- **Identify kicker rows** in [`src/nfl/simulation_corr.py`](src/nfl/simulation_corr.py):
- After `_prepare_simulation_players`, create a mask for `position == "K"` and keep their Sabersim `dk_proj` values.
- **Design a simple kicker DK model**:
- For each kicker and simulation, generate DK points as a noisy variable with mean equal to its Sabersim `dk_proj`.
- Optionally scale and/or correlate this with the simulated team offensive DK total (e.g., more team scoring → slightly higher kicker expectation).
- **Implement kicker DK assignment**:
- After offensive DK points are computed for all players in each sim, overwrite the DK values for rows where `position == "K"` using the kicker model (so they have non-zero variance and appropriate magnitude).
- Keep the implementation minimal (e.g., Normal or lognormal around the projection, with capped negatives) and controlled by a small number of constants to tune later.
- **Keep diagnostics intact**:
- Make sure the kicker rows appear in `dk_sim_vs_proj` and `dk_points_first_100.csv` with reasonable means and std devs.

## Step 3: Add DST DK simulation as a function of opponent offense

- **Identify DST rows and their opponents**:
- In `_prepare_simulation_players`, or just after it, construct a mask for `position == "DST"` and map each DST to its opposing team name for the slate.
- **Define a simple DST DK model tied to opponent offense**:
- For each DST and simulation, compute opponent offensive DK (sum of offensive players on the opposing team in that sim).
- Define DST DK as a noisy function of that quantity, for example: `DST_DK = a + b * (-opp_offense_dk) + noise`, where `a`/`b`/noise are chosen so that the expected value equals the DST’s Sabersim `dk_proj` and variance is in a reasonable range.
- **Implement DST DK assignment** in the main simulation loop:
- After offensive DK points are computed and before collecting `dk_points` into the big matrix, compute DST DK values per sim and write them into the appropriate rows.
- Ensure the implementation handles edge cases (very low/high opponent offense, small or negative projections) and clips to a sane DK range if needed.
- **Verify DST presence and behavior**:
- Check diagnostics (`sim_input_players`, `dk_sim_vs_proj`, `dk_points_first_100.csv`) to confirm DST players have:
- Non-zero variance
- Means close to their projections
- Negative correlation with opponent offensive stars where expected.

## Step 4: Wire through to top1% and validate end-to-end

- **Confirm integration with `top1pct_core`** in [`src/shared/top1pct_core.py`](src/shared/top1pct_core.py):
- Ensure that K and DST appear in `Sabersim_Projections` and that `_build_player_universe` includes them (using the existing `Pos`/fallback logic).
- Verify the position-based flooring logic (`non_dst_k_mask`) already treats K/DST correctly (i.e., they are exempt from the "floor at 0" rule).
- **Sanity-check implications on field modeling**:
- Confirm that the ownership pipeline and field builder can handle K and DST without special cases (they’re just additional players with projections and ownership).
- **Run and compare diagnostics**:
- Rebuild a correlations workbook for a test slate and examine the resulting `Correlation_Matrix` to ensure K and DST are present.
- Run `run_full.sh` on a sample slate and verify that top1% estimation completes with K and DST included, and that no `"not found in the player universe"` errors occur.
- Inspect some sample correlations (e.g., DST vs opposing QB) to check directionality and rough magnitudes are sensible.

## Step 5: Optional tuning and configuration

- **Expose a few tuning knobs** in [`src/nfl/config.py`](src/nfl/config.py) for K and DST simulation (e.g., variance multipliers, coefficients relating DST DK to opponent offense) so behavior can be calibrated later.
- **Iterate using flashback slates and contest histories** to refine K/DST variance and correlation structure if early diagnostics show obvious miscalibration (e.g., unrealistically high DST variance or correlations).