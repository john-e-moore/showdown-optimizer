---
name: quota-balanced-top1pct-integration
overview: Integrate a quota-balanced random field lineup builder into the shared top1pct step so both NFL and NBA pipelines can optionally simulate fields via explicit lineups instead of the current ownership-mixture approximation.
todos: []
---

# Quota-Balanced Field Builder Integration into Top1pct

## Goals

- **Add a shared quota-balanced field lineup builder** that uses ownership projections, projections, and correlations to generate a realistic field of lineups.
- **Integrate it into `top1pct_core`** so NFL and NBA `top1pct` CLIs can optionally use explicit field lineups to estimate top 1% finish rates.
- **Preserve current behavior as the default**, with a clear flag/API to switch to the new method when desired.

## High-Level Design

- **New shared module**: create `src/shared/field_builder.py` (or similar) that implements the quota-balanced random builder.
- Expose a primary function like `build_quota_balanced_field(...) -> pd.DataFrame` returning a `Lineups`-style dataframe (CPT + 5 FLEX, salary, projections, entrant type tag, etc.).
- Take inputs directly from the already-loaded `lineups`, `ownership`, and correlation workbooks (player names, projections, salaries, ownership, corr matrix, positions, and `field_size`).
- **Extend `top1pct_core.run_top1pct`** with a strategy switch:
- Add an enum/string flag `field_model` with options: `"mixture"` (current ownership mixture) and `"explicit"` (quota-balanced field lineups).
- When `field_model == "mixture"`, keep existing `_compute_field_thresholds` path unchanged.
- When `field_model == "explicit"`, call the field builder to generate `field_lineups_df`, simulate scores for both your candidate lineups and the field lineups, and compute empirical 99th-percentile thresholds from the simulated field scores.
- **Wire sport-specific CLIs** (`src/nfl/top1pct_finish_rate.py`, `src/nba/top1pct_finish_rate_nba.py`) to pass the `field_model` flag through from the command line.

## Detailed Steps

### 1. Implement shared quota-balanced field builder

- **Inputs & data extraction** (inside `field_builder.py`):
- Accept: `field_size`, `ownership_df`, `sabersim_proj_df`, `lineups_proj_df`, `corr_df`, and a random seed; optionally accept a config object for tuning.
- From ownership, build per-player CPT and FLEX ownership targets (`p_cpt`, `p_flex`) as fractions; compute quotas `T_CPT[j] = round(p_cpt[j] * field_size)`, `T_FLEX[j] = round(p_flex[j] * field_size)`.
- From projections and correlation inputs, derive `proj[j]`, `salary[j]`, `pos[j]`, `team[j]`, and a correlation look-up `corr(j, k)` consistent with `top1pct_core`’s player universe.
- **Entrant types and hyperparameters**:
- Start with a **single entrant type** (one set of `alpha, beta, gamma, delta`) to match the prompt’s method but keep initial complexity low.
- Hard-code initial defaults in the builder (e.g. `alpha=1.0, beta=1.0, gamma=0.5, delta=0.5`) and leave room to extend later to multiple entrant types.
- **CPT selection**:
- Maintain remaining quotas `R_CPT[j]`, `R_FLEX[j]`.
- For each new field lineup, compute weights `w_CPT[j]` based on (projection/salary) value, remaining CPT quota, and soft rules derived from `pos[j]` (boost QB/RB/WR, penalize K/DST), skipping players with `R_CPT[j] <= 0`.
- Sample CPT proportional to `w_CPT[j] `and decrement `R_CPT`.
- **FLEX selection (5 slots)**:
- Fill slots sequentially, tracking current lineup `L`, remaining salary, and team composition.
- For each slot, enumerate candidates satisfying: not already in `L`, `R_FLEX[j] > 0`, legal positions/teams, and plausible salary feasibility.
- Compute weights `w_FLEX[j] `based on value, remaining FLEX quota, correlation with existing players (`exp(gamma * sum corr(j,k))`), and soft rule penalties (e.g. encourage common stack patterns, discourage extreme salary leaves).
- Sample a player, add to lineup, decrement `R_FLEX`.
- Implement simple dead-end handling: if no candidates, temporarily ignore quotas and/or some rules; only restart the lineup as a last resort.
- **Field generation & output schema**:
- Repeat CPT + FLEX sampling `field_size` times to produce `N` field lineups.
- Produce a `DataFrame` with columns mirroring the optimizer `Lineups` sheet (`cpt`, `flex1`..`flex5`, `salary`, maybe `lineup_id` and `entrant_type`).
- Optionally compute realized CPT/FLEX ownership and log summary diagnostics, but keep I/O minimal so the builder can be used inside `top1pct_core`.

### 2. Integrate explicit field model into `top1pct_core`

- **API extension**:
- Update `run_top1pct` signature to accept `field_model: str = "mixture"` and a small `field_model_config` dict for tuning the builder if needed.
- Leave all existing call sites valid by defaulting `field_model` to the current mixture behavior.
- **Refactor field modeling**:
- Extract the current mixture logic in `_compute_field_thresholds` into a clearly named branch (e.g. `if field_model == "mixture": ...`).
- Add a new branch `if field_model == "explicit":` that:
  - Calls the quota-balanced builder to get `field_lineups_df` using the same player universe / ownership inputs already loaded in `_build_player_universe`.
  - Builds a weight matrix `W_field` for the field lineups using `_build_lineup_weights` or a small adaptation.
  - Uses the same simulated player score matrix `X` to compute a field score matrix `scores_field = X @ W_field.T`.
  - For each simulation `s`, compute the **empirical 99th-percentile score** across `scores_field[s, :] `(or `field_size * 0.99` index) and store these as `thresholds_explicit[s]`.
  - Use `thresholds_explicit` in place of the analytic thresholds when scoring candidate lineups and computing `top1_pct_finish_rate`.
- **Output consistency**:
- Keep the `Lineups_Top1Pct` and `Meta` sheet schemas unchanged for downstream tools.
- In `Meta`, add fields noting `field_model` and (optionally) basic stats about the simulated field (e.g. achieved CPT/FLEX ownership error summary).

### 3. Wire NFL and NBA CLIs to the new option

- **NFL CLI (`src/nfl/top1pct_finish_rate.py`)**:
- Add a `--field-model` argument with choices `"mixture"` and `"explicit"`, defaulting to `"mixture"`.
- Pass this through to `top1pct_core.run_top1pct`.
- **NBA CLI (`src/nba/top1pct_finish_rate_nba.py`)**:
- Mirror the NFL changes (same `--field-model` flag and default), wiring it into the call to `top1pct_core.run_top1pct`.

### 4. Validation and calibration

- **Unit/smoke tests**:
- Add a small synthetic slate test in `tests/` that generates dummy projections, ownership, and correlations, then runs `run_top1pct` with `field_model="mixture"` and `field_model="explicit"` to ensure both paths run end-to-end and produce sensible outputs (no NaNs, non-negative thresholds, probabilities in [0, 100]).
- Add a quick check that the simulated field’s realized CPT/FLEX ownership is reasonably close to the targets on a toy example.
- **Back-of-the-envelope comparisons**:
- On a stored historical slate you already have, run the NFL `top1pct_finish_rate` both ways and compare:
  - Distribution of top1% rates,
  - Behavior for high-owned CPT vs low-owned CPT lineups.
- Use these runs to decide if you want to expose additional knobs (e.g. entrant-type mix, quota tolerance) as CLI flags later.

## Implementation Todos

- **define-builder-api**: Add a new shared module (e.g. `field_builder.py`) and define its public API and inputs based on existing `top1pct_core` data structures.
- **implement-builder-logic**: Implement the quota-balanced random field lineup generator (single entrant type first, quotas + CPT/FLEX sampling, basic dead-end handling).
- **integrate-into-core**: Extend `run_top1pct` with a `field_model` switch and wire in the explicit field simulation path, including empirical 99th-percentile threshold computation.
- **update-clis**: Add a `--field-model` flag to NFL and NBA `top1pct` CLIs and plumb it through.
- **testing-calibration**: Create small tests and run a couple of real-slate comparisons to sanity-check the new method and adjust default hyperparameters if needed.