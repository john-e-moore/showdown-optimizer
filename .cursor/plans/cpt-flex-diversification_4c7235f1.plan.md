---
name: cpt-flex-diversification
overview: Add CPT dollar-exposure targeting and improved FLEX diversification to the NFL Showdown pipeline while preserving the existing top1% and overlap-driven lineup selection.
todos:
  - id: wire-cli-flags
    content: Add new configuration/CLI flags for CPT and FLEX exposure control in nfl fill_dkentries (and optionally diversify_lineups) and plumb them into run() and assignment helpers.
    status: completed
  - id: flex-overlap-core
    content: Optionally extend diversify_core._greedy_diversified_selection to support a separate max_flex_overlap constraint while preserving existing behavior when unset.
    status: completed
  - id: field-ownership-targets
    content: Leverage _load_field_ownership_mapping in nfl fill_dkentries to compute per-player CPT and FLEX field ownership targets for use during assignment.
    status: completed
    dependencies:
      - wire-cli-flags
  - id: exposure-aware-assignment
    content: Replace or extend _assign_lineups_fee_aware in nfl fill_dkentries with an exposure-aware assignment that balances CPT dollar-exposure error, FLEX diversification, and lineup strength.
    status: completed
    dependencies:
      - wire-cli-flags
      - field-ownership-targets
  - id: ownership-summary-enhancements
    content: Update ownership.csv generation to clearly expose field targets vs realized lineup and dollar exposure, and verify outputs on sample runs.
    status: completed
    dependencies:
      - exposure-aware-assignment
  - id: docs-and-validation
    content: Run small-slate experiments to tune weights and overlap settings, then update README NFL docs to describe the new behavior and recommended usage.
    status: completed
    dependencies:
      - ownership-summary-enhancements
---

### CPT & FLEX diversification enhancements

#### Overview
Implement a layered diversification system for NFL Showdown that:
- Keeps the existing **min top1% + max-overlap** greedy selection for high-EV, de-correlated lineups.
- Adjusts **DKEntries assignment** to target **CPT dollar exposure ≈ field CPT ownership**.
- Adds **explicit FLEX diversification controls** (via overlap constraints and/or soft exposure penalties).

#### 1) Add configuration knobs / CLI flags
- **NFL diversify CLI** – extend `[src/nfl/diversify_lineups.py](/home/john/showdown-optimizer/src/nfl/diversify_lineups.py)` only if you want different overlap rules, otherwise leave as-is.
- **DKEntries filler CLI** – in `[src/nfl/fill_dkentries.py](/home/john/showdown-optimizer/src/nfl/fill_dkentries.py)`:
  - Add optional flags for:
    - `--cpt-own-weight`: strength of CPT dollar-exposure matching vs lineup strength (default 1.0 or similar).
    - `--flex-own-weight`: weight for FLEX diversification penalty (default maybe 0.2–0.5, or 0 to disable).
    - `--max-flex-overlap`: optional hard cap on shared FLEX players between any two lineups used in DKEntries assignment (if you want a second layer of control separate from the earlier selection step).
  - Plumb these new args into `run()` and the helper functions that need them (especially the assignment routine).

#### 2) (Optional) Refine diversification overlap to be FLEX-aware
- In `[src/shared/diversify_core.py](/home/john/showdown-optimizer/src/shared/diversify_core.py)`, within `_greedy_diversified_selection`:
  - Currently `overlap` is computed as the size of the intersection of two full player sets (CPT + FLEX).
  - If desired, introduce an additional **FLEX-only overlap** constraint:
    - Extract CPT and FLEX sets separately per lineup when building `_player_set`, or maintain parallel structures (e.g. `_player_set_all`, `_player_set_flex`).
    - For each candidate vs selected lineup, compute:
      - `total_overlap = len(all_players_candidate ∩ all_players_selected)` (existing behavior).
      - `flex_overlap = len(flex_players_candidate ∩ flex_players_selected)`.
    - Enforce both `total_overlap <= max_overlap` (existing) and `flex_overlap <= max_flex_overlap` (new, from config/default).
  - Add an optional `max_flex_overlap` parameter to `run_diversify(...)` and pass through from the NFL wrapper if you want this tunable; otherwise keep a conservative fixed default.

#### 3) Expose field CPT/FLEX ownership targets for assignment
- In `[src/nfl/fill_dkentries.py](/home/john/showdown-optimizer/src/nfl/fill_dkentries.py)` you already have `_load_field_ownership_mapping()` returning, per player:
  - `team`, `field_own_cpt`, `field_own_flex`.
- Extend `_load_field_ownership_mapping` or add a small helper to:
  - Normalize field CPT ownership percentages to a consistent scale (e.g. treat them as raw percents, 0–100).
  - Optionally compute **per-player CPT target dollar exposure** as:
    - `field_own_cpt_normalized[player]` (keep in % terms), to be compared directly to realized CPT dollar exposure (also in %).
- Ensure `run()` calls this helper once and passes the resulting map into the assignment function.

#### 4) Implement CPT dollar-exposure targeting in assignment
- In the same file, modify `_assign_lineups_fee_aware` to become something like `_assign_lineups_exposure_aware` (or keep the name but change internals):
  - **Inputs**: `dk_df`, `records`, `field_ownership_map`, `cpt_own_weight`, `flex_own_weight`.
  - Before the main loop:
    - Compute `entry_indices` and `fee_values` as currently done.
    - Precompute `total_fees = sum(fee_values)`.
    - Initialize tracking of **realized exposures so far**:
      - `cpt_dollar_exposure[player] = 0.0` (accumulated dollars assigned where player is CPT).
      - `flex_dollar_exposure[player] = 0.0` or at least `flex_lineup_count[player] = 0` if you prefer lineup-level exposure for FLEX.
  - For each DK entry processed in descending fee order:
    - For each candidate remaining lineup `rec`:
      - Let `fee` be this entry’s fee.
      - Identify its CPT player `cpt_name = rec.players[0]`.
      - Compute **CPT dollar exposure cost**:
        - Current CPT % for that player: `curr_pct = 100 * cpt_dollar_exposure[cpt_name] / total_fees` (guard against divide by zero).
        - New CPT % if we assign this lineup here: `new_pct = 100 * (cpt_dollar_exposure[cpt_name] + fee) / total_fees`.
        - Target CPT % from field: `target_pct = field_own_cpt_for_player` (fallback 0 if missing).
        - Incremental squared error:
          - `delta_cpt = (new_pct - target_pct)^2 - (curr_pct - target_pct)^2`.
      - Compute **FLEX diversification cost** (optional, softer):
        - For each FLEX player in `rec.players[1:]`, you can use either:
          - A simple **overuse penalty** when their current exposure exceeds field FLEX or a soft cap; or
          - A squared-error term similar to CPT: compare realized FLEX % vs `field_own_flex`.
        - Sum these into `delta_flex` for the lineup.
      - Combine costs with lineup strength:
        - Example score:
          - `score = cpt_own_weight * delta_cpt + flex_own_weight * delta_flex - strength_lambda * rec.strength`.
        - `strength_lambda` can be 1.0 initially (or driven by a config flag) so that higher-strength lineups are preferred among similar exposure costs.
    - Choose the candidate lineup with **minimum `score`**.
    - Assign it to this entry, then update:
      - `cpt_dollar_exposure` and FLEX exposure trackers.
      - Remove that lineup from the remaining pool.
  - Keep the existing fee-tier sanity summary or update it to log CPT and FLEX exposure vs target for quick checks.

#### 5) Adjust ownership.csv to surface realized vs field targets
- You already write `ownership.csv` in `fill_dkentries` via `_write_ownership_summary_csv`.
- Extend that CSV to (optionally) include **target_field_ownership** columns for easier post-run QA:
  - Add `field_ownership_cpt_target` and `field_ownership_flex_target` or just reuse the existing `field_ownership` column and add separate `target_type`.
  - Or simply ensure that `field_ownership` is clearly the field target and `lineup_exposure` / `dollar_exposure` are realized; this file will be your main tool for checking how well the new assignment logic is matching CPT field ownership.

#### 6) Guardrails, defaults, and backward compatibility
- Choose conservative defaults so the new behavior is **opt-in** or at least relatively mild:
  - `cpt_own_weight` default: modest (e.g. 0.5–1.0) so very high-EV outlier lineups can still be overweight if desired.
  - `flex_own_weight` default: small (e.g. 0.2) or `0.0` to disable FLEX exposure matching unless explicitly enabled.
  - Keep `max_overlap` default unchanged; only use new `max_flex_overlap` if you see meaningful redundancy in FLEX.
- Ensure that if **field ownership data is missing** or malformed, the assignment gracefully falls back to:
  - Previous fee-aware behavior; or
  - A simpler uniform-exposure heuristic with a log warning.

#### 7) Testing and validation
- Add or update tests (if you have a test harness) or small scripts to:
  - Run `fill_dkentries` on a tiny synthetic DKEntries CSV and a small diversified workbook where:
    - Field CPT ownership targets are intentionally skewed (e.g. one obvious chalk CPT, several mid-range, a few punt CPTs).
    - Verify that realized CPT **dollar_exposure** in `ownership.csv` roughly tracks the field targets after assignment.
  - Create scenarios with extreme fee tiers (e.g. one very high-fee entry plus many small ones) to sanity-check that the algorithm doesn’t overly sacrifice EV to hit ownership exactly.
- Visually inspect the per-run `ownership.csv` under `outputs/nfl/runs/<ts>/` to confirm:
  - CPT dollar exposure curve is close to projected field CPT curve.
  - FLEX exposure is more spread out or closer to desired levels, given your chosen weights.

#### 8) Documentation
- Update `[README.md](/home/john/showdown-optimizer/README.md)` in the NFL section to briefly describe:
  - The new CPT dollar-exposure matching behavior and FLEX diversification options.
  - The new CLI flags and example commands (e.g. showing `--cpt-own-weight` and `--flex-own-weight`).
  - How to interpret the enhanced `ownership.csv` files and use them to tune your weights and overlap settings.