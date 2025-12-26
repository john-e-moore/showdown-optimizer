---
name: Add avg duplicates
overview: Redefine lineup duplicates to exclude the lineup itself (min 0) and surface entrant-level average duplicates in the flashback workbook, for both NFL and NBA, with the requested column placement.
todos:
  - id: dup-minus-one
    content: Update flashback Simulation 'Duplicates' to be (count - 1) clipped at 0 in flashback_core.run_flashback().
    status: completed
  - id: entrant-avg-dup
    content: Compute and add 'Avg. Duplicates' to entrant summary in flashback_core._build_entrant_summary().
    status: completed
    dependencies:
      - dup-minus-one
  - id: entrant-col-order
    content: Adjust entrant summary base column ordering to insert 'Avg. Duplicates' between 'Entries' and 'Actual ROI'.
    status: completed
    dependencies:
      - entrant-avg-dup
  - id: excel-format
    content: Update xlsx formatting to include 'Avg. Duplicates' as a numeric column in the Entrant summary sheet.
    status: completed
    dependencies:
      - entrant-col-order
  - id: tests
    content: Add/extend a unit test verifying duplicates are computed as group_size-1 and that entrant average duplicates is the mean of those values.
    status: completed
    dependencies:
      - dup-minus-one
      - entrant-avg-dup
---

# Add Avg. Duplicates to flashback

## Goal

- Add a new **`Avg. Duplicates`** column to the **`Entrant summary`** tab for both NFL and NBA flashback sims.
- Insert it **between `Entries` and `Actual ROI`**.
- Redefine **`Duplicates`** in the **`Simulation`** tab to mean **number of other entries with the exact same lineup** (so `count - 1`, minimum 0).

## Where this lives

- Shared pipeline + workbook writing: [`/home/john/showdown-optimizer/src/shared/flashback_core.py`](/home/john/showdown-optimizer/src/shared/flashback_core.py)
- NFL wrapper: [`/home/john/showdown-optimizer/src/nfl/flashback_sim.py`](/home/john/showdown-optimizer/src/nfl/flashback_sim.py) (no change expected)
- NBA wrapper: [`/home/john/showdown-optimizer/src/nba/flashback_sim_nba.py`](/home/john/showdown-optimizer/src/nba/flashback_sim_nba.py) (no change expected)

## Implementation steps

### 1) Change `Simulation` tab Duplicates definition (shared)

- In `run_flashback()`, locate the section that computes duplicates:
- Current behavior: `dup_counts = groupby(lineup_cols).transform("size")` then `simulation_df["Duplicates"] = dup_counts`
- Update it to:
- `simulation_df["Duplicates"] = (dup_counts - 1).clip(lower=0)`
- Keep the existing column placement logic (it already inserts `Duplicates` after `Actual Points` or `Avg Points`).

### 2) Add `Avg. Duplicates` to `Entrant summary`

- In `_build_entrant_summary(simulation_df, flex_role_label)`:
- Add an entrant-level aggregation for duplicates:
  - If `Duplicates` exists on `simulation_df`, compute mean per entrant (same pattern as `Actual ROI` and `Sim ROI` aggregation).
- Ensure the resulting summary column is named exactly **`Avg. Duplicates`**.

### 3) Enforce column ordering (requested placement)

- In `_build_entrant_summary()` final ordering block:
- Build `base_cols` as:
  - `Entrant`, `Entries`, `Avg. Duplicates`, `Actual ROI` (if present), `Avg. Sim ROI` (if present), then existing averages.
- This guarantees `Avg. Duplicates` lands **between `Entries` and `Actual ROI`**; if `Actual ROI` is absent, it will naturally sit right after `Entries`.

### 4) Excel formatting updates

- In `run_flashback()` formatting section for `Entrant summary`:
- Add `Avg. Duplicates` to `entrant_num_cols` so it gets numeric formatting.
- In `Simulation` formatting:
- Keep `Duplicates` as numeric; optionally switch it to an integer-like format if the repo already uses one (otherwise keep existing `0.00` numeric format for consistency).

### 5) Light regression coverage

- Add or extend a small test to validate the new definition:
- With two identical lineups in the same group, each should get `Duplicates == 1` (because there’s one *other* entry).
- A unique lineup should get `Duplicates == 0`.
- Best candidate location: [`/home/john/showdown-optimizer/tests/test_flashback_corr_handling.py`](/home/john/showdown-optimizer/tests/test_flashback_corr_handling.py) if it already covers flashback helpers; otherwise add a new focused test file under `tests/`.

## Acceptance criteria

- **Simulation** sheet: `Duplicates` is **0** for unique lineups, and **(group_size - 1)** for duplicated lineups.
- **Entrant summary** sheet includes **`Avg. Duplicates`** and it appears **between `Entries` and `Actual ROI`**.
- Behavior is identical for NFL and NBA since it’s implemented in the shared core.