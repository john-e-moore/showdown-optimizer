---
name: ROI-first DKEntries
overview: Make DKEntries assignment prefer higher EV ROI (when available) instead of top-1% finish rate, while preserving the existing small projection tie-breaker and keeping fallback behavior unchanged.
todos:
  - id: update-nba-strength
    content: Change `src/nba/fill_dkentries_nba.py` to compute `LineupRecord.strength` from `ev_roi` when present, else fallback to `top1_pct_finish_rate`, keeping the projection tie-breaker.
    status: completed
  - id: update-nfl-strength
    content: Change `src/nfl/fill_dkentries.py` to compute `LineupRecord.strength` from `ev_roi` when present, else fallback to `top1_pct_finish_rate`, keeping the projection tie-breaker.
    status: completed
    dependencies:
      - update-nba-strength
  - id: doc-note
    content: Optionally document ROI-first DKEntries tie-breaker behavior in `README.md`.
    status: completed
    dependencies:
      - update-nfl-strength
---

# ROI-first DKEntries assignment

## Goal

When the diversified sheet contains `ev_roi` (from `--contest-id`/`--payouts-json` runs), DKEntries assignment should **prefer higher ROI** in its tie-breaker scoring instead of `top1_pct_finish_rate`.

## Where the change happens

The DKEntries assignment algorithm (`src/shared/dkentries_core.py`) breaks ties using `LineupRecord.strength` (higher is preferred via `key = (imbalance, -rec.strength)`). The “strength” value is constructed in the sport-specific fillers:

- [`/home/john/showdown-optimizer/src/nba/fill_dkentries_nba.py`](/home/john/showdown-optimizer/src/nba/fill_dkentries_nba.py)
- [`/home/john/showdown-optimizer/src/nfl/fill_dkentries.py`](/home/john/showdown-optimizer/src/nfl/fill_dkentries.py)

Today both compute:

- primary: `top1_pct_finish_rate`
- secondary: `0.001 * lineup_projection`

## Implementation steps

- Update NBA filler (`src/nba/fill_dkentries_nba.py`):
- Detect whether `ev_roi` exists in the diversified sheet (`has_ev_roi = "ev_roi" in df.columns`).
- Set `base_strength` to `row["ev_roi"] `when `has_ev_roi` and the cell is numeric/nonnull; otherwise fall back to `row["top1_pct_finish_rate"] `(when present) else `0.0`.
- Keep the existing projection tie-breaker exactly as-is: `strength = base_strength + 0.001 * proj_component`.
- Update NFL filler (`src/nfl/fill_dkentries.py`) with the same logic.
- (Optional but low-risk) Add a short note in [`/home/john/showdown-optimizer/README.md`](/home/john/showdown-optimizer/README.md) clarifying that DKEntries assignment tie-breaker is ROI-first when `ev_roi` is present.

## Safety / compatibility

- If the diversified sheet has no `ev_roi` column (legacy top1% runs), behavior remains unchanged.
- If `ev_roi` is present but a row has missing/invalid values, that row falls back to `top1_pct_finish_rate` (or `0.0`).

## Quick validation

- Run the pipeline once in legacy mode (no `--contest-id`) and ensure output matches prior behavior.
- Run with `--contest-id` and confirm the diversified workbook/CSV contains `ev_roi`, then ensure filled DKEntries prefers higher-ROI lineups when exposure-imbalance scores tie.