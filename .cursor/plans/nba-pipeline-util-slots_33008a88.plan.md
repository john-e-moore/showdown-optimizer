---
name: nba-pipeline-util-slots
overview: Adjust the shared diversification and NBA DKEntries code so the NBA pipeline correctly handles UTIL-style lineup slots instead of assuming FLEX columns.
todos:
  - id: update-diversify-core-slots
    content: Update shared `diversify_core` to resolve non-CPT slot columns (FLEX vs UTIL) and use them in player-set and exposure logic.
    status: completed
  - id: update-nba-fill-dkentries
    content: Update `fill_dkentries_nba` to load diversified lineups using the same FLEX/UTIL slot resolver and build LineupRecord players accordingly.
    status: completed
    dependencies:
      - update-diversify-core-slots
  - id: regression-tests-nfl-nba
    content: Run and verify tests and small NFL/NBA smoke runs to ensure both pipelines work with the new UTIL-aware behavior.
    status: completed
    dependencies:
      - update-diversify-core-slots
      - update-nba-fill-dkentries
---

## Plan: Make NBA pipeline UTIL-aware in diversification and DKEntries stages

### 1. Make diversification logic slot-name agnostic
- **Introduce non-CPT slot resolver in `diversify_core`**: Add a small helper that mirrors `top1pct_core._resolve_non_cpt_slot_columns`, returning the 5 non-CPT columns as either `flex1`–`flex5` or `util1`–`util5` based on what exists in the `Lineups_Top1Pct` DataFrame.
- **Use resolver in player set builders**: Update `_build_player_set` and `_build_flex_set` in `[src/shared/diversify_core.py](src/shared/diversify_core.py)` to use the resolved non-CPT column list instead of hard-coding `flex1`–`flex5`, so they work for both NFL (FLEX) and NBA (UTIL) top1pct workbooks.
- **Use resolver for exposure computation**: Update `_compute_exposure` to derive the non-CPT slot columns via the same resolver, and iterate over those when counting flex/UTIL exposure, keeping column names in the diversified output unchanged.

### 2. Make NBA DKEntries filling robust to UTIL vs FLEX
- **Relax diversified lineup column requirements**: In `_load_diversified_lineups` in `[src/nba/fill_dkentries_nba.py](src/nba/fill_dkentries_nba.py)`, change the strict requirement on `flex1`–`flex5` to instead resolve non-CPT columns (via importing `top1pct_core._resolve_non_cpt_slot_columns` or a local equivalent) and require `cpt` plus those 5 columns.
- **Build `LineupRecord` players from resolved slots**: When constructing `LineupRecord.players`, append `cpt` followed by the resolved non-CPT columns in order, regardless of whether they are named `flex*` or `util*`, preserving the existing CPT/UTIL semantics for NBA.
- **Ensure downstream ownership summary still works**: Confirm that `_write_ownership_summary_csv` remains unaffected (it works on DKEntries slots, not the diversified workbook) so only the diversified Excel reading logic needs to know about UTIL vs FLEX.

### 3. Keep NFL behavior unchanged and add light validation
- **Preserve NFL FLEX pipeline**: Verify that for an NFL `Lineups_Top1Pct` or diversified workbook with `flex1`–`flex5`, the new resolver returns FLEX columns and that all existing NFL CLI wrappers (`src/nfl/diversify_lineups.py`, `src/shared/diversify_core.py`) behave identically.
- **Add defensive error messages**: Where we now use the resolver, keep or add clear `KeyError` messages when neither FLEX nor UTIL columns are present, so misformatted workbooks fail with actionable feedback.

### 4. Testing and manual verification
- **Run existing lightweight tests**: Re-run `python -m tests_top1pct_field_model` to ensure top1pct + field builder changes still pass after diversification updates.
- **NFL smoke test**: Run the existing NFL end-to-end script (or a minimal NFL top1pct + diversify invocation) to confirm nothing regressed for FLEX-style lineups.
- **NBA end-to-end smoke**: Run `./run_full_nba.sh` on a small slate and confirm all 4 steps complete, and that the diversified Excel/CSV and filled DKEntries CSV contain CPT + 5 UTIL players with sensible overlap and ownership behavior.