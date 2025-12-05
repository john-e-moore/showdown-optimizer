---
name: field-lineups-and-run-directories
overview: Add a Field Lineups sheet in explicit field model outputs and restructure NFL outputs so each full run writes into a per-run directory under outputs/{sport}/runs/{timestamp}.
todos: []
---

## Plan for Field Lineups Tab and Per-Run Output Directories

### 1) Add `Field Lineups` tab for explicit field model

- **Identify lineup metadata schema**: Inspect the existing `Lineups` sheet structure in recent NFL and NBA lineups workbooks to confirm the columns used for lineup projection, salary, stack label, stack pattern, and slot columns (`cpt`, `flex1`–`flex5` for NFL; `cpt`, `util1`–`util5` for NBA going forward).
- **Factor out lineup annotation logic**: Create or reuse a helper in `src/shared/top1pct_core.py` (e.g. `_annotate_lineups_with_meta`) that, given a lineups DataFrame plus projection/salary/team data, computes:
- lineup projection (1.5× CPT projection + 1× each flex/UTIL projection),
- lineup salary (same weighting pattern using salary),
- stack label and stack pattern using the same logic as in the optimizer’s `Lineups` sheet,
- and preserves the `cpt` + flex/UTIL slot columns.
- **Expose player-level salary/team to top1pct core**: Extend `_build_player_universe` (or add a separate helper) in `top1pct_core.py` to also return per-player salary and team mappings derived from the projections inputs, so the annotation helper can compute salary and stack info without re-reading files.
- **Generate field lineups with metadata in explicit mode**: In `run_top1pct`’s `field_model == "explicit"` branch in `top1pct_core.py`:
- After building `field_lineups_df = build_quota_balanced_field(...)`, call the new annotation helper to produce `field_lineups_with_meta`.
- For NFL, keep slot columns as `cpt`, `flex1`–`flex5`; for NBA, map those columns to `cpt`, `util1`–`util5` in the annotated DataFrame before writing.
- **Write `Field Lineups` sheet into top1pct workbook**: When creating the `top1pct_lineups_*.xlsx` file in `run_top1pct`, add a new worksheet (e.g. `"Field_Lineups"`) that is only populated when `field_model == "explicit"` and contains `field_lineups_with_meta` with columns:
- `lineup_projection`, `lineup_salary`, `stack`, `stack_pattern`, followed by the CPT and flex/UTIL slot columns.

### 2) Introduce per-run output directories under `outputs/{sport}/runs/{timestamp}`

- **Define run directory convention**: Standardize on `outputs/{sport}/runs/{timestamp}/` (e.g. `outputs/nfl/runs/20251205_193000/`) and keep existing sport-level roots (e.g. `config.OUTPUTS_DIR`) unchanged for backwards compatibility.
- **Add optional run-dir support to shared core**: Update `src/shared/top1pct_core.run_top1pct` to accept an optional `run_dir: Path | None` (or similar) argument:
- If provided, write the `top1pct_lineups_*.xlsx` into `run_dir` rather than `outputs_dir / "top1pct"`.
- Keep the current default behavior when `run_dir` is `None` so standalone CLI usage still writes to the existing `outputs/{sport}/top1pct/` location.
- **Extend NFL/NBA top1pct wrappers to pass run-dir**: In `src/nfl/top1pct_finish_rate.py` and `src/nba/top1pct_finish_rate_nba.py`:
- Add an optional CLI flag (e.g. `--run-dir`) that, when provided, is converted to a `Path` and passed through to `top1pct_core.run_top1pct`.
- Document that when `--run-dir` is set, the top1pct workbook and its `Field Lineups` tab will be written into that directory.

### 3) Wire NFL `run_full.sh` to a single per-run directory

- **Create the run directory once in `run_full.sh`**: Early in `run_full.sh`, after computing `timestamp`, define `RUN_DIR="outputs/nfl/runs/${timestamp}"` and `mkdir -p "${RUN_DIR}"`.
- **Route correlation output into run-dir**: Adjust the correlation step to set `CORR_EXCEL="${RUN_DIR}/correlations_${timestamp}.xlsx"` (or similar) and pass that into `src.nfl.main --output-excel`, instead of the current `outputs/nfl/correlations/...` path.
- **Route lineups workbook into run-dir**: Update `src/nfl/showdown_optimizer_main` (if needed) to accept an explicit `--output-excel` (or `--output-dir`) argument; in `run_full.sh`, pass a path like `"${RUN_DIR}/lineups_${timestamp}.xlsx"` so the lineups workbook is saved into the same run directory.
- **Route top1pct workbook into run-dir**: Modify the `src/nfl.top1pct_finish_rate` CLI to accept the new `--run-dir` flag and, in `run_full.sh`, call:
- `python -m src.nfl.top1pct_finish_rate --field-size "${FIELD_SIZE}" --run-dir "${RUN_DIR}"`,
- so the resulting `top1pct_lineups_*.xlsx` (including the new `Field Lineups` sheet) lands under the same `RUN_DIR`.
- **Route diversified and DKEntries outputs into run-dir**: For `diversify_lineups` and `fill_dkentries`:
- Update `src/nfl/diversify_lineups.py` to accept an optional `--output-dir` (defaulting to the current `outputs/nfl/dkentries/`), and have it write `diversified.csv` and `ownership.csv` into that directory.
- Ensure `src/nfl/fill_dkentries.py` already accepts `--output-csv` (it is passed from `run_full.sh`); in `run_full.sh`, change `OUTPUT_DKENTRIES_CSV` to be `"${RUN_DIR}/dkentries_${timestamp}.csv"` instead of under `outputs/nfl/dkentries/`.
- Update the final print messages in `run_full.sh` and docs to reflect that diversified and DKEntries CSVs now live alongside the other run artifacts when invoked via `run_full.sh`.
- **Keep flashback outputs separate**: Leave `flashback_core` and sport-specific flashback CLIs unchanged so they continue to write into their existing `outputs/{sport}/flashback`-style locations, satisfying the requirement to keep flashback output separate.

### 4) Align NBA non-CPT slot naming with UTIL convention

- **Audit NBA lineups workbook generation**: Locate where NBA Showdown lineups Excel workbooks are written (likely in `src/nba/showdown_optimizer_main.py` or a shared lineup writer) and confirm the current slot column names in the `Lineups` sheet.
- **Rename NBA flex columns to util in outputs**: Update the NBA lineup writer so that:
- The `Lineups` sheet uses `cpt`, `util1`–`util5` column names in new runs.
- Any downstream NBA code that refers to lineup columns (e.g. NBA’s `top1pct_finish_rate_nba` wrapper or any diversification code) is updated to look for `util` columns instead of `flex` when reading NBA lineups.
- **Ensure consistency with top1pct Field Lineups sheet**: When writing the `Field Lineups` sheet for NBA in `top1pct_core.run_top1pct`, map internal slot columns to `util1`–`util5` so the top1pct workbook matches the NBA lineup workbook convention, while reusing the same scoring logic that treats these as 5 non-CPT flex-style slots.

### 5) Documentation and sanity checks

- **Update README / usage notes**: Briefly document in `README.md` (or a sport-specific section) that:
- Running `run_full.sh` now creates a per-run directory under `outputs/nfl/runs/{timestamp}` containing the correlation workbook, lineups workbook, top1pct workbook (with `Field Lineups` in explicit mode), and the diversified/ownership/DKEntries CSVs.
- Flashback outputs remain in their existing locations.
- **Validation passes**:
- Run `python -m src.nfl.top1pct_finish_rate --field-size <size> --field-model explicit --run-dir outputs/nfl/runs/test_timestamp` and open the resulting workbook to verify the `Field Lineups` sheet exists and has the expected columns and reasonable values.
- Run `./run_full.sh ...` for NFL to confirm all expected artifacts appear under a single `outputs/nfl/runs/{timestamp}` directory.
- For NBA, generate a Showdown run and verify that both the lineup workbook and any top1pct workbook use `util1`–`util5` (not `flex`) in their visible outputs.