---
name: nfl-diversified-output-dir
overview: Move NFL diversified lineups workbooks from outputs/nfl/top1pct/ to outputs/nfl/diversified/ and update all consumers so the pipeline still runs end-to-end.
todos:
  - id: update-core-path
    content: Change `run_diversify` in `src/shared/diversify_core.py` to write diversified workbooks under `outputs/<sport>/diversified/` instead of `outputs/<sport>/top1pct/`.
    status: completed
  - id: update-fill-dkentries-path
    content: Update `_load_diversified_lineups` and CLI help in `src/nfl/fill_dkentries.py` to resolve diversified workbooks from `outputs/nfl/diversified/`.
    status: completed
  - id: review-run-full-and-docs
    content: Review `run_full.sh`, `src/nfl/diversify_lineups.py`, and README/comments to ensure all references to diversified output locations are consistent with the new `outputs/nfl/diversified/` directory, then validate with a full pipeline run.
    status: completed
---

## Plan: Move diversified outputs to `outputs/nfl/diversified/`

### 1) Update core diversification output path
- **Change write directory in `diversify_core`**: In `src/shared/diversify_core.py`, update `run_diversify` so it writes the diversified workbook into a `diversified` subdirectory instead of `top1pct`.
  - Currently it builds `top1pct_dir = outputs_dir / "top1pct"` and `output_path = top1pct_dir / f"top1pct_diversified_{num_lineups}.xlsx"`.
  - Change this to something like `diversified_dir = outputs_dir / "diversified"` and write `top1pct_diversified_{num_lineups}.xlsx` there, keeping the filename pattern the same so downstream code that only cares about sheet names remains valid.
  - Keep `_load_top1pct_lineups` unchanged so it still reads `Lineups_Top1Pct` sheets from `outputs/<sport>/top1pct/top1pct_lineups_*.xlsx`.

### 2) Update DKEntries filler to look in the new diversified directory
- **Adjust diversified-workbook resolution in `fill_dkentries`**: In `src/nfl/fill_dkentries.py`, update `_load_diversified_lineups` so it resolves workbooks from `config.OUTPUTS_DIR / "diversified"` instead of `config.OUTPUTS_DIR / "top1pct"`.
  - The helper `_resolve_latest_excel` can remain the same; it will now just scan the `diversified` directory.
  - Update the docstring and CLI help text for `--diversified-excel` to mention `outputs/nfl/diversified/` as the default location.
  - Ensure the rest of `_load_diversified_lineups` remains unchanged so it still expects the `Lineups_Diversified` sheet and the same column schema.

### 3) Verify that `run_full.sh` and CSV sidecars still work end-to-end
- **Confirm behavior of `run_full.sh`**: `run_full.sh` already passes an `--output-dir` pointing at the run directory to `src.nfl.diversify_lineups`, and `src/nfl/diversify_lineups.py` writes `diversified.csv` and `ownership.csv` into that `output_dir` based on whatever Excel path `run_diversify` returns.
  - After changing the core output directory, confirm that `diversify_lineups.run` still correctly receives the new `outputs/nfl/diversified/top1pct_diversified_X.xlsx` path and that it continues to write `diversified.csv` and `ownership.csv` into `${RUN_DIR}` (no path changes needed there).
  - Ensure `run_full.sh` Step 5 (calling `src.nfl.fill_dkentries`) now picks up the latest workbook from `outputs/nfl/diversified` via the updated `_load_diversified_lineups` without requiring any script changes.

### 4) Clean up references and docs
- **Update comments/docstrings describing paths**: Search for references to `outputs/nfl/top1pct/top1pct_diversified_` and update them to `outputs/nfl/diversified/top1pct_diversified_` where appropriate (e.g., module docstrings or README sections that describe where diversified workbooks live).
- **Optional manual cleanup**: Optionally delete or archive existing `top1pct_diversified_*.xlsx` files from `outputs/nfl/top1pct/` once the change is deployed, to avoid confusion when browsing old outputs.

### 5) Sanity check with a full pipeline run
- **Run `run_full.sh` on a small slate** and verify:
  - A new `top1pct_lineups_*.xlsx` appears in `outputs/nfl/top1pct/` and is used by diversification.
  - A new `top1pct_diversified_*.xlsx` appears in `outputs/nfl/diversified/`.
  - `fill_dkentries` successfully reads from `outputs/nfl/diversified`, writes `dkentries.csv`, `diversified.csv`, and `ownership.csv` under the timestamped `outputs/nfl/dkentries/<timestamp>/` directory, and no `Lineups_Top1Pct`/`Lineups_Diversified` sheet errors occur.