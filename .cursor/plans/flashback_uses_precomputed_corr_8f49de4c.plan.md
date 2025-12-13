---
name: Flashback uses precomputed corr
overview: Update the flashback simulation to load the latest precomputed correlation workbook from data/nba/correlations/ by default (fail fast if missing), using workbook projections + correlation matrix instead of simulating correlations.
todos:
  - id: core-corr-excel
    content: Add corr workbook resolution/loading to flashback_core.run_flashback; default to latest data/<sport>/correlations and fail fast if missing; use workbook Sabersim_Projections + Correlation_Matrix.
    status: completed
  - id: nba-cli-flag
    content: Add --corr-excel to src/nba/flashback_sim_nba.py and plumb into flashback_core.
    status: completed
    dependencies:
      - core-corr-excel
  - id: nfl-cli-flag
    content: Add --corr-excel to src/nfl/flashback_sim.py and plumb into flashback_core (defaults to data/nfl/correlations).
    status: completed
    dependencies:
      - core-corr-excel
  - id: readme-docs
    content: Update README NBA flashback section documenting new default and --corr-excel override.
    status: completed
    dependencies:
      - nba-cli-flag
---

# Flashback correlation workbook default

## Desired behavior

- **NBA flashback (`python -m src.nba.flashback_sim_nba`) loads correlations from a workbook by default**, using the most recent `*.xlsx` found in `data/nba/correlations/`.
- Flashback **fails fast** with a clear error if it cannot find/load that workbook (unless the user explicitly passes an override later).
- When using a correlation workbook, flashback **uses the workbook’s `Sabersim_Projections` sheet for projections (mu/sigma)** and **`Correlation_Matrix` sheet for correlations**, keeping them consistent.
- Flashback still uses `--sabersim-csv` (or latest in `data/nba/sabersim/`) to read **raw Sabersim rows for ownership/salary columns** in the output workbook.

## Implementation steps

### 1) Extend flashback core to accept/load a correlation workbook

- Update the signature of `run_flashback()` in [`/home/john/showdown-optimizer/src/shared/flashback_core.py`](/home/john/showdown-optimizer/src/shared/flashback_core.py) to accept a new optional arg, e.g. `corr_excel: str | None`.
- Add a helper in `flashback_core.py` similar to `_resolve_latest_csv()` but for Excel:
  - Resolve `corr_path` from an explicit `corr_excel` if provided.
  - Otherwise resolve the most recent `*.xlsx` in `config_module.DATA_DIR / "correlations"`.
  - If no file exists, raise `FileNotFoundError` with a message like: `No correlation workbook found in data/nba/correlations/. Generate one or pass --corr-excel`.
- Load the workbook via the existing shared loader in [`/home/john/showdown-optimizer/src/shared/top1pct_core.py`](/home/john/showdown-optimizer/src/shared/top1pct_core.py):
  - Use `top1pct_core._load_corr_workbook(corr_path)` to obtain `sabersim_proj_df` and `corr_df` (expects sheets `Sabersim_Projections` and `Correlation_Matrix`).
- Replace the current “simulate corr from projections” branch in `run_flashback()`:
  - **Delete/skip** `print("Building player correlation matrix via simulation...")` and the call to `simulate_corr_matrix_from_projections()` when a workbook is used.
  - Use `sabersim_proj_df` (workbook projections) as the projections dataframe for `_build_player_universe_from_sabersim_and_lineups(...)`.
- Keep reading `sabersim_raw_df = pd.read_csv(sabersim_path)` for ownership/salary calculations (unchanged).

### 2) Add CLI plumbing: `--corr-excel`

- In [`/home/john/showdown-optimizer/src/nba/flashback_sim_nba.py`](/home/john/showdown-optimizer/src/nba/flashback_sim_nba.py):
  - Add `corr_excel: str | None = None` to `run()` and pass it through to `flashback_core.run_flashback(corr_excel=corr_excel, ...)`.
  - Add an argparse flag `--corr-excel` (optional) explaining:
    - default is latest `data/nba/correlations/*.xlsx`
    - must contain `Sabersim_Projections` and `Correlation_Matrix` sheets.
- Mirror the same change for NFL for consistency in [`/home/john/showdown-optimizer/src/nfl/flashback_sim.py`](/home/john/showdown-optimizer/src/nfl/flashback_sim.py), defaulting to latest `data/nfl/correlations/*.xlsx`.

### 3) Update documentation

- Update the NBA flashback README section around your cursor location in [`/home/john/showdown-optimizer/README.md`](/home/john/showdown-optimizer/README.md) to document:
  - the new default correlation source (`data/nba/correlations/`)
  - how to override with `--corr-excel`
  - the “fail fast if missing/unreadable” behavior
  - a short example invocation.

### 4) Quick validation checklist (manual)

- Run NBA flashback with only `--contest-csv` and `--sabersim-csv` omitted to ensure it:
  - resolves latest contest + latest sabersim + latest correlation workbook
  - errors clearly if correlation workbook is missing
  - no longer prints “Building player correlation matrix via simulation...”
- Run with `--corr-excel <explicit>` to ensure the override works.

## Data flow (after change)

```mermaid
sequenceDiagram
participant CLI as flashback_sim_nba
participant Core as flashback_core
participant CorrDir as data/nba/correlations
participant CorrXlsx as corr_workbook.xlsx
participant SaberCsv as data/nba/sabersim/*.csv

CLI->>Core: run_flashback(contest_csv,sabersim_csv,corr_excel?)
Core->>CorrDir: resolve_latest_xlsx(if corr_excel is None)
Core->>CorrXlsx: load Sabersim_Projections + Correlation_Matrix
Core->>SaberCsv: read raw Sabersim CSV (ownership/salary)
Core->>Core: simulate correlated scores using corr + workbook projections
Core->>CLI: write flashback workbook
```