---
name: Add DK_Lineups sheet
overview: Add a new Excel tab `DK_Lineups` to the generated top1pct workbook. The tab mirrors `Lineups_Top1Pct` but formats lineup slots as `Name (dk_player_id)` by mapping names/roles to IDs from the latest (or provided) DKEntries CSV for each sport; if mapping is unavailable/incomplete, skip the tab with a warning.
todos:
  - id: add-dk-lineups-writer
    content: Add `DK_Lineups` dataframe generation + conditional worksheet writing in `src/shared/top1pct_core.py` using DKEntries name/role→ID mapping and the existing slot-column + name-parsing helpers.
    status: completed
  - id: plumb-dkentries-flag
    content: Add `--dkentries-csv` to NFL/NBA top1pct CLIs and plumb it into `top1pct_core.run_top1pct` so each sport can control which DKEntries template is used.
    status: completed
    dependencies:
      - add-dk-lineups-writer
  - id: docs
    content: Update `README.md` to document the new `DK_Lineups` sheet and the `--dkentries-csv` option + default behavior.
    status: in_progress
    dependencies:
      - plumb-dkentries-flag
  - id: tests
    content: Add a focused unit test that verifies slot formatting and the skip-on-missing-mapping behavior for `DK_Lineups`.
    status: pending
    dependencies:
      - add-dk-lineups-writer
---

# Add `DK_Lineups` sheet to top1pct workbook

## Goal

- Generate a new worksheet **`DK_Lineups`** in the `top1pct_lineups_*.xlsx` workbook.
- The sheet is an **identical copy** of `Lineups_Top1Pct`, except each lineup slot cell shows **`{name} ({dk_player_id})`** instead of **`{name} ({ownership_pct})`**.
- Must work across **all sports** supported by this repo (currently NFL + NBA), and be reusable for future sports.
- If a DKEntries mapping can’t be built, **still write the workbook** but **skip `DK_Lineups`** (and print a warning).

## Where to implement (shared, sport-agnostic)

- Add the functionality in the shared writer in [`src/shared/top1pct_core.py`](/home/john/showdown-optimizer/src/shared/top1pct_core.py) where the workbook is produced:
  - It currently writes `Lineups_Top1Pct` via `lineups_with_top1.to_excel(...)`.
  - We’ll generate a second dataframe `dk_lineups_df` and write it as `sheet_name="DK_Lineups"`.

## Data mapping strategy (DKEntries CSV → IDs)

- Source DK IDs from the **embedded player dictionary** inside a DKEntries CSV using existing helpers:
  - [`src/shared/dkentries_core.py`](/home/john/showdown-optimizer/src/shared/dkentries_core.py)
    - `resolve_latest_dkentries_csv(data_root, explicit=None)`
    - `build_name_role_to_id_map(df, flex_role_label=...)` which returns `(player_name, roster_role) -> dk_player_id`.

### Slot-role handling per sport

- We must map lineup slots by role:
  - **CPT**: role `"CPT"`
  - **Non-CPT**: role is sport-dependent in DKEntries:
    - NFL Showdown uses `"FLEX"`
    - NBA Showdown uses `"UTIL"`
- In `top1pct_core`, infer the non-CPT role label from a **sport hint**:
  - Use `outputs_dir.name.lower()` (already used elsewhere in the file) to detect `"nba"` → `flex_role_label="UTIL"`.
  - Default all other sports to `flex_role_label="FLEX"`.
  - (Plan is extensible: if a future sport uses a different label, it can be added to this mapping table.)

## Formatting logic for `DK_Lineups`

- Implement a small helper in [`src/shared/top1pct_core.py`](/home/john/showdown-optimizer/src/shared/top1pct_core.py):
  - Determine slot columns (`cpt` + either `flex1..flex5` or `util1..util5`) using the existing `_resolve_non_cpt_slot_columns()`.
  - For each slot cell:
    - Parse raw name using existing `_parse_player_name(cell)` (already handles `"Name (34.5%)"`).
    - Lookup DK ID via `name_role_to_id[(name, role)] `where role is `"CPT"` for `cpt` and `flex_role_label` for non-CPT slots.
    - Emit formatted string: `f"{name} ({dk_id})"`.
- **Missing mapping behavior** (per your choice):
  - If we cannot find a usable DKEntries CSV (or mapping is empty), print a warning and do **not** write `DK_Lineups`.
  - If mapping exists but any lineup slot name/role is missing, print a warning (include a small sample of missing keys) and **skip `DK_Lineups`** to avoid producing misleading IDs.

## CLI / configuration changes (so all sports can use it)

- Add an optional argument plumbed through the sport wrappers:
  - `--dkentries-csv` (optional): explicit path to a DKEntries CSV to use for mapping.
  - If omitted, default to **latest** DKEntries CSV under `config.DATA_DIR / 'dkentries'` using `resolve_latest_dkentries_csv`.
- Update wrappers:
  - [`src/nfl/top1pct_finish_rate.py`](/home/john/showdown-optimizer/src/nfl/top1pct_finish_rate.py)
  - [`src/nba/top1pct_finish_rate_nba.py`](/home/john/showdown-optimizer/src/nba/top1pct_finish_rate_nba.py)
  - Both already pass `data_dir=config.DATA_DIR` into `top1pct_core.run_top1pct`; we’ll extend the call signature to also pass `dkentries_csv=args.dkentries_csv`.

## Documentation update

- Update [`README.md`](/home/john/showdown-optimizer/README.md) in the top1pct section(s):
  - Mention the new `DK_Lineups` sheet.
  - Document `--dkentries-csv` and the default resolution behavior.

## Validation (lightweight)

- Add/extend a small unit test to ensure the transformation logic is correct and stable:
  - Prefer a new test file like [`tests/test_top1pct_dk_lineups_sheet.py`](/home/john/showdown-optimizer/tests/test_top1pct_dk_lineups_sheet.py).
  - Use a tiny fake `Lineups_Top1Pct` dataframe with one lineup row and a fake DKEntries dataframe containing the Name/ID/Roster Position dictionary rows.
  - Assert:
    - Output dataframe matches input columns.
    - Slot columns are `Name (id)`.
    - When mapping is incomplete, we skip creation (helper returns `None` / raises a sentinel handled by caller).

## Data flow (for clarity)

```mermaid
flowchart TD
  runTop1pct[run_top1pct] --> writeTop1pct[Write Lineups_Top1Pct]
  runTop1pct --> resolveDkentries[Resolve DKEntries CSV]
  resolveDkentries --> buildMap[Build (Name,Role)->ID map]
  buildMap --> buildDkSheet[Build DK_Lineups dataframe]
  buildDkSheet --> writeDkSheet[Write DK_Lineups]
  resolveDkentries -->|missing_or_bad| warnSkip[Warn and skip DK_Lineups]
  buildDkSheet -->|missing_ids| warnSkip
```