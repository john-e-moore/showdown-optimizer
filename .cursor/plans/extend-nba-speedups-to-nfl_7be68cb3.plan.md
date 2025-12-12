---
name: extend-nba-speedups-to-nfl
overview: Extend the new NBA optimizer speed/field features (profiling, parallel multi-stack, warm-start & solver tolerances, optional field augmentation, and env-based wiring in the run script) to the NFL showdown optimizer and full pipeline.
todos:
  - id: nfl-prof-metrics
    content: Add timing, OPT_TIME logging, and optimizer metrics JSON output to src/nfl/showdown_optimizer_main.py, mirroring the NBA implementation.
    status: completed
  - id: nfl-parallel-stacks
    content: Implement by-stack-pattern parallelization with --num-workers and --parallel-mode in src/nfl/showdown_optimizer_main.py, including per-pattern timing metrics.
    status: completed
    dependencies:
      - nfl-prof-metrics
  - id: nfl-solver-knobs
    content: Expose --use-warm-start, --solver-max-seconds, and --solver-rel-gap in the NFL showdown optimizer CLI and thread them into optimize_showdown_lineups calls.
    status: completed
    dependencies:
      - nfl-prof-metrics
  - id: run-full-nfl-wiring
    content: Extend run_full.sh to pass NUM_WORKERS, PARALLEL_MODE, CHUNK_SIZE, SOLVER_MAX_SECONDS, SOLVER_REL_GAP, and USE_WARM_START env vars through to src/nfl.showdown_optimizer_main.
    status: completed
    dependencies:
      - nfl-parallel-stacks
      - nfl-solver-knobs
  - id: nfl-augment-cli
    content: Create src/nfl/augment_lineups_with_field.py to append quota-balanced field-style lineups to an NFL optimizer lineups workbook using shared field_builder and top1pct_core helpers.
    status: completed
  - id: run-full-nfl-augment
    content: Wire an optional EXTRA_FIELD_LINEUPS augmentation step into run_full.sh between optimization and NFL top1% scoring.
    status: completed
    dependencies:
      - nfl-augment-cli
  - id: docs-update-nfl
    content: Update README.md NFL sections to document new optimizer flags, profiling output, and the optional NFL field augmentation CLI and env vars.
    status: completed
    dependencies:
      - nfl-prof-metrics
      - nfl-parallel-stacks
      - nfl-solver-knobs
      - run-full-nfl-wiring
      - nfl-augment-cli
      - run-full-nfl-augment
---

# Extend NBA optimizer improvements to NFL pipeline

## Goals

- **Mirror the new NBA optimizer and pipeline features for NFL**:
- Lightweight profiling + metrics output around NFL lineup generation.
- **Parallelized multi-stack solving** with CLI flags for worker count/strategy.
- Expose **CBC warm-start and solver tolerance knobs** (time limit, MIP gap) at the NFL CLI and wire them into the shared core.
- Add an **optional field-style lineup augmentation** step into the NFL full pipeline, analogous to the NBA flow.
- Keep defaults backwards-compatible unless you explicitly opt into the new behavior.

## Target codepaths

- **NFL optimizer CLI**: [`src/nfl/showdown_optimizer_main.py`](/home/john/showdown-optimizer/src/nfl/showdown_optimizer_main.py)
- **Shared MILP core (already updated for NBA)**: [`src/shared/optimizer_core.py`](/home/john/showdown-optimizer/src/shared/optimizer_core.py)
- **NFL top1% wrapper**: [`src/nfl/top1pct_finish_rate.py`](/home/john/showdown-optimizer/src/nfl/top1pct_finish_rate.py)
- **NFL full pipeline script**: [`run_full.sh`](/home/john/showdown-optimizer/run_full.sh)
- **Shared field & top1% cores (for augmentation)**: [`src/shared/field_builder.py`](/home/john/showdown-optimizer/src/shared/field_builder.py), [`src/shared/top1pct_core.py`](/home/john/showdown-optimizer/src/shared/top1pct_core.py)

## Step 1: Add NFL optimizer profiling & metrics

- In `src/nfl/showdown_optimizer_main.py`:
- Wrap the existing optimization block with a `time.perf_counter()` timer (similar to NBA) and compute `elapsed` seconds for the whole run.
- Print a **single machine-parsable summary line** after solving, e.g.:  
`OPT_TIME seconds=X num_lineups_requested=Y num_lineups_generated=Z stack_mode=... chunk_size=... parallel_mode=... num_workers=... [solver_max_seconds=... solver_rel_gap=...]`.
- Construct a small `metrics` dict (total seconds, requested/generated lineups, stack mode, chunk size, parallel settings, solver knobs, and optionally per-pattern metrics once parallelization is added).
- Write this out as `lineups_<timestamp>_metrics.json` next to the NFL lineups workbook, mirroring the NBA layout.
- Keep human-readable timing and "Generated N lineups" output unchanged for ergonomics.

## Step 2: Parallelize NFL multi-stack mode (by stack pattern)

- In `src/nfl/showdown_optimizer_main.py`:
- Add CLI flags parallel to NBA:
  - `--num-workers` (int, default 1).
  - `--parallel-mode` with choices `none|by_stack_pattern` (default `none`).
- Refactor the **multi-stack loop** over `STACK_PATTERNS` to:
  - Build a small job description per pattern: `(pattern, n, team_a_count, team_b_count)` based on `STACK_PATTERN_COUNTS` and the parsed `--stack-weights`.
  - Extract a helper function (akin to NBA’s `_optimize_for_stack_pattern`) that:
  - Builds the stack constraint via `build_team_stack_constraint`.
  - Calls `optimize_showdown_lineups` with that constraint set and the NFL-specific `constraint_builders` from `build_custom_constraints()`.
  - Returns `(lineups_for_pattern, label, metrics_dict)` where `label` is from `_build_stack_pattern_label` and `metrics_dict` includes `pattern`, `seconds`, requested/generated counts, etc.
  - When `stack_mode == "multi" and parallel_mode == "by_stack_pattern" and num_workers > 1`:
  - Use a `ThreadPoolExecutor(max_workers=num_workers)` to submit one job per pattern.
  - Collect results in **submission order** to keep deterministic label ordering, then merge all lineups/labels into `all_lineups`/`all_labels`.
  - Otherwise, run the same helper sequentially as today.
- Preserve the **existing cross-pattern deduping** logic (by sorted player ID tuples) unchanged.
- Accumulate the per-pattern `metrics_dict` entries into the overall metrics JSON from Step 1.

## Step 3: Expose warm-start & solver tolerance knobs at NFL CLI

- Since `optimizer_core.optimize_showdown_lineups` already supports `use_warm_start`, `max_seconds`, and `rel_gap` from the NBA work, reuse those hooks for NFL:
- Extend the NFL optimizer CLI parser with:
  - `--use-warm-start` (flag) – passed through when `chunk_size <= 0` to reuse a single growing model efficiently.
  - `--solver-max-seconds` (float | None) – per-solve time limit passed to CBC (`maxSeconds`).
  - `--solver-rel-gap` (float | None) – relative MIP gap (`fracGap`, e.g., `0.005` for 0.5%) allowing early termination.
- When calling `optimize_showdown_lineups` for both single-pass and multi-stack paths, thread these values through in the same way as the NBA optimizer.
- Add concise help-text and keep defaults `None`/off so existing behavior remains fully optimal unless you opt in.
- Include these values in the OPT_TIME line and metrics JSON as done for NBA.

## Step 4: Wire NFL run script for parallelism & solver knobs

- In [`run_full.sh`](/home/john/showdown-optimizer/run_full.sh):
- Introduce environment-variable-based defaults (mirroring `run_full_nba.sh`):
  - `NUM_WORKERS` (default e.g. `5`) and `PARALLEL_MODE` (default `by_stack_pattern`) near the top along with `FIELD_SIZE`, `NUM_LINEUPS`, etc.
  - Optional: `SOLVER_MAX_SECONDS`, `SOLVER_REL_GAP`, `USE_WARM_START`, and `CHUNK_SIZE` to control the NFL optimizer from the script.
- Build argument arrays `OPT_PARALLEL` and `SOLVER_OPTS` similar to NBA:
  - For `STACK_MODE='multi'` and `NUM_WORKERS>1`, append `--parallel-mode`/`--num-workers` to the NFL optimizer call.
  - If solver env vars are set, append `--solver-max-seconds`, `--solver-rel-gap`, and `--use-warm-start` appropriately.
- Update the existing `python -m src.nfl.showdown_optimizer_main ...` invocation to include these arrays and a `--chunk-size` override when `CHUNK_SIZE` is set.
- Keep the run script’s behavior **unchanged** when the new env vars are left unset (default to current sequential, fully optimal behavior).

## Step 5: Add optional NFL field-style augmentation step (analogous to NBA)

- Implement an NFL-specific augmentation CLI, e.g. [`src/nfl/augment_lineups_with_field.py`](/home/john/showdown-optimizer/src/nfl/augment_lineups_with_field.py):
- Mirror the NBA script’s structure but adjust paths/naming for NFL:
  - Args: `--lineups-excel`, `--corr-excel`, `--extra-lineups`, `--output-excel`, `--random-seed`.
  - Use `top1pct_core._load_lineups_workbook`-style logic or direct `pandas` reads to get `Lineups`, `Ownership`, and `Projections` from the NFL optimizer workbook.
  - Use `top1pct_core._load_corr_workbook` to load the NFL correlations workbook.
  - Call `field_builder.build_quota_balanced_field(...)` to generate `extra_lineups` CPT+FLEX NFL lineups.
  - Use `top1pct_core._build_player_universe` and `_annotate_lineups_with_meta(..., sport="nfl")` to map these into the same schema as the NFL `Lineups` sheet, assigning new `rank` values after the existing optimizer lineups and tagging `stack_pattern`/`target_stack_pattern` as a field marker (e.g., `"field"`).
  - Concatenate optimizer + field-style lineups; write an augmented workbook that preserves the original sheets and updates only `Lineups`.
- Integrate this into `run_full.sh` as an **optional step between Steps 2 and 3**:
- Add an env var `EXTRA_FIELD_LINEUPS` (default `0`):
  - When `>0`, call `python -m src.nfl.augment_lineups_with_field` to create an augmented lineups workbook in the run directory, then pass that path into `top1pct_finish_rate` instead of the original lineups file.
  - When `0`, skip augmentation and operate on pure optimizer lineups as today.

## Step 6: Documentation touch-ups

- Update [`README.md`](/home/john/showdown-optimizer/README.md) NFL sections to document:
- New NFL optimizer flags: `--num-workers`, `--parallel-mode`, `--use-warm-start`, `--solver-max-seconds`, `--solver-rel-gap`, and the clarified `--chunk-size` semantics.
- The OPT_TIME line and metrics JSON for NFL, mirroring the NBA docs.
- The new NFL augmentation CLI (`src.nfl.augment_lineups_with_field`) and the `EXTRA_FIELD_LINEUPS`/parallelism-related env vars for `run_full.sh`.

## Implementation todos

- **nfl-prof-metrics**: Add timing, OPT_TIME logging, and metrics JSON to `src/nfl/showdown_optimizer_main.py`.
- **nfl-parallel-stacks**: Implement `--num-workers`/`--parallel-mode` and pattern-level parallelization + per-pattern metrics in `src/nfl/showdown_optimizer_main.py`.
- **nfl-solver-knobs**: Expose `--use-warm-start`, `--solver-max-seconds`, `--solver-rel-gap` at the NFL CLI and plumb them into `optimize_showdown_lineups`.
- **run-full-nfl-wiring**: Extend `run_full.sh` to pass new parallelism and solver flags/env vars into the NFL optimizer.
- **nfl-augment-cli**: Create `src/nfl/augment_lineups_with_field.py` and ensure it produces an augmented lineups workbook compatible with `top1pct_finish_rate`.
- **run-full-nfl-augment**: Add optional augmentation step in `run_full.sh` controlled by `EXTRA_FIELD_LINEUPS` and verify it plays nicely with the existing explicit-field top1% model.
- **docs-update-nfl**: Refresh `README.md` to describe the NFL optimizer’s new flags, profiling output, and augmentation options.