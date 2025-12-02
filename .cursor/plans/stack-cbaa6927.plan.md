<!-- cbaa6927-0099-4f14-96a3-b210a1c90c27 ecefbf21-4c3e-4b6f-bbc7-6b5c7348d6a9 -->
# Multi-stack Showdown lineup generation

## Overview

Implement a "multi-stack" mode for the Showdown optimizer that splits the requested number of lineups across stack patterns (5|1, 4|2, 3|3, 2|4, 1|5), runs the MILP separately for each stack pattern with appropriate constraints, and then merges all lineups into a single Excel workbook. The allocation across patterns will be controlled by user-configurable weights, and the final Excel output will remain compatible with the existing top-1% simulator.

## Files to touch

- [src/showdown_optimizer_main.py](/home/john/showdown-optimizer/src/showdown_optimizer_main.py)
- [src/showdown_constraints.py](/home/john/showdown-optimizer/src/showdown_constraints.py)
- Potentially [src/lineup_optimizer.py](/home/john/showdown-optimizer/src/lineup_optimizer.py) if we need small helper utilities for constraints
- [README.md](/home/john/showdown-optimizer/README.md) (document new CLI usage)

## Step-by-step plan

### 1. Extend CLI and configuration for multi-stack mode

- **Add CLI flags** in `showdown_optimizer_main.py`:
- `--stack-mode` with options like `none` (current behavior) and `multi` (new behavior).
- `--stack-weights` to specify relative weights for the five patterns, e.g. `5|1=0.3,4|2=0.25,3|3=0.2,2|4=0.15,1|5=0.1`.
- **Define reasonable defaults**: if `--stack-mode=multi` is used without `--stack-weights`, default to equal weights (0.2 each) that effectively implement "divide by 5" behavior you described.
- **Parse weights** into a normalized mapping `{"5|1": w1, "4|2": w2, "3|3": w3, "2|4": w4, "1|5": w5}` and validate they are non-negative and not all zero.

### 2. Compute per-pattern lineup counts

- Given `num_lineups` and the normalized weights, compute integer lineup counts per pattern:
- Start with `base_n[p] = floor(num_lineups * weight[p])` for each pattern.
- Distribute any remaining lineups (due to rounding) to patterns in a fixed priority order (e.g., favoring heavier stacks or simply the order `5|1, 4|2, 3|3, 2|4, 1|5`).
- Skip patterns whose computed count is zero to avoid unnecessary solver runs.

### 3. Implement stack pattern constraint builders

- In `showdown_constraints.py` (or a small helper module), implement utilities to build per-run stack constraints using the two-team assumption:
- First, detect the two teams from `player_pool` (already done in `showdown_optimizer_main.py` for exposures) and fix a consistent ordering, e.g. `team_a`, `team_b`.
- For each pattern:
- **5|1 targeting team A**: total players from `team_a` (CPT + FLEX) = 5 and from `team_b` = 1.
- **4|2 targeting team A**: team A count = 4, team B count = 2.
- **3|3**: team A count = 3, team B count = 3 (symmetric; only one run needed).
- **2|4 targeting team B**: team A count = 2, team B count = 4.
- **1|5 targeting team B**: team A count = 1, team B count = 5.
- Expose a function like `build_stack_constraints(pattern: str, primary_team: str, secondary_team: str) -> ConstraintBuilder` that adds the appropriate equality constraints to the MILP using the existing `PlayerPool`, `CPT_SLOT`, and `SLOTS` abstractions.

### 4. Orchestrate multiple optimizer runs in `showdown_optimizer_main`

- Refactor `main()` so that after loading the player pool and base `constraint_builders = build_custom_constraints()`, it:
- If `stack-mode` is `none` (default), preserves existing single-run behavior.
- If `stack-mode` is `multi`:
- Determines the two teams (`team_a`, `team_b`).
- Iterates over the ordered patterns `["5|1", "4|2", "3|3", "2|4", "1|5"]` along with their computed lineup counts.
- For each pattern with `n > 0`, constructs `pattern_constraint_builders = constraint_builders + [stack_constraint_for_this_pattern]`.
- Calls `optimize_showdown_lineups(...)` once per pattern with `num_lineups=n` and `constraint_builders=pattern_constraint_builders`, collecting the resulting `lineups` along with metadata `{pattern, team_a, team_b}`.
- Concatenate all pattern-specific lineups into a single `all_lineups` list, preserving a well-defined ordering (e.g., by overall projection regardless of pattern, or by pattern then projection).

### 5. Ensure uniqueness and annotate stack patterns

- **Uniqueness**:
- Rely on the existing no-duplicate constraints within `optimize_showdown_lineups` to avoid duplicates inside each pattern run.
- As a safety check when combining `all_lineups`, deduplicate by canonical player ID tuples (sorted CPT+FLEX IDs) in Python before computing exposures and writing Excel; in practice, cross-pattern duplicates should not occur because stack patterns are mutually exclusive.
- **Annotation**:
- When building `lineup_rows` for the `Lineups` sheet, add an extra column such as `target_stack_pattern` (e.g., `"5|1_A-heavy"`, `"4|2_A-heavy"`, `"3|3"`, `"2|4_B-heavy"`, `"1|5_B-heavy"`) to record which run produced each lineup.
- Keep the existing `stack` column (derived from actual team counts) unchanged so downstream tooling like `top1pct_finish_rate.py` continues to work without modification.

### 6. Recompute exposures and write combined Excel

- After deduplicating and merging `all_lineups`, reuse the existing exposure and ownership logic in `showdown_optimizer_main.py`:
- Compute `cpt_counts` and `flex_counts` over `all_lineups` combined.
- Build `exposure_df` and `ownership_df` exactly as before.
- Build `lineups_df` from `all_lineups`, including the new `target_stack_pattern` column but preserving all existing columns.
- Continue writing a single Excel workbook with sheets `Projections`, `Ownership`, `Exposure`, and `Lineups` to `outputs/lineups/lineups_<timestamp>.xlsx` so that `top1pct_finish_rate.py` can consume it unchanged.

### 7. Update documentation

- In `README.md`, update the "Running the lineup optimizer" section to:
- Describe the new `--stack-mode multi` behavior and the default equal-weight split.
- Show an example command with custom `--stack-weights` and a large `--num-lineups` (e.g., 1000).
- Explain the meaning of the `target_stack_pattern` column in the `Lineups` sheet and how it interacts with the existing `stack` column.

## Implementation todos

- **cli-flags**: Add `--stack-mode` and `--stack-weights` flags and parsing logic in `showdown_optimizer_main.py`.
- **stack-weights**: Implement weight normalization and integer lineup count allocation across patterns.
- **stack-constraints**: Implement per-pattern stack constraint builders using team-level player counts.
- **multi-run-orchestration**: Refactor `main()` to loop over patterns, run the optimizer once per pattern, and collect all lineups.
- **dedupe-and-annotate**: Deduplicate combined lineups by composition and add a `target_stack_pattern` column when building `lineups_df`.
- **docs-update**: Update `README.md` to document multi-stack usage and any new columns in the Excel output.

### To-dos

- [ ] Add --stack-mode and --stack-weights CLI flags and parsing logic in showdown_optimizer_main.py.
- [ ] Implement weight normalization and integer lineup count allocation across 5|1, 4|2, 3|3, 2|4, 1|5 patterns.
- [ ] Implement per-pattern stack constraint builders that enforce exact team counts for each stack pattern using PlayerPool and PuLP.
- [ ] Refactor showdown_optimizer_main.main() to loop over stack patterns, run optimize_showdown_lineups once per pattern, and collect all lineups.
- [ ] Deduplicate combined lineups by player composition and add a target_stack_pattern column when building the Lineups sheet.
- [ ] Update README.md to document multi-stack mode, stack weight configuration, and the new Excel column semantics.