---
name: Salary-capped field builder
overview: Enforce DraftKings showdown salary constraints (min/max) during explicit field lineup generation without breaking existing quota/correlation logic, and add a 'Field Meta' sheet to the top1pct output workbook summarizing field ownership, salary/projection, and stack distributions.
todos:
  - id: salary-feasibility
    content: Add salary/min-salary feasibility checks to CPT and FLEX sampling and compute lineup salary with CPT 1.5× weighting.
    status: completed
  - id: retry-relax-quotas
    content: Refactor field building to be transactional per lineup, add retry loop, and implement quota relaxation after N attempts while never violating salary constraints.
    status: completed
    dependencies:
      - salary-feasibility
  - id: field-meta-sheet
    content: In run_top1pct explicit mode, compute field ownership/salary/projection/stack distributions and write a new 'Field Meta' sheet into the output workbook.
    status: completed
  - id: smoke-tests
    content: Update/add lightweight smoke tests to cover salary constraints and presence of 'Field Meta' sheet for explicit field mode.
    status: completed
    dependencies:
      - salary-feasibility
      - field-meta-sheet
---

# Enforce salary cap in field builder + add Field Meta sheet

## Goals

- **Field builder must only accept lineups whose total DK showdown salary satisfies** `min_salary <= salary <= salary_cap` using CPT weighting (1.5× CPT salary, 1× each FLEX).
- Preserve existing field construction behavior as much as possible (quotas → value/corr weights → fallbacks), but **never violate salary constraints**.
- In `top1pct_lineups_*.xlsx`, add a **`Field Meta`** tab summarizing the explicit field that was generated.

## Key files to change

- [src/shared/field_builder.py](/home/john/showdown-optimizer/src/shared/field_builder.py)
- [src/shared/top1pct_core.py](/home/john/showdown-optimizer/src/shared/top1pct_core.py)
- (Optional) [tests_top1pct_field_model.py](/home/john/showdown-optimizer/tests_top1pct_field_model.py)

## Implementation plan

## 1) Make salary constraints part of the sampling process (not a post-filter)

**Why**: Today the builder samples lineups and decrements quotas as it goes; a post-hoc “reject if salary too high” would either drift quotas (bad) or require rollback logic (error-prone). Salary-aware sampling avoids generating impossible lineups and keeps quotas consistent.

### 1.1 Add small helper functions in `field_builder.py`

- **Compute showdown lineup salary**: `lineup_salary = 1.5*salary[cpt] + sum(salary[flex])`.
- **Feasibility bounds for remaining slots**:
- For a partial lineup with `used_salary` and `slots_left`, compute:
  - `min_possible = used_salary + sum_of_smallest_k(eligible_salaries, slots_left)`
  - `max_possible = used_salary + sum_of_largest_k(eligible_salaries, slots_left)`
- A partial lineup is feasible iff:
  - `min_possible <= cfg.salary_cap` (can still stay under cap)
  - `max_possible >= cfg.min_salary` (can still reach min spend)
- Keep this as a **cheap bound check** (not an exact knapsack), since we’re sampling many lineups.

### 1.2 Salary-aware CPT sampling

Update `_sample_cpt(...)` to **exclude CPT candidates** that cannot possibly lead to a valid full lineup under min/max salary using the bounds check (5 flex slots remaining).

### 1.3 Salary-aware FLEX sequential sampling

Update `_sample_flex_sequence(...)` to accept (or derive) `used_salary` and enforce:

- **Hard filter** for each candidate at each slot:
- `used_salary + salary[cand] <= cfg.salary_cap`
- and bounds-feasible for the remaining slots after selecting `cand`.
- Apply the same filter in the existing fallback branch (the branch that currently “ignores quotas”).

### 1.4 Stop mutating quotas until a lineup is accepted

Refactor `build_quota_balanced_field(...)` so each lineup is built “transactionally”:

- For each lineup:
- Attempt to sample a lineup using the current `remaining_cpt`/`remaining_flex` **without decrementing them permanently**.
- Only after producing a valid salary line, **commit quota decrements**.

## 2) Add retry + quota-relax behavior (Option B)

Implement the requested behavior in `build_quota_balanced_field(...)`:

- Add config knobs to `FieldBuilderConfig` (defaults conservative):
- `max_attempts_per_lineup: int` (e.g. 50–200)
- `relax_quotas_after_attempts: int` (e.g. 10)
- Attempt loop per lineup:
- **Phase 1 (strict quotas)**: normal behavior.
- **Phase 2 (relax quotas)**: temporarily soften quota enforcement **for the lineup under construction only** (e.g., treat quota weights as flat or reduce effective `beta`), while keeping:
  - no duplicates,
  - correlation/value/rules,
  - **strict salary feasibility**.
- If a lineup still can’t be produced after `max_attempts_per_lineup`, raise a clear error:
  - include `salary_cap`, `min_salary`, and a short diagnostic (e.g. number of players, salary min/max, whether all remaining quotas are exhausted).

## 3) Add `Field Meta` sheet to `top1pct_lineups_*.xlsx`

In [src/shared/top1pct_core.py](/home/john/showdown-optimizer/src/shared/top1pct_core.py) within `run_top1pct(...)`:

- When `field_model == "explicit"` (i.e., `field_lineups_with_meta` is not `None`):
- Compute **CPT ownership** from `field_lineups_df["cpt"]`:
  - count and percent of lineups.
- Compute **FLEX ownership** from the five non-CPT columns of `field_lineups_df` (builder output uses `flex1..flex5`):
  - count and percent across all FLEX slots (denominator = `5 * field_size`).
- Compute **average lineup projection and salary** from `field_lineups_with_meta["lineup_projection"]` and `["lineup_salary"]`.
- Compute **stack distribution** from `field_lineups_with_meta["stack"]`:
  - counts and percent per stack string (e.g., `"4|2"`, `"3|3"`, etc.).
- Write to the existing Excel writer:
- keep existing tabs: `Lineups_Top1Pct`, `Meta`, `Field_Lineups`
- add new tab: **`Field Meta`**
- Layout: write a small summary table at top, then CPT ownership, FLEX ownership, and stack distribution below it using `startrow` offsets.

## 4) Add/adjust smoke coverage (optional but recommended)

Update [tests_top1pct_field_model.py](/home/john/showdown-optimizer/tests_top1pct_field_model.py) to cover the new constraints:

- Build a toy slate with **>= 6 distinct players** (so lineup construction is feasible).
- Pass a `FieldBuilderConfig` with a small `salary_cap`/`min_salary` window and assert all generated lineups satisfy it.
- When running `run_top1pct(field_model="explicit")`, assert the output workbook includes the `Field Meta` sheet.

## Notes / design choices

- We enforce salary constraints **during sampling**, not by post-filtering, to avoid quota drift.
- Quota relaxation only changes *how strongly* we follow remaining quotas; it does **not** permit invalid salaries.
- Stack distribution uses the already-computed `stack` string from `_annotate_lineups_with_meta(...)`, so we don’t introduce a second stack-definition.