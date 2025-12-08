from __future__ import annotations

"""Lightweight sanity tests for top1pct field modeling.

These are not meant to be exhaustive unit tests, but they provide quick
smoke coverage for both the analytic "mixture" field model and the explicit
quota-balanced field model used by ``shared.top1pct_core``.

They can be run ad-hoc via:

    python -m tests_top1pct_field_model

from the project root.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from src.shared import top1pct_core
from src.shared.field_builder import build_quota_balanced_field


def _make_toy_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    # Simple 3-player toy slate with made-up projections and ownership.
    players = ["A", "B", "C"]

    lineups_df = pd.DataFrame(
        [
            {"cpt": "A", "flex1": "B", "flex2": "C", "flex3": "A", "flex4": "B", "flex5": "C"},
            {"cpt": "B", "flex1": "A", "flex2": "C", "flex3": "A", "flex4": "B", "flex5": "C"},
        ]
    )

    ownership_df = pd.DataFrame(
        {
            "player": players,
            "cpt_ownership": [40.0, 35.0, 25.0],
            "flex_ownership": [60.0, 55.0, 45.0],
        }
    )

    projections_df = pd.DataFrame(
        {
            "Name": players,
            "Team": ["T1", "T1", "T2"],
            "Salary": [10000, 9000, 8000],
            "My Proj": [20.0, 18.0, 15.0],
        }
    )

    corr_matrix = np.eye(len(players))
    corr_df = pd.DataFrame(corr_matrix, index=players, columns=players)

    return lineups_df, ownership_df, projections_df, corr_df


def _run_field_builder_smoke() -> None:
    lineups_df, ownership_df, projections_df, corr_df = _make_toy_inputs()

    field_df = build_quota_balanced_field(
        field_size=100,
        ownership_df=ownership_df,
        sabersim_proj_df=projections_df,
        lineups_proj_df=projections_df,
        corr_df=corr_df,
        random_seed=123,
    )

    assert not field_df.empty, "Field builder returned no lineups."
    assert {"cpt", "flex1", "flex2", "flex3", "flex4", "flex5"}.issubset(
        field_df.columns
    )


def _run_top1pct_smoke(field_model: str) -> None:
    # Wire the toy inputs through ``run_top1pct`` using a temporary outputs dir.
    outputs_dir = Path("./_tmp_top1pct_test")
    outputs_dir.mkdir(parents=True, exist_ok=True)

    lineups_df, ownership_df, projections_df, corr_df = _make_toy_inputs()

    lineups_path = outputs_dir / "lineups" / "lineups_toy.xlsx"
    corr_path = outputs_dir / "correlations" / "corr_toy.xlsx"
    lineups_path.parent.mkdir(parents=True, exist_ok=True)
    corr_path.parent.mkdir(parents=True, exist_ok=True)

    with pd.ExcelWriter(lineups_path) as writer:
        lineups_df.to_excel(writer, sheet_name=top1pct_core.LINEUPS_SHEET_NAME, index=False)
        ownership_df.to_excel(writer, sheet_name=top1pct_core.OWNERSHIP_SHEET_NAME, index=False)
        projections_df.to_excel(writer, sheet_name=top1pct_core.PROJECTIONS_SHEET_NAME, index=False)

    with pd.ExcelWriter(corr_path) as writer:
        projections_df.to_excel(writer, sheet_name=top1pct_core.CORR_PROJECTIONS_SHEET_NAME, index=False)
        corr_df.to_excel(writer, sheet_name=top1pct_core.CORR_MATRIX_SHEET_NAME)

    output_path = top1pct_core.run_top1pct(
        field_size=100,
        outputs_dir=outputs_dir,
        lineups_excel=str(lineups_path),
        corr_excel=str(corr_path),
        num_sims=500,
        random_seed=42,
        field_model=field_model,
    )

    assert output_path.is_file(), f"Top1pct output not written for field_model={field_model}."


def main() -> None:
    _run_field_builder_smoke()
    _run_top1pct_smoke("mixture")
    _run_top1pct_smoke("explicit")
    print("tests_top1pct_field_model: OK")


if __name__ == "__main__":
    main()
