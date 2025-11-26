from __future__ import annotations

"""
Lightweight diagnostics helpers for writing dataframe snapshots and summaries.

Snapshots are written under the project-level `diagnostics/` directory, grouped
by pipeline step. Each snapshot consists of:
  - A Parquet file with up to `max_rows` rows.
  - A JSON summary with row/column counts, dtypes, and simple stats for key
    categorical columns (season, week, team, position) when present.
"""

import json
from pathlib import Path
from typing import Iterable, Mapping, Any

import pandas as pd

from . import config


def _safe_ensure_dir(path: Path) -> None:
    """
    Create a directory if diagnostics are enabled.
    """
    path.mkdir(parents=True, exist_ok=True)


def write_df_snapshot(
    df: pd.DataFrame,
    name: str,
    step: str,
    *,
    max_rows: int = 1000,
) -> None:
    """
    Write a small snapshot of `df` plus summary statistics to diagnostics/.

    Parameters
    ----------
    df : pd.DataFrame
        The dataframe to snapshot.
    name : str
        Logical name for this dataframe (e.g. \"player_games\", \"pairwise\").
    step : str
        Pipeline step identifier (e.g. \"load\", \"fantasy_scoring\").
    max_rows : int, optional
        Maximum number of rows to persist in the Parquet snapshot.
    """
    if not config.ENABLE_DIAGNOSTICS:
        return

    step_dir = config.DIAGNOSTICS_DIR / step
    _safe_ensure_dir(step_dir)

    # Data snapshot (CSV only)
    sample_df = df.head(max_rows).copy()
    csv_path = step_dir / f"{name}.csv"
    try:
        sample_df.to_csv(csv_path, index=False)
    except Exception:
        # Best-effort; if CSV writing fails we still try to write the summary.
        pass

    # Summary statistics
    summary: dict[str, Any] = {
        "n_rows": int(df.shape[0]),
        "n_cols": int(df.shape[1]),
        "columns": {col: str(dtype) for col, dtype in df.dtypes.items()},
    }

    # Basic describe for numeric columns
    try:
        num_desc = df.describe(include="number").to_dict()
        summary["numeric_describe"] = num_desc
    except Exception:
        summary["numeric_describe"] = {}

    # Simple value counts for a few key categorical columns, if present.
    key_cat_cols = [
        config.COL_SEASON,
        config.COL_WEEK,
        config.COL_TEAM,
        config.COL_POSITION,
    ]
    cat_counts: dict[str, Mapping[Any, int]] = {}
    for col in key_cat_cols:
        if col in df.columns:
            try:
                vc = df[col].value_counts(dropna=False).head(20)
                cat_counts[col] = {str(k): int(v) for k, v in vc.items()}
            except Exception:
                continue
    summary["categorical_value_counts"] = cat_counts

    summary_path = step_dir / f"{name}_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)



