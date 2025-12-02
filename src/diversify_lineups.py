from __future__ import annotations

"""
Post-process Showdown lineups with top 1% finish rates to select a diversified
subset of X lineups.

This module:
  1. Loads the latest top1pct workbook from outputs/top1pct/*.xlsx (or a
     user-specified workbook path).
  2. Filters to lineups with top1_pct_finish_rate >= min_top1_pct.
  3. Represents each lineup as a set of player names (CPT + 5 FLEX).
  4. Greedily selects up to num_lineups lineups, preferring higher
     top1_pct_finish_rate while enforcing a maximum player-overlap constraint.
  5. Writes the selected lineups to a new Excel workbook under outputs/top1pct/.
"""

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Dict, FrozenSet, List
import pandas as pd

from . import config
from .top1pct_finish_rate import _parse_player_name, _resolve_latest_excel


LINEUPS_TOP1PCT_SHEET_NAME = "Lineups_Top1Pct"


def _load_top1pct_lineups(
    top1pct_excel: str | None,
) -> tuple[Path, pd.DataFrame]:
    """
    Load the Lineups_Top1Pct sheet from the latest or specified top1pct workbook.
    """
    top1pct_dir = config.OUTPUTS_DIR / "top1pct"
    workbook_path = _resolve_latest_excel(top1pct_dir, top1pct_excel)

    xls = pd.ExcelFile(workbook_path)
    try:
        lineups_df = pd.read_excel(xls, sheet_name=LINEUPS_TOP1PCT_SHEET_NAME)
    except ValueError as exc:
        raise KeyError(
            f"Top1pct workbook missing '{LINEUPS_TOP1PCT_SHEET_NAME}' sheet: "
            f"{workbook_path}"
        ) from exc

    return workbook_path, lineups_df


def _build_player_set(row: pd.Series) -> FrozenSet[str]:
    """
    Build a frozenset of player names for a lineup row (CPT + 5 FLEX).
    """
    cols = ["cpt"] + [f"flex{j}" for j in range(1, 6)]
    names: List[str] = []
    for col in cols:
        if col not in row:
            raise KeyError(f"Expected column '{col}' in Lineups_Top1Pct sheet.")
        names.append(_parse_player_name(row[col]))
    return frozenset(names)


def _greedy_diversified_selection(
    df: pd.DataFrame,
    num_lineups: int,
    max_overlap: int,
    min_top1_pct: float,
) -> pd.DataFrame:
    """
    Greedily select up to num_lineups diversified lineups.

    Strategy:
      1. Filter to lineups with top1_pct_finish_rate >= min_top1_pct.
      2. Sort remaining lineups by top1_pct_finish_rate desc, then by
         lineup_projection desc as a tie-breaker when available.
      3. Iterate in sorted order, accepting a lineup if its overlap (number of
         shared players) with every already-selected lineup is <= max_overlap.
    """
    if "top1_pct_finish_rate" not in df.columns:
        raise KeyError(
            "Expected 'top1_pct_finish_rate' column in Lineups_Top1Pct sheet."
        )

    candidates = df[df["top1_pct_finish_rate"] >= min_top1_pct].copy()
    if candidates.empty:
        # No lineups meet the threshold; return an empty DataFrame with same schema.
        return candidates

    # Build player sets for diversification.
    candidates = candidates.copy()
    candidates["_player_set"] = candidates.apply(_build_player_set, axis=1)

    sort_cols: List[str] = ["top1_pct_finish_rate"]
    ascending: List[bool] = [False]
    if "lineup_projection" in candidates.columns:
        sort_cols.append("lineup_projection")
        ascending.append(False)

    candidates.sort_values(by=sort_cols, ascending=ascending, inplace=True)

    selected_rows: List[Dict[str, object]] = []
    selected_sets: List[FrozenSet[str]] = []

    for _, row in candidates.iterrows():
        player_set: FrozenSet[str] = row["_player_set"]
        # Enforce max_overlap constraint against all previously selected lineups.
        if any(len(player_set & s) > max_overlap for s in selected_sets):
            continue

        selected_rows.append(row.to_dict())
        selected_sets.append(player_set)

        if len(selected_rows) >= num_lineups:
            break

    if not selected_rows:
        # No diversified lineups could be selected under the constraints.
        # Return an empty DataFrame with the original schema (sans helper column).
        result = candidates.drop(columns=["_player_set"])
        return result.iloc[0:0]

    result_df = pd.DataFrame(selected_rows)
    result_df = result_df.drop(columns=["_player_set"], errors="ignore")
    # Preserve column order from the original DataFrame where possible.
    cols_original = [c for c in df.columns if c in result_df.columns]
    extras = [c for c in result_df.columns if c not in cols_original]
    return result_df[cols_original + extras]


def _compute_exposure(selected_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute CPT / FLEX / total exposure across the diversified lineups.

    Exposure is reported as a percentage of selected lineups:
      - cpt_exposure: 100 * (# of lineups where player is CPT) / N
      - flex_exposure: 100 * (# of lineups where player appears in any FLEX) / N
      - total_exposure: cpt_exposure + flex_exposure
    """
    if selected_df.empty:
        return pd.DataFrame(
            columns=["player", "cpt_exposure", "flex_exposure", "total_exposure"]
        )

    required_cols = ["cpt"] + [f"flex{j}" for j in range(1, 6)]
    missing = [c for c in required_cols if c not in selected_df.columns]
    if missing:
        raise KeyError(
            f"Expected lineup columns {missing} to compute exposure in "
            "Lineups_Diversified sheet."
        )

    n_lineups = len(selected_df)
    cpt_counts: Dict[str, int] = defaultdict(int)
    flex_counts: Dict[str, int] = defaultdict(int)

    for _, row in selected_df.iterrows():
        # CPT
        cpt_name = _parse_player_name(row["cpt"])
        cpt_counts[cpt_name] += 1

        # FLEX slots
        for col in [f"flex{j}" for j in range(1, 6)]:
            flex_name = _parse_player_name(row[col])
            flex_counts[flex_name] += 1

    players = sorted(set(cpt_counts) | set(flex_counts))
    rows: List[Dict[str, float | str]] = []
    for name in players:
        cpt_share = cpt_counts[name] / n_lineups if n_lineups > 0 else 0.0
        flex_share = flex_counts[name] / n_lineups if n_lineups > 0 else 0.0
        total_share = cpt_share + flex_share
        rows.append(
            {
                "player": name,
                "cpt_exposure": 100.0 * cpt_share,
                "flex_exposure": 100.0 * flex_share,
                "total_exposure": 100.0 * total_share,
            }
        )

    exposure_df = pd.DataFrame(rows)
    exposure_df.sort_values(
        by=["total_exposure", "player"], ascending=[False, True], inplace=True
    )
    return exposure_df


def run(
    num_lineups: int,
    *,
    min_top1_pct: float = 1.0,
    max_overlap: int = 4,
    top1pct_excel: str | None = None,
) -> Path:
    """
    Execute diversification over top1pct lineups.

    Args:
        num_lineups: Number of lineups (X) to select.
        min_top1_pct: Minimum top1_pct_finish_rate (in percent) to keep.
        max_overlap: Maximum number of overlapping players allowed between any
            pair of selected lineups (0–6 for Showdown).
        top1pct_excel: Optional explicit path to a top1pct workbook. If None,
            the latest .xlsx file in outputs/top1pct/ is used.

    Returns:
        Path to the written diversified lineups Excel workbook.
    """
    workbook_path, lineups_df = _load_top1pct_lineups(top1pct_excel)

    print(f"Using top1pct workbook: {workbook_path}")
    print(
        "Selecting up to "
        f"{num_lineups} lineups with "
        f"top1_pct_finish_rate >= {min_top1_pct:.2f}% "
        f"and max_overlap <= {max_overlap}"
    )

    selected_df = _greedy_diversified_selection(
        df=lineups_df,
        num_lineups=num_lineups,
        max_overlap=max_overlap,
        min_top1_pct=min_top1_pct,
    )

    exposure_df = _compute_exposure(selected_df)

    outputs_dir = config.OUTPUTS_DIR / "top1pct"
    outputs_dir.mkdir(parents=True, exist_ok=True)

    # Include num_lineups in the filename; timestamp is already part of the
    # underlying top1pct workbook used for selection.
    output_path = outputs_dir / f"top1pct_diversified_{num_lineups}.xlsx"

    print(f"Writing {len(selected_df)} diversified lineups to {output_path}...")
    with pd.ExcelWriter(output_path, engine="xlsxwriter") as writer:
        selected_df.to_excel(
            writer,
            sheet_name="Lineups_Diversified",
            index=False,
        )
        exposure_df.to_excel(
            writer,
            sheet_name="Exposure",
            index=False,
        )

    print("Done.")
    return output_path


def _parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Select a diversified subset of NFL Showdown lineups based on "
            "top 1% finish rate and player-overlap constraints."
        )
    )
    parser.add_argument(
        "--num-lineups",
        type=int,
        required=True,
        help="Number of lineups (X) to select.",
    )
    parser.add_argument(
        "--min-top1-pct",
        type=float,
        default=1.0,
        help=(
            "Minimum top1_pct_finish_rate (in percent) required to keep a "
            "lineup (default: 1.0)."
        ),
    )
    parser.add_argument(
        "--max-overlap",
        type=int,
        default=4,
        help=(
            "Maximum number of overlapping players allowed between any pair of "
            "selected lineups (0–6 for Showdown; default: 4)."
        ),
    )
    parser.add_argument(
        "--top1pct-excel",
        type=str,
        default=None,
        help=(
            "Optional path to a top1pct workbook under outputs/top1pct/. "
            "If omitted, the most recent .xlsx file in that directory is used."
        ),
    )
    return parser.parse_args(argv)


def main(argv: List[str] | None = None) -> None:
    args = _parse_args(argv)
    run(
        num_lineups=args.num_lineups,
        min_top1_pct=args.min_top1_pct,
        max_overlap=args.max_overlap,
        top1pct_excel=args.top1pct_excel,
    )


if __name__ == "__main__":
    main()


