from __future__ import annotations

"""
Sport-agnostic core for diversifying DraftKings Showdown top1pct lineups.

This module operates on the standard top1pct Excel schema and writes
diversified lineups back under a caller-provided outputs directory.
"""

from pathlib import Path
from typing import Dict, FrozenSet, List, Tuple

import pandas as pd


LINEUPS_TOP1PCT_SHEET_NAME = "Lineups_Top1Pct"


def _resolve_latest_excel(directory: Path, explicit: str | None) -> Path:
    """
    Resolve an Excel file path, preferring an explicit argument when provided.
    """
    if explicit:
        path = Path(explicit)
        if not path.is_file():
            raise FileNotFoundError(f"Specified Excel file does not exist: {path}")
        return path

    candidates = sorted(directory.glob("*.xlsx"), key=lambda p: p.stat().st_mtime)
    if not candidates:
        raise FileNotFoundError(f"No .xlsx files found in directory: {directory}")
    return candidates[-1]


def _parse_player_name(cell: str) -> str:
    """
    Extract player name from a cell like 'Deebo Samuel (34.5%)'.
    """
    if not isinstance(cell, str):
        return str(cell)
    split_idx = cell.find(" (")
    if split_idx == -1:
        return cell.strip()
    return cell[:split_idx].strip()


def _load_top1pct_lineups(
    outputs_dir: Path,
    top1pct_excel: str | None,
) -> tuple[Path, pd.DataFrame]:
    """
    Load the Lineups_Top1Pct sheet from the latest or specified top1pct workbook.
    """
    top1pct_dir = outputs_dir / "top1pct"
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
    Build a frozenset of player names for a lineup row (CPT + 5 flex-style slots).
    """
    cols = ["cpt"] + [f"flex{j}" for j in range(1, 6)]
    names: List[str] = []
    for col in cols:
        if col not in row:
            raise KeyError(f"Expected column '{col}' in Lineups_Top1Pct sheet.")
        names.append(_parse_player_name(row[col]))
    return frozenset(names)


def _build_flex_set(row: pd.Series) -> FrozenSet[str]:
    """
    Build a frozenset of FLEX-only player names for a lineup row.
    """
    cols = [f"flex{j}" for j in range(1, 6)]
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
    max_flex_overlap: int | None = None,
    cpt_max_share: dict[str, float] | None = None,
) -> pd.DataFrame:
    """
    Greedily select up to num_lineups diversified lineups.

    Strategy:
      1. Filter to lineups with top1_pct_finish_rate >= min_top1_pct.
      2. Sort remaining lineups by top1_pct_finish_rate desc, then by
         lineup_projection desc as a tie-breaker when available.
      3. Iterate in sorted order, accepting a lineup if:
           - total overlap (CPT + FLEX) with every already-selected lineup is
             <= max_overlap; and
           - when max_flex_overlap is not None, FLEX-only overlap with every
             already-selected lineup is <= max_flex_overlap.
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
    candidates["_flex_set"] = candidates.apply(_build_flex_set, axis=1)

    sort_cols: List[str] = ["top1_pct_finish_rate"]
    ascending: List[bool] = [False]
    if "lineup_projection" in candidates.columns:
        sort_cols.append("lineup_projection")
        ascending.append(False)

    candidates.sort_values(by=sort_cols, ascending=ascending, inplace=True)

    selected_rows: List[Dict[str, object]] = []
    selected_sets: List[FrozenSet[str]] = []
    selected_flex_sets: List[FrozenSet[str]] = []
    # Track CPT counts to optionally enforce per-player CPT share caps.
    cpt_counts: Dict[str, int] = {}
    selected_count = 0

    for _, row in candidates.iterrows():
        # Parse CPT name for this candidate lineup.
        cpt_name = _parse_player_name(row["cpt"])
        player_set: FrozenSet[str] = row["_player_set"]
        flex_set: FrozenSet[str] = row["_flex_set"]
        # Enforce max_overlap constraint against all previously selected lineups.
        if any(len(player_set & s) > max_overlap for s in selected_sets):
            continue

        # Optionally enforce a FLEX-only overlap constraint.
        if max_flex_overlap is not None:
            if any(len(flex_set & s) > max_flex_overlap for s in selected_flex_sets):
                continue

        # Optionally enforce a CPT exposure cap relative to the diversified set.
        if cpt_max_share is not None:
            current_cpt_count = cpt_counts.get(cpt_name, 0)
            future_selected_count = selected_count + 1
            future_share = (current_cpt_count + 1) / future_selected_count
            max_share = cpt_max_share.get(cpt_name, 1.0)
            if future_share > max_share:
                continue

        selected_rows.append(row.to_dict())
        selected_sets.append(player_set)
        selected_flex_sets.append(flex_set)
        selected_count += 1
        cpt_counts[cpt_name] = cpt_counts.get(cpt_name, 0) + 1

        if len(selected_rows) >= num_lineups:
            break

    if not selected_rows:
        # No diversified lineups could be selected under the constraints.
        # Return an empty DataFrame with the original schema (sans helper columns).
        result = candidates.drop(columns=["_player_set", "_flex_set"])
        return result.iloc[0:0]

    result_df = pd.DataFrame(selected_rows)
    result_df = result_df.drop(columns=["_player_set", "_flex_set"], errors="ignore")
    # Preserve column order from the original DataFrame where possible.
    cols_original = [c for c in df.columns if c in result_df.columns]
    extras = [c for c in result_df.columns if c not in cols_original]
    return result_df[cols_original + extras]


def _compute_exposure(selected_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute CPT / flex-style / total exposure across the diversified lineups.

    Exposure is reported as a percentage of selected lineups:
      - cpt_exposure: 100 * (# of lineups where player is CPT) / N
      - flex_exposure: 100 * (# of lineups where player appears in any non-CPT
        flex-style slot) / N
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
    from collections import defaultdict

    cpt_counts: Dict[str, int] = defaultdict(int)
    flex_counts: Dict[str, int] = defaultdict(int)

    for _, row in selected_df.iterrows():
        # CPT
        cpt_name = _parse_player_name(row["cpt"])
        cpt_counts[cpt_name] += 1

        # Flex-style slots
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


def run_diversify(
    num_lineups: int,
    outputs_dir: Path,
    *,
    min_top1_pct: float = 1.0,
    max_overlap: int = 4,
    top1pct_excel: str | None = None,
    max_flex_overlap: int | None = None,
    cpt_max_share: dict[str, float] | None = None,
) -> Path:
    """
    Execute diversification over top1pct lineups for a given sport.

    Args:
        num_lineups: Number of lineups (X) to select.
        outputs_dir: Root outputs directory for the sport
                     (e.g. outputs/nfl or outputs/nba).
        min_top1_pct: Minimum top1_pct_finish_rate (in percent) to keep.
        max_overlap: Maximum overlap allowed between any pair of lineups.
        top1pct_excel: Optional explicit path to a top1pct workbook.
        max_flex_overlap: Optional maximum FLEX-only overlap allowed between
            any pair of selected lineups. If None, only total overlap is used.

    Returns:
        Path to the written diversified lineups Excel workbook.
    """
    workbook_path, lineups_df = _load_top1pct_lineups(outputs_dir, top1pct_excel)

    print(f"Using top1pct workbook: {workbook_path}")
    print(
        "Selecting up to "
        f"{num_lineups} lineups with "
        f"top1_pct_finish_rate >= {min_top1_pct:.2f}% "
        f"and max_overlap <= {max_overlap}"
        + (
            f" and max_flex_overlap <= {max_flex_overlap}"
            if max_flex_overlap is not None
            else ""
        )
    )

    selected_df = _greedy_diversified_selection(
        df=lineups_df,
        num_lineups=num_lineups,
        max_overlap=max_overlap,
        min_top1_pct=min_top1_pct,
        max_flex_overlap=max_flex_overlap,
        cpt_max_share=cpt_max_share,
    )

    exposure_df = _compute_exposure(selected_df)

    # Write diversified lineups under a dedicated 'diversified' subdirectory so
    # they do not collide with the raw top1pct workbooks.
    diversified_dir = outputs_dir / "diversified"
    diversified_dir.mkdir(parents=True, exist_ok=True)

    # Include num_lineups in the filename; timestamp is already part of the
    # underlying top1pct workbook used for selection.
    output_path = diversified_dir / f"top1pct_diversified_{num_lineups}.xlsx"

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


__all__ = ["run_diversify"]


