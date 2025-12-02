from __future__ import annotations

"""
Fill a DraftKings DKEntries CSV with diversified Showdown lineups.

This script:
  1. Resolves a DKEntries*.csv template (latest under data/dkentries/ by default).
  2. Resolves the latest diversified lineups workbook under outputs/top1pct/.
  3. Maps each lineup (CPT + 5 FLEX names) to DraftKings player IDs using the
     Name/ID/Roster Position dictionary embedded in the DKEntries CSV.
  4. Assigns lineups to DK entries in a fee-aware way, spreading player exposure
     across distinct Entry Fee tiers while favoring stronger lineups in
     higher-fee contests.
  5. Writes a DK-upload-ready CSV under outputs/dkentries/ with lineup slots
     formatted as '{player_name} ({player_id})'.
"""

import argparse
import csv
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from . import config, dkentries_utils
from .top1pct_finish_rate import _parse_player_name, _resolve_latest_excel


LINEUPS_DIVERSIFIED_SHEET = "Lineups_Diversified"
ENTRY_ID_COLUMN = "Entry ID"
ENTRY_FEE_COLUMN = "Entry Fee"


@dataclass(frozen=True)
class LineupRecord:
    idx: int
    players: Tuple[str, ...]  # CPT first, then FLEX1..FLEX5
    strength: float


def _find_mapping_columns(df: pd.DataFrame) -> Tuple[str, str, str]:
    """
    Locate the columns corresponding to DK's player dictionary:
      - Name
      - ID
      - Roster Position (CPT / FLEX)

    These appear in the DKEntries CSV rows after the actual entries. We locate
    them by scanning for cells equal to the header labels.
    """
    name_col: Optional[str] = None
    id_col: Optional[str] = None
    roster_pos_col: Optional[str] = None

    # Use position-based indexing + NumPy to avoid pandas' ambiguous truth-value
    # behaviour when dealing with duplicate column names or non-standard dtypes.
    for j, col in enumerate(df.columns):
        series = df.iloc[:, j].astype(str)
        values = series.to_numpy()

        if name_col is None and (values == "Name").any():
            name_col = col
        if id_col is None and (values == "ID").any():
            id_col = col
        if roster_pos_col is None and (values == "Roster Position").any():
            roster_pos_col = col

    missing = [lbl for lbl, col in [("Name", name_col), ("ID", id_col), ("Roster Position", roster_pos_col)] if col is None]
    if missing:
        raise KeyError(
            "Failed to locate DK player dictionary columns in DKEntries CSV; "
            f"could not find headers {missing} in any column."
        )

    return name_col, id_col, roster_pos_col


def _build_name_role_to_id_map(df: pd.DataFrame) -> Dict[Tuple[str, str], str]:
    """
    Build a mapping (player_name, roster_role) -> dk_player_id using the
    embedded player dictionary in the DKEntries CSV.

    roster_role is one of {"CPT", "FLEX"}.
    """
    name_col, id_col, roster_pos_col = _find_mapping_columns(df)

    mask = df[roster_pos_col].isin(["CPT", "FLEX"])
    if not mask.any():
        raise ValueError(
            "DKEntries CSV appears to be missing CPT/FLEX player dictionary rows."
        )

    mapping: Dict[Tuple[str, str], str] = {}
    for _, row in df[mask].iterrows():
        raw_name = row[name_col]
        raw_id = row[id_col]
        roster_role = row[roster_pos_col]

        if pd.isna(raw_name) or pd.isna(raw_id) or pd.isna(roster_role):
            continue

        name = str(raw_name).strip()
        role = str(roster_role).strip().upper()
        player_id = str(raw_id).strip()

        if not name or not player_id or role not in {"CPT", "FLEX"}:
            continue

        key = (name, role)
        if key in mapping and mapping[key] != player_id:
            # In the unlikely event of conflicting IDs, keep the first and warn.
            print(
                f"Warning: multiple IDs found for ({name!r}, {role}); "
                f"keeping {mapping[key]!r}, ignoring {player_id!r}."
            )
            continue
        mapping[key] = player_id

    if not mapping:
        raise ValueError(
            "Failed to build any (Name, Roster Position) -> ID mappings from "
            "DKEntries CSV."
        )

    return mapping


def _get_lineup_slot_columns(df: pd.DataFrame) -> List[int]:
    """
    Identify the 6 contiguous columns corresponding to CPT + 5 FLEX slots.

    We locate the first 'CPT' column and then take its positional index plus the
    next 5 indices. Returning positional indices (rather than column labels)
    avoids pandas' ambiguous behaviour when there are duplicate column names
    such as multiple 'FLEX' columns in the DKEntries template.
    """
    cols = list(df.columns)
    try:
        idx_cpt = cols.index("CPT")
    except ValueError as exc:
        raise KeyError("DKEntries CSV is missing required 'CPT' column.") from exc

    if idx_cpt + 5 >= len(cols):
        raise ValueError(
            "DKEntries CSV does not appear to contain 6 lineup slot columns "
            "(CPT + 5 FLEX)."
        )

    # Return integer column indices for CPT followed by 5 FLEX slots.
    return list(range(idx_cpt, idx_cpt + 6))


def _load_diversified_lineups(
    diversified_excel: Optional[str] = None,
) -> Tuple[Path, pd.DataFrame, List[LineupRecord]]:
    """
    Load diversified lineups from the latest (or specified) top1pct workbook.
    """
    top1pct_dir = config.OUTPUTS_DIR / "top1pct"
    workbook_path = _resolve_latest_excel(top1pct_dir, diversified_excel)

    xls = pd.ExcelFile(workbook_path)
    try:
        df = pd.read_excel(xls, sheet_name=LINEUPS_DIVERSIFIED_SHEET)
    except ValueError as exc:
        raise KeyError(
            f"Diversified workbook missing '{LINEUPS_DIVERSIFIED_SHEET}' sheet: "
            f"{workbook_path}"
        ) from exc

    required_cols = ["cpt"] + [f"flex{j}" for j in range(1, 6)]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise KeyError(
            f"Diversified lineups sheet missing required columns {missing}."
        )

    # Build LineupRecord list with a simple strength metric:
    #   primary: top1_pct_finish_rate (higher is better)
    #   secondary: lineup_projection when available
    has_top1 = "top1_pct_finish_rate" in df.columns
    has_proj = "lineup_projection" in df.columns

    records: List[LineupRecord] = []
    for idx, row in df.iterrows():
        names: List[str] = []
        names.append(_parse_player_name(row["cpt"]))
        for j in range(1, 6):
            names.append(_parse_player_name(row[f"flex{j}"]))

        base_strength = float(row["top1_pct_finish_rate"]) if has_top1 else 0.0
        proj_component = float(row["lineup_projection"]) if has_proj and pd.notna(row["lineup_projection"]) else 0.0
        strength = base_strength + 0.001 * proj_component

        records.append(
            LineupRecord(
                idx=int(idx),
                players=tuple(names),
                strength=strength,
            )
        )

    return workbook_path, df, records


def _build_player_exposure(records: Sequence[LineupRecord]) -> Tuple[Dict[str, int], int]:
    """
    Compute total appearance counts per player across all diversified lineups.
    """
    counts: Counter[str] = Counter()
    for rec in records:
        counts.update(rec.players)
    return dict(counts), len(records)


def _assign_lineups_fee_aware(
    dk_df: pd.DataFrame,
    records: Sequence[LineupRecord],
) -> Dict[int, LineupRecord]:
    """
    Assign lineups to DK entries in a fee-aware way.

    Strategy:
      - Group DK entries by Entry Fee tiers.
      - Process entries in descending Entry Fee, breaking ties by original row order.
      - For each entry, among remaining unassigned lineups:
          * Minimize an imbalance score that measures how concentrated each
            player's exposure would become in this fee tier relative to a
            uniform target across tiers.
          * Break ties by preferring stronger lineups.
    """
    if ENTRY_ID_COLUMN not in dk_df.columns:
        raise KeyError(f"DKEntries CSV is missing '{ENTRY_ID_COLUMN}' column.")
    if ENTRY_FEE_COLUMN not in dk_df.columns:
        raise KeyError(f"DKEntries CSV is missing '{ENTRY_FEE_COLUMN}' column.")

    entry_mask = dk_df[ENTRY_ID_COLUMN].notna() & (
        dk_df[ENTRY_ID_COLUMN].astype(str).str.strip() != ""
    )
    entry_indices = [int(i) for i in dk_df.index[entry_mask]]

    if not entry_indices:
        raise ValueError("DKEntries CSV contains no rows with a non-empty Entry ID.")

    if len(records) < len(entry_indices):
        raise ValueError(
            f"Not enough diversified lineups ({len(records)}) to cover "
            f"{len(entry_indices)} DK entries."
        )

    # Normalize Entry Fee to numeric for sorting / tiering.
    fees = dk_df.loc[entry_indices, ENTRY_FEE_COLUMN]
    try:
        fee_values = fees.astype(float)
    except ValueError:
        # Fallback: strip '$' and commas then convert.
        fee_values = (
            fees.astype(str)
            .str.replace("$", "", regex=False)
            .str.replace(",", "", regex=False)
            .astype(float)
        )

    fee_by_row: Dict[int, float] = {int(idx): float(fee_values.loc[idx]) for idx in entry_indices}
    unique_fees = sorted({v for v in fee_by_row.values()})
    num_tiers = max(1, len(unique_fees))

    # Precompute global exposure counts and per-player target per tier.
    total_counts, _ = _build_player_exposure(records)
    target_per_tier: Dict[str, float] = {
        p: total / num_tiers for p, total in total_counts.items()
    }

    # State: which lineups are still available, and per-tier exposure counts.
    remaining: Dict[int, LineupRecord] = {rec.idx: rec for rec in records}
    # player -> fee -> count
    exposure_by_fee: Dict[str, Dict[float, int]] = defaultdict(lambda: defaultdict(int))

    # Order entries: high fee first, then by row index.
    sorted_entries = sorted(
        entry_indices,
        key=lambda idx: (-fee_by_row[idx], idx),
    )

    assignment: Dict[int, LineupRecord] = {}

    for row_idx in sorted_entries:
        fee = fee_by_row[row_idx]

        best_key: Optional[Tuple[float, float]] = None
        best_rec: Optional[LineupRecord] = None

        for rec in remaining.values():
            imbalance = 0.0
            for name in rec.players:
                current = exposure_by_fee[name][fee]
                new = current + 1
                target = target_per_tier.get(name, 0.0)
                diff = new - target
                imbalance += diff * diff

            key = (imbalance, -rec.strength)
            if best_key is None or key < best_key:
                best_key = key
                best_rec = rec

        if best_rec is None:
            raise RuntimeError("Internal error: failed to select a lineup for entry.")

        assignment[row_idx] = best_rec
        # Update exposure.
        for name in best_rec.players:
            exposure_by_fee[name][fee] += 1

        # Remove from remaining.
        del remaining[best_rec.idx]

    # Optional: print a small exposure summary for sanity checking.
    print("Fee-tier exposure summary (player appearances per tier):")
    for player, fee_counts in sorted(exposure_by_fee.items()):
        tiers = ", ".join(f"${int(f)}: {c}" for f, c in sorted(fee_counts.items()))
        print(f"  {player}: {tiers}")

    return assignment


def _apply_assignments_to_dkentries(
    dk_df: pd.DataFrame,
    slot_cols: Sequence[int],
    assignment: Mapping[int, LineupRecord],
    name_role_to_id: Mapping[Tuple[str, str], str],
) -> pd.DataFrame:
    """
    Overwrite CPT/FLEX columns in dk_df with assigned lineups, formatting
    each cell as '{player_name} ({player_id})'.
    """
    result = dk_df.copy()

    for row_idx, rec in assignment.items():
        if len(slot_cols) != 6 or len(rec.players) != 6:
            raise ValueError("Expected 6 lineup slots and 6 players per lineup.")

        for j, col_idx in enumerate(slot_cols):
            name = rec.players[j]
            role = "CPT" if j == 0 else "FLEX"
            key = (name, role)
            if key not in name_role_to_id:
                raise KeyError(
                    f"Player {name!r} with role {role!r} not found in DKEntries "
                    "Name/ID/Roster Position dictionary."
                )
            player_id = name_role_to_id[key]
            formatted = f"{name} ({player_id})"
            # Use positional indexing so duplicate column labels (e.g., multiple
            # 'FLEX' columns) are updated independently.
            result.iat[row_idx, col_idx] = formatted

    return result


def run(
    *,
    dkentries_csv: Optional[str] = None,
    diversified_excel: Optional[str] = None,
    output_csv: Optional[str] = None,
) -> Path:
    """
    Execute DKEntries filling with diversified lineups.

    Args:
        dkentries_csv: Optional explicit DKEntries CSV path. If None, the latest
            DKEntries*.csv under data/dkentries/ is used.
        diversified_excel: Optional explicit path to a top1pct workbook. If
            None, the latest .xlsx under outputs/top1pct/ is used.
        output_csv: Optional explicit output path. If None, a timestamped file
            under outputs/dkentries/ is created.
    """
    dkentries_path = dkentries_utils.resolve_latest_dkentries_csv(dkentries_csv)
    print(f"Using DKEntries template: {dkentries_path}")

    diversified_path, diversified_df, records = _load_diversified_lineups(
        diversified_excel=diversified_excel
    )
    print(f"Using diversified lineups workbook: {diversified_path}")
    print(f"Loaded {len(records)} diversified lineups.")

    # Read DKEntries with csv first to tolerate variable-width rows (entry rows
    # plus a wider player dictionary block), then promote to a DataFrame with
    # padded columns.
    with dkentries_path.open("r", encoding="utf-8", newline="") as f:
        reader = list(csv.reader(f))

    if not reader:
        raise ValueError(f"DKEntries CSV at {dkentries_path} is empty.")

    header = reader[0]
    max_len = max(len(row) for row in reader)
    extra_cols = [f"__extra_{i}" for i in range(max_len - len(header))]
    columns = header + extra_cols

    rows_padded: List[List[str]] = []
    for row in reader[1:]:
        padded = list(row) + [""] * (max_len - len(row))
        rows_padded.append(padded)

    dk_df = pd.DataFrame(rows_padded, columns=columns)
    slot_cols = _get_lineup_slot_columns(dk_df)
    name_role_to_id = _build_name_role_to_id_map(dk_df)

    assignment = _assign_lineups_fee_aware(dk_df, records)
    filled_df = _apply_assignments_to_dkentries(
        dk_df=dk_df,
        slot_cols=slot_cols,
        assignment=assignment,
        name_role_to_id=name_role_to_id,
    )

    outputs_dir = config.OUTPUTS_DIR / "dkentries"
    outputs_dir.mkdir(parents=True, exist_ok=True)

    if output_csv is not None:
        output_path = Path(output_csv)
    else:
        # Timestamp based on diversified workbook path for stable grouping.
        stem = diversified_path.stem
        output_path = outputs_dir / f"DKEntries_filled_{stem}.csv"

    print(f"Writing filled DKEntries CSV to {output_path} ...")
    filled_df.to_csv(output_path, index=False)
    print("Done.")
    return output_path


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fill a DraftKings DKEntries CSV with diversified NFL Showdown "
            "lineups and write a DK-upload-ready CSV."
        )
    )
    parser.add_argument(
        "--dkentries-csv",
        type=str,
        default=None,
        help=(
            "Optional explicit path to a DKEntries CSV. If omitted, the latest "
            "DKEntries*.csv under data/dkentries/ is used."
        ),
    )
    parser.add_argument(
        "--diversified-excel",
        type=str,
        default=None,
        help=(
            "Optional explicit path to a top1pct workbook containing "
            f"'{LINEUPS_DIVERSIFIED_SHEET}' sheet. If omitted, the latest "
            ".xlsx under outputs/top1pct/ is used."
        ),
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default=None,
        help=(
            "Optional explicit output CSV path. If omitted, a timestamped file "
            "is written under outputs/dkentries/."
        ),
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    args = _parse_args(argv)
    run(
        dkentries_csv=args.dkentries_csv,
        diversified_excel=args.diversified_excel,
        output_csv=args.output_csv,
    )


if __name__ == "__main__":
    main()


