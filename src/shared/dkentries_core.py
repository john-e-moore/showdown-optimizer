from __future__ import annotations

"""
Sport-agnostic DKEntries helpers shared by NFL and NBA.

This module provides:
  - Resolving the latest DKEntries*.csv template under a given data root.
  - Counting real entries (rows with a non-empty Entry ID).
  - Parsing the embedded player dictionary (Name / ID / Roster Position).
  - Identifying CPT + 5 FLEX slot columns.
  - Fee-aware assignment of diversified lineups to entries.
  - Applying assignments back to a DKEntries dataframe.
"""

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


DKENTRIES_SUBDIR = "dkentries"
DKENTRIES_PATTERN = "DKEntries*.csv"
ENTRY_ID_COLUMN = "Entry ID"
ENTRY_FEE_COLUMN = "Entry Fee"


def resolve_latest_dkentries_csv(
    data_root: Path,
    explicit: Optional[str] = None,
) -> Path:
    """
    Resolve the path to a DKEntries CSV under a given data_root.

    If `explicit` is provided, it is treated as a path (absolute or relative
    to the project root) and must exist.

    Otherwise, the most recent `DKEntries*.csv` under data_root / 'dkentries'
    is used.
    """
    if explicit:
        path = Path(explicit)
        if not path.is_file():
            raise FileNotFoundError(f"Specified DKEntries CSV does not exist: {path}")
        return path

    dkentries_dir = data_root / DKENTRIES_SUBDIR
    candidates = sorted(
        dkentries_dir.glob(DKENTRIES_PATTERN),
        key=lambda p: p.stat().st_mtime,
    )
    if not candidates:
        raise FileNotFoundError(
            f"No DKEntries CSV files matching pattern {DKENTRIES_PATTERN!r} "
            f"found under {dkentries_dir}"
        )
    return candidates[-1]


def count_real_entries(path: Path) -> int:
    """
    Count the number of actual entries in a DKEntries CSV.

    We treat any row with a non-empty `Entry ID` as a real entry.
    """
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        try:
            header = next(reader)
        except StopIteration:
            return 0

        try:
            idx_entry_id = header.index(ENTRY_ID_COLUMN)
        except ValueError as exc:
            raise KeyError(
                f"DKEntries CSV at {path} is missing required column "
                f"{ENTRY_ID_COLUMN!r}."
            ) from exc

        count = 0
        for row in reader:
            if idx_entry_id >= len(row):
                continue
            val = str(row[idx_entry_id]).strip()
            if val:
                count += 1

    return count


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

    missing = [
        lbl
        for lbl, col in [
            ("Name", name_col),
            ("ID", id_col),
            ("Roster Position", roster_pos_col),
        ]
        if col is None
    ]
    if missing:
        raise KeyError(
            "Failed to locate DK player dictionary columns in DKEntries CSV; "
            f"could not find headers {missing} in any column."
        )

    return name_col, id_col, roster_pos_col


def build_name_role_to_id_map(
    df: pd.DataFrame,
    *,
    flex_role_label: str = "FLEX",
) -> Dict[Tuple[str, str], str]:
    """
    Build a mapping (player_name, roster_role) -> dk_player_id using the
    embedded player dictionary in the DKEntries CSV.

    By default this treats CPT plus FLEX rows as valid dictionary entries.
    For sports like NBA where DraftKings uses a different label (e.g. ``UTIL``)
    for non-CPT lineup slots, pass ``flex_role_label=\"UTIL\"``.
    """
    name_col, id_col, roster_pos_col = _find_mapping_columns(df)

    flex_role_label_up = flex_role_label.upper()
    valid_roles = {"CPT", flex_role_label_up}

    mask = df[roster_pos_col].astype(str).str.upper().isin(valid_roles)
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

        if not name or not player_id or role not in valid_roles:
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


def get_lineup_slot_columns(df: pd.DataFrame) -> List[int]:
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


@dataclass(frozen=True)
class LineupRecord:
    idx: int
    players: Tuple[str, ...]  # CPT first, then FLEX1..FLEX5
    strength: float


def build_player_exposure(records: Sequence[LineupRecord]) -> Tuple[Dict[str, int], int]:
    """
    Compute total appearance counts per player across all diversified lineups.
    """
    from collections import Counter

    counts: Counter[str] = Counter()
    for rec in records:
        counts.update(rec.players)
    return dict(counts), len(records)


def assign_lineups_fee_aware(
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

    fee_by_row: Dict[int, float] = {
        int(idx): float(fee_values.loc[idx]) for idx in entry_indices
    }
    unique_fees = sorted({v for v in fee_by_row.values()})
    num_tiers = max(1, len(unique_fees))

    # Precompute global exposure counts and per-player target per tier.
    total_counts, _ = build_player_exposure(records)
    target_per_tier: Dict[str, float] = {
        p: total / num_tiers for p, total in total_counts.items()
    }

    # State: which lineups are still available, and per-tier exposure counts.
    remaining: Dict[int, LineupRecord] = {rec.idx: rec for rec in records}
    # player -> fee -> count
    from collections import defaultdict

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


def apply_assignments_to_dkentries(
    dk_df: pd.DataFrame,
    slot_cols: Sequence[int],
    assignment: Mapping[int, LineupRecord],
    name_role_to_id: Mapping[Tuple[str, str], str],
    *,
    flex_role_label: str = "FLEX",
) -> pd.DataFrame:
    """
    Overwrite CPT / flex-style columns in ``dk_df`` with assigned lineups,
    formatting each cell as ``\"{player_name} ({player_id})\"``.

    The first slot in ``slot_cols`` is always treated as ``CPT``; all remaining
    slots use ``flex_role_label`` (\"FLEX\" for NFL, \"UTIL\" for NBA, etc.)
    when looking up player IDs in ``name_role_to_id``.
    """
    result = dk_df.copy()

    for row_idx, rec in assignment.items():
        if len(slot_cols) != 6 or len(rec.players) != 6:
            raise ValueError("Expected 6 lineup slots and 6 players per lineup.")

        for j, col_idx in enumerate(slot_cols):
            name = rec.players[j]
            role = "CPT" if j == 0 else flex_role_label
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


__all__ = [
    "resolve_latest_dkentries_csv",
    "count_real_entries",
    "build_name_role_to_id_map",
    "get_lineup_slot_columns",
    "LineupRecord",
    "build_player_exposure",
    "assign_lineups_fee_aware",
    "apply_assignments_to_dkentries",
]


