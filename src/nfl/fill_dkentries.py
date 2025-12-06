from __future__ import annotations

"""
Fill a DraftKings DKEntries CSV with diversified NFL Showdown lineups.

This script:
  1. Resolves a DKEntries*.csv template (latest under data/nfl/dkentries/ by default).
  2. Resolves the latest diversified lineups workbook under outputs/nfl/diversified/.
  3. Maps each lineup (CPT + 5 FLEX names) to DraftKings player IDs using the
     Name/ID/Roster Position dictionary embedded in the DKEntries CSV.
  4. Assigns lineups to DK entries in a fee-aware way, spreading player exposure
     across distinct Entry Fee tiers while favoring stronger lineups in
     higher-fee contests.
  5. Writes a DK-upload-ready CSV under outputs/nfl/dkentries/ with lineup slots
     formatted as '{player_name} ({player_id})'.
"""

import argparse
import csv
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from ..shared import dkentries_core
from . import config, dkentries_utils
from .top1pct_finish_rate import run as run_top1pct  # noqa: F401 (for CLI symmetry)


LINEUPS_DIVERSIFIED_SHEET = "Lineups_Diversified"
ENTRY_ID_COLUMN = dkentries_core.ENTRY_ID_COLUMN
ENTRY_FEE_COLUMN = dkentries_core.ENTRY_FEE_COLUMN


@dataclass(frozen=True)
class LineupRecord:
    idx: int
    players: Tuple[str, ...]  # CPT first, then FLEX1..FLEX5
    strength: float


def _parse_player_name(cell: str) -> str:
    """
    Extract the raw player name from a lineup cell like 'Deebo Samuel (34.5%)'.
    """
    if not isinstance(cell, str):
        return str(cell)
    # Split at the first ' (' if present.
    split_idx = cell.find(" (")
    if split_idx == -1:
        return cell.strip()
    return cell[:split_idx].strip()


def _resolve_latest_excel(directory: Path, explicit: str | None) -> Path:
    """
    Resolve an Excel file path, preferring an explicit argument when provided.

    If explicit is None, pick the most recent *.xlsx file in `directory`.
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


def _load_diversified_lineups(
    diversified_excel: Optional[str] = None,
) -> Tuple[Path, pd.DataFrame, List[LineupRecord]]:
    """
    Load diversified lineups from the latest (or specified) diversified workbook.
    """
    diversified_dir = config.OUTPUTS_DIR / "diversified"
    workbook_path = _resolve_latest_excel(diversified_dir, diversified_excel)

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
        proj_component = (
            float(row["lineup_projection"])
            if has_proj and pd.notna(row["lineup_projection"])
            else 0.0
        )
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


def _compute_realized_exposure(
    dk_df: pd.DataFrame,
    slot_cols: Sequence[int],
) -> Tuple[Dict[Tuple[str, str], Dict[str, float]], int, float]:
    """
    Compute realized lineup and dollar exposure per (player, role) across
    the filled DKEntries DataFrame.

    Returns:
        exposure: mapping (player_name, role) -> {
            "lineup_exposure": percentage of DK entries containing the player
                               in that role,
            "dollar_exposure": percentage of total entry fees allocated to
                               lineups where the player appears in that role,
        }
        num_entries: total number of DK entries with a non-empty Entry ID.
        total_fees: total sum of Entry Fee across those entries.
    """
    if ENTRY_ID_COLUMN not in dk_df.columns:
        raise KeyError(f"DKEntries CSV is missing '{ENTRY_ID_COLUMN}' column.")
    if ENTRY_FEE_COLUMN not in dk_df.columns:
        raise KeyError(f"DKEntries CSV is missing '{ENTRY_FEE_COLUMN}' column.")

    entry_mask = dk_df[ENTRY_ID_COLUMN].notna() & (
        dk_df[ENTRY_ID_COLUMN].astype(str).str.strip() != ""
    )
    entry_indices = [int(i) for i in dk_df.index[entry_mask]]
    num_entries = len(entry_indices)
    if num_entries == 0:
        raise ValueError("DKEntries CSV contains no rows with a non-empty Entry ID.")

    # Normalize Entry Fee to numeric for weighting.
    fees = dk_df.loc[entry_indices, ENTRY_FEE_COLUMN]
    try:
        fee_values = fees.astype(float)
    except ValueError:
        fee_values = (
            fees.astype(str)
            .str.replace("$", "", regex=False)
            .str.replace(",", "", regex=False)
            .astype(float)
        )
    total_fees = float(fee_values.sum())

    counts: Dict[Tuple[str, str], int] = {}
    fee_sums: Dict[Tuple[str, str], float] = {}

    for row_idx in entry_indices:
        fee = float(fee_values.loc[row_idx])

        # CPT slot.
        cpt_cell = dk_df.iat[row_idx, slot_cols[0]]
        cpt_name = _parse_player_name(str(cpt_cell))
        key_cpt = (cpt_name, "CPT")
        counts[key_cpt] = counts.get(key_cpt, 0) + 1
        fee_sums[key_cpt] = fee_sums.get(key_cpt, 0.0) + fee

        # FLEX slots.
        for col_idx in slot_cols[1:]:
            flex_cell = dk_df.iat[row_idx, col_idx]
            flex_name = _parse_player_name(str(flex_cell))
            key_flex = (flex_name, "FLEX")
            counts[key_flex] = counts.get(key_flex, 0) + 1
            fee_sums[key_flex] = fee_sums.get(key_flex, 0.0) + fee

    exposure: Dict[Tuple[str, str], Dict[str, float]] = {}
    for key, cnt in counts.items():
        fee_sum = fee_sums.get(key, 0.0)
        lineup_exposure = 100.0 * cnt / float(num_entries) if num_entries > 0 else 0.0
        dollar_exposure = 100.0 * fee_sum / total_fees if total_fees > 0.0 else 0.0
        exposure[key] = {
            "lineup_exposure": lineup_exposure,
            "dollar_exposure": dollar_exposure,
        }

    return exposure, num_entries, total_fees


def _load_field_ownership_mapping() -> Dict[str, Dict[str, object]]:
    """
    Load per-player team and projected field ownership from the latest
    lineups workbook's Projections sheet (original SaberSim CSV).

    Returns:
        mapping: player_name -> {
            "team": team_abbrev or "",
            "field_own_cpt": projected CPT ownership (%),
            "field_own_flex": projected FLEX ownership (%),
        }

    Notes:
        - When only a single aggregate ownership column (e.g. 'My Own') is
          available, we currently apply the same value to both CPT and FLEX
          roles for convenience.
    """
    outputs_dir = config.OUTPUTS_DIR
    lineups_dir = outputs_dir / "lineups"
    lineups_path = _resolve_latest_excel(lineups_dir, explicit=None)

    xls = pd.ExcelFile(lineups_path)
    try:
        sabersim_df = pd.read_excel(xls, sheet_name="Projections")
    except ValueError as exc:
        raise KeyError(
            "Lineups workbook missing 'Projections' sheet: "
            f"{lineups_path}"
        ) from exc

    required_cols = {"Name", "Team", "My Proj", "My Own"}
    missing = required_cols.difference(sabersim_df.columns)
    if missing:
        raise KeyError(
            "Projections sheet missing required columns "
            f"{sorted(missing)}. Expected at least {sorted(required_cols)}."
        )

    def _to_pct(value: float) -> float:
        # Interpret SaberSim 'My Own' as already in percentage units (e.g., 0.67%).
        try:
            return float(value)
        except (TypeError, ValueError):
            return 0.0

    grouped = sabersim_df.groupby(["Name", "Team"])

    mapping: Dict[str, Dict[str, object]] = {}

    for (name, team), g in grouped:
        if len(g) == 0:
            continue
        g_sorted = g.sort_values(by="My Proj", ascending=False)
        cpt_row = g_sorted.iloc[0]
        cpt_pct = _to_pct(cpt_row["My Own"])

        if len(g_sorted) > 1:
            flex_row = g_sorted.iloc[1]
            flex_pct = _to_pct(flex_row["My Own"])
        else:
            flex_pct = 0.0

        name_str = str(name).strip()
        team_str = str(team).strip()

        # Keep first occurrence per player name.
        if name_str not in mapping:
            mapping[name_str] = {
                "team": team_str,
                "field_own_cpt": cpt_pct,
                "field_own_flex": flex_pct,
            }

    return mapping


def _write_ownership_summary_csv(
    filled_df: pd.DataFrame,
    slot_cols: Sequence[int],
    base_dir: Path,
) -> Path:
    """
    Write per-player ownership summary CSV alongside the filled DKEntries CSV.

    Columns:
        - player: player name
        - team: team abbreviation (when available)
        - role: 'CPT' or 'FLEX'
        - field_ownership: projected field ownership (%) for that role
        - lineup_exposure: percentage of DK entries containing the player
                           in that role
        - dollar_exposure: percentage of total entry fees allocated to
                           lineups where the player appears in that role
    """
    exposure, _, _ = _compute_realized_exposure(filled_df, slot_cols)
    ownership_map = _load_field_ownership_mapping()

    rows: List[Dict[str, object]] = []
    for (player_name, role), metrics in exposure.items():
        info = ownership_map.get(player_name, {})
        team = str(info.get("team", "") or "")
        if role == "CPT":
            field_own = float(info.get("field_own_cpt", 0.0))
        else:
            field_own = float(info.get("field_own_flex", 0.0))

        rows.append(
            {
                "player": player_name,
                "team": team,
                "role": role,
                "field_ownership": field_own,
                "lineup_exposure": float(metrics.get("lineup_exposure", 0.0)),
                "dollar_exposure": float(metrics.get("dollar_exposure", 0.0)),
            }
        )

    if not rows:
        # No exposure rows; still write an empty CSV with headers for consistency.
        ownership_df = pd.DataFrame(
            columns=[
                "player",
                "team",
                "role",
                "field_ownership",
                "lineup_exposure",
                "dollar_exposure",
            ]
        )
    else:
        ownership_df = pd.DataFrame(rows)
        # Sort by dollar exposure desc, then player name.
        ownership_df.sort_values(
            by=["dollar_exposure", "player"],
            ascending=[False, True],
            inplace=True,
        )

    base_dir.mkdir(parents=True, exist_ok=True)

    output_path = base_dir / "ownership.csv"
    print(f"Writing ownership summary CSV to {output_path} ...")
    ownership_df.to_csv(output_path, index=False)

    return output_path


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

    fee_by_row: Dict[int, float] = {
        int(idx): float(fee_values.loc[idx]) for idx in entry_indices
    }
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
    Execute DKEntries filling with diversified NFL Showdown lineups.

    Args:
        dkentries_csv: Optional explicit DKEntries CSV path. If None, the latest
            DKEntries*.csv under data/nfl/dkentries/ is used.
        diversified_excel: Optional explicit path to a diversified lineups
            workbook. If None, the latest .xlsx under outputs/nfl/diversified/
            is used.
        output_csv: Optional explicit output path. If None, a timestamped file
            under outputs/nfl/dkentries/ is created.
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
    slot_cols = dkentries_core.get_lineup_slot_columns(dk_df)
    name_role_to_id = dkentries_core.build_name_role_to_id_map(dk_df, flex_role_label="FLEX")

    assignment = _assign_lineups_fee_aware(dk_df, records)
    filled_df = _apply_assignments_to_dkentries(
        dk_df=dk_df,
        slot_cols=slot_cols,
        assignment=assignment,
        name_role_to_id=name_role_to_id,
    )

    # Resolve output location.
    if output_csv is not None:
        # Honor an explicit output path (e.g. when called from run_full.sh with
        # a per-run directory like outputs/nfl/runs/<timestamp>/).
        output_path = Path(output_csv)
        base_dir = output_path.parent
        base_dir.mkdir(parents=True, exist_ok=True)
    else:
        # Default to a timestamped folder under outputs/nfl/dkentries/.
        outputs_root = config.OUTPUTS_DIR / "dkentries"
        outputs_root.mkdir(parents=True, exist_ok=True)
        ts_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_dir = outputs_root / ts_str
        base_dir.mkdir(parents=True, exist_ok=True)
        # Write filled DKEntries CSV as dkentries.csv inside the timestamped folder.
        output_path = base_dir / "dkentries.csv"
    print(f"Writing filled DKEntries CSV to {output_path} ...")
    filled_df.to_csv(output_path, index=False)
    print("Done.")

    # Also write the diversified lineups as diversified.csv for convenience.
    diversified_csv_path = base_dir / "diversified.csv"
    print(f"Writing diversified lineups CSV to {diversified_csv_path} ...")
    diversified_df.to_csv(diversified_csv_path, index=False)

    # Write ownership summary sidecar CSV.
    _write_ownership_summary_csv(
        filled_df=filled_df,
        slot_cols=slot_cols,
        base_dir=base_dir,
    )

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
            "DKEntries*.csv under data/nfl/dkentries/ is used."
        ),
    )
    parser.add_argument(
        "--diversified-excel",
        type=str,
        default=None,
        help=(
            "Optional explicit path to a diversified lineups workbook "
            f"containing '{LINEUPS_DIVERSIFIED_SHEET}' sheet. If omitted, the "
            "latest .xlsx under outputs/nfl/diversified/ is used."
        ),
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default=None,
        help=(
            "Optional explicit output CSV path. If omitted, a timestamped file "
            "is written under outputs/nfl/dkentries/."
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



