from __future__ import annotations

"""
Fill a DraftKings DKEntries CSV with diversified NBA Showdown lineups.

This mirrors the NFL `fill_dkentries` script but:
  - Uses NBA data/outputs roots via `nba.config`.
  - Uses the shared DKEntries core for template resolution and assignments.
"""

import argparse
import csv
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd

from ..shared import dkentries_core, top1pct_core
from . import config


LINEUPS_DIVERSIFIED_SHEET = "Lineups_Diversified"


@dataclass(frozen=True)
class LineupRecord:
    idx: int
    # players are stored as [CPT, UTIL1..UTIL5] for NBA Showdown
    players: Tuple[str, ...]
    strength: float


def _load_diversified_lineups(
    diversified_excel: Optional[str] = None,
) -> Tuple[Path, pd.DataFrame, List[LineupRecord]]:
    """
    Load diversified lineups from the latest (or specified) top1pct workbook.
    """
    top1pct_dir = config.OUTPUTS_DIR / "top1pct"
    workbook_path = top1pct_core._resolve_latest_excel(top1pct_dir, diversified_excel)

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
        names.append(top1pct_core._parse_player_name(row["cpt"]))
        for j in range(1, 6):
            names.append(top1pct_core._parse_player_name(row[f"flex{j}"]))

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


def _write_ownership_summary_csv(
    filled_df: pd.DataFrame,
    slot_cols: Sequence[int],
    base_dir: Path,
) -> Path:
    """
    Write per-player ownership summary CSV alongside the filled DKEntries CSV.
    """
    # Reuse NFL helper logic by importing the function at call site to avoid
    # circular imports.
    from ..fill_dkentries import _compute_realized_exposure, _load_field_ownership_mapping

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


def run(
    *,
    dkentries_csv: Optional[str] = None,
    diversified_excel: Optional[str] = None,
    output_csv: Optional[str] = None,
) -> Path:
    """
    Execute DKEntries filling with diversified NBA lineups.
    """
    dkentries_path = dkentries_core.resolve_latest_dkentries_csv(
        data_root=config.DATA_DIR,
        explicit=dkentries_csv,
    )
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
    # NBA DKEntries use CPT / UTIL roster positions rather than CPT / FLEX.
    name_role_to_id = dkentries_core.build_name_role_to_id_map(
        dk_df, flex_role_label="UTIL"
    )

    # Convert local LineupRecord objects to shared core LineupRecord for reuse.
    shared_records = [
        dkentries_core.LineupRecord(idx=rec.idx, players=rec.players, strength=rec.strength)
        for rec in records
    ]

    assignment = dkentries_core.assign_lineups_fee_aware(dk_df, shared_records)
    filled_df = dkentries_core.apply_assignments_to_dkentries(
        dk_df=dk_df,
        slot_cols=slot_cols,
        assignment=assignment,
        name_role_to_id=name_role_to_id,
        flex_role_label="UTIL",
    )

    outputs_root = config.OUTPUTS_DIR / "dkentries"
    outputs_root.mkdir(parents=True, exist_ok=True)

    # Determine timestamp directory for this run.
    ts_str: Optional[str] = None
    if output_csv is not None:
        out_path_requested = Path(output_csv)
        m = re.search(r"(\d{8}_\d{6})", out_path_requested.stem)
        if m and out_path_requested.parent == outputs_root:
            ts_str = m.group(1)

    if ts_str is None:
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
            "Fill a DraftKings DKEntries CSV with diversified NBA Showdown "
            "lineups and write a DK-upload-ready CSV."
        )
    )
    parser.add_argument(
        "--dkentries-csv",
        type=str,
        default=None,
        help=(
            "Optional explicit path to a DKEntries CSV. If omitted, the latest "
            "DKEntries*.csv under data/nba/dkentries/ is used."
        ),
    )
    parser.add_argument(
        "--diversified-excel",
        type=str,
        default=None,
        help=(
            "Optional explicit path to a top1pct workbook containing "
            f"'{LINEUPS_DIVERSIFIED_SHEET}' sheet. If omitted, the latest "
            ".xlsx under outputs/nba/top1pct/ is used."
        ),
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default=None,
        help=(
            "Optional explicit output CSV path. If omitted, a timestamped file "
            "is written under outputs/nba/dkentries/."
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


