from __future__ import annotations

"""
Utilities for working with DraftKings DKEntries CSV templates.

This module provides helpers to:
  - Locate the latest DKEntries*.csv file under data/dkentries/.
  - Count the number of actual entries (rows with a non-empty Entry ID).
  - Expose a tiny CLI so shell scripts (e.g. run_full.sh) can query the
    entry count.
"""

import argparse
from pathlib import Path
from typing import Optional

import pandas as pd

from . import config


DKENTRIES_SUBDIR = "dkentries"
DKENTRIES_PATTERN = "DKEntries*.csv"
ENTRY_ID_COLUMN = "Entry ID"


def resolve_latest_dkentries_csv(explicit: Optional[str] = None) -> Path:
    """
    Resolve the path to a DKEntries CSV.

    If `explicit` is provided, it is treated as a path (absolute or relative
    to the project root) and must exist.

    Otherwise, the most recent `DKEntries*.csv` under data/dkentries/ is used.
    """
    if explicit:
        path = Path(explicit)
        if not path.is_file():
            raise FileNotFoundError(f"Specified DKEntries CSV does not exist: {path}")
        return path

    dkentries_dir = config.DATA_DIR / DKENTRIES_SUBDIR
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
    df = pd.read_csv(path)
    if ENTRY_ID_COLUMN not in df.columns:
        raise KeyError(
            f"DKEntries CSV at {path} is missing required column "
            f"{ENTRY_ID_COLUMN!r}."
        )

    entry_id_series = df[ENTRY_ID_COLUMN]
    mask = entry_id_series.notna() & (entry_id_series.astype(str).str.strip() != "")
    return int(mask.sum())


def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Helpers for DraftKings DKEntries CSV templates."
    )
    parser.add_argument(
        "--dkentries-csv",
        type=str,
        default=None,
        help=(
            "Optional explicit path to a DKEntries CSV. "
            "If omitted, the latest DKEntries*.csv under data/dkentries/ is used."
        ),
    )
    parser.add_argument(
        "--count-entries",
        action="store_true",
        help=(
            "When set, print the number of actual entries (non-empty Entry ID rows) "
            "to stdout and exit."
        ),
    )
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> None:
    args = _parse_args(argv)

    dkentries_path = resolve_latest_dkentries_csv(args.dkentries_csv)

    if args.count_entries:
        n = count_real_entries(dkentries_path)
        # Print just the integer so shell callers can capture it cleanly.
        print(n)
        return

    # If no explicit action is requested, default to printing the resolved path
    # for debugging / interactive use.
    print(dkentries_path)


if __name__ == "__main__":
    main()


