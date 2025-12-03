from __future__ import annotations

"""
NFL-specific wrappers around shared DKEntries helpers.

This module preserves the existing CLI interface while delegating the
core DKEntries resolution/counting logic to `src.shared.dkentries_core`.
"""

import argparse
from pathlib import Path
from typing import Optional

from . import config
from .shared import dkentries_core


ENTRY_ID_COLUMN = dkentries_core.ENTRY_ID_COLUMN


def resolve_latest_dkentries_csv(explicit: Optional[str] = None) -> Path:
    """
    Resolve the path to a DKEntries CSV for NFL.
    """
    return dkentries_core.resolve_latest_dkentries_csv(
        data_root=config.DATA_DIR,
        explicit=explicit,
    )


def count_real_entries(path: Path) -> int:
    """
    Count the number of actual entries in a DKEntries CSV.
    """
    return dkentries_core.count_real_entries(path)


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


