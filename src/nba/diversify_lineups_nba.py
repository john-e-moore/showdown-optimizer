from __future__ import annotations

"""
NBA wrapper for diversifying Showdown top1pct lineups.

This mirrors the NFL CLI but delegates the selection and exposure logic
to `src.shared.diversify_core.run_diversify` and uses NBA paths.
"""

import argparse
from pathlib import Path
from typing import List

import pandas as pd

from ..shared import diversify_core
from . import config, fill_dkentries_nba


def run(
    num_lineups: int,
    *,
    min_top1_pct: float = 1.0,
    max_overlap: int = 4,
    top1pct_excel: str | None = None,
    output_dir: str | None = None,
    sort_by: str = "top1_pct_finish_rate",
    max_flex_overlap: int | None = None,
    cpt_field_cap_multiplier: float | None = None,
    lineups_excel: str | None = None,
):
    """
    NBA wrapper around the shared diversification core.
    """
    outputs_dir = config.OUTPUTS_DIR

    # When running as part of an end-to-end pipeline, top1pct_finish_rate_nba can
    # write a run-scoped workbook under a per-run directory (e.g.
    # outputs/nba/runs/<ts>/). If the caller passed an output_dir (the run
    # directory) and did not override top1pct_excel explicitly, prefer the latest
    # top1pct_lineups_*.xlsx in that directory instead of falling back to
    # outputs/nba/top1pct/.
    resolved_top1pct_excel = top1pct_excel
    if resolved_top1pct_excel is None and output_dir is not None:
        run_dir = Path(output_dir)
        candidates = sorted(
            run_dir.glob("top1pct_lineups_*.xlsx"),
            key=lambda p: p.stat().st_mtime,
        )
        if candidates:
            resolved_top1pct_excel = str(candidates[-1])

    # Build an optional CPT max-share map from projected field ownership when
    # the caller has requested a field-aware cap.
    cpt_max_share: dict[str, float] | None = None
    if cpt_field_cap_multiplier is not None and cpt_field_cap_multiplier > 0.0:
        # Resolve the lineups workbook used for field ownership. Prefer a
        # run-scoped lineups workbook under the provided output_dir when
        # available, otherwise fall back to the latest under outputs/nba/lineups/.
        resolved_lineups_excel = lineups_excel
        if resolved_lineups_excel is None and output_dir is not None:
            run_dir = Path(output_dir)
            lineups_candidates = sorted(
                run_dir.glob("lineups_*.xlsx"),
                key=lambda p: p.stat().st_mtime,
            )
            if lineups_candidates:
                resolved_lineups_excel = str(lineups_candidates[-1])

        try:
            field_map = fill_dkentries_nba.load_field_ownership_mapping(
                lineups_excel=resolved_lineups_excel
            )
            cpt_max_share = {}
            for name, info in field_map.items():
                try:
                    field_own_cpt = float(info.get("field_own_cpt", 0.0))
                except (TypeError, ValueError):
                    field_own_cpt = 0.0
                if field_own_cpt <= 0.0:
                    continue
                max_share = (field_own_cpt / 100.0) * float(cpt_field_cap_multiplier)
                if max_share > 1.0:
                    max_share = 1.0
                cpt_max_share[name] = max_share
        except Exception as exc:  # pragma: no cover - defensive
            # If anything goes wrong loading field ownership, fall back to
            # diversification without CPT caps rather than failing the run.
            print(
                "Warning: failed to build CPT field-aware caps; proceeding without "
                f"CPT caps. Error: {exc}"
            )
            cpt_max_share = None

    excel_path = diversify_core.run_diversify(
        num_lineups=num_lineups,
        outputs_dir=outputs_dir,
        min_top1_pct=min_top1_pct,
        max_overlap=max_overlap,
        top1pct_excel=resolved_top1pct_excel,
        sort_by=sort_by,
        max_flex_overlap=max_flex_overlap,
        cpt_max_share=cpt_max_share,
    )

    # When output_dir is provided, write convenient CSV sidecars into that
    # directory mirroring the NFL pipeline:
    #   - diversified.csv: diversified lineups
    #   - ownership.csv: CPT/FLEX exposure summary
    if output_dir is not None:
        out_dir_path = Path(output_dir)
        out_dir_path.mkdir(parents=True, exist_ok=True)

        xls = pd.ExcelFile(excel_path)
        try:
            diversified_df = pd.read_excel(xls, sheet_name="Lineups_Diversified")
        except ValueError as exc:
            raise KeyError(
                "Diversified workbook missing 'Lineups_Diversified' sheet: "
                f"{excel_path}"
            ) from exc
        try:
            exposure_df = pd.read_excel(xls, sheet_name="Exposure")
        except ValueError as exc:
            raise KeyError(
                "Diversified workbook missing 'Exposure' sheet: "
                f"{excel_path}"
            ) from exc
        diversified_csv_path = out_dir_path / "diversified.csv"
        ownership_csv_path = out_dir_path / "ownership.csv"
        print(f"Writing diversified CSV to {diversified_csv_path} ...")
        diversified_df.to_csv(diversified_csv_path, index=False)
        print(f"Writing ownership CSV to {ownership_csv_path} ...")
        exposure_df.to_csv(ownership_csv_path, index=False)

    return excel_path


def _parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Select a diversified subset of NBA Showdown lineups based on "
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
            "Optional path to a top1pct workbook under outputs/nba/top1pct/. "
            "If omitted, the most recent .xlsx file in that directory is used, "
            "or a run-scoped workbook under --output-dir when provided."
        ),
    )
    parser.add_argument(
        "--sort-by",
        type=str,
        choices=["top1_pct_finish_rate", "ev_roi"],
        default="top1_pct_finish_rate",
        help=(
            "Column to sort candidate lineups by before greedy diversification. "
            "Use 'ev_roi' when top1pct scoring was run with --contest-id."
        ),
    )
    parser.add_argument(
        "--max-flex-overlap",
        type=int,
        default=None,
        help=(
            "Optional maximum number of overlapping FLEX/UTIL players allowed "
            "between any pair of selected lineups (0–5 for Showdown). If "
            "omitted, only total player overlap (--max-overlap) is enforced."
        ),
    )
    parser.add_argument(
        "--cpt-field-cap-multiplier",
        type=float,
        default=2.0,
        help=(
            "Multiple of projected field CPT ownership to use as a max CPT share "
            "cap within the diversified set. Set <= 0 to disable CPT caps."
        ),
    )
    parser.add_argument(
        "--lineups-excel",
        type=str,
        default=None,
        help=(
            "Optional explicit path to a lineups workbook whose Projections sheet "
            "provides projected field ownership. If omitted, a run-scoped "
            "lineups_*.xlsx under the output run directory is preferred when "
            "available; otherwise, the latest .xlsx under outputs/nba/lineups/ "
            "is used."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help=(
            "Optional directory in which to write diversified.csv and "
            "ownership.csv sidecar files for this diversification run."
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
        output_dir=args.output_dir,
        sort_by=args.sort_by,
        max_flex_overlap=args.max_flex_overlap,
        cpt_field_cap_multiplier=args.cpt_field_cap_multiplier,
        lineups_excel=args.lineups_excel,
    )


if __name__ == "__main__":
    main()


