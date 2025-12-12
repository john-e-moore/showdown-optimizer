from __future__ import annotations

"""
Augment optimized NFL Showdown lineups with quota-balanced field-style lineups.

This script:
  1) Loads an existing optimizer lineups workbook.
  2) Builds additional CPT+FLEX lineups using the shared quota-balanced field
     builder wired through the same ownership/projections/correlation inputs
     used by the top1% pipeline.
  3) Appends these field-style lineups to the original Lineups sheet and writes
     an augmented workbook.

The augmented workbook can then be passed directly into the NFL top1% CLI.
"""

import argparse
from pathlib import Path
from typing import List

import pandas as pd

from ..shared import field_builder, top1pct_core


def _parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Augment an NFL optimizer lineups workbook with quota-balanced "
            "field-style Showdown lineups."
        )
    )
    parser.add_argument(
        "--lineups-excel",
        type=str,
        required=True,
        help="Path to the optimizer lineups workbook to augment.",
    )
    parser.add_argument(
        "--corr-excel",
        type=str,
        required=True,
        help=(
            "Path to the correlations workbook providing Sabersim projections "
            "and the player correlation matrix."
        ),
    )
    parser.add_argument(
        "--extra-lineups",
        type=int,
        required=True,
        help="Number of additional field-style lineups to generate and append.",
    )
    parser.add_argument(
        "--output-excel",
        type=str,
        default=None,
        help=(
            "Optional explicit path for the augmented lineups workbook. "
            "If omitted, a sibling file with '_augmented' suffix is written "
            "next to the input lineups workbook."
        ),
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=None,
        help="Optional RNG seed for the quota-balanced field builder.",
    )
    return parser.parse_args(argv)


def _augment_lineups(
    lineups_path: Path,
    corr_path: Path,
    extra_lineups: int,
    output_path: Path | None,
    random_seed: int | None,
) -> Path:
    if extra_lineups <= 0:
        raise ValueError("--extra-lineups must be positive.")

    if not lineups_path.is_file():
        raise FileNotFoundError(f"Lineups workbook not found: {lineups_path}")
    if not corr_path.is_file():
        raise FileNotFoundError(f"Correlation workbook not found: {corr_path}")

    print(f"Loading optimizer lineups workbook from {lineups_path}...")
    xls = pd.ExcelFile(lineups_path)
    lineups_df, ownership_df, projections_df = top1pct_core._load_lineups_workbook(
        lineups_path
    )
    exposure_df: pd.DataFrame | None = None
    if "Exposure" in xls.sheet_names:
        exposure_df = pd.read_excel(xls, sheet_name="Exposure")

    print(f"Loading correlation workbook from {corr_path}...")
    sabersim_proj_df, corr_df = top1pct_core._load_corr_workbook(corr_path)

    # Build a player universe aligned with the top1% core for consistent
    # projections/salaries/teams when annotating new lineups.
    (
        player_names,
        mu,
        _sigma,
        _cpt_own,
        _flex_own,
        _positions,
        salaries,
        teams,
    ) = top1pct_core._build_player_universe(
        lineups_df=lineups_df,
        ownership_df=ownership_df,
        sabersim_proj_df=sabersim_proj_df,
        corr_df=corr_df,
        lineups_proj_df=projections_df,
    )

    print(
        f"Building {extra_lineups} quota-balanced field-style lineups "
        "using shared field builder..."
    )
    field_lineups_df = field_builder.build_quota_balanced_field(
        field_size=extra_lineups,
        ownership_df=ownership_df,
        sabersim_proj_df=sabersim_proj_df,
        lineups_proj_df=projections_df,
        corr_df=corr_df,
        random_seed=random_seed,
    )

    # Annotate field lineups with projection/salary/stack metadata in an NFL
    # schema (CPT + flex1–flex5).
    field_with_meta = top1pct_core._annotate_lineups_with_meta(
        lineups_df=field_lineups_df,
        player_names=player_names,
        mu=mu,
        salaries=salaries,
        teams=teams,
        sport="nfl",
    )

    # Align columns with the existing Lineups sheet.
    max_rank = int(lineups_df["rank"].max()) if "rank" in lineups_df.columns else len(
        lineups_df
    )
    num_new = len(field_with_meta)
    # For NFL, the top1pct core uses 'stack_pattern' as the metadata label;
    # NFL lineups currently use 'target_stack_pattern', so map accordingly.
    if "stack_pattern" in field_with_meta.columns:
        field_with_meta = field_with_meta.rename(
            columns={"stack_pattern": "target_stack_pattern"}
        )
    else:
        field_with_meta["target_stack_pattern"] = ""

    field_with_meta.insert(
        0,
        "rank",
        list(range(max_rank + 1, max_rank + 1 + num_new)),
    )
    # Tag these as coming from the field model.
    field_with_meta["target_stack_pattern"] = field_with_meta[
        "target_stack_pattern"
    ].astype(str)
    field_with_meta["target_stack_pattern"] = field_with_meta[
        "target_stack_pattern"
    ].where(
        field_with_meta["target_stack_pattern"] != "",
        "field",
    )

    # Reindex to match the existing Lineups columns, filling any missing values.
    desired_cols = list(lineups_df.columns)
    field_for_sheet = field_with_meta.reindex(columns=desired_cols, fill_value="")

    augmented_lineups = pd.concat(
        [lineups_df, field_for_sheet], ignore_index=True, axis=0
    )

    if output_path is None:
        output_path = lineups_path.with_name(lineups_path.stem + "_augmented.xlsx")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Writing augmented lineups workbook to {output_path}...")
    with pd.ExcelWriter(output_path, engine="xlsxwriter") as writer:
        projections_df.to_excel(
            writer, sheet_name=top1pct_core.PROJECTIONS_SHEET_NAME, index=False
        )
        ownership_df.to_excel(
            writer, sheet_name=top1pct_core.OWNERSHIP_SHEET_NAME, index=False
        )
        if exposure_df is not None:
            exposure_df.to_excel(writer, sheet_name="Exposure", index=False)
        augmented_lineups.to_excel(
            writer, sheet_name=top1pct_core.LINEUPS_SHEET_NAME, index=False
        )

    return output_path


def main(argv: List[str] | None = None) -> None:
    args = _parse_args(argv)
    output_path = _augment_lineups(
        lineups_path=Path(args.lineups_excel),
        corr_path=Path(args.corr_excel),
        extra_lineups=args.extra_lineups,
        output_path=Path(args.output_excel) if args.output_excel is not None else None,
        random_seed=args.random_seed,
    )
    print(f"Augmented lineups workbook written to: {output_path}")


if __name__ == "__main__":
    main()


