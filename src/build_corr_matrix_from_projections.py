from __future__ import annotations

"""
Sabersim projections loading utilities.

This module provides a helper to load a Sabersim Showdown CSV and return a
FLEX-only dataframe suitable for downstream simulation.
"""

from pathlib import Path

import pandas as pd

from . import config


SABERSIM_NAME_COL = "Name"
SABERSIM_TEAM_COL = "Team"
SABERSIM_POS_COL = "Pos"
SABERSIM_SALARY_COL = "Salary"
SABERSIM_DK_PROJ_COL = "My Proj"


def load_sabersim_projections(path: str | Path) -> pd.DataFrame:
    """
    Load Sabersim projections and drop Captain (CPT) rows, keeping FLEX only.

    Heuristic:
      - For each (Name, Team) pair, keep the row with the LOWER salary, which
        corresponds to FLEX in DraftKings Showdown.
    """
    df = pd.read_csv(path)

    # Basic sanity checks
    for col in [SABERSIM_NAME_COL, SABERSIM_TEAM_COL, SABERSIM_SALARY_COL]:
        if col not in df.columns:
            raise KeyError(
                f"Sabersim CSV missing required column '{col}'. "
                "Please check the file schema."
            )

    # Keep lowest-salary row per player-team as FLEX
    df = (
        df.sort_values(by=[SABERSIM_NAME_COL, SABERSIM_TEAM_COL, SABERSIM_SALARY_COL])
        .groupby([SABERSIM_NAME_COL, SABERSIM_TEAM_COL], as_index=False)
        .first()
    )

    # Filter to offensive positions where possible
    if SABERSIM_POS_COL in df.columns:
        df = df[df[SABERSIM_POS_COL].isin(config.OFFENSIVE_POSITIONS)]

    return df



def _prepare_player_features_from_sabersim(
    sabersim_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Construct per-player features aligned with training-time schema.

    We approximate historical stats using projections:
      - mu_player ~= projected DK points
      - sigma_player ~= a constant (e.g., 10.0)
      - season-to-date means ~= projected stats
      - game-level totals derived from team projected DK totals
    """
    df = sabersim_df.copy()

    # Canonical basic identifiers
    df.rename(
        columns={
            SABERSIM_TEAM_COL: config.COL_TEAM,
            SABERSIM_POS_COL: config.COL_POSITION,
        },
        inplace=True,
    )

    # Projected DK points
    if SABERSIM_DK_PROJ_COL not in df.columns:
        raise KeyError(
            f"Sabersim CSV missing DK projection column '{SABERSIM_DK_PROJ_COL}'."
        )
    df[config.COL_DK_POINTS] = df[SABERSIM_DK_PROJ_COL].astype(float)

    # Projected box-score stats (if present)
    proj_stat_map = {
        "Pass Yds": config.COL_PASS_YARDS,
        "Rush Yds": config.COL_RUSH_YARDS,
        "Rec Yds": config.COL_REC_YARDS,
        "Rec": config.COL_RECEPTIONS,
    }
    for raw, canonical in proj_stat_map.items():
        if raw in df.columns:
            df[canonical] = df[raw].astype(float)
        else:
            df[canonical] = 0.0

    # Team-level DK projections as a proxy for team totals
    team_totals = df.groupby(config.COL_TEAM)[config.COL_DK_POINTS].transform("sum")
    df["team_dk_total"] = team_totals

    unique_teams = df[config.COL_TEAM].unique()
    if len(unique_teams) == 2:
        t1, t2 = unique_teams
        totals_by_team = (
            df.groupby(config.COL_TEAM)[config.COL_DK_POINTS].sum().to_dict()
        )
        total_points = float(totals_by_team.get(t1, 0.0) + totals_by_team.get(t2, 0.0))
        point_diff = float(abs(totals_by_team.get(t1, 0.0) - totals_by_team.get(t2, 0.0)))
    else:
        # Fallback for unexpected cases
        total_points = float(df[config.COL_DK_POINTS].sum())
        point_diff = 0.0

    df["total_points"] = total_points
    df["point_diff"] = point_diff

    # Approximate home/away: arbitrarily treat the first team as home
    home_team = unique_teams[0] if len(unique_teams) > 0 else None
    df["is_home"] = (df[config.COL_TEAM] == home_team).astype(int)

    # Approximate per-player mu and sigma from projections
    df[config.COL_MU_PLAYER] = df[config.COL_DK_POINTS]
    df[config.COL_SIGMA_PLAYER] = 10.0  # heuristic constant

    # Approximate season-to-date means using projected stats
    df[f"{config.COL_DK_POINTS}_mean_szn_to_date"] = df[config.COL_DK_POINTS]
    df[f"{config.COL_PASS_YARDS}_mean_szn_to_date"] = df[config.COL_PASS_YARDS]
    df[f"{config.COL_RUSH_YARDS}_mean_szn_to_date"] = df[config.COL_RUSH_YARDS]
    df[f"{config.COL_REC_YARDS}_mean_szn_to_date"] = df[config.COL_REC_YARDS]
    df[f"{config.COL_RECEPTIONS}_mean_szn_to_date"] = df[config.COL_RECEPTIONS]

    # Season/week placeholders (treated as test-time data)
    df[config.COL_SEASON] = 2025
    df[config.COL_WEEK] = 1

    # Keep key columns plus player display name
    df["player_name"] = df[SABERSIM_NAME_COL]

    return df


def build_corr_matrix_from_sabersim(
    model, sabersim_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Given a trained model and a Sabersim projections dataframe (FLEX rows only),
    build pairwise features for all player pairs, predict correlation y_hat,
    clamp to [-1, 1], and assemble into a square correlation matrix DataFrame
    indexed and columned by player name.
    """
    players_df = _prepare_player_features_from_sabersim(sabersim_df)

    # Build all unordered player pairs via self-merge
    players_df = players_df.reset_index(drop=True)
    players_df["idx"] = players_df.index

    pair_df = players_df.merge(
        players_df,
        how="inner",
        left_on="idx",
        right_on="idx",
        suffixes=("_A", "_B"),
    )

    # Alternatively, construct a true Cartesian product and filter to i < j
    pair_df = players_df.merge(
        players_df,
        how="cross",
        suffixes=("_A", "_B"),
    )
    pair_df = pair_df[pair_df["idx_A"] < pair_df["idx_B"]].copy()

    # Assemble feature matrix X with the same columns used during training
    X = pd.DataFrame(index=pair_df.index)

    # Categorical features
    X["A_" + config.COL_POSITION] = pair_df[f"{config.COL_POSITION}_A"]
    X["B_" + config.COL_POSITION] = pair_df[f"{config.COL_POSITION}_B"]
    X["A_" + config.COL_TEAM] = pair_df[f"{config.COL_TEAM}_A"]
    X["B_" + config.COL_TEAM] = pair_df[f"{config.COL_TEAM}_B"]

    # Numerical features
    X["A_is_home"] = pair_df["is_home_A"]
    X["B_is_home"] = pair_df["is_home_B"]

    X["A_" + config.COL_MU_PLAYER] = pair_df[f"{config.COL_MU_PLAYER}_A"]
    X["B_" + config.COL_MU_PLAYER] = pair_df[f"{config.COL_MU_PLAYER}_B"]

    X["A_" + config.COL_SIGMA_PLAYER] = pair_df[f"{config.COL_SIGMA_PLAYER}_A"]
    X["B_" + config.COL_SIGMA_PLAYER] = pair_df[f"{config.COL_SIGMA_PLAYER}_B"]

    X["A_" + config.COL_DK_POINTS] = pair_df[f"{config.COL_DK_POINTS}_A"]
    X["B_" + config.COL_DK_POINTS] = pair_df[f"{config.COL_DK_POINTS}_B"]

    X[
        "A_" + config.COL_DK_POINTS + "_mean_szn_to_date"
    ] = pair_df[f"{config.COL_DK_POINTS}_mean_szn_to_date_A"]
    X[
        "B_" + config.COL_DK_POINTS + "_mean_szn_to_date"
    ] = pair_df[f"{config.COL_DK_POINTS}_mean_szn_to_date_B"]

    X[
        "A_" + config.COL_PASS_YARDS + "_mean_szn_to_date"
    ] = pair_df[f"{config.COL_PASS_YARDS}_mean_szn_to_date_A"]
    X[
        "B_" + config.COL_PASS_YARDS + "_mean_szn_to_date"
    ] = pair_df[f"{config.COL_PASS_YARDS}_mean_szn_to_date_B"]

    X[
        "A_" + config.COL_RUSH_YARDS + "_mean_szn_to_date"
    ] = pair_df[f"{config.COL_RUSH_YARDS}_mean_szn_to_date_A"]
    X[
        "B_" + config.COL_RUSH_YARDS + "_mean_szn_to_date"
    ] = pair_df[f"{config.COL_RUSH_YARDS}_mean_szn_to_date_B"]

    X[
        "A_" + config.COL_REC_YARDS + "_mean_szn_to_date"
    ] = pair_df[f"{config.COL_REC_YARDS}_mean_szn_to_date_A"]
    X[
        "B_" + config.COL_REC_YARDS + "_mean_szn_to_date"
    ] = pair_df[f"{config.COL_REC_YARDS}_mean_szn_to_date_B"]

    X[
        "A_" + config.COL_RECEPTIONS + "_mean_szn_to_date"
    ] = pair_df[f"{config.COL_RECEPTIONS}_mean_szn_to_date_A"]
    X[
        "B_" + config.COL_RECEPTIONS + "_mean_szn_to_date"
    ] = pair_df[f"{config.COL_RECEPTIONS}_mean_szn_to_date_B"]

    X[config.COL_SEASON] = pair_df[f"{config.COL_SEASON}_A"]
    X[config.COL_WEEK] = pair_df[f"{config.COL_WEEK}_A"]
    X["total_points"] = pair_df["total_points_A"]
    X["point_diff"] = pair_df["point_diff_A"]

    # Ensure all expected columns are present; if any are missing because they
    # were not used during training, align with the training feature list.
    all_features: List[str] = CATEGORICAL_FEATURES + [
        f for f in NUMERICAL_FEATURES if f in X.columns
    ]
    X = X[all_features]

    # Predict pairwise correlation and clamp to [-1, 1]
    y_hat = model.predict(X)
    y_hat = np.clip(y_hat, -1.0, 1.0)

    # Build symmetric correlation matrix
    player_names = players_df["player_name"].tolist()
    n = len(player_names)
    corr_mat = pd.DataFrame(
        np.eye(n), index=player_names, columns=player_names, dtype=float
    )

    name_A = pair_df["player_name_A"].values
    name_B = pair_df["player_name_B"].values

    for a, b, corr in zip(name_A, name_B, y_hat):
        corr_mat.at[a, b] = corr
        corr_mat.at[b, a] = corr

    return corr_mat


