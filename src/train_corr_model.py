from __future__ import annotations

"""
Training utilities for the pairwise correlation regression model.
"""

from typing import List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from . import config, diagnostics


# Feature definitions shared between training and inference
CATEGORICAL_FEATURES: List[str] = [
    "A_" + config.COL_POSITION,
    "B_" + config.COL_POSITION,
    "A_" + config.COL_TEAM,
    "B_" + config.COL_TEAM,
]

NUMERICAL_FEATURES: List[str] = [
    "A_is_home",
    "B_is_home",
    "A_" + config.COL_MU_PLAYER,
    "B_" + config.COL_MU_PLAYER,
    "A_" + config.COL_SIGMA_PLAYER,
    "B_" + config.COL_SIGMA_PLAYER,
    "A_" + config.COL_DK_POINTS,
    "B_" + config.COL_DK_POINTS,
    "A_" + config.COL_DK_POINTS + "_mean_szn_to_date",
    "B_" + config.COL_DK_POINTS + "_mean_szn_to_date",
    "A_" + config.COL_PASS_YARDS + "_mean_szn_to_date",
    "B_" + config.COL_PASS_YARDS + "_mean_szn_to_date",
    "A_" + config.COL_RUSH_YARDS + "_mean_szn_to_date",
    "B_" + config.COL_RUSH_YARDS + "_mean_szn_to_date",
    "A_" + config.COL_REC_YARDS + "_mean_szn_to_date",
    "B_" + config.COL_REC_YARDS + "_mean_szn_to_date",
    "A_" + config.COL_RECEPTIONS + "_mean_szn_to_date",
    "B_" + config.COL_RECEPTIONS + "_mean_szn_to_date",
    config.COL_SEASON,
    config.COL_WEEK,
    "total_points",
    "point_diff",
]


def _time_based_split(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split pairwise dataset into train/val/test using season ranges from config.
    """
    season_series = df[config.COL_SEASON]
    train_mask = season_series.isin(config.TRAIN_SEASONS)
    val_mask = season_series.isin(config.VAL_SEASONS)
    test_mask = season_series.isin(config.TEST_SEASONS)

    train_df = df[train_mask].copy()
    val_df = df[val_mask].copy()
    test_df = df[test_mask].copy()
    return train_df, val_df, test_df


def _build_pipeline() -> Pipeline:
    """
    Build a scikit-learn Pipeline with preprocessing and a tree-based regressor.
    """
    # Use a dense output for compatibility with HistGradientBoostingRegressor.
    try:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:  # fallback for older scikit-learn versions
        ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "cat",
                ohe,
                CATEGORICAL_FEATURES,
            ),
            ("num", "passthrough", NUMERICAL_FEATURES),
        ],
        remainder="drop",
    )

    try:
        base_model = HistGradientBoostingRegressor(
            max_depth=6,
            learning_rate=0.1,
            max_iter=200,
            random_state=42,
        )
    except Exception:  # pragma: no cover - fallback if not available
        base_model = RandomForestRegressor(
            n_estimators=200,
            max_depth=None,
            n_jobs=-1,
            random_state=42,
        )

    pipe = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", base_model),
        ]
    )
    return pipe


def train_corr_regressor(pairwise_df: pd.DataFrame) -> Pipeline:
    """
    Train the correlation regression model and save it to disk.

    Steps:
      1. Separate features X and target y.
      2. Perform time-based train/val/test split by season.
      3. Fit a tree-based regressor wrapped in a Pipeline.
      4. Evaluate on val and test sets (RMSE, MAE) and print metrics.
      5. Save the fitted pipeline to models/corr_model.pkl.
    """
    # Ensure all required feature columns exist; if some rolling features are
    # missing, they will be silently ignored later when selecting columns.
    available_num_features = [
        f for f in NUMERICAL_FEATURES if f in pairwise_df.columns
    ]
    X_cols = CATEGORICAL_FEATURES + available_num_features

    df = pairwise_df.copy()
    if "y" not in df.columns:
        raise KeyError("Pairwise dataframe must contain target column 'y'.")

    # Diagnostics for full pairwise dataset
    diagnostics.write_df_snapshot(df, name="pairwise_full", step="train_input")

    train_df, val_df, test_df = _time_based_split(df)

    # Basic split diagnostics
    print(
        f"Train/Val/Test sizes: "
        f"{len(train_df)} / {len(val_df)} / {len(test_df)} rows"
    )

    # Guard against empty training data and provide helpful guidance.
    if len(train_df) == 0:
        seasons = df[config.COL_SEASON].dropna()
        if seasons.empty:
            raise ValueError(
                "No season information available in pairwise dataframe; cannot "
                "perform time-based split for training."
            )
        min_season = int(seasons.min())
        max_season = int(seasons.max())
        raise ValueError(
            "Training split is empty: no rows with seasons in TRAIN_SEASONS. "
            f"Available seasons range from {min_season} to {max_season}. "
            "Update TRAIN_SEASONS/VAL_SEASONS/TEST_SEASONS in config.py or "
            "adjust the splitting logic to match your data."
        )

    X_train = train_df[X_cols]
    y_train = train_df["y"]

    X_val = val_df[X_cols]
    y_val = val_df["y"]

    X_test = test_df[X_cols]
    y_test = test_df["y"]

    # Diagnostics for split datasets
    diagnostics.write_df_snapshot(train_df, name="train", step="train_split")
    diagnostics.write_df_snapshot(val_df, name="val", step="train_split")
    diagnostics.write_df_snapshot(test_df, name="test", step="train_split")

    pipeline = _build_pipeline()
    pipeline.fit(X_train, y_train)

    def _eval_split(name: str, X: pd.DataFrame, y: pd.Series) -> None:
        if len(X) == 0:
            print(f"{name}: no samples")  # pragma: no cover - diagnostic
            return
        preds = pipeline.predict(X)
        rmse = float(np.sqrt(mean_squared_error(y, preds)))
        mae = float(mean_absolute_error(y, preds))
        print(f"{name} RMSE={rmse:.4f}, MAE={mae:.4f}")

    _eval_split("Train", X_train, y_train)
    _eval_split("Validation", X_val, y_val)
    _eval_split("Test", X_test, y_test)

    # Persist pipeline
    config.ensure_directories()
    joblib.dump(pipeline, config.CORR_MODEL_PATH)

    return pipeline


