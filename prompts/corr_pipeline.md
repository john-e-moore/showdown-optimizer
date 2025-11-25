# Coding Agent Prompt: NFL Player Correlation Model Pipeline (Python)

You are an expert Python/ML engineer. Build the **first stage** of a pipeline for NFL DFS Showdown Captain Mode lineup optimization.

The goal of this stage is to:
1. Ingest historical NFL data (box scores + game scores) from **nflverse** (Parquet, regular season, 2005+).
2. Train a **regression model** that predicts **pairwise fantasy point correlation** between two players in a single game, using the **z-score product trick** described below.
3. Load a **Sabersim projections CSV** (for a single DraftKings Showdown game) and use the trained model to produce a full **player correlation matrix** across *both* teams.
4. Output an **Excel file** with:
   - Sheet 1: original Sabersim projections
   - Sheet 2: correlation matrix (values in [-1, 1]).

Later stages (not in this task) will use this correlation matrix to generate lineups optimized for top 1% finishes in Showdown contests.

---

## 1. Project structure

Create the following directory/file structure:

```text
project_root/
  data/
    nfl_raw/          # parquet files from nflverse (player-game stats, games/schedule)
    nfl_processed/    # cleaned player-game dataset with fantasy points & z-scores
    sabersim/         # Sabersim projection CSVs
  models/
    corr_model.pkl    # trained correlation regression model
  outputs/
    correlations/
      showdown_corr_matrix.xlsx  # Excel with projections + correlation matrix
  src/
    config.py
    data_loading.py
    fantasy_scoring.py
    feature_engineering.py
    build_pairwise_dataset.py
    train_corr_model.py
    build_corr_matrix_from_projections.py
    main.py
```

You may add helper modules if needed, but keep things relatively simple and well organized.

Assume the user will place the **Sabersim projections CSV** at:

```text
data/sabersim/NFL_2025-11-24-815pm_DK_SHOWDOWN_CAR-@-SF.csv
```

(That file path already exists in the environment and is equivalent to `/mnt/data/NFL_2025-11-24-815pm_DK_SHOWDOWN_CAR-@-SF.csv`.)

---

## 2. Data ingestion (nflverse, Parquet)

### 2.1 Inputs

Use **nflverse**-style Parquet data files (do *not* hard-code remote URLs; instead, parameterize file paths in `config.py` so the user can point to local Parquet files). For this task, assume the user has or will download Parquet versions of:

- **Player game stats** (one row per player per game)
- **Games/schedule** data (one row per game)

In `config.py`, define (at minimum):

```python
NFL_PLAYER_GAMES_PARQUET = "data/nfl_raw/player_stats.parquet"
NFL_GAMES_PARQUET = "data/nfl_raw/games.parquet"
SABERSIM_CSV = "data/sabersim/NFL_2025-11-24-815pm_DK_SHOWDOWN_CAR-@-SF.csv"
OUTPUT_CORR_EXCEL = "outputs/correlations/showdown_corr_matrix.xlsx"
MIN_PLAYER_GAMES = 8  # minimum games per player to compute stable z-scores
```

### 2.2 Loading & filtering (in `data_loading.py`)

Implement functions:

```python
def load_player_game_stats(path: str) -> pd.DataFrame: ...
def load_games(path: str) -> pd.DataFrame: ...
```

Requirements:

- Load Parquet files using pandas.
- Filter to **regular-season games only** (no preseason/postseason).
- Filter seasons to **2005 and later**.
- Ensure there is a unique `game_id` column used to join player stats with game metadata.

If necessary, allow for flexible column names by defining a small mapping in `config.py` (e.g., `COL_GAME_ID`, `COL_PLAYER_ID`, etc.).

---

## 3. Fantasy scoring (DraftKings Classic-like)

In `fantasy_scoring.py`, implement a function that computes *offensive* DraftKings fantasy points from box score fields for each player-game row. For now, ignore defense/special teams and fumbles; focus on offense-only positions (QB, RB, WR, TE, and optionally K).

Use the following DK-like scoring rules:

- Passing
  - +0.04 points per passing yard
  - +4 points per passing TD
  - –1 point per interception
  - +3-point bonus for 300+ passing yards
- Rushing
  - +0.1 points per rushing yard
  - +6 points per rushing TD
  - +3-point bonus for 100+ rushing yards
- Receiving
  - +1 point per reception (PPR)
  - +0.1 points per receiving yard
  - +6 points per receiving TD
  - +3-point bonus for 100+ receiving yards

Implement:

```python
def compute_dk_points_offense(df: pd.DataFrame) -> pd.DataFrame:
    '''
    Adds a "dk_points" column to the player-game dataframe, using
    the above scoring rules and standard column names like:
    pass_yards, pass_tds, interceptions,
    rush_yards, rush_tds,
    rec_yards, rec_tds, receptions.
    '''
```

If raw column names differ (e.g., `passing_yards` vs `pass_yards`), centralize mappings in `config.py` so they can be easily changed.

---

## 4. Standardization & z-scores

In `feature_engineering.py`, implement functions to compute **per-player** fantasy point means and standard deviations, and then per-game z-scores:

```python
def add_player_dk_stats(df: pd.DataFrame, min_games: int) -> pd.DataFrame:
    '''
    Given a player-game dataframe with columns:
      - player_id, dk_points, season, week, game_id, team, opponent, position
    1. Compute per-player mean (mu_i) and std (sigma_i) of dk_points across all games.
    2. Filter out players with < min_games, or set sigma_i to a reasonable default
       (e.g., shrink toward positional averages).
       For a first version, it's OK to just drop players with < min_games.
    3. Create "mu_player", "sigma_player", and standardized "z" = (dk_points - mu_player)/sigma_player.
    4. Return the augmented dataframe with these columns.
    '''
```

Make sure to:

- Avoid sigma = 0: if a player has constant scores, set a small epsilon (e.g., 0.1) to avoid division by zero.
- Keep only players that will be used downstream (offensive positions + maybe K).

Store the processed player-game dataframe to `data/nfl_processed/player_games_with_z.parquet` for reuse.

---

## 5. Build pairwise training dataset (z_i * z_j trick)

In `build_pairwise_dataset.py`, construct the model training dataset based on **pairwise products of standardized scores**.

### 5.1 Target definition

For each game `g` and each unordered player pair `(i, j)` that both appear in that game (offensive positions only):

- Let `z_i` and `z_j` be their standardized fantasy scores in game `g`.
- Define the target:
  ```python
  y = z_i * z_j
  ```

### 5.2 Features

Build features that describe Player A, Player B, and the game context. Keep it simple but informative for the first version.

**Player-level features (for both A and B, with prefixes `A_` and `B_`):**

- Position (one-hot encoded or categorical)
- Team
- Home vs away (if games data has a home/away flag)
- Player’s overall mean `mu_player` and std `sigma_player`
- Season-to-date rolling means for:
  - dk_points
  - passing yards (if QB)
  - rushing yards (if RB/QB)
  - receiving yards and receptions (if WR/TE)
- Simple usage proxies (e.g., share of team dk_points or yards up to that game)

Make sure **rolling features use only prior games** (no leakage). If rolling windows are complex, start with season-to-date average up to the previous game.

**Game-level features (no prefix or use `G_`):**

- Season, week
- Total points scored in that game (home_score + away_score)
- Point differential (abs(home_score - away_score))
- Indicator for divisional matchup (if such info is easily accessible; optional)

### 5.3 Implementation

Implement:

```python
def build_pairwise_dataset(player_games_df: pd.DataFrame, games_df: pd.DataFrame) -> pd.DataFrame:
    '''
    Returns a dataframe with columns for:
      - A_player_id, B_player_id
      - A_* features, B_* features
      - Game-level features
      - target y = z_A * z_B
    Only includes offensive players and games after 2005 regular season.
    '''
```

This dataset will be used to train the correlation regression model.

---

## 6. Model training (correlation regressor)

In `train_corr_model.py`, train a relatively simple but appropriate regression model on the pairwise dataset.

### 6.1 Model choice

Use a **tree-based regression model** from scikit-learn, e.g. `HistGradientBoostingRegressor` or `GradientBoostingRegressor`. This is flexible, handles non-linearities, and is easy to tune.

Assume scikit-learn is available. If not, fall back to `RandomForestRegressor`.

### 6.2 Train/validation/test split

- Perform a **time-based split**:
  - Train on older seasons (e.g. 2005–2017)
  - Validate on mid seasons (e.g. 2018–2020)
  - Test on recent seasons (e.g. 2021+)
- Split based on season (and optionally week), not random rows.

### 6.3 Training script

Implement:

```python
def train_corr_regressor(pairwise_df: pd.DataFrame):
    '''
    1. Separate features X and target y.
    2. Encode categorical features (e.g., position, team) using one-hot or similar.
    3. Split into train/val/test based on season.
    4. Train a tree-based regressor (e.g. HistGradientBoostingRegressor).
    5. Evaluate on val and test sets (RMSE, maybe MAE).
    6. Save the fitted model and any encoders/transformers to "models/corr_model.pkl".
    '''
```

Use `joblib` or `pickle` for model persistence. Ensure any preprocessing (encoders, scalers) is saved as part of a single pipeline using `sklearn.pipeline.Pipeline`.

Clamp model predictions to the range [-1, 1] at inference time.

---

## 7. Building a correlation matrix from Sabersim projections

In `build_corr_matrix_from_projections.py`, implement functionality to:

1. Load the **Sabersim projections CSV** from `config.SABERSIM_CSV`.
2. Build **pairwise feature rows** for all player pairs in that file.
3. Use the trained model to predict `y_hat` = estimated correlation for each pair.
4. Assemble these into a square correlation matrix.

### 7.1 Sabersim projections loading

Assume the Sabersim CSV contains (at least):

- Player identifiers: `Name`, `Team`, `Pos`
- Box-score-like projections: `Pass Yds`, `Pass TD`, `Rush Yds`, `Rush TD`, `Rec`, `Rec Yds`, `Rec TD`, etc.
- A DraftKings fantasy projection: `My Proj` (or similar)
- Ownership, percentiles, etc. (not strictly needed for this step)

You **must**:

- Drop the **Captain (CPT)** rows, keeping only FLEX rows (as indicated by lower `Salary` for the same player name).
- For each player, treat the projections as **expected stats for this game**, analogous to pregame expectations used in training.

### 7.2 Feature construction from projections

Construct features aligned as closely as possible with the training features:
- For Player A and Player B:
  - Position, team, projected DK points (`My Proj`), and the projected box-score columns (`Pass Yds`, `Rec Yds`, etc.).
  - You can approximate usage/role-based features using projections (e.g. player’s share of team projected DK points or yards).
- Game-level features:
  - Because Sabersim projections are for a single game, you will likely not have actual points scored. Use proxies such as:
    - Sum of all offensive players’ projected DK points per team as a proxy for team total.
    - If available, use an external Vegas total/spread; otherwise, you can omit or approximate.

Implement a function:

```python
def build_corr_matrix_from_sabersim(model, sabersim_df: pd.DataFrame) -> pd.DataFrame:
    '''
    Given a trained model and a Sabersim projections dataframe (FLEX rows only),
    build pairwise features for all player pairs, predict correlation y_hat,
    clamp to [-1, 1], and assemble into a square correlation matrix DataFrame
    indexed and columned by player name.
    '''
```

Ensure the correlation matrix is symmetric with 1.0 on the diagonal (even if the model’s self-prediction is not exactly 1).

---

## 8. Final Excel output (projections + correlation matrix)

In `main.py`, implement a CLI or simple entry point that:

1. Runs the full pipeline if needed (or assume historical model is already trained):
   - Load & process historical data.
   - Compute DK points and z-scores.
   - Build pairwise dataset.
   - Train and save model (or load existing).
2. Loads the trained model from `models/corr_model.pkl`.
3. Loads the Sabersim projections CSV from `config.SABERSIM_CSV`.
4. Builds the correlation matrix using `build_corr_matrix_from_sabersim(...)`.
5. Writes an Excel file to `OUTPUT_CORR_EXCEL` with:
   - Sheet `"Sabersim_Projections"`: original (or lightly cleaned) Sabersim data (FLEX rows only).
   - Sheet `"Correlation_Matrix"`: the correlation matrix DataFrame.

Use `pandas.ExcelWriter` with the `xlsxwriter` engine. Include indices (player names) as row/column labels in the correlation sheet.

---

## 9. General coding and documentation guidelines

- Use Python 3.10+.
- Prefer `pandas` and `numpy` for data handling; `scikit-learn` for models.
- Add clear docstrings and type hints for all functions.
- Make all paths configurable through `config.py`, no hard-coded absolute paths.
- Write code so that later we can easily plug this into a larger Showdown lineup optimization pipeline (the main artifact we care about is the correlation matrix for the players in a Sabersim projections sheet).

At the end, produce:
- All `.py` modules in `src/` as described.
- A short `README.md` in `project_root/` summarizing how to run `main.py` to train the model and generate the correlation Excel file for a Sabersim showdown slate.