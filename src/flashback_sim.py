from __future__ import annotations

"""
Flashback contest simulation for completed NFL Showdown slates.

This script:
  1. Loads a completed DraftKings Showdown contest CSV from data/contests/.
  2. Loads Sabersim projections for the same slate from data/sabersim/.
  3. Builds a player correlation matrix from Sabersim projections using the
     existing simulation-based correlation pipeline.
  4. Simulates correlated player DK outcomes and scores every actual contest
     lineup.
  5. Writes an Excel workbook with:
       - Standings: original contest CSV.
       - Simulation: per-lineup CPT/FLEX players, Top 1% / Top 5% / Top 20%
         finish rates, and average DK points.
       - Entrant summary: averages of these metrics across entries for each
         entrant (cleaned EntryName).
       - Player summary: draft rates and simulation performance by role
         (CPT vs FLEX).

Usage example:

    python -m src.flashback_sim

or with explicit inputs:

    python -m src.flashback_sim \\
        --contest-csv data/contests/my_contest.csv \\
        --sabersim-csv data/sabersim/my_slate.csv \\
        --num-sims 20000
"""

import argparse
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd

from . import build_corr_matrix_from_projections, config, simulation_corr


FLASHBACK_SUBDIR = "flashback"


@dataclass
class ParsedLineup:
    """Structured representation of a single contest lineup."""

    entrant: str
    entry_name_raw: str
    cpt: str
    flex: Tuple[str, str, str, str, str]


def _resolve_latest_csv(directory: Path, explicit: str | None) -> Path:
    """
    Resolve a CSV path, preferring an explicit argument when provided.

    If explicit is None, pick the most recent *.csv file in `directory`.
    """
    if explicit:
        path = Path(explicit)
        if not path.is_file():
            raise FileNotFoundError(f"Specified CSV file does not exist: {path}")
        return path

    candidates = sorted(directory.glob("*.csv"), key=lambda p: p.stat().st_mtime)
    if not candidates:
        raise FileNotFoundError(f"No .csv files found in directory: {directory}")
    return candidates[-1]


def _clean_entrant_name(entry_name: str) -> str:
    """
    Clean an EntryName like 'Fantassin (4/5)' down to just 'Fantassin'.
    """
    if not isinstance(entry_name, str):
        return str(entry_name)
    # Remove a trailing parenthetical like " (4/5)".
    return re.sub(r"\s*\([^)]*\)\s*$", "", entry_name).strip()


def _parse_lineup_string(lineup: str) -> Tuple[str, Tuple[str, str, str, str, str]]:
    """
    Parse a DraftKings Showdown lineup string into CPT and five FLEX names.

    Example lineup string:
        "CPT A.J. Brown FLEX Jalen Hurts FLEX DeVonta Smith FLEX D'Andre Swift "
        "FLEX Jake Elliott FLEX Luther Burden III"
    """
    if not isinstance(lineup, str):
        raise ValueError(f"Lineup value is not a string: {lineup!r}")

    tokens = lineup.split()
    roles = {"CPT", "FLEX"}

    current_role: str | None = None
    current_name_tokens: List[str] = []
    cpt_name: str | None = None
    flex_names: List[str] = []

    def flush_current() -> None:
        nonlocal cpt_name, flex_names, current_role, current_name_tokens
        if current_role is None:
            return
        name = " ".join(current_name_tokens).strip()
        if not name:
            return
        if current_role == "CPT":
            if cpt_name is not None:
                raise ValueError(
                    f"Multiple CPT players found in lineup string: {lineup!r}"
                )
            cpt_name = name
        elif current_role == "FLEX":
            flex_names.append(name)
        current_role = None
        current_name_tokens = []

    for tok in tokens:
        if tok in roles:
            # Starting a new role; flush the previous one.
            flush_current()
            current_role = tok
            current_name_tokens = []
        else:
            current_name_tokens.append(tok)

    # Flush the final role.
    flush_current()

    if cpt_name is None:
        raise ValueError(f"No CPT player found in lineup string: {lineup!r}")
    if len(flex_names) != 5:
        raise ValueError(
            f"Expected 5 FLEX players in lineup string, found {len(flex_names)}: "
            f"{lineup!r}"
        )

    return cpt_name, (flex_names[0], flex_names[1], flex_names[2], flex_names[3], flex_names[4])


def _load_contest_lineups(contest_csv: Path) -> Tuple[pd.DataFrame, List[ParsedLineup]]:
    """
    Load contest standings CSV and parse lineups into structured form.

    Returns:
        standings_df: original contest standings dataframe.
        parsed_lineups: list of ParsedLineup objects for each entry.
    """
    standings_df = pd.read_csv(contest_csv)

    required_cols = {"EntryName", "Lineup"}
    missing = required_cols.difference(standings_df.columns)
    if missing:
        raise KeyError(
            f"Contest CSV missing required columns {sorted(missing)}. "
            f"Expected at least {sorted(required_cols)}."
        )

    parsed: List[ParsedLineup] = []
    for _, row in standings_df.iterrows():
        entry_name_raw = row["EntryName"]
        lineup_str = row["Lineup"]
        entrant = _clean_entrant_name(str(entry_name_raw))
        cpt, flex = _parse_lineup_string(str(lineup_str))
        parsed.append(
            ParsedLineup(
                entrant=entrant,
                entry_name_raw=str(entry_name_raw),
                cpt=cpt,
                flex=flex,
            )
        )

    return standings_df, parsed


def _build_player_universe_and_weights(
    parsed_lineups: Sequence[ParsedLineup],
) -> Tuple[List[str], Dict[str, int], np.ndarray, pd.DataFrame]:
    """
    Build the player universe and lineup weight matrix from parsed lineups.

    Returns:
        player_names: list of unique player names in model order.
        player_index: mapping from player name to index.
        W: weight matrix of shape (num_lineups, num_players) where CPT has
           weight 1.5 and each FLEX has weight 1.0.
        lineups_df: dataframe with Entrant, EntryName, CPT, Flex1..Flex5 columns.
    """
    # Unique player names in order of appearance.
    player_names: List[str] = []
    seen: set[str] = set()

    def _add_name(name: str) -> None:
        if name not in seen:
            seen.add(name)
            player_names.append(name)

    rows: List[Dict[str, object]] = []
    for pl in parsed_lineups:
        _add_name(pl.cpt)
        for name in pl.flex:
            _add_name(name)

        rows.append(
            {
                "Entrant": pl.entrant,
                "EntryName": pl.entry_name_raw,
                "CPT": pl.cpt,
                "Flex1": pl.flex[0],
                "Flex2": pl.flex[1],
                "Flex3": pl.flex[2],
                "Flex4": pl.flex[3],
                "Flex5": pl.flex[4],
            }
        )

    player_index: Dict[str, int] = {name: i for i, name in enumerate(player_names)}

    num_lineups = len(parsed_lineups)
    num_players = len(player_names)
    W = np.zeros((num_lineups, num_players), dtype=float)

    for k, pl in enumerate(parsed_lineups):
        # CPT weight 1.5
        W[k, player_index[pl.cpt]] += 1.5
        # FLEX weights 1.0
        for name in pl.flex:
            W[k, player_index[name]] += 1.0

    lineups_df = pd.DataFrame(rows)
    return player_names, player_index, W, lineups_df


def _build_player_universe_from_sabersim_and_lineups(
    sabersim_df: pd.DataFrame,
    corr_df: pd.DataFrame,
    lineup_player_names: Iterable[str],
) -> Tuple[List[str], np.ndarray, np.ndarray]:
    """
    Build unified player universe, means, and std devs from Sabersim + lineups.

    Universe construction:
      1) Players in the correlation matrix (Sabersim-based) in their existing order.
      2) Remaining players from the Sabersim dataframe.
      3) Any additional players that appear only in the contest lineups.

    For each player:
      - mu: Sabersim 'My Proj' if available, else 0.0.
      - sigma: 'dk_std' if available; otherwise a heuristic
               max(1.0, 0.7 * max(mu, 0)).
    """
    # Player means and optional std devs from Sabersim projections.
    if build_corr_matrix_from_projections.SABERSIM_NAME_COL not in sabersim_df.columns:
        raise KeyError(
            "Sabersim projections missing 'Name' column; expected under "
            f"{build_corr_matrix_from_projections.SABERSIM_NAME_COL!r}."
        )
    if build_corr_matrix_from_projections.SABERSIM_DK_PROJ_COL not in sabersim_df.columns:
        raise KeyError(
            "Sabersim projections missing DK projection column "
            f"{build_corr_matrix_from_projections.SABERSIM_DK_PROJ_COL!r}."
        )

    sabersim = sabersim_df.copy()
    sabersim["Name"] = sabersim[
        build_corr_matrix_from_projections.SABERSIM_NAME_COL
    ].astype(str)
    sabersim_indexed = sabersim.set_index("Name")

    mu_saber = sabersim_indexed[
        build_corr_matrix_from_projections.SABERSIM_DK_PROJ_COL
    ].astype(float)

    if "dk_std" in sabersim_indexed.columns:
        dk_std_saber = sabersim_indexed["dk_std"].astype(float)
    else:
        dk_std_saber = pd.Series(index=sabersim_indexed.index, dtype=float)

    # Universe names: correlations first, then Sabersim-only, then lineup-only.
    base_names = [str(x) for x in corr_df.index.tolist()]
    universe_names: List[str] = []
    seen: set[str] = set()

    def _add_name(n: str) -> None:
        if n not in seen:
            seen.add(n)
            universe_names.append(n)

    for n in base_names:
        _add_name(n)
    for n in sabersim_indexed.index.tolist():
        _add_name(n)
    for n in lineup_player_names:
        _add_name(str(n))

    mu_list: List[float] = []
    sigma_list: List[float] = []

    for name in universe_names:
        if name in mu_saber.index:
            mu_val = float(mu_saber.at[name])
        else:
            mu_val = 0.0

        if name in dk_std_saber.index and pd.notna(dk_std_saber.at[name]):
            sigma_val = float(dk_std_saber.at[name])
        else:
            sigma_val = max(1.0, 0.7 * max(mu_val, 0.0))

        mu_list.append(mu_val)
        sigma_list.append(sigma_val)

    mu = np.array(mu_list, dtype=float)
    sigma = np.array(sigma_list, dtype=float)
    return universe_names, mu, sigma


def _build_full_correlation(
    universe_names: List[str],
    corr_df: pd.DataFrame,
) -> np.ndarray:
    """
    Build a full correlation matrix over the unified player universe.

    Players that are not present in the input matrix are treated as
    independent (corr=0 with others, corr=1 with self).
    """
    base_names = [str(x) for x in corr_df.index.tolist()]
    base_index: Dict[str, int] = {name: i for i, name in enumerate(base_names)}

    n_total = len(universe_names)

    corr_full = np.eye(n_total, dtype=float)
    base_corr = corr_df.to_numpy(dtype=float)

    for i_univ, name_i in enumerate(universe_names):
        i_base = base_index.get(name_i)
        if i_base is None:
            continue
        for j_univ, name_j in enumerate(universe_names):
            j_base = base_index.get(name_j)
            if j_base is None:
                continue
            corr_full[i_univ, j_univ] = float(base_corr[i_base, j_base])

    # Ensure symmetry and unit diagonal.
    corr_full = 0.5 * (corr_full + corr_full.T)
    np.fill_diagonal(corr_full, 1.0)
    return corr_full


def _make_psd(cov: np.ndarray) -> np.ndarray:
    """
    Ensure covariance matrix is positive semi-definite by adjusting the diagonal.
    """
    for _ in range(5):
        try:
            np.linalg.cholesky(cov)
            return cov
        except np.linalg.LinAlgError:
            eigvals = np.linalg.eigvalsh(cov)
            min_eig = float(eigvals.min())
            if min_eig >= 0.0:
                jitter = 1e-8
            else:
                jitter = -min_eig + 1e-8
            cov = cov + np.eye(cov.shape[0], dtype=float) * jitter
    # Final attempt; may still raise.
    np.linalg.cholesky(cov)
    return cov


def _simulate_player_scores(
    mu: np.ndarray,
    sigma: np.ndarray,
    corr_full: np.ndarray,
    num_sims: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Draw correlated player DK scores from a multivariate normal.

    Returns:
        X: array of shape (num_sims, P) with DK scores.
    """
    outer = np.outer(sigma, sigma)
    cov = corr_full * outer
    cov = _make_psd(cov)
    X = rng.multivariate_normal(mean=mu, cov=cov, size=num_sims)
    return X


def _compute_lineup_finish_rates(
    scores: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute top 1%, top 5%, top 20% finish rates and average points per lineup.

    Args:
        scores: array of shape (num_sims, num_lineups) with DK scores.

    Returns:
        top1: array of shape (num_lineups,) with Top 1% finish rates (fractions).
        top5: array of shape (num_lineups,) with Top 5% finish rates (fractions).
        top20: array of shape (num_lineups,) with Top 20% finish rates (fractions).
        avg_points: array of shape (num_lineups,) with mean DK points.
    """
    if scores.ndim != 2:
        raise ValueError("scores must be a 2D array of shape (num_sims, num_lineups).")

    # Quantiles across the field for each simulation.
    q1 = np.quantile(scores, 0.99, axis=1)
    q5 = np.quantile(scores, 0.95, axis=1)
    q20 = np.quantile(scores, 0.80, axis=1)

    mask1 = scores >= q1[:, None]
    mask5 = scores >= q5[:, None]
    mask20 = scores >= q20[:, None]

    top1 = mask1.mean(axis=0)
    top5 = mask5.mean(axis=0)
    top20 = mask20.mean(axis=0)
    avg_points = scores.mean(axis=0)

    return top1, top5, top20, avg_points


def _build_entrant_summary(
    simulation_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build entrant-level summary statistics from per-lineup simulation results.
    """
    grouped = simulation_df.groupby("Entrant", as_index=False)
    summary = grouped.agg(
        {
            "Top 1%": "mean",
            "Top 5%": "mean",
            "Top 20%": "mean",
            "Avg Points": "mean",
        }
    )
    summary.rename(
        columns={
            "Top 1%": "Avg. Top 1%",
            "Top 5%": "Avg. Top 5%",
            "Top 20%": "Avg. Top 20%",
            "Avg Points": "Avg Points",
        },
        inplace=True,
    )
    return summary


def _build_player_summary(
    simulation_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build player-level summary statistics from per-lineup simulation results.

    Expected columns in simulation_df:
        Entrant, CPT, Flex1..Flex5, Top 1%, Top 20%.
    """
    num_lineups = len(simulation_df)
    if num_lineups == 0:
        return pd.DataFrame(
            columns=[
                "Player",
                "CPT draft %",
                "FLEX draft %",
                "CPT top 1% rate",
                "FLEX top 1% rate",
                "CPT top 20% rate",
                "FLEX top 20% rate",
            ]
        )

    # Long-form table for role/player combinations.
    records: List[Dict[str, object]] = []
    for _, row in simulation_df.iterrows():
        top1 = float(row["Top 1%"])
        top20 = float(row["Top 20%"])
        cpt = str(row["CPT"])
        records.append(
            {
                "Player": cpt,
                "role": "CPT",
                "Top 1%": top1,
                "Top 20%": top20,
            }
        )
        for col in ["Flex1", "Flex2", "Flex3", "Flex4", "Flex5"]:
            name = str(row[col])
            records.append(
                {
                    "Player": name,
                    "role": "FLEX",
                    "Top 1%": top1,
                    "Top 20%": top20,
                }
            )

    long_df = pd.DataFrame(records)

    # Draft percentages by role.
    cpt_counts = long_df[long_df["role"] == "CPT"]["Player"].value_counts()
    flex_counts = long_df[long_df["role"] == "FLEX"]["Player"].value_counts()

    all_players = sorted(set(long_df["Player"].astype(str)))
    rows: List[Dict[str, object]] = []

    for player in all_players:
        # Draft percentages.
        cpt_draft = float(cpt_counts.get(player, 0)) / float(num_lineups)
        flex_draft = float(flex_counts.get(player, 0)) / float(num_lineups)

        # Role-specific performance metrics.
        sub_cpt = long_df[(long_df["Player"] == player) and (long_df["role"] == "CPT")]
        sub_flex = long_df[(long_df["Player"] == player) and (long_df["role"] == "FLEX")]

        def _safe_mean(series: pd.Series) -> float:
            return float(series.mean()) if not series.empty else 0.0

        rows.append(
            {
                "Player": player,
                "CPT draft %": cpt_draft,
                "FLEX draft %": flex_draft,
                "CPT top 1% rate": _safe_mean(sub_cpt["Top 1%"]),
                "FLEX top 1% rate": _safe_mean(sub_flex["Top 1%"]),
                "CPT top 20% rate": _safe_mean(sub_cpt["Top 20%"]),
                "FLEX top 20% rate": _safe_mean(sub_flex["Top 20%"]),
            }
        )

    return pd.DataFrame(rows)


def run(
    contest_csv: str | None = None,
    sabersim_csv: str | None = None,
    num_sims: int = 20_000,
    random_seed: int | None = None,
) -> Path:
    """
    Execute the flashback contest simulation pipeline.

    Returns:
        Path to the written output Excel workbook.
    """
    # Resolve input paths.
    contest_dir = config.DATA_DIR / "contests"
    sabersim_dir = config.SABERSIM_DIR

    contest_path = _resolve_latest_csv(contest_dir, contest_csv)
    sabersim_path = _resolve_latest_csv(sabersim_dir, sabersim_csv)

    print(f"Using contest CSV: {contest_path}")
    print(f"Using Sabersim CSV: {sabersim_path}")

    # Load contest standings and parse lineups.
    standings_df, parsed_lineups = _load_contest_lineups(contest_path)
    if not parsed_lineups:
        raise ValueError("No lineups parsed from contest CSV.")

    player_names_from_lineups, player_index_from_lineups, W, sim_lineups_df = (
        _build_player_universe_and_weights(parsed_lineups)
    )

    # Load Sabersim projections and build correlation matrix via simulation.
    sabersim_df = build_corr_matrix_from_projections.load_sabersim_projections(
        sabersim_path
    )
    print("Building player correlation matrix via simulation...")
    corr_df = simulation_corr.simulate_corr_matrix_from_projections(sabersim_df)

    # Build unified player universe and correlation matrix.
    universe_names, mu, sigma = _build_player_universe_from_sabersim_and_lineups(
        sabersim_df=sabersim_df,
        corr_df=corr_df,
        lineup_player_names=player_names_from_lineups,
    )
    corr_full = _build_full_correlation(universe_names, corr_df)

    # Align weight matrix W to universe order (may expand with zero columns).
    index_in_universe: Dict[str, int] = {name: i for i, name in enumerate(universe_names)}
    num_lineups, _ = W.shape
    num_players_univ = len(universe_names)
    W_univ = np.zeros((num_lineups, num_players_univ), dtype=float)
    for name, old_idx in player_index_from_lineups.items():
        new_idx = index_in_universe.get(name)
        if new_idx is not None:
            W_univ[:, new_idx] = W[:, old_idx]

    # Simulate correlated player scores.
    rng = np.random.default_rng(
        random_seed if random_seed is not None else config.SIM_RANDOM_SEED
    )
    print(
        f"Simulating correlated DK scores for {len(universe_names)} players across "
        f"{num_sims} simulations..."
    )
    X = _simulate_player_scores(
        mu=mu,
        sigma=sigma,
        corr_full=corr_full,
        num_sims=num_sims,
        rng=rng,
    )

    # Score lineups and compute finish rates.
    print("Scoring contest lineups and computing finish rates...")
    scores = X @ W_univ.T  # shape (num_sims, num_lineups)
    top1, top5, top20, avg_points = _compute_lineup_finish_rates(scores)

    # Build Simulation sheet.
    simulation_df = sim_lineups_df.copy()
    simulation_df["Top 1%"] = top1
    simulation_df["Top 5%"] = top5
    simulation_df["Top 20%"] = top20
    simulation_df["Avg Points"] = avg_points

    # Entrant and player summaries.
    entrant_summary_df = _build_entrant_summary(simulation_df)
    player_summary_df = _build_player_summary(simulation_df)

    # Output path.
    flashback_dir = config.OUTPUTS_DIR / FLASHBACK_SUBDIR
    flashback_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = flashback_dir / f"flashback_{timestamp}.xlsx"

    print(f"Writing flashback results to {output_path}...")
    with pd.ExcelWriter(output_path, engine="xlsxwriter") as writer:
        standings_df.to_excel(writer, sheet_name="Standings", index=False)
        simulation_df.to_excel(writer, sheet_name="Simulation", index=False)
        entrant_summary_df.to_excel(writer, sheet_name="Entrant summary", index=False)
        player_summary_df.to_excel(writer, sheet_name="Player summary", index=False)

    print("Done.")
    return output_path


def _parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Flashback contest simulation for completed NFL Showdown slates "
            "using Sabersim projections and a correlation matrix."
        )
    )
    parser.add_argument(
        "--contest-csv",
        type=str,
        default=None,
        help=(
            "Path to contest standings CSV under data/contests/. "
            "If omitted, the most recent .csv file in that directory is used."
        ),
    )
    parser.add_argument(
        "--sabersim-csv",
        type=str,
        default=None,
        help=(
            "Path to Sabersim projections CSV under data/sabersim/. "
            "If omitted, the most recent .csv file in that directory is used."
        ),
    )
    parser.add_argument(
        "--num-sims",
        type=int,
        default=20_000,
        help="Number of Monte Carlo simulations to run (default: 20000).",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=None,
        help=(
            "Optional random seed for reproducibility. "
            "Defaults to config.SIM_RANDOM_SEED when omitted."
        ),
    )
    return parser.parse_args(argv)


def main(argv: List[str] | None = None) -> None:
    args = _parse_args(argv)
    run(
        contest_csv=args.contest_csv,
        sabersim_csv=args.sabersim_csv,
        num_sims=args.num_sims,
        random_seed=args.random_seed,
    )


if __name__ == "__main__":
    main()



