from __future__ import annotations

"""
Sport-agnostic flashback contest simulation core for completed Showdown slates.

This module implements the heavy lifting for:
  - Loading contest standings and parsing CPT plus flex-style lineups.
  - Loading Sabersim projections and building a correlation matrix.
  - Simulating correlated DK scores for all players.
  - Scoring every contest lineup across simulations.
  - Computing finish rates, simulated ROI, and summary tables.

It is parameterised by:
  - config_module: provides DATA_DIR, OUTPUTS_DIR, SABERSIM_DIR, SIM_RANDOM_SEED.
  - load_sabersim_projections: function(path) -> flex-style Sabersim dataframe.
  - simulate_corr_matrix_from_projections: function(sabersim_df) -> corr_df.

NFL and NBA wrappers should live in sport-specific modules and call
`run_flashback` with their own config and loader functions.
"""

import json
import re
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple, Callable, Any

import numpy as np
import pandas as pd


DK_PAYOUT_URL_TEMPLATE = (
    "https://api.draftkings.com/contests/v1/contests/{contest_id}?format=json"
)
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
    Parse a DraftKings Showdown lineup string into one CPT and five flex-role names.

    DraftKings uses ``CPT`` plus either ``FLEX`` (NFL) or ``UTIL`` (NBA) tokens
    in the exported lineup strings; both FLEX and UTIL are treated as flex-style
    slots here.
    """
    if not isinstance(lineup, str):
        raise ValueError(f"Lineup value is not a string: {lineup!r}")

    tokens = lineup.split()
    # Accept both FLEX and UTIL as flex-style roles.
    roles = {"CPT", "FLEX", "UTIL"}

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
        else:
            # Any non-CPT role token (FLEX, UTIL, etc.) is treated as a flex slot.
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
            "Expected 5 non-CPT (flex-style) players in lineup string, found "
            f"{len(flex_names)}: {lineup!r}"
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
        lineup_val = row["Lineup"]

        # Some rows may not have a valid lineup string (NaN/blank); skip them.
        if pd.isna(lineup_val):
            continue
        lineup_str = str(lineup_val)
        if not lineup_str.strip():
            continue

        entrant = _clean_entrant_name(str(entry_name_raw))
        cpt, flex = _parse_lineup_string(lineup_str)
        parsed.append(
            ParsedLineup(
                entrant=entrant,
                entry_name_raw=str(entry_name_raw),
                cpt=cpt,
                flex=flex,
            )
        )

    return standings_df, parsed


def _download_payout_json(contest_id: str, dest_path: Path) -> bool:
    """
    Download payout JSON for a contest from DraftKings and write it to dest_path.

    Returns:
        True if the file was downloaded and written successfully; False otherwise.
    """
    url = DK_PAYOUT_URL_TEMPLATE.format(contest_id=contest_id)
    print(f"Downloading DraftKings payout structure from {url} ...")

    try:
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(url, timeout=10) as resp:
            status = getattr(resp, "status", None)
            if status is not None and status != 200:
                print(
                    f"Warning: DraftKings payout request for contest {contest_id} "
                    f"returned HTTP status {status}; skipping ROI computation."
                )
                return False
            data = resp.read()
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        print(
            f"Warning: Failed to download DraftKings payout JSON for contest {contest_id} "
            f"from {url}: {exc}. Skipping ROI computation."
        )
        return False

    try:
        payload = json.loads(data)
    except json.JSONDecodeError as exc:
        print(
            f"Warning: Failed to parse DraftKings payout JSON for contest {contest_id} "
            f"from {url}: {exc}. Skipping ROI computation."
        )
        return False

    try:
        with dest_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, sort_keys=True)
    except OSError as exc:
        print(
            f"Warning: Failed to write payout JSON to {dest_path}: {exc}. "
            "Skipping ROI computation."
        )
        return False

    print(f"Wrote DraftKings payout JSON to {dest_path}")
    return True


def _load_payout_structure(
    payouts_csv: str | None,
    contest_id: str,
    num_entries: int,
    *,
    data_dir: Path,
) -> Tuple[np.ndarray, float] | None:
    """
    Load DraftKings payout structure for a contest and build a rank→payout array.
    """
    payouts_dir = data_dir / "payouts"

    if payouts_csv is not None:
        path = Path(payouts_csv)
        if not path.is_file():
            raise FileNotFoundError(
                f"Specified payout file does not exist: {path} "
                "(from --payouts-csv)."
            )
    else:
        path = payouts_dir / f"payouts-{contest_id}.json"
        if not path.is_file():
            print(
                f"Warning: No payout file found at inferred path {path}. "
                "Attempting to download from DraftKings payouts API..."
            )
            if not _download_payout_json(str(contest_id), path):
                print(
                    "Warning: Failed to obtain payout file from DraftKings. "
                    "Skipping ROI computation."
                )
                return None

    try:
        with path.open("r", encoding="utf-8") as f:
            raw = json.load(f)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Failed to parse payout JSON at {path}: {exc}") from exc

    try:
        detail = raw["contestDetail"]
    except (TypeError, KeyError) as exc:
        raise ValueError(
            f"Payout JSON at {path} missing 'contestDetail' section."
        ) from exc

    try:
        entry_fee = float(detail.get("entryFee", 0.0))
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Payout JSON at {path} has non-numeric 'entryFee' field."
        ) from exc

    payout_summary = detail.get("payoutSummary", [])
    payouts = np.zeros(num_entries, dtype=float)

    if not isinstance(payout_summary, list):
        raise ValueError(
            f"Payout JSON at {path} has unexpected 'payoutSummary' format; "
            "expected a list."
        )

    for tier in payout_summary:
        try:
            min_pos = int(tier.get("minPosition", 0))
            max_pos = int(tier.get("maxPosition", min_pos))
        except (TypeError, ValueError):
            continue

        if max_pos < 1 or min_pos < 1:
            continue

        payout_descs = tier.get("payoutDescriptions") or []
        value: float = 0.0
        if isinstance(payout_descs, list):
            for desc in payout_descs:
                if not isinstance(desc, dict):
                    continue
                if "value" in desc:
                    try:
                        value = float(desc["value"])
                    except (TypeError, ValueError):
                        continue
                    break
        if value <= 0.0:
            continue

        for pos in range(min_pos, max_pos + 1):
            idx = pos - 1
            if 0 <= idx < num_entries:
                payouts[idx] = value

    return payouts, entry_fee


def _build_player_universe_and_weights(
    parsed_lineups: Sequence[ParsedLineup],
) -> Tuple[List[str], Dict[str, int], np.ndarray, pd.DataFrame]:
    """
    Build the player universe and lineup weight matrix from parsed lineups.
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
        # Flex-style weights 1.0
        for name in pl.flex:
            W[k, player_index[name]] += 1.0

    lineups_df = pd.DataFrame(rows)
    return player_names, player_index, W, lineups_df


def _build_player_universe_from_sabersim_and_lineups(
    sabersim_df: pd.DataFrame,
    corr_df: pd.DataFrame,
    lineup_player_names: Iterable[str],
    *,
    name_col: str,
    dk_proj_col: str,
) -> Tuple[List[str], np.ndarray, np.ndarray]:
    """
    Build unified player universe, means, and std devs from Sabersim + lineups.
    """
    if name_col not in sabersim_df.columns:
        raise KeyError(
            f"Sabersim projections missing Name column {name_col!r}."
        )
    if dk_proj_col not in sabersim_df.columns:
        raise KeyError(
            f"Sabersim projections missing DK projection column {dk_proj_col!r}."
        )

    sabersim = sabersim_df.copy()
    sabersim["Name"] = sabersim[name_col].astype(str)
    sabersim_indexed = sabersim.set_index("Name")

    mu_saber = sabersim_indexed[dk_proj_col].astype(float)

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


def _compute_sim_roi(
    scores: np.ndarray,
    payouts_by_rank: np.ndarray,
    entry_fee: float,
) -> np.ndarray:
    """
    Compute simulated ROI for each lineup given a payout structure.
    """
    if scores.ndim != 2:
        raise ValueError("scores must be a 2D array of shape (num_sims, num_lineups).")
    if entry_fee <= 0:
        raise ValueError("entry_fee must be positive for ROI computation.")

    num_sims, num_lineups = scores.shape

    payouts = np.asarray(payouts_by_rank, dtype=float).reshape(-1)
    if payouts.size == 0:
        return np.zeros(num_lineups, dtype=float)

    # Ensure we have payouts defined for at least num_lineups positions.
    if payouts.size < num_lineups:
        payouts_full = np.zeros(num_lineups, dtype=float)
        payouts_full[:payouts.size] = payouts
    else:
        payouts_full = payouts[:num_lineups]

    # Allocate payout matrix with DK-style tie splitting applied per simulation.
    payout_matrix = np.zeros_like(scores, dtype=float)

    for sim_idx in range(num_sims):
        row_scores = scores[sim_idx]
        # Order lineups by score descending.
        order = np.argsort(-row_scores)
        sorted_scores = row_scores[order]

        payouts_this_sim = np.zeros(num_lineups, dtype=float)

        start = 0
        while start < num_lineups:
            # Find end of tie group [start, end) in sorted order.
            end = start + 1
            while end < num_lineups and sorted_scores[end] == sorted_scores[start]:
                end += 1

            lo = start
            hi = end
            group_len = hi - lo

            if group_len <= 0:
                start = end
                continue

            if lo >= payouts_full.size:
                avg_payout = 0.0
            else:
                slice_hi = min(hi, payouts_full.size)
                total = float(payouts_full[lo:slice_hi].sum()) if slice_hi > lo else 0.0
                avg_payout = total / float(group_len)

            tied_indices = order[lo:hi]
            payouts_this_sim[tied_indices] = avg_payout

            start = end

        payout_matrix[sim_idx] = payouts_this_sim

    roi_matrix = (payout_matrix - entry_fee) / entry_fee

    sim_roi = roi_matrix.mean(axis=0)
    return sim_roi


def _build_sabersim_ownership_and_salary(
    sabersim_raw_df: pd.DataFrame,
    *,
    name_col: str,
    team_col: str,
    salary_col: str,
    proj_col: str,
) -> Dict[Tuple[str, str], Dict[str, float]]:
    """
    Build CPT / flex-style ownership and salary mapping from the raw Sabersim CSV.
    """
    required_cols = {name_col, team_col, salary_col, proj_col, "My Own"}
    missing = required_cols.difference(sabersim_raw_df.columns)
    if missing:
        raise KeyError(
            "Sabersim CSV missing required columns for ownership/salary "
            f"computation {sorted(missing)}."
        )

    df = sabersim_raw_df.copy()
    df[name_col] = df[name_col].astype(str)
    df[team_col] = df[team_col].astype(str)
    df[proj_col] = df[proj_col].astype(float)
    df["My Own"] = df["My Own"].astype(float)
    df[salary_col] = df[salary_col].astype(float)

    mapping: Dict[Tuple[str, str], Dict[str, float]] = {}

    grouped = df.groupby([name_col, team_col])
    for (name, team), g in grouped:
        if g.empty:
            continue
        g_sorted = g.sort_values(by=proj_col, ascending=False)
        cpt_row = g_sorted.iloc[0]
        cpt_own = float(cpt_row["My Own"])
        cpt_salary = float(cpt_row[salary_col])

        flex_own = 0.0
        flex_salary = 0.0
        if len(g_sorted) > 1:
            flex_row = g_sorted.iloc[1]
            flex_own = float(flex_row["My Own"])
            flex_salary = float(flex_row[salary_col])

        mapping[(str(name), str(team))] = {
            "cpt_own_pct": cpt_own,
            "flex_own_pct": flex_own,
            "cpt_salary": cpt_salary,
            "flex_salary": flex_salary,
        }

    return mapping


def _build_projected_ownership_by_player(
    ownership_salary_by_name_team: Dict[Tuple[str, str], Dict[str, float]],
) -> Dict[str, Dict[str, float]]:
    """
    Aggregate projected CPT / flex-style ownership by player name across teams.
    """
    projected: Dict[str, Dict[str, float]] = {}
    for (name, _team), info in ownership_salary_by_name_team.items():
        record = projected.setdefault(
            name, {"cpt_proj_own": 0.0, "flex_proj_own": 0.0}
        )
        record["cpt_proj_own"] += float(info.get("cpt_own_pct", 0.0))
        record["flex_proj_own"] += float(info.get("flex_own_pct", 0.0))
    return projected


def _add_ownership_columns_to_simulation(
    simulation_df: pd.DataFrame,
    ownership_salary_by_name_team: Dict[Tuple[str, str], Dict[str, float]],
    name_to_team: Dict[str, str],
) -> pd.DataFrame:
    """
    Add Sum Ownership and Salary-weighted Ownership columns to simulation_df.
    """
    if simulation_df.empty:
        result = simulation_df.copy()
        result["Sum Ownership"] = pd.Series(dtype=float)
        result["Salary-weighted Ownership"] = pd.Series(dtype=float)
        return result

    sum_own_values: List[float] = []
    swo_values: List[float] = []

    for _, row in simulation_df.iterrows():
        total_own = 0.0
        swo_raw = 0.0

        # CPT slot.
        cpt_name = str(row["CPT"])
        cpt_team = name_to_team.get(cpt_name, "")
        cpt_info = ownership_salary_by_name_team.get((cpt_name, cpt_team))
        if cpt_info is not None:
            cpt_own = float(cpt_info.get("cpt_own_pct", 0.0))
            cpt_salary = float(cpt_info.get("cpt_salary", 0.0))
            total_own += cpt_own
            swo_raw += cpt_salary * cpt_own

        # FLEX slots.
        for col in ["Flex1", "Flex2", "Flex3", "Flex4", "Flex5"]:
            flex_name = str(row[col])
            flex_team = name_to_team.get(flex_name, "")
            flex_info = ownership_salary_by_name_team.get((flex_name, flex_team))
            if flex_info is None:
                continue
            flex_own = float(flex_info.get("flex_own_pct", 0.0))
            flex_salary = float(flex_info.get("flex_salary", 0.0))
            total_own += flex_own
            swo_raw += flex_salary * flex_own

        sum_own_values.append(total_own)
        # Scale by 1,000 for readability.
        swo_values.append(swo_raw / 1000.0)

    result = simulation_df.copy()
    result["Sum Ownership"] = sum_own_values
    result["Salary-weighted Ownership"] = swo_values
    return result


def _build_entrant_summary(
    simulation_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build entrant-level summary statistics from per-lineup simulation results.
    """
    # Mean performance metrics by entrant.
    grouped = simulation_df.groupby("Entrant", as_index=False)
    agg_spec: Dict[str, str] = {
        "Top 1%": "mean",
        "Top 5%": "mean",
        "Top 20%": "mean",
        "Avg Points": "mean",
    }
    if "Sim ROI" in simulation_df.columns:
        agg_spec["Sim ROI"] = "mean"

    summary = grouped.agg(agg_spec)
    summary.rename(
        columns={
            "Top 1%": "Avg. Top 1%",
            "Top 5%": "Avg. Top 5%",
            "Top 20%": "Avg. Top 20%",
            "Avg Points": "Avg Points",
            "Sim ROI": "Avg. Sim ROI",
        },
        inplace=True,
    )

    # Entry counts per entrant.
    entries_df = (
        simulation_df.groupby("Entrant", as_index=False)
        .agg(Entries=("Entrant", "count"))
    )

    summary = entries_df.merge(summary, on="Entrant", how="left")

    # Final column ordering.
    cols = ["Entrant", "Entries"]
    if "Avg. Sim ROI" in summary.columns:
        cols.append("Avg. Sim ROI")
    cols.extend(["Avg. Top 1%", "Avg. Top 5%", "Avg. Top 20%", "Avg Points"])
    summary = summary[cols]
    return summary


def _build_player_summary(
    simulation_df: pd.DataFrame,
    projected_ownership_by_player: Dict[str, Dict[str, float]],
    *,
    flex_role_label: str,
) -> pd.DataFrame:
    """
    Build player-level summary statistics from per-lineup simulation results.
    """
    num_lineups = len(simulation_df)
    if num_lineups == 0:
        flex_label_up = flex_role_label.upper()
        return pd.DataFrame(
            columns=[
                "Player",
                "CPT draft %",
                "CPT proj ownership",
                f"{flex_label_up} draft %",
                f"{flex_label_up} proj ownership",
                "CPT Sim ROI",
                f"{flex_label_up} Sim ROI",
                "CPT top 1% rate",
                f"{flex_label_up} top 1% rate",
                "CPT top 20% rate",
                f"{flex_label_up} top 20% rate",
            ]
        )

    # Long-form table for role/player combinations.
    records: List[Dict[str, object]] = []
    for _, row in simulation_df.iterrows():
        top1 = float(row["Top 1%"])
        top20 = float(row["Top 20%"])
        sim_roi = float(row["Sim ROI"]) if "Sim ROI" in simulation_df.columns else 0.0
        cpt = str(row["CPT"])
        records.append(
            {
                "Player": cpt,
                "role": "CPT",
                "Top 1%": top1,
                "Top 20%": top20,
                "Sim ROI": sim_roi,
            }
        )
        for col in ["Flex1", "Flex2", "Flex3", "Flex4", "Flex5"]:
            name = str(row[col])
            records.append(
                {
                    "Player": name,
                    "role": flex_role_label.upper(),
                    "Top 1%": top1,
                    "Top 20%": top20,
                    "Sim ROI": sim_roi,
                }
            )

    long_df = pd.DataFrame(records)

    # Draft percentages by role.
    flex_label_up = flex_role_label.upper()
    cpt_counts = long_df[long_df["role"] == "CPT"]["Player"].value_counts()
    flex_counts = long_df[long_df["role"] == flex_label_up]["Player"].value_counts()

    all_players = sorted(set(long_df["Player"].astype(str)))
    rows: List[Dict[str, object]] = []

    for player in all_players:
        # Draft percentages.
        cpt_draft = float(cpt_counts.get(player, 0)) / float(num_lineups)
        flex_draft = float(flex_counts.get(player, 0)) / float(num_lineups)

        # Role-specific performance metrics.
        mask_player = long_df["Player"] == player
        sub_cpt = long_df[mask_player & (long_df["role"] == "CPT")]
        sub_flex = long_df[mask_player & (long_df["role"] == flex_label_up)]

        def _safe_mean(series: pd.Series) -> float:
            return float(series.mean()) if not series.empty else 0.0

        proj = projected_ownership_by_player.get(
            player, {"cpt_proj_own": 0.0, "flex_proj_own": 0.0}
        )

        rows.append(
            {
                "Player": player,
                "CPT draft %": cpt_draft,
                "CPT proj ownership": float(proj.get("cpt_proj_own", 0.0)),
                f"{flex_label_up} draft %": flex_draft,
                f"{flex_label_up} proj ownership": float(
                    proj.get("flex_proj_own", 0.0)
                ),
                "CPT Sim ROI": _safe_mean(sub_cpt["Sim ROI"])
                if "Sim ROI" in sub_cpt.columns
                else 0.0,
                f"{flex_label_up} Sim ROI": _safe_mean(sub_flex["Sim ROI"])
                if "Sim ROI" in sub_flex.columns
                else 0.0,
                "CPT top 1% rate": _safe_mean(sub_cpt["Top 1%"]),
                f"{flex_label_up} top 1% rate": _safe_mean(sub_flex["Top 1%"]),
                "CPT top 20% rate": _safe_mean(sub_cpt["Top 20%"]),
                f"{flex_label_up} top 20% rate": _safe_mean(sub_flex["Top 20%"]),
            }
        )

    return pd.DataFrame(rows)


def run_flashback(
    *,
    contest_csv: str | None,
    sabersim_csv: str | None,
    num_sims: int,
    random_seed: int | None,
    payouts_csv: str | None,
    config_module: Any,
    load_sabersim_projections: Callable[[str | Path], pd.DataFrame],
    simulate_corr_matrix_from_projections: Callable[[pd.DataFrame], pd.DataFrame],
    name_col: str,
    team_col: str,
    salary_col: str,
    dk_proj_col: str,
    flex_role_label: str = "FLEX",
) -> Path:
    """
    Execute the flashback contest simulation pipeline for a given sport.

    Args:
        flex_role_label: Label used for non-CPT lineup slots when producing
            player-level summaries (e.g., \"FLEX\" for NFL, \"UTIL\" for NBA).
    """
    # Normalize flex role label for downstream use.
    flex_role_label_up = flex_role_label.upper()

    # Resolve input paths.
    contest_dir = config_module.DATA_DIR / "contests"
    sabersim_dir = config_module.SABERSIM_DIR

    contest_path = _resolve_latest_csv(contest_dir, contest_csv)
    sabersim_path = _resolve_latest_csv(sabersim_dir, sabersim_csv)

    # Extract contest_id from the contest standings filename.
    contest_filename = contest_path.name
    m = re.search(r"contest-standings-(\d+)\.csv$", contest_filename)
    if m:
        contest_id = m.group(1)
    else:
        # Fallback: use the stem if pattern does not match.
        contest_id = contest_path.stem

    print(f"Using contest CSV: {contest_path}")
    print(f"Using Sabersim CSV: {sabersim_path}")

    # Load contest standings and parse lineups.
    standings_df, parsed_lineups = _load_contest_lineups(contest_path)
    if not parsed_lineups:
        raise ValueError("No lineups parsed from contest CSV.")

    num_entries = len(standings_df)

    # Optional: load payout structure for ROI computation.
    payout_result = _load_payout_structure(
        payouts_csv=payouts_csv,
        contest_id=str(contest_id),
        num_entries=num_entries,
        data_dir=config_module.DATA_DIR,
    )
    if payout_result is not None:
        payouts_by_rank, entry_fee = payout_result
    else:
        payouts_by_rank = None
        entry_fee = None

    player_names_from_lineups, player_index_from_lineups, W, sim_lineups_df = (
        _build_player_universe_and_weights(parsed_lineups)
    )

    # Load Sabersim projections and build correlation matrix via simulation.
    sabersim_raw_df = pd.read_csv(sabersim_path)
    sabersim_df = load_sabersim_projections(sabersim_path)
    print("Building player correlation matrix via simulation...")
    corr_df = simulate_corr_matrix_from_projections(sabersim_df)

    # Build unified player universe and correlation matrix.
    universe_names, mu, sigma = _build_player_universe_from_sabersim_and_lineups(
        sabersim_df=sabersim_df,
        corr_df=corr_df,
        lineup_player_names=player_names_from_lineups,
        name_col=name_col,
        dk_proj_col=dk_proj_col,
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
        random_seed if random_seed is not None else config_module.SIM_RANDOM_SEED
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

    # Optional: simulated ROI per lineup given payout structure.
    sim_roi: np.ndarray | None = None
    if payouts_by_rank is not None and entry_fee is not None and entry_fee > 0:
        sim_roi = _compute_sim_roi(
            scores=scores,
            payouts_by_rank=payouts_by_rank,
            entry_fee=float(entry_fee),
        )

    # Build Simulation sheet.
    simulation_df = sim_lineups_df.copy()
    if sim_roi is not None:
        simulation_df["Sim ROI"] = sim_roi
    simulation_df["Top 1%"] = top1
    simulation_df["Top 5%"] = top5
    simulation_df["Top 20%"] = top20
    simulation_df["Avg Points"] = avg_points

    # Actual Points: pull from the original contest standings CSV.
    if "EntryName" in simulation_df.columns and "Points" in standings_df.columns:
        points_df = (
            standings_df[["EntryName", "Points"]]
            .drop_duplicates(subset=["EntryName"])
            .copy()
        )
        points_df["Points"] = pd.to_numeric(points_df["Points"], errors="coerce")
        simulation_df = simulation_df.merge(points_df, on="EntryName", how="left")
        simulation_df.rename(columns={"Points": "Actual Points"}, inplace=True)

        # Ensure Actual Points appears immediately after 'Avg Points'.
        cols = list(simulation_df.columns)
        if "Avg Points" in cols and "Actual Points" in cols:
            cols.remove("Actual Points")
            avg_idx = cols.index("Avg Points")
            cols.insert(avg_idx + 1, "Actual Points")
            simulation_df = simulation_df[cols]

        # Optional: Actual ROI based on realized rank and payout structure.
        if payouts_by_rank is not None and entry_fee is not None and entry_fee > 0:
            if "Rank" in standings_df.columns:
                rank_df = standings_df[["EntryName", "Rank"]].copy()
                rank_df["Rank"] = pd.to_numeric(
                    rank_df["Rank"], errors="coerce"
                )

                def _rank_to_payout(rank_val: float) -> float:
                    if pd.isna(rank_val):
                        return 0.0
                    try:
                        r_int = int(rank_val)
                    except (TypeError, ValueError):
                        return 0.0
                    if r_int < 1 or r_int > len(payouts_by_rank):
                        return 0.0
                    return float(payouts_by_rank[r_int - 1])

                rank_df["Actual ROI"] = rank_df["Rank"].map(_rank_to_payout)
                rank_df["Actual ROI"] = (
                    rank_df["Actual ROI"] - float(entry_fee)
                ) / float(entry_fee)
                actual_roi_df = (
                    rank_df[["EntryName", "Actual ROI"]]
                    .drop_duplicates(subset=["EntryName"])
                    .copy()
                )
                simulation_df = simulation_df.merge(
                    actual_roi_df, on="EntryName", how="left"
                )

                # Place Actual ROI immediately after Actual Points when present.
                cols = list(simulation_df.columns)
                if "Actual ROI" in cols and "Actual Points" in cols:
                    cols.remove("Actual ROI")
                    anchor_idx = cols.index("Actual Points")
                    cols.insert(anchor_idx + 1, "Actual ROI")
                    simulation_df = simulation_df[cols]

    # Duplicates: how many times this exact lineup (CPT + 5 FLEX) was entered.
    lineup_cols = ["CPT", "Flex1", "Flex2", "Flex3", "Flex4", "Flex5"]
    if all(col in simulation_df.columns for col in lineup_cols):
        dup_counts = simulation_df.groupby(lineup_cols)["Entrant"].transform("size")
        # Ensure Duplicates appears immediately after 'Actual Points' (or
        # 'Avg Points' if Actual Points is unavailable).
        simulation_df["Duplicates"] = dup_counts
        cols = list(simulation_df.columns)
        anchor_col = "Actual Points" if "Actual Points" in cols else "Avg Points"
        if anchor_col in cols:
            anchor_idx = cols.index(anchor_col)
            # Move Duplicates to right after the anchor column.
            cols.remove("Duplicates")
            cols.insert(anchor_idx + 1, "Duplicates")
            simulation_df = simulation_df[cols]

    # Ensure Sim ROI, when present, appears immediately before 'Top 1%'.
    cols = list(simulation_df.columns)
    if "Sim ROI" in cols and "Top 1%" in cols:
        cols.remove("Sim ROI")
        top1_idx = cols.index("Top 1%")
        cols.insert(top1_idx, "Sim ROI")
        simulation_df = simulation_df[cols]

    # Compute stack label per lineup using Sabersim team info.
    required_cols = {name_col, team_col}
    missing = required_cols.difference(sabersim_df.columns)
    if missing:
        raise KeyError(
            "Sabersim projections missing required columns for stack computation "
            f"{sorted(missing)}."
        )

    sabersim_names = sabersim_df[name_col].astype(str)
    sabersim_teams = sabersim_df[team_col].astype(str)
    name_to_team: Dict[str, str] = (
        pd.DataFrame({"Name": sabersim_names, "Team": sabersim_teams})
        .drop_duplicates(subset=["Name"])
        .set_index("Name")["Team"]
        .to_dict()
    )

    stack_labels: List[str] = []
    from collections import Counter

    for _, row in simulation_df.iterrows():
        player_names = [
            str(row["CPT"]),
            str(row["Flex1"]),
            str(row["Flex2"]),
            str(row["Flex3"]),
            str(row["Flex4"]),
            str(row["Flex5"]),
        ]
        teams: List[str] = []
        for name in player_names:
            team = name_to_team.get(name)
            if team:
                teams.append(team)

        if not teams:
            stack_labels.append("")
            continue

        team_counts = Counter(teams)
        counts_sorted = sorted(team_counts.values(), reverse=True)
        pattern = "|".join(str(c) for c in counts_sorted)

        # Determine heavy team if there is a unique max count.
        top_team, top_count = max(team_counts.items(), key=lambda kv: kv[1])
        if list(team_counts.values()).count(top_count) == 1:
            label = f"{pattern} {top_team}-heavy"
        else:
            label = pattern
        stack_labels.append(label)

    # Insert 'Stack' as the 3rd column (between EntryName and CPT).
    cols = list(simulation_df.columns)
    if "EntryName" in cols:
        insert_at = cols.index("EntryName") + 1
        simulation_df.insert(insert_at, "Stack", stack_labels)
    else:
        # Fallback: prepend Stack if EntryName is missing for some reason.
        simulation_df.insert(0, "Stack", stack_labels)

    # Ownership-driven lineup metrics and summaries.
    ownership_salary_by_name_team = _build_sabersim_ownership_and_salary(
        sabersim_raw_df,
        name_col=name_col,
        team_col=team_col,
        salary_col=salary_col,
        proj_col=dk_proj_col,
    )
    projected_ownership_by_player = _build_projected_ownership_by_player(
        ownership_salary_by_name_team
    )
    simulation_df = _add_ownership_columns_to_simulation(
        simulation_df=simulation_df,
        ownership_salary_by_name_team=ownership_salary_by_name_team,
        name_to_team=name_to_team,
    )

    # Entrant and player summaries.
    entrant_summary_df = _build_entrant_summary(simulation_df)
    player_summary_df = _build_player_summary(
        simulation_df,
        projected_ownership_by_player,
        flex_role_label=flex_role_label_up,
    )

    # Output path.
    flashback_dir = config_module.OUTPUTS_DIR / FLASHBACK_SUBDIR
    flashback_dir.mkdir(parents=True, exist_ok=True)
    output_path = flashback_dir / f"flashback_{contest_id}.xlsx"

    print(f"Writing flashback results to {output_path}...")
    with pd.ExcelWriter(output_path, engine="xlsxwriter") as writer:
        sabersim_raw_df.to_excel(writer, sheet_name="Projections", index=False)
        standings_df.to_excel(writer, sheet_name="Standings", index=False)
        simulation_df.to_excel(writer, sheet_name="Simulation", index=False)
        entrant_summary_df.to_excel(writer, sheet_name="Entrant summary", index=False)
        player_summary_df.to_excel(writer, sheet_name="Player summary", index=False)

    print("Done.")
    return output_path


__all__ = [
    "ParsedLineup",
    "run_flashback",
]


