from __future__ import annotations

"""
NFL-specific namespace.

All NFL-facing code for the showdown optimizer lives under this package:

- `src.nfl.config` – NFL configuration and filesystem paths.
- `src.nfl.main` – NFL correlation pipeline (builds correlations from Sabersim).
- `src.nfl.showdown_optimizer_main` – NFL Showdown lineup optimizer CLI.
- `src.nfl.top1pct_finish_rate` – NFL top 1% finish-rate estimator.
- `src.nfl.diversify_lineups` – Diversify NFL Showdown lineups.
- `src.nfl.flashback_sim` – Flashback contest simulator for completed NFL slates.
- `src.nfl.fill_dkentries` – Fill DraftKings DKEntries CSVs for NFL Showdown.
- `src.nfl.download_nfl_data` – Download and normalize historical NFL data.

Shared, sport-agnostic cores live under `src.shared`, while NBA-specific
code lives under `src.nba`.
"""


