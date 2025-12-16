import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.shared import flashback_core, top1pct_core


class TestFlashbackCorrHandling(unittest.TestCase):
    def test_resolve_corr_excel_relative_to_corr_dir(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            corr_dir = Path(td) / "corr"
            corr_dir.mkdir(parents=True, exist_ok=True)
            target = corr_dir / "my_corr.xlsx"
            target.write_bytes(b"")  # existence only; not a valid workbook

            resolved = flashback_core._resolve_corr_excel_path(corr_dir, "my_corr.xlsx")
            self.assertEqual(resolved, target)

    def test_resolve_corr_excel_missing_shows_both_candidates(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            corr_dir = Path(td) / "corr"
            corr_dir.mkdir(parents=True, exist_ok=True)

            with self.assertRaises(FileNotFoundError) as ctx:
                flashback_core._resolve_corr_excel_path(corr_dir, "missing.xlsx")

            msg = str(ctx.exception)
            self.assertIn("missing.xlsx", msg)
            self.assertIn(str(corr_dir / "missing.xlsx"), msg)

    def test_nfl_flashback_autocompute_writes_workbook(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            data_dir = root / "data" / "nfl"
            sabersim_dir = data_dir / "sabersim"
            contests_dir = data_dir / "contests"
            payouts_dir = data_dir / "payouts"
            outputs_dir = root / "outputs" / "nfl"
            corr_dir = outputs_dir / "correlations"

            sabersim_dir.mkdir(parents=True, exist_ok=True)
            contests_dir.mkdir(parents=True, exist_ok=True)
            payouts_dir.mkdir(parents=True, exist_ok=True)
            corr_dir.mkdir(parents=True, exist_ok=True)

            # Minimal contest standings.
            contest_path = contests_dir / "contest-standings-123.csv"
            contest_df = pd.DataFrame(
                [
                    {
                        "EntryName": "User1",
                        "Lineup": "CPT A FLEX B FLEX C FLEX D FLEX E FLEX F",
                        "Points": 0.0,
                    }
                ]
            )
            contest_df.to_csv(contest_path, index=False)

            # Provide a minimal payouts JSON so tests don't hit the network.
            (payouts_dir / "payouts-123.json").write_text(
                '{"contestDetail": {"entryFee": 0, "payoutSummary": []}}',
                encoding="utf-8",
            )

            # Minimal Sabersim CSV (one row per player is enough; flex ownership will be 0.0).
            sabersim_path = sabersim_dir / "sabersim.csv"
            sabersim_raw = pd.DataFrame(
                [
                    {"Name": "A", "Team": "TA", "Salary": 10000, "My Proj": 10.0, "My Own": 5.0},
                    {"Name": "B", "Team": "TA", "Salary": 9000, "My Proj": 9.0, "My Own": 5.0},
                    {"Name": "C", "Team": "TA", "Salary": 8000, "My Proj": 8.0, "My Own": 5.0},
                    {"Name": "D", "Team": "TB", "Salary": 7000, "My Proj": 7.0, "My Own": 5.0},
                    {"Name": "E", "Team": "TB", "Salary": 6000, "My Proj": 6.0, "My Own": 5.0},
                    {"Name": "F", "Team": "TB", "Salary": 5000, "My Proj": 5.0, "My Own": 5.0},
                ]
            )
            sabersim_raw.to_csv(sabersim_path, index=False)

            class DummyNFLConfig:
                DATA_DIR = data_dir
                SABERSIM_DIR = sabersim_dir
                OUTPUTS_DIR = outputs_dir
                CORRELATIONS_DIR = corr_dir
                SIM_RANDOM_SEED = 0
                FLASHBACK_COMPUTE_CORR_WHEN_MISSING = True

            def load_sabersim_projections(path: str | Path) -> pd.DataFrame:
                return pd.read_csv(path)

            def simulate_corr_matrix_from_projections(sabersim_df: pd.DataFrame) -> pd.DataFrame:
                names = sabersim_df["Name"].astype(str).tolist()
                n = len(names)
                return pd.DataFrame(np.eye(n), index=names, columns=names, dtype=float)

            before = set(corr_dir.glob("flashback_corr_123_*.xlsx"))

            out_path = flashback_core.run_flashback(
                contest_csv=str(contest_path),
                sabersim_csv=str(sabersim_path),
                corr_excel=None,
                num_sims=5,
                random_seed=0,
                payouts_csv=None,
                config_module=DummyNFLConfig,
                load_sabersim_projections=load_sabersim_projections,
                simulate_corr_matrix_from_projections=simulate_corr_matrix_from_projections,
                name_col="Name",
                team_col="Team",
                salary_col="Salary",
                dk_proj_col="My Proj",
                flex_role_label="FLEX",
            )

            self.assertTrue(out_path.is_file())

            after = set(corr_dir.glob("flashback_corr_123_*.xlsx"))
            new_files = sorted(after - before)
            self.assertGreaterEqual(len(new_files), 1)

            # Workbook should be readable in the expected format.
            sabersim_proj_df, corr_df = top1pct_core._load_corr_workbook(new_files[-1])
            self.assertIn("Name", sabersim_proj_df.columns)
            self.assertEqual(corr_df.shape[0], corr_df.shape[1])


if __name__ == "__main__":
    unittest.main()


