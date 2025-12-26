import unittest

import pandas as pd

from src.shared import flashback_core


class TestFlashbackDuplicates(unittest.TestCase):
    def test_duplicates_excluding_self_and_entrant_avg_duplicates(self) -> None:
        # Two identical lineups for EntrantA, one unique lineup for EntrantB.
        base = {
            "Top 1%": 0.1,
            "Top 5%": 0.2,
            "Top 20%": 0.3,
            "Avg Points": 100.0,
        }
        lineup1 = {"CPT": "A", "Flex1": "B", "Flex2": "C", "Flex3": "D", "Flex4": "E", "Flex5": "F"}
        lineup2 = {"CPT": "A", "Flex1": "B", "Flex2": "C", "Flex3": "D", "Flex4": "E", "Flex5": "G"}

        df = pd.DataFrame(
            [
                {"Entrant": "EntrantA", **lineup1, **base},
                {"Entrant": "EntrantA", **lineup1, **base},
                {"Entrant": "EntrantB", **lineup2, **base},
            ]
        )

        lineup_cols = ["CPT", "Flex1", "Flex2", "Flex3", "Flex4", "Flex5"]
        df["Duplicates"] = flashback_core._compute_lineup_duplicates_excluding_self(
            df, lineup_cols=lineup_cols
        )

        self.assertEqual(df["Duplicates"].tolist(), [1, 1, 0])

        entrant_summary = flashback_core._build_entrant_summary(df, flex_role_label="FLEX")

        # Column placement: Avg. Duplicates should appear right after Entries.
        cols = list(entrant_summary.columns)
        self.assertIn("Entries", cols)
        self.assertIn("Avg. Duplicates", cols)
        self.assertEqual(cols.index("Avg. Duplicates"), cols.index("Entries") + 1)

        # Entrant-level values.
        by_entrant = entrant_summary.set_index("Entrant")
        self.assertEqual(int(by_entrant.loc["EntrantA", "Entries"]), 2)
        self.assertAlmostEqual(float(by_entrant.loc["EntrantA", "Avg. Duplicates"]), 1.0)
        self.assertEqual(int(by_entrant.loc["EntrantB", "Entries"]), 1)
        self.assertAlmostEqual(float(by_entrant.loc["EntrantB", "Avg. Duplicates"]), 0.0)


if __name__ == "__main__":
    unittest.main()


