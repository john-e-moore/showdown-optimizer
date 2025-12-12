import unittest

from src.nba.showdown_constraints import (
    cpt_min_minutes,
    max_low_proj_utils,
    min_lineup_salary,
)
from src.shared.optimizer_core import Player, PlayerPool, optimize_showdown_lineups


class TestNBAHardConstraints(unittest.TestCase):
    def test_cpt_projected_minutes_floor(self) -> None:
        # Player with <24 projected minutes has the best projection and would be CPT
        # without the constraint; the constraint should force a different CPT.
        players = [
            Player("A|TA", "A", "TA", "SG", 9000, 60.0, proj_min=23.0),
            Player("B|TA", "B", "TA", "PG", 8800, 55.0, proj_min=35.0),
            Player("C|TA", "C", "TA", "SF", 7000, 40.0, proj_min=30.0),
            Player("D|TB", "D", "TB", "PF", 8200, 50.0, proj_min=34.0),
            Player("E|TB", "E", "TB", "C", 6500, 38.0, proj_min=28.0),
            Player("F|TB", "F", "TB", "SG", 6000, 35.0, proj_min=26.0),
            Player("G|TB", "G", "TB", "PG", 5500, 30.0, proj_min=25.0),
        ]
        pool = PlayerPool(players)

        lineups = optimize_showdown_lineups(
            player_pool=pool,
            num_lineups=1,
            salary_cap=50_000,
            constraint_builders=[cpt_min_minutes(24.0)],
            chunk_size=0,
        )
        self.assertEqual(len(lineups), 1)
        self.assertNotEqual(lineups[0].cpt.player_id, "A|TA")
        self.assertGreaterEqual(lineups[0].cpt.proj_min or 0.0, 24.0)

    def test_max_one_low_proj_util(self) -> None:
        # Force needing two cheap UTILs; only one of them is allowed to be <=8 proj.
        players = [
            # Expensive core
            Player("A|TA", "A", "TA", "SG", 10000, 45.0, proj_min=30.0),
            Player("B|TA", "B", "TA", "PG", 9800, 44.0, proj_min=30.0),
            Player("C|TB", "C", "TB", "SF", 9600, 43.0, proj_min=30.0),
            Player("D|TB", "D", "TB", "PF", 9400, 42.0, proj_min=30.0),
            # Cheap fillers
            Player("L1|TA", "L1", "TA", "G", 1000, 7.0, proj_min=30.0),   # low proj
            Player("L2|TB", "L2", "TB", "F", 1000, 7.5, proj_min=30.0),   # low proj
            Player("OK|TB", "OK", "TB", "C", 1200, 12.0, proj_min=30.0),  # acceptable
        ]
        pool = PlayerPool(players)

        lineups = optimize_showdown_lineups(
            player_pool=pool,
            num_lineups=1,
            salary_cap=50_000,
            constraint_builders=[max_low_proj_utils(max_count=1, threshold=8.0)],
            chunk_size=0,
        )
        self.assertEqual(len(lineups), 1)
        lu = lineups[0]
        low_utils = [p for p in lu.flex if p.dk_proj <= 8.0]
        self.assertLessEqual(len(low_utils), 1)

    def test_min_lineup_salary_enforced_and_infeasible(self) -> None:
        # Feasible: lineup can hit >= 48,500 under cap.
        feasible_players = [
            # Constructed so the optimal lineup can land in [48,500, 50,000].
            # Example feasible composition:
            #   CPT(A=9000 -> 13500) + FLEX(B=8000,C=7500,D=7000,E=7000,F=7000) = 50,000
            Player("A|TA", "A", "TA", "SG", 9000, 50.0, proj_min=30.0),
            Player("B|TA", "B", "TA", "PG", 8000, 48.0, proj_min=30.0),
            Player("C|TA", "C", "TA", "SF", 7500, 46.0, proj_min=30.0),
            Player("D|TB", "D", "TB", "PF", 7000, 45.0, proj_min=30.0),
            Player("E|TB", "E", "TB", "C", 7000, 44.0, proj_min=30.0),
            Player("F|TB", "F", "TB", "G", 7000, 43.0, proj_min=30.0),
            Player("G|TB", "G", "TB", "F", 6500, 42.0, proj_min=30.0),
        ]
        pool = PlayerPool(feasible_players)
        lineups = optimize_showdown_lineups(
            player_pool=pool,
            num_lineups=1,
            salary_cap=50_000,
            constraint_builders=[min_lineup_salary(48_500)],
            chunk_size=0,
        )
        self.assertEqual(len(lineups), 1)
        self.assertGreaterEqual(lineups[0].salary(), 48_500)

        # Infeasible: even max salary lineup can't reach 48,500.
        infeasible_players = [
            Player("A|TA", "A", "TA", "SG", 5000, 30.0, proj_min=30.0),
            Player("B|TA", "B", "TA", "PG", 5000, 29.0, proj_min=30.0),
            Player("C|TA", "C", "TA", "SF", 5000, 28.0, proj_min=30.0),
            Player("D|TB", "D", "TB", "PF", 5000, 27.0, proj_min=30.0),
            Player("E|TB", "E", "TB", "C", 5000, 26.0, proj_min=30.0),
            Player("F|TB", "F", "TB", "G", 5000, 25.0, proj_min=30.0),
            Player("G|TB", "G", "TB", "F", 5000, 24.0, proj_min=30.0),
        ]
        pool2 = PlayerPool(infeasible_players)
        lineups2 = optimize_showdown_lineups(
            player_pool=pool2,
            num_lineups=1,
            salary_cap=50_000,
            constraint_builders=[min_lineup_salary(48_500)],
            chunk_size=0,
        )
        self.assertEqual(len(lineups2), 0)


if __name__ == "__main__":
    unittest.main()


