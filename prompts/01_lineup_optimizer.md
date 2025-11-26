File: 01_lineup_optimizer.md

NFL Showdown Lineup Optimizer (Captain Mode)
===========================================

Goal
----
Implement a DraftKings NFL Showdown (Captain Mode) lineup optimizer that:

- Reads SaberSim projections from data/sabersim/NFL_*.csv
- Builds valid lineups with 1 CPT and 5 FLEX
- Maximizes mean projected lineup points
- Supports constraint rules such as:
  - Minimum X players from a given team
  - Prohibit certain player combinations (negative correlation rules)
  - Conditional rules like "if QB from Team A is CPT, prohibit DST from Team B"
- Returns the top N lineups under these constraints


1. Data inputs and assumptions
------------------------------

1. SaberSim projections CSV:

- Path pattern: data/sabersim/NFL_*.csv (always a single game showdown file)
- Columns (assumed; adapt your loader if names differ):
  - player_id: unique id (string or int)
  - player_name
  - team: team abbreviation, for example KC, SF
  - position: QB, RB, WR, TE, DST, K, etc.
  - dk_salary: integer DraftKings salary
  - dk_proj: mean projected DK fantasy points
  - optionally dk_std: standard deviation of projected points (used elsewhere, not required here)
  - optionally is_cpt_eligible: boolean, default True if missing
  - optionally is_flex_eligible: boolean, default True if missing

2. Roster rules (DraftKings Showdown):

- 6 total roster spots
- 1 CPT slot
  - Scores 1.5 times fantasy points
  - Costs 1.5 times salary (round or cast to int after multiplication)
- 5 FLEX slots
  - Score 1.0 times fantasy points
  - Cost normal salary
- 6 distinct players per lineup
- Salary cap, typically 50000 (make configurable)


2. Core data structures
-----------------------

You only need simple data classes.

Player:

- player_id: string
- name: string
- team: string
- position: string
- dk_salary: int
- dk_proj: float
- dk_std: float or None
- is_cpt_eligible: bool
- is_flex_eligible: bool

PlayerPool:

- players: list of Player
- helper methods:
  - by_team(team) -> list of Player
  - by_position(position) -> list of Player
  - get(player_id) -> Player

Lineup:

- cpt: Player
- flex: list of exactly 5 Player
- methods:
  - salary() -> int
    - 1.5 times CPT salary (round to int) plus sum of FLEX salaries
  - projection() -> float
    - 1.5 times CPT projection plus sum of FLEX projections
  - as_tuple_ids() -> tuple of player ids, CPT first then FLEX in a stable order


3. Optimization model (MILP)
----------------------------

Use a mixed-integer linear program (MILP) solver such as pulp or OR-Tools. Below assumes pulp.

Decision variables:

- For each player i and each lineup slot s in {CPT, F1, F2, F3, F4, F5}:
  - x[i, s] is a binary variable:
    - 1 if player i is placed in slot s
    - 0 otherwise

Slots:

- "CPT" is the captain slot
- "F1" through "F5" are five FLEX slots


3.1 Objective: maximize mean projection
---------------------------------------

For each player i:

- CPT contribution: 1.5 * dk_proj[i] * x[i, CPT]
- FLEX contribution: dk_proj[i] * sum over flex slots x[i, Fj]

Total objective:

- Maximize sum over players of
  1.5 * dk_proj[i] * x[i, CPT] + dk_proj[i] * sum over FLEX slots x[i, Fj]


3.2 Base constraints
--------------------

1) Exactly one CPT:

- Sum over players of x[i, CPT] = 1

2) Exactly 5 FLEX players:

- Sum over players and FLEX slots of x[i, Fj] = 5

3) Each player appears in at most one slot:

For every player i:

- x[i, CPT] + x[i, F1] + x[i, F2] + x[i, F3] + x[i, F4] + x[i, F5] <= 1

4) Salary cap:

For each player i:

- Effective CPT salary contribution: 1.5 * dk_salary[i] * x[i, CPT]
- FLEX salary contribution: dk_salary[i] * sum over FLEX slots x[i, Fj]

Constraint:

- Sum over players of (1.5 * salary_i * x[i, CPT] + salary_i * sum FLEX x[i, Fj]) <= salary_cap

5) Eligibility:

- If player is not CPT eligible:
  - impose x[i, CPT] = 0
- If player is not FLEX eligible:
  - impose x[i, Fj] = 0 for all FLEX slots Fj


4. Custom DFS constraints
-------------------------

You will implement small helper functions that add constraints to the same MILP model. The general pattern:

- Each helper function has signature:
  - build_constraint(prob, x, player_pool)

where:

- prob is the pulp LpProblem
- x is the dictionary mapping (player_id, slot) to the pulp variable
- player_pool gives access to players and their attributes

You then pass a list of these constraint builder functions into the top level optimize function.

4.1 Minimum players from a team
-------------------------------

To enforce "at least k players from team T":

- Define an expression team_count_T:

  team_count_T = sum over players i on team T, and slots s in all 6 slots of x[i, s]

- Add constraint:

  team_count_T >= k

Similarly, you can add a maximum players constraint by using <= instead of >=.

4.2 Mutually exclusive groups
-----------------------------

Example: "prohibit playing RB from Team A with DST from Team B".

Let:

- Group G1 be all players with team = team_A and position = RB
- Group G2 be all players with team = team_B and position = DST

Compute:

- g1 = sum over player i in G1 and all slots s of x[i, s]
- g2 = sum over player j in G2 and all slots s of x[j, s]

To prevent any lineup that contains at least one from G1 and at least one from G2, use:

- g1 + g2 <= 1

This forces at most one of these groups to be present in any lineup.

4.3 Conditional rule: if QB from team A is CPT, prohibit DST from team B
-------------------------------------------------------------------------

Define:

- qb_cpt = sum over players i where team = team_A and position = QB of x[i, CPT]
- dst_any = sum over players j where team = team_B and position = DST and all slots s of x[j, s]

Constraint:

- qb_cpt + dst_any <= 1

Interpretation:

- If qb_cpt is 1 (team A QB is CPT), then dst_any must be 0 (no DST from team B anywhere).
- If dst_any is 1 (at least one DST from team B), then qb_cpt must be 0 (no team A QB in CPT).
- It also allows lineups with neither, which is fine.


5. Solving and extracting lineups
---------------------------------

Steps for a single optimal lineup:

1) Build PlayerPool from the SaberSim CSV file.

2) Create MILP model:

- prob, x = build_showdown_model(player_pool, salary_cap)

3) Add base constraints:

- add_single_cpt_constraint
- add_flex_count_constraint
- add_unique_player_constraint
- add_salary_cap_constraint
- add_eligibility_constraints

4) Add custom constraints:

- For each constraint builder in a list, call builder(prob, x, player_pool)

5) Add objective (maximize mean projection).

6) Solve with pulp:

- prob.solve(pulp.PULP_CBC_CMD(msg=False))

7) Extract lineup:

- For any variable x[i, s] with value >= 0.5:
  - If s is CPT, that player is the captain
  - If s is a FLEX slot, add that player to the FLEX list

8) Build Lineup object and return it.


5.1 Generating the top N lineups
--------------------------------

To get multiple distinct optimal lineups, repeat solve and add a "no duplicate lineup" constraint after each solution.

Let lineup L have:

- Players used: the set of player ids that appear in that lineup (CPT plus FLEX)

Add a constraint that prevents all of those same players from being used together again.

One simple method:

- Define used_players to be the set of player ids in L
- For all slots s in {CPT, F1, F2, F3, F4, F5}, sum the variables x[pid, s] over pid in used_players
- Constraint:

  sum over pid in used_players and all slots s of x[pid, s] <= 5

Because a valid lineup must contain exactly 6 distinct players, this constraint ensures that at least one of those six players is different the next time the solver finds a solution.

Algorithm to get N lineups:

- Initialize lineups = empty list
- For iteration from 1 to N:
  - Solve the current model
  - If the solver status is not optimal, break the loop
  - Extract lineup L
  - Append L to lineups
  - Add a no-duplicate constraint based on L so that the next solution differs by at least one player


6. High level function shape
----------------------------

You can expose a function like:

- optimize_showdown_lineups(
    projections_path_pattern: str,
    num_lineups: int,
    salary_cap: int = 50000,
    constraint_builders: list of callables or None = None
  ) -> list of Lineup

The function should:

1) Load SaberSim projections from paths matching projections_path_pattern.
2) Build Player objects and a PlayerPool.
3) Build MILP model and variables.
4) Add base constraints and custom constraints.
5) Iteratively solve and extract the top num_lineups by mean projection, adding a no-duplicate constraint after each extracted lineup.
6) Return the list of Lineup objects.

This optimizer will be the front end to your top 1 percent finish rate simulation code.