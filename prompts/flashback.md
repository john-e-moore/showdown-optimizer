**Objective**
We currently run game simulations and then do some statistics with ownership percentages to assign a top 1% finish probability to each lineup. 

After the contest is complete, we have the actual lineups from the contest. I want to simulate outcomes for these actual lineups - frequency finishing top 1%, top 5%, top 20%. It will also output various summary statistics for the contest and simulation. Create a python script that does this.

**Description**
You will need to parse contest data from the 'data/contests/' directory. The spreadsheet contains headers. Lineups are in the 'Lineup' column in the following format (example): "CPT A.J. Brown FLEX Jalen Hurts FLEX DeVonta Smith FLEX D'Andre Swift FLEX Jake Elliott FLEX Luther Burden III".

Allow flags for number of simulations (default 20,000), contest standings input filepath (default to most recent .csv in 'data/contests/'), sabersim projections filepath (default to most recent .csv in 'data/sabersim/').

Generate the correlation matrix in the same way we already do, then use that to simulate player fantasy point outcomes, then map those onto the actual lineups from the contest.


**Output**
An excel spreadsheet with the following tabs:
- 'Standings': A copy of the contest standings from the input spreadsheet
- 'Simulation': Columns: Entrant, CPT, Flex1, Flex2, Flex3, Flex4, Flex5, Top 1%, Top 5%, Top 20%, Avg Points
 - 'Entrant' corresponds to the 'EntryName' column in the input spreadsheet. Remove the parentheses and the numbers inside. So "Fantassin (4/5)" would become "Fantassin".
- 'Entrant summary': 'Entrant', 'Avg. Top 1%', 'Avg. Top 5%', 'Avg. Top 20%', 'Avg Points'. An entrant may have multiple entries in the contest; average the statistics of all their lineups.
- 'Player summary': 'Player' (e.g. A.J. Brown, Jake Elliott), 'CPT draft %', 'FLEX draft %', 'CPT top 1% rate', 'FLEX top 1% rate', 'CPT top 20% rate', 'FLEX top 20% rate'

Note these summaries are all for our simulation results, not just the contest results.