#!/usr/bin/env bash

# Run the full NFL Showdown pipeline for a single slate:
#   1) Build correlation matrix from a SaberSim projections CSV
#   2) Generate optimized lineups from that same CSV
#   3) Estimate top 1% finish probabilities for those lineups
#
# Usage:
#   ./run_full.sh PATH_TO_SABERSIM_CSV [FIELD_SIZE] [NUM_LINEUPS] [SALARY_CAP]
#
# Examples:
#   ./run_full.sh data/sabersim/NFL_2025-11-27-820pm_DK_SHOWDOWN_TB-@-DAL.csv
#   ./run_full.sh data/sabersim/NFL_2025-11-27-820pm_DK_SHOWDOWN_TB-@-DAL.csv 23529 150 50000
#
# Notes:
#   - FIELD_SIZE defaults to 23529 if not provided.
#   - NUM_LINEUPS defaults to 150.
#   - SALARY_CAP defaults to 50000.
#   - The script assumes it is run from the project root, or that Python
#     can find the `src` package from the current directory.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

if [[ "${#}" -lt 1 ]]; then
  echo "Usage: $0 PATH_TO_SABERSIM_CSV [FIELD_SIZE] [NUM_LINEUPS] [SALARY_CAP]" >&2
  exit 1
fi

SABERSIM_CSV="$1"
FIELD_SIZE="${2:-10000}"
NUM_LINEUPS="${3:-1000}"
SALARY_CAP="${4:-50000}"

if [[ ! -f "${SABERSIM_CSV}" ]]; then
  echo "Error: Sabersim CSV not found at '${SABERSIM_CSV}'" >&2
  exit 1
fi

timestamp="$(date +%Y%m%d_%H%M%S)"
CORR_EXCEL="outputs/correlations/showdown_corr_matrix_${timestamp}.xlsx"

echo "================================================================"
echo "Step 1/3: Building correlation matrix from '${SABERSIM_CSV}'"
echo "         -> Output: ${CORR_EXCEL}"
echo "================================================================"

python -m src.main \
  --sabersim-csv "${SABERSIM_CSV}" \
  --output-excel "${CORR_EXCEL}"

echo
echo "================================================================"
echo "Step 2/3: Generating ${NUM_LINEUPS} lineups (salary cap ${SALARY_CAP})"
echo "         from '${SABERSIM_CSV}'"
echo "================================================================"

python -m src.showdown_optimizer_main \
  --sabersim-glob "${SABERSIM_CSV}" \
  --num-lineups "${NUM_LINEUPS}" \
  --salary-cap "${SALARY_CAP}"

echo
echo "================================================================"
echo "Step 3/3: Estimating top 1% finish probabilities"
echo "         Field size: ${FIELD_SIZE}"
echo "         Using latest lineups & correlations workbooks."
echo "================================================================"

python -m src.top1pct_finish_rate \
  --field-size "${FIELD_SIZE}"

echo
echo "All steps completed."


