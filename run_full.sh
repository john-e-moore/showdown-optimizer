#!/usr/bin/env bash

# Run the full NFL Showdown pipeline for a single slate:
#   1) Build correlation matrix from a SaberSim projections CSV
#   2) Generate optimized lineups from that same CSV
#   3) Estimate top 1% finish probabilities for those lineups
#   4) Select a diversified subset of lineups based on top 1% finish rate
#
# Usage:
#   ./run_full.sh PATH_TO_SABERSIM_CSV [FIELD_SIZE] [NUM_LINEUPS] [SALARY_CAP] [STACK_MODE] [STACK_WEIGHTS] [DIVERSIFIED_NUM]
#
# Examples:
#   ./run_full.sh data/sabersim/NFL_2025-11-27-820pm_DK_SHOWDOWN_TB-@-DAL.csv
#   ./run_full.sh data/sabersim/NFL_2025-11-27-820pm_DK_SHOWDOWN_TB-@-DAL.csv 23529 150 50000
#   ./run_full.sh data/sabersim/NFL_2025-11-27-820pm_DK_SHOWDOWN_TB-@-DAL.csv 23529 1000 50000 multi
#   ./run_full.sh data/sabersim/NFL_2025-11-27-820pm_DK_SHOWDOWN_TB-@-DAL.csv 23529 1000 50000 multi "5|1=0.4,4|2=0.3,3|3=0.2,2|4=0.1,1|5=0.0"
#
# Notes:
#   - FIELD_SIZE defaults to 23529 if not provided.
#   - NUM_LINEUPS defaults to 150.
#   - SALARY_CAP defaults to 50000.
#   - STACK_MODE defaults to 'none' (single run); set to 'multi' to enable multi-stack mode.
#   - STACK_WEIGHTS is optional; when provided in multi-stack mode it is passed
#     through to --stack-weights (e.g. '5|1=0.4,4|2=0.3,3|3=0.2,2|4=0.1,1|5=0.0').
#     If STACK_MODE='multi' and STACK_WEIGHTS is omitted, the script defaults to
#     equal 20% weights for each pattern:
#       '5|1=0.2,4|2=0.2,3|3=0.2,2|4=0.2,1|5=0.2'.
#   - DIVERSIFIED_NUM is the number of diversified lineups to select from the
#     top1pct output; it defaults to NUM_LINEUPS when omitted.
#   - The script assumes it is run from the project root, or that Python can
#     find the `src` package from the current directory.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

if [[ "${#}" -lt 1 ]]; then
  echo "Usage: $0 PATH_TO_SABERSIM_CSV [FIELD_SIZE] [NUM_LINEUPS] [SALARY_CAP] [STACK_MODE] [STACK_WEIGHTS]" >&2
  exit 1
fi

SABERSIM_CSV="$1"
FIELD_SIZE="${2:-500}"
NUM_LINEUPS="${3:-1000}"
SALARY_CAP="${4:-50000}"
STACK_MODE="${5:-multi}"
STACK_WEIGHTS="${6-}"
DIVERSIFIED_NUM="${7:-$NUM_LINEUPS}"

if [[ ! -f "${SABERSIM_CSV}" ]]; then
  echo "Error: Sabersim CSV not found at '${SABERSIM_CSV}'" >&2
  exit 1
fi

timestamp="$(date +%Y%m%d_%H%M%S)"
CORR_EXCEL="outputs/correlations/showdown_corr_matrix_${timestamp}.xlsx"

echo "================================================================"
echo "Step 1/4: Building correlation matrix from '${SABERSIM_CSV}'"
echo "         -> Output: ${CORR_EXCEL}"
echo "================================================================"

python -m src.main \
  --sabersim-csv "${SABERSIM_CSV}" \
  --output-excel "${CORR_EXCEL}"

echo
echo "================================================================"
echo "Step 2/4: Generating ${NUM_LINEUPS} lineups (salary cap ${SALARY_CAP})"
echo "         from '${SABERSIM_CSV}'"
if [[ "${STACK_MODE}" != "none" ]]; then
  if [[ -z "${STACK_WEIGHTS}" ]]; then
    STACK_WEIGHTS="5|1=0.2,4|2=0.2,3|3=0.2,2|4=0.2,1|5=0.2"
  fi
  echo "         stack mode: ${STACK_MODE}"
  echo "         stack weights: ${STACK_WEIGHTS}"
fi
echo "================================================================"

OPT_STACK_MODE=()
if [[ "${STACK_MODE}" != "none" ]]; then
  OPT_STACK_MODE=(--stack-mode "${STACK_MODE}")
fi

OPT_STACK_WEIGHTS=()
if [[ -n "${STACK_WEIGHTS}" ]]; then
  OPT_STACK_WEIGHTS=(--stack-weights "${STACK_WEIGHTS}")
fi

python -m src.showdown_optimizer_main \
  --sabersim-glob "${SABERSIM_CSV}" \
  --num-lineups "${NUM_LINEUPS}" \
  --salary-cap "${SALARY_CAP}" \
  "${OPT_STACK_MODE[@]}" \
  "${OPT_STACK_WEIGHTS[@]}"

echo
echo "================================================================"
echo "Step 3/4: Estimating top 1% finish probabilities"
echo "         Field size: ${FIELD_SIZE}"
echo "         Using latest lineups & correlations workbooks."
echo "================================================================"

python -m src.top1pct_finish_rate \
  --field-size "${FIELD_SIZE}"

echo
echo "================================================================"
echo "Step 4/4: Selecting diversified lineups"
echo "         Target diversified lineups: ${DIVERSIFIED_NUM}"
echo "         Min top1% finish rate: 1.0%"
echo "         Max player overlap: 4"
echo "================================================================"

python -m src.diversify_lineups \
  --num-lineups "${DIVERSIFIED_NUM}" \
  --min-top1-pct 1.0 \
  --max-overlap 4

echo
echo "All steps completed."


