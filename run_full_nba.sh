#!/usr/bin/env bash

# Run the full NBA Showdown pipeline for a single slate:
#   1) Build correlation matrix from a Sabersim projections CSV
#   2) Generate optimized lineups from that same CSV
#   3) Estimate top 1% finish probabilities for those lineups
#   4) Select a diversified subset of lineups based on top 1% finish rate
#   5) Fill a DKEntries CSV with the diversified lineups
#
# Usage:
#   ./run_full_nba.sh PATH_TO_SABERSIM_CSV [FIELD_SIZE] [NUM_LINEUPS] [SALARY_CAP] [STACK_MODE] [STACK_WEIGHTS] [DIVERSIFIED_NUM]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

if [[ "${#}" -lt 1 ]]; then
  echo "Usage: $0 PATH_TO_SABERSIM_CSV [FIELD_SIZE] [NUM_LINEUPS] [SALARY_CAP] [STACK_MODE] [STACK_WEIGHTS] [DIVERSIFIED_NUM]" >&2
  exit 1
fi

SABERSIM_CSV="$1"
FIELD_SIZE="${2:-500}"
NUM_LINEUPS="${3:-1000}"
SALARY_CAP="${4:-50000}"
STACK_MODE="${5:-multi}"
STACK_WEIGHTS="${6-}"

# If a 7th argument is provided, treat it as an explicit override for the
# number of diversified lineups to select. Otherwise, default to the number
# of actual entries in the latest DKEntries*.csv under data/nba/dkentries/.
DIVERSIFIED_NUM_CLI="${7-}"
if [[ -n "${DIVERSIFIED_NUM_CLI}" ]]; then
  DIVERSIFIED_NUM="${DIVERSIFIED_NUM_CLI}"
else
  echo "Resolving diversified lineup count from latest NBA DKEntries CSV..."
  DIVERSIFIED_NUM="$(python -m src.dkentries_utils --dkentries-csv "data/nba/dkentries" --count-entries || true)"
  # Fallback: if the above fails, just use NUM_LINEUPS.
  if [[ -z "${DIVERSIFIED_NUM}" ]]; then
    DIVERSIFIED_NUM="${NUM_LINEUPS}"
  fi
fi

if [[ ! -f "${SABERSIM_CSV}" ]]; then
  echo "Error: Sabersim CSV not found at '${SABERSIM_CSV}'" >&2
  exit 1
fi

timestamp="$(date +%Y%m%d_%H%M%S)"
CORR_EXCEL="outputs/nba/correlations/showdown_corr_matrix_${timestamp}.xlsx"

echo "================================================================"
echo "NBA Step 1/5: Building correlation matrix from '${SABERSIM_CSV}'"
echo "         -> Output: ${CORR_EXCEL}"
echo "================================================================"

python -m src.nba.main \
  --sabersim-csv "${SABERSIM_CSV}" \
  --output-excel "${CORR_EXCEL}"

echo
echo "================================================================"
echo "NBA Step 2/5: Generating ${NUM_LINEUPS} lineups (salary cap ${SALARY_CAP})"
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

python -m src.nba.showdown_optimizer_main \
  --sabersim-glob "${SABERSIM_CSV}" \
  --num-lineups "${NUM_LINEUPS}" \
  --salary-cap "${SALARY_CAP}" \
  "${OPT_STACK_MODE[@]}" \
  "${OPT_STACK_WEIGHTS[@]}"

echo
echo "================================================================"
echo "NBA Step 3/5: Estimating top 1% finish probabilities"
echo "         Field size: ${FIELD_SIZE}"
echo "         Using latest NBA lineups & correlations workbooks."
echo "================================================================"

python -m src.nba.top1pct_finish_rate_nba \
  --field-size "${FIELD_SIZE}"

echo
echo "================================================================"
echo "NBA Step 4/5: Selecting diversified lineups"
echo "         Target diversified lineups: ${DIVERSIFIED_NUM}"
echo "         Min top1% finish rate: 1.0%"
echo "         Max player overlap: 4"
echo "================================================================"

python -m src.nba.diversify_lineups_nba \
  --num-lineups "${DIVERSIFIED_NUM}" \
  --min-top1-pct 1.0 \
  --max-overlap 4

echo
echo "================================================================"
echo "NBA Step 5/5: Filling DKEntries CSV with diversified lineups"
echo "         Using latest NBA DKEntries template under data/nba/dkentries/"
echo "         Writing DK-upload-ready CSV under outputs/nba/dkentries/"
echo "================================================================"

python -m src.nba.fill_dkentries_nba

echo
echo "NBA pipeline completed."


