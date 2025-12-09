#!/usr/bin/env bash

# Run the full NBA Showdown pipeline for a single slate using a
# manually-provided player correlation workbook:
#   1) Generate optimized lineups from a Sabersim projections CSV
#   2) Estimate top 1% finish probabilities using an explicit field model
#      and the provided correlation workbook
#   3) Select a diversified subset of lineups based on top 1% finish rate
#   4) Fill a DKEntries CSV with the diversified lineups
#
# Usage:
#   ./run_full_nba.sh PATH_TO_SABERSIM_CSV PATH_TO_CORR_EXCEL \
#       [FIELD_SIZE] [NUM_LINEUPS] [SALARY_CAP] [STACK_MODE] [STACK_WEIGHTS] [DIVERSIFIED_NUM]
#
# Examples:
#   ./run_full_nba.sh \
#     data/nba/sabersim/NBA_2025-12-02-1100pm_DK_SHOWDOWN_OKC-@-GSW.csv \
#     outputs/nba/correlations/my_corr_matrix.xlsx
#
#   ./run_full_nba.sh \
#     data/nba/sabersim/NBA_2025-12-02-1100pm_DK_SHOWDOWN_OKC-@-GSW.csv \
#     outputs/nba/correlations/my_corr_matrix.xlsx \
#     23529 2000 50000 multi "5|1=0.4,4|2=0.3,3|3=0.2,2|4=0.1,1|5=0.0" 150
#
# Notes:
#   - PATH_TO_CORR_EXCEL should be an Excel workbook with:
#       * Sheet 'Sabersim_Projections' containing at least columns
#         ['Name', 'My Proj'] (optionally 'dk_std', 'Pos')
#       * Sheet 'Correlation_Matrix' containing a square correlation
#         matrix whose index/columns are player names matching 'Name'
#   - FIELD_SIZE defaults to 500 if not provided.
#   - NUM_LINEUPS defaults to 2000.
#   - SALARY_CAP defaults to 50000.
#   - STACK_MODE defaults to 'multi'; set to 'none' to disable multi-stack mode.
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

if [[ "${#}" -lt 2 ]]; then
  echo "Usage: $0 PATH_TO_SABERSIM_CSV PATH_TO_CORR_EXCEL [FIELD_SIZE] [NUM_LINEUPS] [SALARY_CAP] [STACK_MODE] [STACK_WEIGHTS] [DIVERSIFIED_NUM]" >&2
  exit 1
fi

SABERSIM_CSV="$1"
CORR_EXCEL="$2"
FIELD_SIZE="${3:-500}"
NUM_LINEUPS="${4:-2000}"
SALARY_CAP="${5:-50000}"
STACK_MODE="${6:-multi}"
STACK_WEIGHTS="${7-}"
DIVERSIFIED_NUM="${8:-${NUM_LINEUPS}}"

if [[ ! -f "${SABERSIM_CSV}" ]]; then
  echo "Error: Sabersim CSV not found at '${SABERSIM_CSV}'" >&2
  exit 1
fi

if [[ ! -f "${CORR_EXCEL}" ]]; then
  echo "Error: correlation Excel workbook not found at '${CORR_EXCEL}'" >&2
  exit 1
fi

echo "================================================================"
echo "Step 1/4: Generating ${NUM_LINEUPS} NBA Showdown lineups"
echo "         from Sabersim CSV: ${SABERSIM_CSV}"
echo "         salary cap: ${SALARY_CAP}"
echo "================================================================"

OPT_STACK_MODE=()
if [[ "${STACK_MODE}" != "none" ]]; then
  OPT_STACK_MODE=(--stack-mode "${STACK_MODE}")
fi

OPT_STACK_WEIGHTS=()
if [[ "${STACK_MODE}" != "none" ]]; then
  if [[ -z "${STACK_WEIGHTS}" ]]; then
    STACK_WEIGHTS="5|1=0.2,4|2=0.2,3|3=0.2,2|4=0.2,1|5=0.2"
  fi
  OPT_STACK_WEIGHTS=(--stack-weights "${STACK_WEIGHTS}")
  echo "Using stack mode: ${STACK_MODE}"
  echo "Using stack weights: ${STACK_WEIGHTS}"
else
  echo "Using stack mode: none (single optimization pass)"
fi

python -m src.nba.showdown_optimizer_main \
  --sabersim-glob "${SABERSIM_CSV}" \
  --num-lineups "${NUM_LINEUPS}" \
  --salary-cap "${SALARY_CAP}" \
  "${OPT_STACK_MODE[@]}" \
  "${OPT_STACK_WEIGHTS[@]}"

echo
echo "================================================================"
echo "Step 2/4: Estimating top 1% finish probabilities (explicit field)"
echo "         Field size: ${FIELD_SIZE}"
echo "         Correlation workbook: ${CORR_EXCEL}"
echo "================================================================"

python -m src.nba.top1pct_finish_rate_nba \
  --field-size "${FIELD_SIZE}" \
  --corr-excel "${CORR_EXCEL}" \
  --num-sims 100000 \
  --field-model "explicit"

echo
echo "================================================================"
echo "Step 3/4: Selecting diversified lineups"
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
echo "Step 4/4: Filling DKEntries CSV with diversified lineups"
echo "         Using latest DKEntries template under data/nba/dkentries/"
echo "         Writing DK-upload-ready CSV under outputs/nba/dkentries/"
echo "================================================================"

python -m src.nba.fill_dkentries_nba

echo
echo "All NBA steps completed."



