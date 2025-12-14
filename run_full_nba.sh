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
#     top1pct output; by default it is resolved from the number of entries in
#     the latest NBA DKEntries CSV (under data/nba/dkentries/) when not
#     provided explicitly.
#   - Optional environment variables:
#       * NUM_WORKERS: when >1 and STACK_MODE='multi', enable parallelization
#         across stack patterns in the optimizer CLI.
#       * PARALLEL_MODE: parallelization strategy passed through to the
#         optimizer (defaults to 'by_stack_pattern' when NUM_WORKERS>1).
#       * SOLVER_MAX_SECONDS: optional per-solve CBC time limit passed to the
#         optimizer CLI.
#       * SOLVER_REL_GAP: optional relative MIP gap (e.g., 0.005 for 0.5%)
#         passed to the optimizer CLI.
#       * USE_WARM_START: when set to 1, enable CBC warm starts in the
#         optimizer's single-model path.
#       * EXTRA_FIELD_LINEUPS: when >0, insert an augmentation step that adds
#         this many quota-balanced field-style lineups to the optimizer
#         workbook before running top1% scoring.
#       * CHUNK_SIZE: optional override for the optimizer --chunk-size flag
#         (defaults to 0 to reuse a single growing model as in prior runs).
#   - The script assumes it is run from the project root, or that Python can
#     find the `src` package from the current directory.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

# Optional flags (can appear anywhere after the required CSV + corr workbook):
#   --contest-id <id>    Fetch DK contest payout structure and compute EV ROI
#   --payouts-json <path> Use a pre-downloaded DK contest JSON instead of fetching
CONTEST_ID=""
PAYOUTS_JSON=""
POSITIONAL=()
while [[ "$#" -gt 0 ]]; do
  case "$1" in
    --contest-id)
      CONTEST_ID="${2:-}"
      shift 2
      ;;
    --payouts-json)
      PAYOUTS_JSON="${2:-}"
      shift 2
      ;;
    *)
      POSITIONAL+=("$1")
      shift
      ;;
  esac
done
set -- "${POSITIONAL[@]}"

if [[ "${#}" -lt 2 ]]; then
  echo "Usage: $0 PATH_TO_SABERSIM_CSV PATH_TO_CORR_EXCEL [FIELD_SIZE] [NUM_LINEUPS] [SALARY_CAP] [STACK_MODE] [STACK_WEIGHTS] [DIVERSIFIED_NUM] [MAX_FLEX_OVERLAP] [CPT_FIELD_CAP_MULTIPLIER] [--contest-id ID] [--payouts-json PATH]" >&2
  exit 1
fi

SABERSIM_CSV="$1"
CORR_EXCEL="$2"
FIELD_SIZE_ARG="${3-}"
# If contest-id is provided, allow FIELD_SIZE to be omitted (infer via DK API).
if [[ -n "${CONTEST_ID}" ]]; then
  FIELD_SIZE="${FIELD_SIZE_ARG-}"
else
  FIELD_SIZE="${FIELD_SIZE_ARG:-9803}"
fi
NUM_LINEUPS="${4:-1000}"
SALARY_CAP="${5:-50000}"
STACK_MODE="${6:-multi}"
STACK_WEIGHTS="${7-}"

# Default parallelization settings for multi-stack runs. These can be
# overridden by exporting NUM_WORKERS or PARALLEL_MODE before invoking the
# script, e.g. NUM_WORKERS=1 to force sequential behavior.
NUM_WORKERS="${NUM_WORKERS:-5}"
PARALLEL_MODE="${PARALLEL_MODE:-by_stack_pattern}"

# If an 8th argument is provided, treat it as an explicit override for the
# number of diversified lineups to select. Otherwise, default to the number
# of actual entries in the latest DKEntries*.csv under data/nba/dkentries/.
DIVERSIFIED_NUM_CLI="${8-}"
if [[ -n "${DIVERSIFIED_NUM_CLI}" ]]; then
  DIVERSIFIED_NUM="${DIVERSIFIED_NUM_CLI}"
else
  echo "Resolving diversified lineup count from latest NBA DKEntries CSV..."
  DIVERSIFIED_NUM="$(python -m src.nba.dkentries_utils --count-entries)"
fi

MAX_FLEX_OVERLAP="${9:-5}"
CPT_FIELD_CAP_MULTIPLIER="${10:-1.5}"
CHUNK_SIZE_ENV="${CHUNK_SIZE:-100}"

if [[ ! -f "${SABERSIM_CSV}" ]]; then
  echo "Error: Sabersim CSV not found at '${SABERSIM_CSV}'" >&2
  exit 1
fi

if [[ ! -f "${CORR_EXCEL}" ]]; then
  echo "Error: correlation Excel workbook not found at '${CORR_EXCEL}'" >&2
  exit 1
fi

timestamp="$(date +%Y%m%d_%H%M%S)"
RUN_DIR="outputs/nba/runs/${timestamp}"
mkdir -p "${RUN_DIR}"
LINEUPS_EXCEL="${RUN_DIR}/lineups_${timestamp}.xlsx"

echo "================================================================"
echo "Step 1/4: Generating ${NUM_LINEUPS} NBA Showdown lineups"
echo "         from Sabersim CSV: ${SABERSIM_CSV}"
echo "         salary cap: ${SALARY_CAP}"
echo "         Run directory: ${RUN_DIR}"
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

OPT_PARALLEL=()
if [[ "${STACK_MODE}" == "multi" ]]; then
  NUM_WORKERS_ENV="${NUM_WORKERS}"
  PARALLEL_MODE_ENV="${PARALLEL_MODE}"
  if [[ "${NUM_WORKERS_ENV}" -gt 1 ]]; then
    OPT_PARALLEL=(--parallel-mode "${PARALLEL_MODE_ENV}" --num-workers "${NUM_WORKERS_ENV}")
    echo "Using parallel mode: ${PARALLEL_MODE_ENV} with ${NUM_WORKERS_ENV} workers"
  fi
fi

SOLVER_OPTS=()
if [[ -n "${SOLVER_MAX_SECONDS:-}" ]]; then
  SOLVER_OPTS+=(--solver-max-seconds "${SOLVER_MAX_SECONDS}")
fi
if [[ -n "${SOLVER_REL_GAP:-}" ]]; then
  SOLVER_OPTS+=(--solver-rel-gap "${SOLVER_REL_GAP}")
fi
if [[ "${USE_WARM_START:-0}" == "1" ]]; then
  SOLVER_OPTS+=(--use-warm-start)
fi

python -m src.nba.showdown_optimizer_main \
  --sabersim-glob "${SABERSIM_CSV}" \
  --num-lineups "${NUM_LINEUPS}" \
  --salary-cap "${SALARY_CAP}" \
  --output-excel "${LINEUPS_EXCEL}" \
  --chunk-size "${CHUNK_SIZE_ENV}" \
  "${OPT_STACK_MODE[@]}" \
  "${OPT_STACK_WEIGHTS[@]}" \
  "${OPT_PARALLEL[@]}" \
  "${SOLVER_OPTS[@]}"

EXTRA_FIELD_LINEUPS_ENV="${EXTRA_FIELD_LINEUPS:-0}"
if [[ "${EXTRA_FIELD_LINEUPS_ENV}" -gt 0 ]]; then
  echo
  echo "================================================================"
  echo "Optional augmentation: adding ${EXTRA_FIELD_LINEUPS_ENV} field-style lineups"
  echo "         Correlation workbook: ${CORR_EXCEL}"
  echo "         Base lineups workbook: ${LINEUPS_EXCEL}"
  echo "================================================================"

  AUGMENTED_LINEUPS_EXCEL="${RUN_DIR}/lineups_${timestamp}_augmented.xlsx"
  python -m src.nba.augment_lineups_with_field \
    --lineups-excel "${LINEUPS_EXCEL}" \
    --corr-excel "${CORR_EXCEL}" \
    --extra-lineups "${EXTRA_FIELD_LINEUPS_ENV}" \
    --output-excel "${AUGMENTED_LINEUPS_EXCEL}"

  LINEUPS_EXCEL="${AUGMENTED_LINEUPS_EXCEL}"
fi

echo
echo "================================================================"
echo "Step 2/4: Estimating top 1% finish probabilities (explicit field)"
if [[ -n "${CONTEST_ID}" ]]; then
  echo "         Contest id: ${CONTEST_ID} (field size inferred unless overridden)"
else
  echo "         Field size: ${FIELD_SIZE}"
fi
echo "         Correlation workbook: ${CORR_EXCEL}"
echo "         Lineups workbook: ${LINEUPS_EXCEL}"
echo "================================================================"

TOP1_ARGS=()
if [[ -n "${FIELD_SIZE}" ]]; then
  TOP1_ARGS+=(--field-size "${FIELD_SIZE}")
fi
if [[ -n "${CONTEST_ID}" ]]; then
  TOP1_ARGS+=(--contest-id "${CONTEST_ID}")
fi
if [[ -n "${PAYOUTS_JSON}" ]]; then
  TOP1_ARGS+=(--payouts-json "${PAYOUTS_JSON}")
fi

python -m src.nba.top1pct_finish_rate_nba \
  --lineups-excel "${LINEUPS_EXCEL}" \
  --corr-excel "${CORR_EXCEL}" \
  --num-sims 100000 \
  --field-model "explicit" \
  --run-dir "${RUN_DIR}" \
  "${TOP1_ARGS[@]}"

echo
echo "================================================================"
echo "Step 3/4: Selecting diversified lineups"
echo "         Target diversified lineups: ${DIVERSIFIED_NUM}"
if [[ -n "${CONTEST_ID}" ]]; then
  echo "         Sorting by: ev_roi"
  echo "         Min top1% finish rate: 0.0% (disabled for EV ROI mode)"
else
  echo "         Sorting by: top1_pct_finish_rate"
  echo "         Min top1% finish rate: 1.0%"
fi
echo "         Max player overlap: 4"
echo "         Max FLEX/UTIL overlap: ${MAX_FLEX_OVERLAP}"
echo "         CPT field cap multiplier: ${CPT_FIELD_CAP_MULTIPLIER}"
echo "================================================================"

SORT_BY="top1_pct_finish_rate"
MIN_TOP1_PCT="1.0"
if [[ -n "${CONTEST_ID}" ]]; then
  SORT_BY="ev_roi"
  MIN_TOP1_PCT="0.0"
fi

python -m src.nba.diversify_lineups_nba \
  --num-lineups "${DIVERSIFIED_NUM}" \
  --min-top1-pct "${MIN_TOP1_PCT}" \
  --sort-by "${SORT_BY}" \
  --max-overlap 5 \
  --max-flex-overlap "${MAX_FLEX_OVERLAP}" \
  --cpt-field-cap-multiplier "${CPT_FIELD_CAP_MULTIPLIER}" \
  --lineups-excel "${LINEUPS_EXCEL}" \
  --output-dir "${RUN_DIR}"

echo
echo "================================================================"
echo "Step 4/4: Filling DKEntries CSV with diversified lineups"
echo "         Using latest DKEntries template under data/nba/dkentries/"
echo "         Writing DK-upload-ready CSV under ${RUN_DIR}/"
echo "================================================================"

OUTPUT_DKENTRIES_CSV="${RUN_DIR}/dkentries_${timestamp}.csv"

python -m src.nba.fill_dkentries_nba \
  --output-csv "${OUTPUT_DKENTRIES_CSV}"

echo
echo "All NBA steps completed."



