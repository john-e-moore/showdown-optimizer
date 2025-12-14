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
#   ./run_full.sh data/nfl/sabersim/NFL_2025-11-27-820pm_DK_SHOWDOWN_TB-@-DAL.csv
#   ./run_full.sh data/nfl/sabersim/NFL_2025-11-27-820pm_DK_SHOWDOWN_TB-@-DAL.csv 23529 150 50000
#   ./run_full.sh data/nfl/sabersim/NFL_2025-11-27-820pm_DK_SHOWDOWN_TB-@-DAL.csv 23529 1000 50000 multi
#   ./run_full.sh data/nfl/sabersim/NFL_2025-11-27-820pm_DK_SHOWDOWN_TB-@-DAL.csv 23529 1000 50000 multi "5|1=0.4,4|2=0.3,3|3=0.2,2|4=0.1,1|5=0.0"
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
#         (defaults to 0 to reuse a single growing model when unset).
#   - The script assumes it is run from the project root, or that Python can
#     find the `src` package from the current directory.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

# Optional flags (can appear anywhere after the required CSV):
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

if [[ "${#}" -lt 1 ]]; then
  echo "Usage: $0 PATH_TO_SABERSIM_CSV [FIELD_SIZE] [NUM_LINEUPS] [SALARY_CAP] [STACK_MODE] [STACK_WEIGHTS] [DIVERSIFIED_NUM] [--contest-id ID] [--payouts-json PATH]" >&2
  exit 1
fi

SABERSIM_CSV="$1"
FIELD_SIZE_ARG="${2-}"
# If contest-id is provided, allow FIELD_SIZE to be omitted (infer via DK API).
if [[ -n "${CONTEST_ID}" ]]; then
  FIELD_SIZE="${FIELD_SIZE_ARG-}"
else
  FIELD_SIZE="${FIELD_SIZE_ARG:-250}"
fi
NUM_LINEUPS="${3:-2000}"
SALARY_CAP="${4:-50000}"
STACK_MODE="${5:-multi}"
STACK_WEIGHTS="${6-}"

# Default parallelization settings for multi-stack runs. These can be
# overridden by exporting NUM_WORKERS or PARALLEL_MODE before invoking the
# script, e.g. NUM_WORKERS=1 to force sequential behavior.
NUM_WORKERS="${NUM_WORKERS:-5}"
PARALLEL_MODE="${PARALLEL_MODE:-by_stack_pattern}"

CHUNK_SIZE_ENV="${CHUNK_SIZE:-100}"

# If a 7th argument is provided, treat it as an explicit override for the
# number of diversified lineups to select. Otherwise, default to the number
# of actual entries in the latest DKEntries*.csv under data/nfl/dkentries/.
DIVERSIFIED_NUM_CLI="${7-}"
if [[ -n "${DIVERSIFIED_NUM_CLI}" ]]; then
  DIVERSIFIED_NUM="${DIVERSIFIED_NUM_CLI}"
else
  echo "Resolving diversified lineup count from latest DKEntries CSV..."
  DIVERSIFIED_NUM="$(python -m src.nfl.dkentries_utils --count-entries)"
fi

if [[ ! -f "${SABERSIM_CSV}" ]]; then
  echo "Error: Sabersim CSV not found at '${SABERSIM_CSV}'" >&2
  exit 1
fi

timestamp="$(date +%Y%m%d_%H%M%S)"
RUN_DIR="outputs/nfl/runs/${timestamp}"
mkdir -p "${RUN_DIR}"
CORR_EXCEL="${RUN_DIR}/correlations_${timestamp}.xlsx"

echo "================================================================"
echo "Step 1/4: Building correlation matrix from '${SABERSIM_CSV}'"
echo "         -> Output: ${CORR_EXCEL}"
echo "================================================================"

python -m src.nfl.main \
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

OPT_PARALLEL=()
if [[ "${STACK_MODE}" == "multi" ]]; then
  NUM_WORKERS_ENV="${NUM_WORKERS}"
  PARALLEL_MODE_ENV="${PARALLEL_MODE}"
  if [[ "${NUM_WORKERS_ENV}" -gt 1 ]]; then
    OPT_PARALLEL=(--parallel-mode "${PARALLEL_MODE_ENV}" --num-workers "${NUM_WORKERS_ENV}")
    echo "         parallel mode: ${PARALLEL_MODE_ENV} with ${NUM_WORKERS_ENV} workers"
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

LINEUPS_EXCEL="${RUN_DIR}/lineups_${timestamp}.xlsx"

python -m src.nfl.showdown_optimizer_main \
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
  python -m src.nfl.augment_lineups_with_field \
    --lineups-excel "${LINEUPS_EXCEL}" \
    --corr-excel "${CORR_EXCEL}" \
    --extra-lineups "${EXTRA_FIELD_LINEUPS_ENV}" \
    --output-excel "${AUGMENTED_LINEUPS_EXCEL}"

  LINEUPS_EXCEL="${AUGMENTED_LINEUPS_EXCEL}"
fi

echo
echo "================================================================"
echo "Step 3/4: Estimating top 1% finish probabilities"
if [[ -n "${CONTEST_ID}" ]]; then
  echo "         Contest id: ${CONTEST_ID} (field size inferred unless overridden)"
else
  echo "         Field size: ${FIELD_SIZE}"
fi
echo "         Using run-scoped lineups & correlations workbooks."
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

python -m src.nfl.top1pct_finish_rate \
  --lineups-excel "${LINEUPS_EXCEL}" \
  --corr-excel "${CORR_EXCEL}" \
  --field-model "explicit" \
  --run-dir "${RUN_DIR}" \
  "${TOP1_ARGS[@]}"

echo
echo "================================================================"
echo "Step 4/4: Selecting diversified lineups"
echo "         Target diversified lineups: ${DIVERSIFIED_NUM}"
if [[ -n "${CONTEST_ID}" ]]; then
  echo "         Sorting by: ev_roi"
  echo "         Min top1% finish rate: 0.0% (disabled for EV ROI mode)"
else
  echo "         Sorting by: top1_pct_finish_rate"
  echo "         Min top1% finish rate: 0.25%"
fi
echo "         Max player overlap: 5"
echo "================================================================"

SORT_BY="top1_pct_finish_rate"
MIN_TOP1_PCT="1.0"
if [[ -n "${CONTEST_ID}" ]]; then
  SORT_BY="ev_roi"
  MIN_TOP1_PCT="0.0"
fi

python -m src.nfl.diversify_lineups \
  --num-lineups "${DIVERSIFIED_NUM}" \
  --min-top1-pct "${MIN_TOP1_PCT}" \
  --sort-by "${SORT_BY}" \
  --max-overlap 5 \
  --max-flex-overlap 5 \
  --cpt-field-cap-multiplier 1.5 \
  --lineups-excel "${LINEUPS_EXCEL}" \
  --output-dir "${RUN_DIR}"

echo
echo "================================================================"
echo "Step 5: Filling DKEntries CSV with diversified lineups"
echo "         Using latest DKEntries template under data/nfl/dkentries/"
echo "         Writing DK-upload-ready CSV under ${RUN_DIR}/"
echo "================================================================"

OUTPUT_DKENTRIES_CSV="${RUN_DIR}/dkentries_${timestamp}.csv"

python -m src.nfl.fill_dkentries \
  --output-csv "${OUTPUT_DKENTRIES_CSV}" \
  --lineups-excel "${LINEUPS_EXCEL}"

echo
echo "All steps completed."


