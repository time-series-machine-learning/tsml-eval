#!/bin/bash

set -euo pipefail

# Complete the Multiverse-core resample-0 results that more memory can still reach,
# on Iridis. One CPU classifier per invocation, chosen with MV_CLASSIFIER.
#
# The leaderboard scores only the datasets every listed estimator finished, now 56 of
# 66 across 24 estimators. Ten are dropped, and most of them are not waiting on
# compute at all:
#
#   Alzheimers, EigenWorms,      ConvTran only, CUDA OOM on the A100. Queued on the
#   PhotoStimulation, Locust2022 H200 gap fill, plus TS2Vec on Locust2022.
#   AustraliaRainfall_disc       RDST and ROCKET hit LAPACK integer overflow in
#   Tiselac                      RidgeClassifierCV's SVD, aeon issue 3738. No memory
#                                tier fixes that, and MRHydra hits it on Tiselac too.
#   PenDigits                    MRHydra needs n_timepoints >= 9; the series are 8.
#   EmoPain                      aeon's check_collection_variance rejects it before
#                                fit, for every aeon classifier.
#
# That leaves two datasets this script can actually finish:
#
#   STEW          HC2, TDE     both only "cancelled before completion", never observed
#                              to fail, so their real cost is unknown
#   USCActivity   HC2, TDE, RDST
#
# Five experiments. STEW alone takes the scored set from 56 to 57, USCActivity to 58,
# and the four GPU datasets above take it to 62 if they land.
#
# One honest caveat on USCActivity. TDE failed there on time, against Hali's 7 day
# limit, and Iridis caps at 60 hours, so more memory does not address it. It is
# included because a run cancelled or thrashing for memory can present as a timeout,
# but if TDE times out cleanly here the dataset needs Hali or a faster estimator, not
# a higher tier.
#
# Starting tiers are set from how each one failed, not guessed. The tiers are
# (4 8 16 32 64 128 256 620) GiB, and memory is requested per CPU, so a higher tier
# means fewer experiments per node; with five experiments that costs nothing.
#
#   RDST      recorded "OOM at 64GB" on USCActivity, exhausting what was then the top
#             tier, so start at 7 (256 GiB) with 620 still above it.
#   HC2, TDE  were cancelled or timed out rather than seen to run out of memory, so
#             there is no measured floor. Start at 6 (128 GiB), which is above
#             anything they were given before.
#
# Usage:
#   MV_CLASSIFIER=HC2  bash run_multiverse_core_gaps_classifier.sh
#   MV_CLASSIFIER=TDE  bash run_multiverse_core_gaps_classifier.sh
#   MV_CLASSIFIER=RDST bash run_multiverse_core_gaps_classifier.sh
#   ... --dry-run      to show the plan without submitting

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_dir=$(cd -- "${script_dir}/../../../.." && pwd)
dataset_list_name="MultivariateClassification2-CoreGapsCPU.txt"

MV_CLASSIFIER="${MV_CLASSIFIER:?set MV_CLASSIFIER to FreshPRINCE, HC2, TDE or 1NN-DTW}"

case "${MV_CLASSIFIER}" in
    HC2)  results_category="Hybrid";          start_tier="6" ;;
    TDE)  results_category="DictionaryBased"; start_tier="6" ;;
    RDST) results_category="ShapeletBased";   start_tier="7" ;;
    *)
        echo "ERROR: ${MV_CLASSIFIER} has no gap that more memory can close." >&2
        echo "Expected one of: HC2, TDE, RDST" >&2
        exit 1
        ;;
esac

classifier_lc=$(printf '%s' "${MV_CLASSIFIER}" | tr 'A-Z' 'a-z')

export MV_RESULTS_CATEGORY="${results_category}"
export MV_WORKFLOW_KEY="core-gaps-${classifier_lc}"
export MV_SUBMISSION_LABEL="MVCoreGaps"
export MV_MAX_FOLDS="1"
export MV_START_FOLD="1"
# Curated dataset lists live in ~/DataSetLists on the cluster, so prefer that copy and
# let it be edited without touching the repository. A copy also ships in the repository
# so a fresh checkout runs without any manual step; the home copy wins when present.
if [[ -z "${MV_DATASET_LIST:-}" ]]; then
    home_dataset_list="${HOME}/DataSetLists/${dataset_list_name}"
    repo_dataset_list="${repo_dir}/_tsml_research_resources/dataset_lists/${dataset_list_name}"
    if [[ -s "${home_dataset_list}" ]]; then
        MV_DATASET_LIST="${home_dataset_list}"
    elif [[ -s "${repo_dataset_list}" ]]; then
        MV_DATASET_LIST="${repo_dataset_list}"
        echo "No ${home_dataset_list}; using the copy in the repository."
        echo "  cp ${repo_dataset_list} ${home_dataset_list}"
    else
        echo "ERROR: no dataset list at ${home_dataset_list} or ${repo_dataset_list}" >&2
        exit 1
    fi
fi
export MV_DATASET_LIST
echo "Dataset list: ${MV_DATASET_LIST}"

# Size class does not predict the memory these need, so set every class to the same
# tier rather than letting the on-disk size choose.
export large_dataset_start_tier="${start_tier}"
export medium_dataset_start_tier="${start_tier}"
export default_dataset_start_tier="${start_tier}"

# One retry for a transient node failure, without looping on a deterministic error.
export max_attempts_per_experiment="2"
export max_failed_attempts="2"

exec bash "${script_dir}/run_multiverse_interval_classifier.sh" "$@"
