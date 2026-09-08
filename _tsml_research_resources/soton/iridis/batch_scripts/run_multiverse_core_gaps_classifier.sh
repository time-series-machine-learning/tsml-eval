#!/bin/bash

set -euo pipefail

# Complete the Multiverse-core resample-0 results that keep six datasets out of the
# leaderboard, on Iridis. One CPU classifier per invocation, chosen with MV_CLASSIFIER.
#
# The leaderboard scores only datasets every estimator finished, currently 51 of 66.
# These six are each short by one or two CPU results and nothing else is queued to
# produce them:
#
#   FaceDetection     FreshPRINCE
#   FordChallenge     FreshPRINCE
#   Skoda             FreshPRINCE
#   STEW              HC2, TDE
#   BIDMC32HR_disc    1NN-DTW      (LiteTIME is in the IridisX GPU gap fill)
#   BIDMC32SpO2_disc  1NN-DTW      (LiteTIME is in the IridisX GPU gap fill)
#
# Eight experiments. The first four datasets take the common set from 51 to 55; the
# BIDMC pair follows once the GPU side lands LiteTIME on them.
#
# All four runs share one six-dataset list rather than a list per classifier. The
# runner submits an experiment only when its testResample0.csv is missing or empty,
# so each classifier picks up just its own gaps. The list is read from
# ~/DataSetLists, where the curated lists live, falling back to the copy in the
# repository so a fresh checkout works with no manual step.
#
# Starting tiers are set from how each one failed, not guessed. The tiers are
# (4 8 16 32 64 128 256 620) GiB, and memory is requested per CPU, so a higher tier
# means fewer experiments per node; with eight experiments in total that costs
# nothing.
#
#   FreshPRINCE  recorded "OOM at 128GB" over eight attempts on all three of its
#                datasets, so tier 6 is known to be too small. Start at 7 (256 GiB)
#                and let the runner escalate to 8 (620 GiB) rather than starting at
#                the ceiling with nowhere to go.
#   HC2, TDE     were only "cancelled before completion" on STEW, never observed to
#                fail, so there is no evidence for a high tier. Start at 5 (64 GiB).
#   1NN-DTW      failed on time, not memory, and against Hali's 7 day limit. Iridis
#                caps at 60 hours, so this is the one entry more likely to fail here
#                than there. Tier 3 is ample; if it times out the answer is Hali, not
#                a higher tier.
#
# Usage:
#   MV_CLASSIFIER=FreshPRINCE bash run_multiverse_core_gaps_classifier.sh
#   MV_CLASSIFIER=HC2         bash run_multiverse_core_gaps_classifier.sh
#   MV_CLASSIFIER=TDE         bash run_multiverse_core_gaps_classifier.sh
#   MV_CLASSIFIER=1NN-DTW     bash run_multiverse_core_gaps_classifier.sh
#   ... --dry-run             to show the plan without submitting

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_dir=$(cd -- "${script_dir}/../../../.." && pwd)
dataset_list_name="MultivariateClassification6-CoreGapsCPU.txt"

MV_CLASSIFIER="${MV_CLASSIFIER:?set MV_CLASSIFIER to FreshPRINCE, HC2, TDE or 1NN-DTW}"

case "${MV_CLASSIFIER}" in
    FreshPRINCE) results_category="FeatureBased";     start_tier="7" ;;
    HC2)         results_category="Hybrid";           start_tier="5" ;;
    TDE)         results_category="DictionaryBased";  start_tier="5" ;;
    1NN-DTW)     results_category="DistanceBased";    start_tier="3" ;;
    *)
        echo "ERROR: ${MV_CLASSIFIER} has no gap in this set." >&2
        echo "Expected one of: FreshPRINCE, HC2, TDE, 1NN-DTW" >&2
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

# Size class does not predict the memory these need: FreshPRINCE's three are ordinary
# sized on disk and still exhausted 128 GiB. Set every class to the same tier.
export large_dataset_start_tier="${start_tier}"
export medium_dataset_start_tier="${start_tier}"
export default_dataset_start_tier="${start_tier}"

# One retry for a transient node failure, without looping on a deterministic error.
export max_attempts_per_experiment="2"
export max_failed_attempts="2"

exec bash "${script_dir}/run_multiverse_interval_classifier.sh" "$@"
