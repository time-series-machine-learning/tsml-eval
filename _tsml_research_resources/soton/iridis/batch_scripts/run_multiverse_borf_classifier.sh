#!/bin/bash

set -euo pipefail

# Run aeon's BORF transform with the paper's RidgeClassifierCV head on resample
# 0 of all 66 Multiverse-core datasets on the Iridis6 batch partition. The
# shared controller skips valid results, retries resource failures, and chains
# refills. Results are written to Results/Multiverse/DictionaryBased/BORF.
#
# Check the cluster checkout and optional dependency before submitting any
# Slurm jobs rather than allowing every experiment to fail.

username="ajb2u23"
local_path="/iridisfs/home/${username}"
tsml_eval_dir="${local_path}/Code/tsml-eval"
aeon_dir="${local_path}/Code/aeon"
python_path="/home/${username}/.conda/envs/tsml-eval/bin/python"
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

if ! PYTHONPATH="${aeon_dir}:${tsml_eval_dir}" "${python_path}" -c \
    'from aeon.transformations.collection.dictionary_based import BORF; import sparse' \
    >/dev/null 2>&1; then
    echo "ERROR: BORF is unavailable in ${aeon_dir}." >&2
    echo "Use an aeon checkout that exports the BORF transformer and install sparse." >&2
    exit 1
fi

export MV_CLASSIFIER="BORF"
export MV_RESULTS_CATEGORY="DictionaryBased"
export MV_RESULTS_ROOT="${MV_BORF_RESULTS_ROOT:-${local_path}/Results/Multiverse/DictionaryBased}"
export MV_WORKFLOW_KEY="borf"
export MV_SUBMISSION_LABEL="MVBORF"
export MV_MAX_FOLDS="${MV_MAX_FOLDS:-1}"
export MV_START_FOLD="${MV_START_FOLD:-1}"
export MV_DATASET_LIST="${MV_DATASET_LIST:-${tsml_eval_dir}/_tsml_research_resources/dataset_lists/MultivariateClassification66-MultiverseMini.txt}"
export MV_EXTRA_SOURCE_FILES="tsml_eval/_wip/classification/_borf.py"

exec bash "${script_dir}/run_multiverse_interval_classifier.sh" "$@"
