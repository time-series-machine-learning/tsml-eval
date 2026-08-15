#!/bin/bash

set -euo pipefail

# Run the four full-budget HC2 components on the saved GEAR-Auto transform of
# FaceDetection. Each component writes native TRAIN estimates and TEST
# predictions to a new result family; existing pipeline results are untouched.

username="ajb2u23"
local_path="/iridisfs/home/${username}"
queue="batch"
max_time="60:00:00"
memory_per_cpu="30G"
mail="NONE"
mailto="${username}@soton.ac.uk"

tsml_eval_dir="${local_path}/Code/tsml-eval"
aeon_dir="${local_path}/Code/aeon"
python_path="/home/${username}/.conda/envs/tsml-eval/bin/python"
worker="${tsml_eval_dir}/tsml_eval/_wip/eeg_cote/run_gear_auto_transformed_component.py"

# This is the accelerated ajb/hc2 revision used by the established pipeline
# experiments. Descendants are accepted (for example, the later Arsenal SVD
# fallback), but stock/release aeon checkouts are rejected.
required_aeon_hc2_commit="cac5ddf09d9ecbb56171450bbdd477fb645519c0"

transformed_data_root="${local_path}/Results/Transforms/GEAR-Auto"
results_root="${local_path}/Results/ChannelSelectionPipeline"
output_root="${results_root}/output"
numba_cache_dir="${local_path}/Code/.cache/tsml-eval"

dataset="FaceDetection"
components=("Arsenal" "DrCIF" "STC" "TDE")

train_data="${transformed_data_root}/${dataset}/${dataset}_TRAIN.ts"
test_data="${transformed_data_root}/${dataset}/${dataset}_TEST.ts"
for required in "${python_path}" "${worker}" "${train_data}" "${test_data}"; do
    if [[ ! -s "${required}" ]]; then
        echo "ERROR: required file is missing or empty: ${required}" >&2
        exit 1
    fi
done
for repository in "${tsml_eval_dir}" "${aeon_dir}"; do
    if [[ ! -d "${repository}/.git" ]]; then
        echo "ERROR: Git checkout is missing: ${repository}" >&2
        exit 1
    fi
done
if ! git -C "${aeon_dir}" merge-base --is-ancestor \
    "${required_aeon_hc2_commit}" HEAD; then
    echo "ERROR: aeon does not contain the accelerated ajb/hc2 implementation." >&2
    echo "Required ancestor: ${required_aeon_hc2_commit}" >&2
    echo "Current commit:    $(git -C "${aeon_dir}" rev-parse HEAD)" >&2
    exit 1
fi

mkdir -p "${results_root}" "${output_root}" "${numba_cache_dir}"
tsml_eval_commit=$(git -C "${tsml_eval_dir}" rev-parse HEAD)
aeon_commit=$(git -C "${aeon_dir}" rev-parse HEAD)

# Fail before submission if the installed aeon lacks any native train method.
PYTHONNOUSERSITE=1 PYTHONPATH="${aeon_dir}:${tsml_eval_dir}" \
"${python_path}" - <<'PY'
from tsml_eval._wip.eeg_cote.run_gear_auto_transformed_component import (
    COMPONENTS,
    _make_component,
)

for component in COMPONENTS:
    estimator = _make_component(component, random_state=0)
    assert estimator.get_tag("capability:train_estimate", False, False)
    print(component, type(estimator).__name__)
PY

run_id=$(date +%Y%m%d%H%M%S)
batch_id="${run_id}-gear-auto-facedetection-components"
submission_dir="${results_root}/batch-submissions/${batch_id}"
command_file="${submission_dir}/commands.txt"
submission_file="${submission_dir}/submit.sub"
mkdir -p "${submission_dir}"
: > "${command_file}"

command_count=0
for component in "${components[@]}"; do
    result_name="GEAR-Auto-Native-${component}"
    prediction_dir="${results_root}/${result_name}/Predictions/${dataset}"
    train_result="${prediction_dir}/trainResample0.csv"
    test_result="${prediction_dir}/testResample0.csv"
    if [[ -s "${train_result}" && -s "${test_result}" ]]; then
        echo "Skipping complete: ${result_name}/${dataset}/resample0"
        continue
    fi

    log_dir="${output_root}/${result_name}"
    mkdir -p "${log_dir}"
    command=(
        "${python_path}" -u "${worker}"
        "${transformed_data_root}" "${results_root}"
        "${component}" "${dataset}" --resample-id 0
    )
    printf -v command_line '%q ' "${command[@]}"
    printf '%s> %q 2>&1\n' \
        "${command_line}" \
        "${log_dir}/output-${dataset}-${run_id}.txt" \
        >> "${command_file}"
    command_count=$((command_count + 1))
done

if ((command_count == 0)); then
    echo "All four GEAR-Auto FaceDetection component results are complete."
    exit 0
fi

# One independent single-threaded process per outstanding component.
cpu_count=${command_count}

cat > "${submission_file}" <<EOF
#!/bin/bash
#SBATCH --mail-type=${mail}
#SBATCH --mail-user=${mailto}
#SBATCH --job-name=eeg-gear-auto-fd-components
#SBATCH --partition=${queue}
#SBATCH --time=${max_time}
#SBATCH --output=${submission_dir}/%A.out
#SBATCH --error=${submission_dir}/%A.err
#SBATCH --nodes=1
#SBATCH --ntasks=${cpu_count}
#SBATCH --mem-per-cpu=${memory_per_cpu}

. /etc/profile
set -eo pipefail
cd "${tsml_eval_dir}" || exit 1

unset PYTHONHOME
export PYTHONNOUSERSITE=1
export PYTHONPATH="${aeon_dir}:${tsml_eval_dir}"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export NUMBA_NUM_THREADS=1
export LOKY_MAX_CPU_COUNT=1
export PYTHONUNBUFFERED=1
export NUMBA_CACHE_DIR="${numba_cache_dir}"
mkdir -p "\${NUMBA_CACHE_DIR}"

current_tsml_eval_commit=\$(git -C "${tsml_eval_dir}" rev-parse HEAD)
current_aeon_commit=\$(git -C "${aeon_dir}" rev-parse HEAD)
if [[ "\${current_tsml_eval_commit}" != "${tsml_eval_commit}" ]]; then
    echo "ERROR: tsml-eval changed after submission." >&2
    exit 1
fi
if [[ "\${current_aeon_commit}" != "${aeon_commit}" ]]; then
    echo "ERROR: aeon changed after submission." >&2
    exit 1
fi

echo "Dataset:          ${dataset}"
echo "Input transform:  ${transformed_data_root}/${dataset}"
echo "Outstanding:      ${command_count} component(s)"
echo "Allocated tasks:  \${SLURM_NTASKS}"
echo "Memory per task:  ${memory_per_cpu}"
echo "tsml-eval commit: \${current_tsml_eval_commit}"
echo "aeon commit:      \${current_aeon_commit}"
echo "HC2 speed-ups:    verified (${required_aeon_hc2_commit} is an ancestor)"
echo "Command file:     ${command_file}"
staskfarm "${command_file}"
EOF

sbatch_output=$(sbatch "${submission_file}")
echo "${sbatch_output}"
echo "Submitted ${command_count} FaceDetection component(s) on one node."
echo "Results: ${results_root}/GEAR-Auto-Native-<Component>/Predictions/${dataset}"
echo "Submission records: ${submission_dir}"

# Slurm copied the submission script; staskfarm still needs the command file.
rm -f "${submission_file}"
