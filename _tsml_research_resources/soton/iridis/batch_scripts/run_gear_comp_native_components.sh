#!/bin/bash

set -euo pipefail

# Generate a new, non-overwriting GEAR-Comp result family using HC2's native
# component train estimates. Results are written as GEAR-Comp-Native-Arsenal,
# -DrCIF, -STC and -TDE. The existing GEAR-Comp-* external-CV files are retained.
#
# Typical four-node sequence:
#   RUN_SET=fast bash run_gear_comp_native_components.sh
#   RUN_SET=slow bash run_gear_comp_native_components.sh

run_set="${RUN_SET:-all}"
run_set="${run_set,,}"

username="ajb2u23"
local_path="/iridisfs/home/${username}"
queue="batch"
max_num_submitted=200
max_time="60:00:00"
mail="NONE"
mailto="${username}@soton.ac.uk"

tsml_eval_dir="${local_path}/Code/tsml-eval"
aeon_dir="${local_path}/Code/aeon"
python_path="/home/${username}/.conda/envs/tsml-eval/bin/python"
worker="${tsml_eval_dir}/tsml_eval/_wip/eeg_cote/run_native_gear_component.py"

data_root="${local_path}/Data/EEG"
results_root="${local_path}/Results/ChannelSelectionPipeline"
output_root="${results_root}/output"
numba_cache_dir="${local_path}/Code/.cache/tsml-eval"

fast_datasets=(
    "EyesOpenShut"
    "FingerMovements"
    "HandMovementDirection"
    "ButtonPress"
    "LowCost"
    "FeedbackButton"
    "MindReading"
    "FibroLiverpool"
    "FibroUEA"
    "PhotoStimulation"
    "MotorImagery"
    "PronouncedSpeech"
    "InnerSpeech"
    "VisualSpeech"
    "Alzheimers"
    "FaceDetection"
    "OpenCloseFist"
    "ImaginedFeetHands"
    "ImaginedOpenCloseFist"
    "FeetHands"
    "SongFamiliarity"
)
slow_datasets=(
    "SitStand"
    "ShortIntervalTask"
    "MatchingPennies"
    "LongIntervalTask"
)

# component|dataset group|maximum concurrent tasks|GiB per task
# Memory values retain the headroom established from the earlier GEAR-Comp
# component runs. Native fit-predict is approximately one component fit, not
# ten complete pipeline fits.
all_specs=(
    "Arsenal|fast|21|6"
    "DrCIF|fast|21|12"
    "STC|fast|21|6"
    "TDE|fast|20|30"
    "Arsenal|slow|4|20"
    "DrCIF|slow|4|35"
    "STC|slow|4|10"
    "TDE|slow|4|150"
)

select_specs() {
    case "${run_set}" in
        all)
            specs=("${all_specs[@]}")
            ;;
        fast)
            specs=("${all_specs[@]:0:4}")
            ;;
        slow)
            specs=("${all_specs[@]:4:4}")
            ;;
        arsenal|drcif|stc|tde)
            specs=()
            for spec in "${all_specs[@]}"; do
                IFS="|" read -r component _ _ _ <<< "${spec}"
                if [[ "${component,,}" == "${run_set}" ]]; then
                    specs+=("${spec}")
                fi
            done
            ;;
        arsenal-fast|drcif-fast|stc-fast|tde-fast|arsenal-slow|drcif-slow|stc-slow|tde-slow)
            wanted_component="${run_set%-*}"
            wanted_group="${run_set##*-}"
            specs=()
            for spec in "${all_specs[@]}"; do
                IFS="|" read -r component group _ _ <<< "${spec}"
                if [[ "${component,,}" == "${wanted_component}" \
                      && "${group}" == "${wanted_group}" ]]; then
                    specs+=("${spec}")
                fi
            done
            ;;
        *)
            echo "ERROR: unknown RUN_SET '${run_set}'." >&2
            echo "Use all, fast, slow, a component, or component-fast/component-slow." >&2
            exit 2
            ;;
    esac
}

select_specs

native_result_complete() {
    local result_name="$1"
    local train_file="$2"
    local test_file="$3"
    local train_header
    local test_header

    [[ -s "${train_file}" && -s "${test_file}" ]] || return 1
    IFS= read -r train_header < "${train_file}"
    IFS= read -r test_header < "${test_file}"
    [[ "${train_header}" == *",${result_name} (GEARNativeComponentPipeline),TRAIN,"* \
        && "${test_header}" == *",${result_name} (GEARNativeComponentPipeline),TEST,"* ]]
}

for required in "${python_path}" "${worker}" "${data_root}"; do
    if [[ ! -e "${required}" ]]; then
        echo "ERROR: required path is missing: ${required}" >&2
        exit 1
    fi
done
for repository in "${tsml_eval_dir}" "${aeon_dir}"; do
    if [[ ! -d "${repository}/.git" ]]; then
        echo "ERROR: Git checkout is missing: ${repository}" >&2
        exit 1
    fi
done

mkdir -p "${results_root}" "${output_root}" "${numba_cache_dir}"
tsml_eval_commit=$(git -C "${tsml_eval_dir}" rev-parse HEAD)
aeon_commit=$(git -C "${aeon_dir}" rev-parse HEAD)

PYTHONNOUSERSITE=1 PYTHONPATH="${aeon_dir}:${tsml_eval_dir}" \
"${python_path}" - <<'PY'
import aeon
import aeon_neuro
import tsml_eval
from tsml_eval._wip.eeg_cote.run_native_gear_component import (
    GEARNativeComponentPipeline,
)

for component in ("Arsenal", "DrCIF", "STC", "TDE"):
    estimator = GEARNativeComponentPipeline(component, random_state=0, n_jobs=1)
    assert estimator.get_tag("capability:train_estimate")
    print(component, type(estimator).__name__)
print("aeon:      ", aeon.__file__)
print("aeon-neuro:", aeon_neuro.__file__)
print("tsml-eval: ", tsml_eval.__file__)
PY

run_id=$(date +%Y%m%d%H%M%S)
submission_dir="${results_root}/batch-submissions/${run_id}-gear-comp-native"
mkdir -p "${submission_dir}"

submitted=0
for spec in "${specs[@]}"; do
    IFS="|" read -r component group max_tasks memory_gib <<< "${spec}"
    if [[ "${group}" == "fast" ]]; then
        datasets=("${fast_datasets[@]}")
    else
        datasets=("${slow_datasets[@]}")
    fi

    component_slug="${component,,}"
    result_name="GEAR-Comp-Native-${component}"
    log_dir="${output_root}/${result_name}"
    command_file="${submission_dir}/commands-${component_slug}-${group}.txt"
    submission_file="${submission_dir}/submit-${component_slug}-${group}.sub"
    mkdir -p "${log_dir}"
    : > "${command_file}"

    command_count=0
    for dataset in "${datasets[@]}"; do
        prediction_dir="${results_root}/${result_name}/Predictions/${dataset}"
        train_file="${prediction_dir}/trainResample0.csv"
        test_file="${prediction_dir}/testResample0.csv"
        if native_result_complete \
            "${result_name}" "${train_file}" "${test_file}"; then
            echo "Skipping complete native result: ${result_name}/${dataset}"
            continue
        fi

        command=(
            "${python_path}" -u "${worker}"
            "${data_root}" "${results_root}" "${component}" "${dataset}"
            --resample-id 0
        )
        printf -v command_line '%q ' "${command[@]}"
        printf '%s> %q 2>&1\n' \
            "${command_line}" \
            "${log_dir}/output-${dataset}-${run_id}.txt" \
            >> "${command_file}"
        command_count=$((command_count + 1))
    done

    if ((command_count == 0)); then
        echo "No incomplete cells for ${component}/${group}; no job submitted."
        continue
    fi
    cpu_count=${command_count}
    if ((cpu_count > max_tasks)); then
        cpu_count=${max_tasks}
    fi

    cat > "${submission_file}" <<EOF
#!/bin/bash
#SBATCH --mail-type=${mail}
#SBATCH --mail-user=${mailto}
#SBATCH --job-name=eeg-gear-native-${component_slug}-${group}
#SBATCH --partition=${queue}
#SBATCH --time=${max_time}
#SBATCH --output=${submission_dir}/%A-${component_slug}-${group}.out
#SBATCH --error=${submission_dir}/%A-${component_slug}-${group}.err
#SBATCH --nodes=1
#SBATCH --ntasks=${cpu_count}
#SBATCH --mem-per-cpu=${memory_gib}G

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

echo "Component:        ${component}"
echo "Dataset group:    ${group}"
echo "Native mechanism: enabled"
echo "Host:             \$(hostname)"
echo "Slurm job ID:     \${SLURM_JOB_ID}"
echo "Allocated tasks:  \${SLURM_NTASKS}"
echo "Command count:    ${command_count}"
echo "Memory per task:  ${memory_gib} GiB"
echo "Results:          ${results_root}/${result_name}"
echo "Command file:     ${command_file}"
staskfarm "${command_file}"
EOF

    while true; do
        num_jobs=$(squeue --noheader --user="${username}" --partition="${queue}" \
            --states=RUNNING,PENDING | wc -l)
        if ((num_jobs < max_num_submitted)); then
            break
        fi
        echo "Waiting 60 seconds: ${num_jobs} jobs are running or pending."
        sleep 60
    done

    sbatch_output=$(sbatch "${submission_file}")
    echo "${sbatch_output}"
    echo "Submitted ${component}/${group}: ${command_count} commands, " \
        "${cpu_count} tasks, ${memory_gib} GiB/task."
    submitted=$((submitted + 1))
    rm -f "${submission_file}"
done

echo "Submitted ${submitted} native GEAR-Comp task-farm job(s)."
echo "Submission records: ${submission_dir}"
