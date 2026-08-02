#!/bin/bash

set -euo pipefail

# Run OpenCloseFist leave-one-subject-out component experiments on one Iridis
# batch node. The default "fast" set runs Arsenal and STC first and writes both
# train and test predictions for later HC2-from-file construction.

# ==============================================================================
# Run set: override with RUN_SET=fast, drcif, tde, mrhydra,
# tselect-fast, tselect-drcif, tselect-tde, detachrocket-fast,
# detachrocket-drcif, detachrocket-tde, gmarv3-fast, gmarv3-drcif,
# gmarv3-tde, gear-comp-fast, gear-comp-drcif, gear-comp-tde, or gear-auto.
# ==============================================================================

# fast:  Arsenal and STC, 6 GiB/process, up to 96 concurrent processes.
# drcif: DrCIF only, 22 GiB/process, up to 28 concurrent processes.
# tde:   TDE only, 30 GiB/process, up to 20 concurrent processes.
# mrhydra: MrHydra only, 19 GiB/process, up to 32 concurrent processes.
# The selector sets use the corresponding flat transform-component pipelines
# and retain train files so HC2 can later be reconstructed by fold.
run_set="${RUN_SET:-fast}"
run_set="${run_set,,}"

# ==============================================================================
# Experiment and Slurm configuration
# ==============================================================================

dataset="OpenCloseFist"
result_dataset="${dataset}LOSO"

# Subject identifiers in the published OpenCloseFist ID files are 0..104.
first_subject=0
last_subject=104

max_num_submitted=200
queue="batch"
max_time="60:00:00"

username="ajb2u23"
mail="NONE"
mailto="${username}@soton.ac.uk"

local_path="/iridisfs/home/${username}"

case "${run_set}" in
    fast)
        # factory-name|result-directory-name
        component_specs=(
            "Arsenal|Arsenal"
            "Full-STC|STC"
        )
        max_cpus_to_use=96
        memory_per_cpu_gib=6
        ;;
    drcif)
        component_specs=(
            "DrCIF|DrCIF"
        )
        max_cpus_to_use=28
        memory_per_cpu_gib=22
        ;;
    tde)
        component_specs=(
            "TDE|TDE"
        )
        max_cpus_to_use=20
        memory_per_cpu_gib=30
        ;;
    mrhydra)
        component_specs=(
            "MrHydra|MrHydra"
        )
        max_cpus_to_use=32
        memory_per_cpu_gib=19
        ;;
    tselect-fast)
        component_specs=(
            "TSelect-Arsenal|TSelect-Arsenal"
            "TSelect-STC|TSelect-STC"
        )
        max_cpus_to_use=96
        memory_per_cpu_gib=6
        ;;
    tselect-drcif)
        component_specs=(
            "TSelect-DrCIF|TSelect-DrCIF"
        )
        max_cpus_to_use=60
        memory_per_cpu_gib=10
        ;;
    tselect-tde)
        component_specs=(
            "TSelect-TDE|TSelect-TDE"
        )
        max_cpus_to_use=20
        memory_per_cpu_gib=30
        ;;
    detachrocket-fast)
        component_specs=(
            "DetachRocket-Arsenal|DetachRocket-Arsenal"
            "DetachRocket-STC|DetachRocket-STC"
        )
        max_cpus_to_use=30
        memory_per_cpu_gib=20
        ;;
    detachrocket-drcif)
        component_specs=(
            "DetachRocket-DrCIF|DetachRocket-DrCIF"
        )
        max_cpus_to_use=30
        memory_per_cpu_gib=20
        ;;
    detachrocket-tde)
        component_specs=(
            "DetachRocket-TDE|DetachRocket-TDE"
        )
        max_cpus_to_use=20
        memory_per_cpu_gib=30
        ;;
    gmarv3-fast)
        component_specs=(
            "GMARv3-Arsenal|GMARv3-Arsenal"
            "GMARv3-STC|GMARv3-STC"
        )
        max_cpus_to_use=76
        memory_per_cpu_gib=8
        ;;
    gmarv3-drcif)
        component_specs=(
            "GMARv3-DrCIF|GMARv3-DrCIF"
        )
        max_cpus_to_use=60
        memory_per_cpu_gib=10
        ;;
    gmarv3-tde)
        component_specs=(
            "GMARv3-TDE|GMARv3-TDE"
        )
        max_cpus_to_use=20
        memory_per_cpu_gib=30
        ;;
    gear-comp-fast)
        component_specs=(
            "GEAR-Comp-Arsenal|GEAR-Comp-Arsenal"
            "GEAR-Comp-STC|GEAR-Comp-STC"
        )
        # The completed LOSO folds peaked at 4.14 GiB for Arsenal and
        # 1.94 GiB for STC. Eight GiB leaves a substantial safety margin while
        # allowing 76 independent folds to use one 620-GiB node efficiently.
        max_cpus_to_use=76
        memory_per_cpu_gib=8
        ;;
    gear-comp-drcif)
        component_specs=(
            "GEAR-Comp-DrCIF|GEAR-Comp-DrCIF"
        )
        max_cpus_to_use=30
        memory_per_cpu_gib=20
        ;;
    gear-comp-tde)
        component_specs=(
            "GEAR-Comp-TDE|GEAR-Comp-TDE"
        )
        max_cpus_to_use=20
        memory_per_cpu_gib=30
        ;;
    gear-auto)
        component_specs=(
            "GEAR-Auto-HC2|GEAR-Auto-HC2"
        )
        max_cpus_to_use=20
        memory_per_cpu_gib=30
        ;;
    *)
        echo "ERROR: unknown run_set: ${run_set}"
        echo "Use fast, drcif, tde, mrhydra, tselect-fast, tselect-drcif,"
        echo "tselect-tde, detachrocket-fast, detachrocket-drcif,"
        echo "detachrocket-tde, gmarv3-fast, gmarv3-drcif, gmarv3-tde,"
        echo "gear-comp-fast, gear-comp-drcif, gear-comp-tde, or gear-auto."
        exit 1
        ;;
esac

memory_per_cpu="${memory_per_cpu_gib}G"
job_name="eeg-ocf-loso-${run_set}"

# ==============================================================================
# Repository, environment, data, and result locations
# ==============================================================================

tsml_eval_dir="${local_path}/Code/tsml-eval"
aeon_dir="${local_path}/Code/aeon"

script_file_path="${tsml_eval_dir}/tsml_eval/_wip/eeg_loso.py"

env_name="tsml-eval"
python_path="/home/${username}/.conda/envs/${env_name}/bin/python"

data_dir="${local_path}/Data/EEG"
dataset_dir="${data_dir}/${dataset}"

results_dir="${local_path}/Results/ChannelSelectionLOSO"
out_dir="${results_dir}/output"
numba_cache_dir="${local_path}/Code/.cache/${env_name}"

# ==============================================================================
# Validate configuration
# ==============================================================================

if [[ ! -x "${python_path}" ]]; then
    echo "ERROR: Python executable not found or not executable:"
    echo "  ${python_path}"
    exit 1
fi
if [[ ! -f "${script_file_path}" ]]; then
    echo "ERROR: LOSO experiment script not found:"
    echo "  ${script_file_path}"
    exit 1
fi
for repository in "${tsml_eval_dir}" "${aeon_dir}"; do
    if [[ ! -d "${repository}/.git" ]]; then
        echo "ERROR: Git checkout not found:"
        echo "  ${repository}"
        exit 1
    fi
done
for suffix in TRAIN TEST; do
    data_file="${dataset_dir}/${dataset}_${suffix}.ts"
    id_file="${dataset_dir}/${dataset}_id_${suffix}.txt"
    if [[ ! -s "${data_file}" || ! -s "${id_file}" ]]; then
        echo "ERROR: missing or empty ${suffix} data/ID files:"
        echo "  ${data_file}"
        echo "  ${id_file}"
        exit 1
    fi
done
if ((first_subject < 0 || last_subject < first_subject)); then
    echo "ERROR: invalid subject range ${first_subject}..${last_subject}."
    exit 1
fi
if ((max_cpus_to_use < 1 || max_cpus_to_use > 192)); then
    echo "ERROR: max_cpus_to_use must be between 1 and 192."
    exit 1
fi
if ((max_cpus_to_use * memory_per_cpu_gib > 620)); then
    echo "ERROR: requested task memory exceeds the 620 GiB safety ceiling."
    echo "  CPUs:           ${max_cpus_to_use}"
    echo "  Memory per CPU: ${memory_per_cpu}"
    exit 1
fi

mkdir -p "${results_dir}" "${out_dir}" "${numba_cache_dir}"

tsml_eval_commit=$(git -C "${tsml_eval_dir}" rev-parse HEAD)
aeon_commit=$(git -C "${aeon_dir}" rev-parse HEAD)

echo "Run set:           ${run_set}"
echo "Dataset:           ${dataset}"
echo "Result dataset:    ${result_dataset}"
echo "Subjects:          ${first_subject}..${last_subject}"
echo "Components:        ${#component_specs[@]}"
echo "Maximum CPUs:      ${max_cpus_to_use}"
echo "Memory per CPU:    ${memory_per_cpu}"
echo "Maximum task RAM:  $((max_cpus_to_use * memory_per_cpu_gib)) GiB"
echo "Results:           ${results_dir}"
echo "tsml-eval commit:  ${tsml_eval_commit}"
echo "aeon commit:       ${aeon_commit}"
echo

# Use the exact source checkouts for submission-time validation as well as in
# the generated Slurm job. Without this, a login shell can resolve ``aeon`` as
# an incomplete namespace package and leave ``aeon.__file__`` as None.
unset PYTHONHOME
export PYTHONNOUSERSITE=1
export PYTHONPATH="${aeon_dir}:${tsml_eval_dir}"

# Validate IDs and classifier factories before creating the command list.
"${python_path}" "${script_file_path}" \
    "${data_dir}" \
    "${results_dir}" \
    "Arsenal" \
    "${first_subject}" \
    --dataset "${dataset}" \
    --classifier-name "Arsenal" \
    --validate-only

"${python_path}" - "${component_specs[@]}" <<'PY'
import sys

from tsml_eval.experiments import get_classifier_by_name

for specification in sys.argv[1:]:
    factory_name, result_name = specification.split("|", maxsplit=1)
    classifier = get_classifier_by_name(factory_name, random_state=0, n_jobs=1)
    print(
        f"{result_name}: factory={factory_name}, "
        f"class={classifier.__class__.__name__}"
    )
PY

# ==============================================================================
# Generate commands for incomplete subject/component pairs
# ==============================================================================

run_id=$(date +%Y%m%d%H%M%S)
batch_id="${run_id}-${dataset}-loso-${run_set}"
submission_dir="${results_dir}/batch-submissions/${run_id}"
command_file="${submission_dir}/generatedCommandList-${batch_id}.txt"
submission_file="${submission_dir}/generatedSubmissionFile-${batch_id}.sub"

mkdir -p "${submission_dir}"
: > "${command_file}"

cmd_count=0

# Interleave components by subject so each task-farm wave has a balanced mix.
for ((subject = first_subject; subject <= last_subject; subject++)); do
    for specification in "${component_specs[@]}"; do
        IFS='|' read -r factory_name result_name <<< "${specification}"

        test_file="${results_dir}/${result_name}/Predictions/${result_dataset}/testResample${subject}.csv"
        train_file="${results_dir}/${result_name}/Predictions/${result_dataset}/trainResample${subject}.csv"

        # HC2 components need train and test files for later from-file
        # construction. MrHydra and the final GEAR-Auto HC2 pipeline require
        # only test predictions, so avoid writing an unnecessary train file.
        require_train_file=true
        if [[ "${result_name}" == "MrHydra" ||
              "${result_name}" == "GEAR-Auto-HC2" ]]; then
            require_train_file=false
        fi

        if [[ -s "${test_file}" ]] &&
            { [[ "${require_train_file}" == false ]] ||
              [[ -s "${train_file}" ]]; }; then
            continue
        fi

        component_out_dir="${out_dir}/${result_name}"
        mkdir -p "${component_out_dir}"
        experiment_output="${component_out_dir}/output-${result_dataset}-${subject}-${batch_id}.txt"

        command=(
            "${python_path}"
            -u
            "${script_file_path}"
            "${data_dir}"
            "${results_dir}"
            "${factory_name}"
            "${subject}"
            --dataset "${dataset}"
            --classifier-name "${result_name}"
        )
        if [[ "${require_train_file}" == false ]]; then
            command+=("--no-train-file")
        fi

        printf -v command_line '%q ' "${command[@]}"
        printf '%s> %q 2>&1\n' \
            "${command_line}" \
            "${experiment_output}" \
            >> "${command_file}"

        cmd_count=$((cmd_count + 1))
    done
done

if ((cmd_count == 0)); then
    echo "All ${run_set} LOSO component results already exist; nothing submitted."
    exit 0
fi

cpu_count=${cmd_count}
if ((cpu_count > max_cpus_to_use)); then
    cpu_count=${max_cpus_to_use}
fi

# Recheck the actual request when fewer commands remain.
if ((cpu_count * memory_per_cpu_gib > 620)); then
    echo "ERROR: actual task memory exceeds the 620 GiB safety ceiling."
    exit 1
fi

# ==============================================================================
# Create and submit the one-node task farm
# ==============================================================================

preflight_factories=()
for specification in "${component_specs[@]}"; do
    IFS='|' read -r factory_name _ <<< "${specification}"
    preflight_factories+=("${factory_name}")
done
printf -v preflight_factory_args '%q ' "${preflight_factories[@]}"

cat > "${submission_file}" <<EOF
#!/bin/bash
#SBATCH --mail-type=${mail}
#SBATCH --mail-user=${mailto}
#SBATCH --job-name=${job_name}
#SBATCH --partition=${queue}
#SBATCH --time=${max_time}
#SBATCH --output=${submission_dir}/%A-${batch_id}.out
#SBATCH --error=${submission_dir}/%A-${batch_id}.err
#SBATCH --nodes=1
#SBATCH --ntasks=${cpu_count}
#SBATCH --mem-per-cpu=${memory_per_cpu}

. /etc/profile
set -e

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
export MEMRECORD_INTERVAL=30
mkdir -p "\${NUMBA_CACHE_DIR}"

current_tsml_eval_commit=\$(git -C "${tsml_eval_dir}" rev-parse HEAD)
current_aeon_commit=\$(git -C "${aeon_dir}" rev-parse HEAD)

if [[ "\${current_tsml_eval_commit}" != "${tsml_eval_commit}" ]]; then
    echo "ERROR: tsml-eval changed after submission."
    echo "Expected: ${tsml_eval_commit}"
    echo "Current:  \${current_tsml_eval_commit}"
    exit 1
fi
if [[ "\${current_aeon_commit}" != "${aeon_commit}" ]]; then
    echo "ERROR: aeon changed after submission."
    echo "Expected: ${aeon_commit}"
    echo "Current:  \${current_aeon_commit}"
    exit 1
fi

echo "Run set:           ${run_set}"
echo "Dataset:           ${dataset}"
echo "Result dataset:    ${result_dataset}"
echo "Host:              \$(hostname)"
echo "Slurm job ID:      \${SLURM_JOB_ID}"
echo "Allocated tasks:   \${SLURM_NTASKS}"
echo "Command count:     ${cmd_count}"
echo "Memory per CPU:    ${memory_per_cpu}"
echo "Maximum task RAM:  $((cpu_count * memory_per_cpu_gib)) GiB"
echo "Python:            ${python_path}"
echo "tsml-eval commit:  \${current_tsml_eval_commit}"
echo "aeon commit:       \${current_aeon_commit}"
echo "Command file:      ${command_file}"
echo

"${python_path}" - ${preflight_factory_args} <<'PY'
import sys

import aeon
import tsml_eval
from tsml_eval.experiments import get_classifier_by_name

print("Python:   ", sys.executable)
print("aeon:     ", aeon.__file__)
print("tsml-eval:", tsml_eval.__file__)
for name in sys.argv[1:]:
    classifier = get_classifier_by_name(name, random_state=0, n_jobs=1)
    print(f"{name}: {classifier.__class__.__name__}")
print("Import and classifier-factory checks succeeded")
PY

staskfarm "${command_file}"
EOF

while true; do
    num_jobs=$(
        squeue \
            --noheader \
            --user="${username}" \
            --partition="${queue}" \
            --states=RUNNING,PENDING |
            wc -l
    )
    if ((num_jobs < max_num_submitted)); then
        break
    fi
    echo "Waiting 60 seconds: ${num_jobs} jobs are running or pending."
    sleep 60
done

if ! sbatch_output=$(sbatch "${submission_file}"); then
    echo "ERROR: failed to submit ${submission_file}" >&2
    exit 1
fi

echo "${sbatch_output}"
echo "Submitted OpenCloseFist LOSO ${run_set} task farm:"
echo "  Outstanding commands: ${cmd_count}"
echo "  Requested CPUs:       ${cpu_count}"
echo "  Memory per CPU:       ${memory_per_cpu}"
echo "  Maximum task RAM:     $((cpu_count * memory_per_cpu_gib)) GiB"
echo "  Wall time:            ${max_time}"
echo "  Results:              ${results_dir}"
echo "  Submission directory: ${submission_dir}"

# Slurm has copied the submission script. Keep the command list because
# staskfarm reads it when the allocation starts.
rm -f "${submission_file}"
