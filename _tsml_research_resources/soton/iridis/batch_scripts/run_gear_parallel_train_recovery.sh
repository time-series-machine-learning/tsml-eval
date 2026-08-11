#!/bin/bash

set -euo pipefail

# Recover archive pipeline train files whose serial ten-fold CV runs exceed the
# 60-hour Iridis batch limit. Each dataset gets one node and one
# independent process per deterministic CV fold. The folds are combined into a
# standard tsml trainResample0.csv after staskfarm has completed.

run_set="${RUN_SET:-all}"
run_set="${run_set,,}"

username="ajb2u23"
queue="batch"
max_time="60:00:00"
mail="NONE"
mailto="${username}@soton.ac.uk"
local_path="/iridisfs/home/${username}"

tsml_eval_dir="${local_path}/Code/tsml-eval"
aeon_dir="${local_path}/Code/aeon"
python_path="/home/${username}/.conda/envs/tsml-eval/bin/python"
worker="${tsml_eval_dir}/tsml_eval/_wip/eeg_parallel_train_cv.py"

data_dir="${local_path}/Data/EEG"
results_dir="${local_path}/Results/ChannelSelectionPipeline"
recovery_dir="${results_dir}/parallel-train-recovery"
numba_cache_dir="${local_path}/Code/.cache/tsml-eval"

n_splits=10
resample_id=0

# classifier|dataset|GiB per fold
#
# MatchingPennies STC previously peaked below 1 GiB for a full fit. Six GiB
# includes data and Python overhead. SitStand TDE peaked at 34.7 GiB; 50 GiB
# per fold gives headroom while keeping ten folds below a red node's memory.
all_specs=(
    "GEAR-Comp-STC|MatchingPennies|6"
    "GEAR-Comp-TDE|SitStand|50"
)
# DrCIF is opt-in while the existing serial recovery is active. Thirty GiB
# gives headroom over the 21.2 GiB peak measured for its slow archive fits.
drcif_spec="GEAR-Comp-DrCIF|LongIntervalTask|30"
bpso_specs=(
    "BPSO-Arsenal|LongIntervalTask|20"
    "BPSO-DrCIF|LongIntervalTask|35"
    "BPSO-STC|LongIntervalTask|12"
    "BPSO-TDE|LongIntervalTask|60"
)

case "${run_set}" in
    all)
        specs=("${all_specs[@]}")
        ;;
    stc)
        specs=("${all_specs[0]}")
        ;;
    tde)
        specs=("${all_specs[1]}")
        ;;
    drcif)
        specs=("${drcif_spec}")
        ;;
    bpso)
        specs=("${bpso_specs[@]}")
        ;;
    *)
        echo "ERROR: unknown RUN_SET '${run_set}'." >&2
        echo "Use all, stc, tde, drcif, or bpso." >&2
        exit 2
        ;;
esac

for path in "${python_path}" "${worker}"; do
    if [[ ! -e "${path}" ]]; then
        echo "ERROR: required path does not exist: ${path}" >&2
        exit 1
    fi
done
for path in "${data_dir}" "${results_dir}" "${tsml_eval_dir}/.git" "${aeon_dir}/.git"; do
    if [[ ! -e "${path}" ]]; then
        echo "ERROR: required directory does not exist: ${path}" >&2
        exit 1
    fi
done

mkdir -p "${recovery_dir}" "${numba_cache_dir}"

tsml_eval_commit=$(git -C "${tsml_eval_dir}" rev-parse HEAD)
aeon_commit=$(git -C "${aeon_dir}" rev-parse HEAD)
run_id=$(date +%Y%m%d%H%M%S)
submission_dir="${recovery_dir}/batch-submissions/${run_id}"
mkdir -p "${submission_dir}"

submitted=0
for spec in "${specs[@]}"; do
    IFS="|" read -r classifier dataset memory_gib <<< "${spec}"
    train_file="${results_dir}/${classifier}/Predictions/${dataset}/trainResample${resample_id}.csv"

    if [[ -s "${train_file}" ]]; then
        echo "Skipping complete result: ${classifier}/${dataset}"
        continue
    fi

    slug="${classifier}-${dataset}"
    slug="${slug,,}"
    partial_dir="${recovery_dir}/${classifier}/${dataset}/resample${resample_id}"
    output_dir="${recovery_dir}/output/${classifier}/${dataset}"
    command_file="${submission_dir}/commands-${slug}.txt"
    submission_file="${submission_dir}/submit-${slug}.sub"
    mkdir -p "${partial_dir}" "${output_dir}"
    : > "${command_file}"

    command_count=0
    for ((fold = 0; fold < n_splits; fold++)); do
        partial_file="${partial_dir}/fold${fold}.npz"
        if [[ -s "${partial_file}" ]]; then
            echo "Retaining completed partial: ${classifier}/${dataset}/fold${fold}"
            continue
        fi

        command=(
            "${python_path}" -u "${worker}" fold
            "${data_dir}" "${results_dir}" "${classifier}" "${dataset}"
            "${partial_dir}" "${fold}"
            --resample-id "${resample_id}"
            --n-splits "${n_splits}"
        )
        printf -v command_line '%q ' "${command[@]}"
        printf '%s> %q 2>&1\n' \
            "${command_line}" \
            "${output_dir}/fold${fold}-${run_id}.txt" \
            >> "${command_file}"
        command_count=$((command_count + 1))
    done

    cpu_count=${command_count}
    if ((cpu_count == 0)); then
        cpu_count=1
    fi

    cat > "${submission_file}" <<EOF
#!/bin/bash
#SBATCH --mail-type=${mail}
#SBATCH --mail-user=${mailto}
#SBATCH --job-name=eeg-gear-cv-${slug}
#SBATCH --partition=${queue}
#SBATCH --time=${max_time}
#SBATCH --output=${submission_dir}/%A-${slug}.out
#SBATCH --error=${submission_dir}/%A-${slug}.err
#SBATCH --nodes=1
#SBATCH --ntasks=${cpu_count}
#SBATCH --mem-per-cpu=${memory_gib}G

. /etc/profile
set -euo pipefail

cd "${tsml_eval_dir}"
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

echo "Classifier:       ${classifier}"
echo "Dataset:          ${dataset}"
echo "CV folds:         ${n_splits}"
echo "Outstanding:      ${command_count}"
echo "Allocated tasks:  \${SLURM_NTASKS}"
echo "Memory per task:  ${memory_gib} GiB"
echo "Partial results:  ${partial_dir}"

"${python_path}" - <<'PY'
import aeon
import aeon_neuro
import tsml_eval

print("aeon:      ", aeon.__file__)
print("aeon-neuro:", aeon_neuro.__file__)
print("tsml-eval: ", tsml_eval.__file__)
PY

if (( ${command_count} > 0 )); then
    staskfarm "${command_file}"
fi

"${python_path}" -u "${worker}" combine \
    "${data_dir}" "${results_dir}" "${classifier}" "${dataset}" \
    "${partial_dir}" \
    --resample-id "${resample_id}" \
    --n-splits "${n_splits}"
EOF

    sbatch "${submission_file}"
    submitted=$((submitted + 1))
    echo "Submitted ${classifier}/${dataset}:"
    echo "  Fold processes:  ${command_count}"
    echo "  Requested tasks: ${cpu_count}"
    echo "  Memory per task: ${memory_gib} GiB"
    echo "  Maximum memory:  $((cpu_count * memory_gib)) GiB"
done

echo "Submitted ${submitted} parallel train-recovery job(s)."
echo "Submission files: ${submission_dir}"
