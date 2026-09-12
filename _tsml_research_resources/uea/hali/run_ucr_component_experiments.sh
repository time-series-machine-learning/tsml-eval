#!/bin/bash
# Run one HC2 component classifier over the 112 UCR datasets on Hali, with train files.
#
#   ./run_ucr_component_experiments.sh <classifier> <category> [results_subdir]
#
#   ./run_ucr_component_experiments.sh Arsenal   ConvolutionBased
#   ./run_ucr_component_experiments.sh DrCIF-500 IntervalBased
#   ./run_ucr_component_experiments.sh Arsenal   ConvolutionBased Arsenal-sept
#
# Why this exists. The stored UCR component results were each generated on a different
# date from a different commit of ajb/hc2, and the stored HC2 predates all of them, so an
# ensemble rebuilt from those files is not the HC2 that was run. Regenerating a component
# on current code makes the comparison meaningful. Two components are known to be stale:
#
#   Arsenal    ran 07/26; 9 commits have touched _arsenal.py/_rocket.py since, and 9 more
#              landed between the HC2 run and it, including the ajb/arsenal merge with
#              the member-weighting bug fix.
#   DrCIF-500  does not exist for UCR at all. HC2 builds DrCIF with n_estimators=500
#              while the stored UCR DrCIF used the class default of 200.
#
# Train files (-tr) are always requested: the CAWPE weights come from the component train
# accuracy estimates, so a test file alone cannot be used to rebuild an ensemble.
#
# Submits one Slurm array per dataset, one array element per resample. Existing results
# are skipped, so the script is safe to rerun to fill gaps. A resample whose test file
# exists but whose train file is missing is resubmitted, since it is unusable otherwise.
#
# Set results_subdir to write somewhere other than <category>/<classifier>, e.g. to keep
# a rerun alongside the existing results instead of skipping them.

set -u

classifier="${1:-}"
category="${2:-}"
results_subdir="${3:-$classifier}"

if [ -z "$classifier" ] || [ -z "$category" ]; then
    echo "usage: $0 <classifier> <category> [results_subdir]"
    echo "   e.g: $0 Arsenal ConvolutionBased"
    echo
    echo "Set WAIT_FOR to a job-name pattern to hold submission until the jobs already"
    echo "queued for that pattern have finished, e.g."
    echo "   WAIT_FOR='DrCIF-500_|Arsenal-sept_' $0 Arsenal-fixedweight ConvolutionBased"
    echo "Submission then begins only once no running or pending job of yours matches."
    exit 1
fi

# Optionally hold until earlier work has drained. Slurm dependencies are per job, and
# the earlier runs are hundreds of separate arrays, so waiting on the queue itself is
# simpler and does not need their job ids to be recorded anywhere.
if [ -n "${WAIT_FOR:-}" ]; then
    echo "Waiting for running or pending jobs matching '${WAIT_FOR}' to finish."
    while true; do
        remaining=$(squeue --noheader --array --user="$USER" \
            --states=RUNNING,PENDING --format='%j' 2>/dev/null \
            | grep -cE "${WAIT_FOR}")
        if [ "${remaining:-0}" -eq 0 ]; then
            echo "$(date '+%F %T') none left, submitting."
            break
        fi
        echo "$(date '+%F %T') ${remaining} task(s) still queued, checking again in 10 minutes."
        sleep 600
    done
fi

# ---------------------------------------------------------------------------
# Settings. Check the paths marked VERIFY against your Hali layout before running.
# ---------------------------------------------------------------------------

username="ajb"
mailto="$username@uea.ac.uk"
mail="NONE"                                 # NONE, BEGIN, END, FAIL, REQUEUE, ALL

account="cmp"
partition="compute"
qos="uea-core-default"
max_time="7-00:00:00"
max_memory=8000                             # MB per task

start_fold=1
max_folds=30

local_path="/gpfs/home/${username}"
repo_dir="${local_path}/Code/tsml-eval"
data_dir="${local_path}/Data/UCR/"                                                  # VERIFY
datasets="${local_path}/DataSetLists/UCR.txt"
results_dir="${local_path}/Results/UCR/${category}/"
out_dir="${local_path}/Output/UCR/"

script_file_path="${repo_dir}/tsml_eval/experiments/classification_experiments.py"

# Hali conda environment.
module_name="python/anaconda/2024.10/3.12.7"
conda_sh="/gpfs/software/hali/python/anaconda/2024.10/etc/profile.d/conda.sh"
env_name="tsml-eval"
numba_cache_dir="${local_path}/Code/.cache/numba/tsml-eval"

# ---------------------------------------------------------------------------

if [ ! -f "$datasets" ]; then
    echo "Dataset list not found: $datasets"
    exit 1
fi
if [ ! -f "$script_file_path" ]; then
    echo "Experiment script not found: $script_file_path"
    exit 1
fi

# Record which commit produced these results. The whole reason for this rerun is that
# the stored results do not say what code made them.
provenance="${results_dir}${results_subdir}/PROVENANCE.txt"
mkdir -p "${results_dir}${results_subdir}" "${out_dir}${results_subdir}/"
{
    echo "classifier:  ${classifier}"
    echo "submitted:   $(date -u +'%Y-%m-%dT%H:%M:%SZ')"
    echo "tsml-eval:   $(git -C "${repo_dir}" rev-parse --short HEAD 2>/dev/null) \
($(git -C "${repo_dir}" rev-parse --abbrev-ref HEAD 2>/dev/null))"
    echo "aeon:        $(python -c 'import aeon; print(aeon.__version__, aeon.__file__)' 2>/dev/null)"
} >> "${provenance}"
echo "Wrote provenance to ${provenance}"

count=0
total_tasks=0
while read -r dataset; do
    [ -z "$dataset" ] && continue

    array_jobs=""
    for (( i=start_fold-1; i<max_folds; i++ )); do
        test_file="${results_dir}${results_subdir}/Predictions/${dataset}/testResample${i}.csv"
        train_file="${results_dir}${results_subdir}/Predictions/${dataset}/trainResample${i}.csv"
        # A test file alone is not enough: without the train file there is no accuracy
        # estimate, so the resample cannot contribute a CAWPE weight.
        if [ -f "$test_file" ] && [ -f "$train_file" ]; then
            continue
        fi
        if [ -z "${array_jobs}" ]; then
            array_jobs="$((i + 1))"
        else
            array_jobs="${array_jobs},$((i + 1))"
        fi
    done

    [ -z "${array_jobs}" ] && continue

    mkdir -p "${out_dir}${results_subdir}/${dataset}/"

    echo "#!/bin/bash
#SBATCH --account=${account}
#SBATCH --qos=${qos}
#SBATCH --mail-type=${mail}
#SBATCH --mail-user=${mailto}
#SBATCH -p ${partition}
#SBATCH -t ${max_time}
#SBATCH --job-name=${results_subdir}_${dataset}
#SBATCH --array=${array_jobs}
#SBATCH --mem=${max_memory}M
#SBATCH -o ${out_dir}${results_subdir}/${dataset}/%A-%a.out
#SBATCH -e ${out_dir}${results_subdir}/${dataset}/%A-%a.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1

# One core per task, so keep every numerical library single threaded. Rocket-family
# transforms were only recently made thread safe for ensemble use, so this also keeps
# the run comparable with the single-threaded stored results.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=\"\"
export NUMBA_CACHE_DIR=${numba_cache_dir}
mkdir -p \$NUMBA_CACHE_DIR

module add ${module_name}
source ${conda_sh}
conda activate ${env_name}

export PYTHONPATH=${repo_dir}:\$PYTHONPATH

python -u ${script_file_path} ${data_dir} ${results_dir} ${classifier} ${dataset} \\
    \$((\$SLURM_ARRAY_TASK_ID - 1)) -tr" > "generatedFile_${results_subdir}.sub"

    n=$(awk -F, '{print NF}' <<< "${array_jobs}")
    echo "${count} ${results_subdir}/${dataset}: ${n} resamples [${array_jobs}]"
    sbatch < "generatedFile_${results_subdir}.sub"
    count=$((count + 1))
    total_tasks=$((total_tasks + n))

done < "${datasets}"

rm -f "generatedFile_${results_subdir}.sub"
echo "Submitted ${total_tasks} tasks across ${count} datasets for ${classifier}."
