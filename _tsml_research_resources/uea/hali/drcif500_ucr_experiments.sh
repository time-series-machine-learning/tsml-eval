#!/bin/bash
# Run DrCIF-500 on the 112 UCR datasets on Hali, with train files.
#
# HC2 builds its interval component as DrCIFClassifier(n_estimators=500), but the
# standalone DrCIF results on the UCR archive were generated with the class default of
# 200 trees. Ensembles rebuilt from those component files are therefore not HC2. This
# script generates the missing DrCIF-500 results so the rebuild can use the same
# component HC2 does.
#
# Train files (-tr) are required: the CAWPE weights come from the component train
# accuracy estimates, so a test file alone is useless for rebuilding an ensemble.
#
# Submits one Slurm array per dataset, each array element being one resample. Existing
# results are skipped, so the script is safe to rerun to fill gaps.

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

classifier="DrCIF-500"                      # tsml-eval alias, n_estimators=500
category="IntervalBased"

local_path="/gpfs/home/${username}"
repo_dir="${local_path}/Code/tsml-eval"
data_dir="${local_path}/Data/UCR/"                                    # VERIFY
datasets="${local_path}/DataSetLists/UnivariateClassification112-UCR2018Clean.txt"  # VERIFY
results_dir="${local_path}/Results/UCR/${category}/"
out_dir="${local_path}/Output/UCR/"

script_file_path="${repo_dir}/tsml_eval/experiments/classification_experiments.py"

# Hali conda environment.
module_name="python/anaconda/2024.10/3.12.7"
conda_sh="/gpfs/software/hali/python/anaconda/2024.10/etc/profile.d/conda.sh"
env_name="tsml-eval"
numba_cache_dir="${local_path}/Code/.cache/numba/tsml-eval"

generate_train_files="true"                 # -tr, required for ensemble rebuilds
predefined_folds="false"
normalise_data="false"

# ---------------------------------------------------------------------------

if [ ! -f "$datasets" ]; then
    echo "Dataset list not found: $datasets"
    exit 1
fi

mkdir -p "${out_dir}${classifier}/"

count=0
submitted=0
while read dataset; do
    [ -z "$dataset" ] && continue

    # Build the list of resamples that have no test result yet. Rerunning the script
    # only submits the gaps.
    array_jobs=""
    for (( i=start_fold-1; i<max_folds; i++ )); do
        if [ -f "${results_dir}${classifier}/Predictions/${dataset}/testResample${i}.csv" ]; then
            if [ "${generate_train_files}" == "true" ] && \
               [ ! -f "${results_dir}${classifier}/Predictions/${dataset}/trainResample${i}.csv" ]; then
                : # test present but train missing, still needs running
            else
                continue
            fi
        fi
        if [ -z "${array_jobs}" ]; then
            array_jobs="$((i + 1))"
        else
            array_jobs="${array_jobs},$((i + 1))"
        fi
    done

    if [ -z "${array_jobs}" ]; then
        continue
    fi

    mkdir -p "${out_dir}${classifier}/${dataset}/"

    echo "#!/bin/bash
#SBATCH --account=${account}
#SBATCH --qos=${qos}
#SBATCH --mail-type=${mail}
#SBATCH --mail-user=${mailto}
#SBATCH -p ${partition}
#SBATCH -t ${max_time}
#SBATCH --job-name=${classifier}_${dataset}
#SBATCH --array=${array_jobs}
#SBATCH --mem=${max_memory}M
#SBATCH -o ${out_dir}${classifier}/${dataset}/%A-%a.out
#SBATCH -e ${out_dir}${classifier}/${dataset}/%A-%a.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1

# One core per task, so keep every numerical library single threaded.
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
    \$((\$SLURM_ARRAY_TASK_ID - 1)) -tr" > generatedFile.sub

    echo "${count} ${classifier}/${dataset} resamples ${array_jobs}"
    sbatch < generatedFile.sub
    submitted=$((submitted + 1))
    count=$((count + 1))

done < "${datasets}"

rm -f generatedFile.sub
echo "Submitted arrays for ${submitted} datasets."
