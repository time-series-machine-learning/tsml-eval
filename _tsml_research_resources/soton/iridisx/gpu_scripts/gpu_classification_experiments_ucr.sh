#!/bin/bash
# GPU classification experiments on IridisX for the UCR univariate archive.
# Check and edit all options before the first run!
# While reading is fine, please dont write anything to the default directories in this script
#
# Slurm settings below are confirmed from sinfo/sacctmgr on IridisX.

# Start and end for resamples. 30 is the full resample protocol, 5 is a useful first pass
max_folds=5
start_fold=1

# To avoid hitting the cluster queue limit we have a higher level queue.
# Keep this low for GPU jobs, there are far fewer GPUs than CPU cores
max_num_submitted=12

# IridisX GPU partitions, confirmed from sinfo. Do NOT use "batch", it does not exist
# on IridisX, and "amd" is the general CPU partition with no GPUs.
#   partition        nodes  gres              timelimit   notes
#   swarm_a100       5      gpu:a100swarm:4   5-00:00:00  ECS/ORC staff and PGR only
#   swarm_h100       2      gpu:h100swarm:8   5-00:00:00  ECS/ORC staff and PGR only
#   a100             12     gpu:a100:2        2-12:00:00  open to all
#   scavenger_4a100  5      gpu:a100swarm:4     12:00:00  PREEMPTIBLE
#   scavenger_8h100  2      gpu:h100swarm:8     12:00:00  PREEMPTIBLE
#   mi300x           1      gpu:mi300x:8      2-12:00:00  AMD, will NOT run CUDA
# Preemption is tolerable here: a killed resample leaves no results file, so rerunning
# this script resubmits exactly the missing work
queue="swarm_a100"

# The gres TYPE differs by partition and must match it exactly. swarm_a100 uses
# "a100swarm", the open a100 partition uses plain "a100". Getting this wrong is a
# submission error, not a silent fallback
gres="gpu:a100swarm:1"

# No account or QoS is needed for the default association. If a swarm submission is
# rejected for QoS reasons, set qos to the relevant one (ecsa100 for swarm_a100,
# ecsh100 for swarm_h100). Leave empty to omit the directive
account=""
qos=""

# Enter your username and email here
username="ajb2u23"
mail="NONE"
mailto="$username@soton.ac.uk"

# MB for jobs, increase incrementally and try not to use more than you need.
# Deep learners need more host memory than the CPU classifiers.
# swarm_a100 nodes have 48 cores and 900000 MB usable, default 17700 MB/core
max_memory=16000

# CPUs per job for the data pipeline. The GPU does the training, this is for loading.
# 4 of 48 cores alongside 1 of 4 GPUs leaves the node usable by others
cpus_per_task=4

# swarm_a100 and swarm_h100 allow up to 5-00:00:00, a100 up to 2-12:00:00, and the
# scavenger partitions only 12:00:00. Request what you need, accurate walltimes are
# scheduled sooner
max_time="2-00:00:00"

# Start point for the script i.e. 3 datasets, 3 classifiers = 9 jobs to submit, start_point=5 will skip to job 5
start_point=1

# /home and /scratch are global Storage Scale mounts on login and compute nodes.
# /iridisfs/home and /iridisfs/scratch are the same directories
local_path="/home/$username/"

# Datasets to use and directory of data files.
# UnivariateClassification112-UCR2018Clean.txt is the usual list, the full 128 archive
# is UnivariateClassification128-UCR2018.txt. Both are in ../../../dataset_lists/
data_dir="$local_path/Data/"
datasets="$local_path/DataSetLists/UnivariateClassification112-UCR2018Clean.txt"

# Results and output file write location. Change these to reflect your own file structure
results_dir="$local_path/ClassificationResults/results/"
out_dir="$local_path/ClassificationResults/output/"

# The python script we are running. This is a separate checkout to the CPU one, on the
# ajb/hc2 branch. It must match the checkout that tsml-eval-gpu is pip installed from
script_file_path="$local_path/Code/tsml-eval-gpu/tsml_eval/experiments/classification_experiments.py"

# GPU conda environment. Keep this separate from the CPU "tsml-eval" environment,
# see iridisx_python.md for how to build it
env_name="tsml-eval-gpu"

# conda/python3 is the only conda module on IridisX
conda_module="conda/python3"

# Classifiers to loop over. Must be separated by a space
# Deep learning options in tsml_eval/experiments/_get_classifier.py:
#   CNN FCN MLP Encoder ResNet SingleInception InceptionTime H-InceptionTime
#   LITETime IndividualLITE DisjointCNN
classifiers_to_run="InceptionTime LITETime"

# You can add extra arguments here. See tsml_eval/utils/arguments.py parse_args
# You will have to add any variable to the python call close to the bottom of the script
# and possibly to the options handling below

# generate a results file for the train data as well as test, usually slower
generate_train_files="false"

# If set for true, looks for <problem><fold>_TRAIN.ts file. This is useful for running tsml-java resamples
predefined_folds="false"

# Normalise data before fit/predict
normalise_data="false"

# ======================================================================================
# 	Experiment configuration end
# ======================================================================================

# Set to -tr to generate test files
generate_train_files=$([ "${generate_train_files,,}" == "true" ] && echo "-tr" || echo "")

# Set to -pr to use predefined folds
predefined_folds=$([ "${predefined_folds,,}" == "true" ] && echo "-pr" || echo "")

# Set to -rn to normalise data
normalise_data=$([ "${normalise_data,,}" == "true" ] && echo "-rn" || echo "")

# Optional directives. These fall back to a comment rather than an empty string,
# as a blank line would end Slurm's #SBATCH parsing and drop every later directive
account_directive=$([ -n "${account}" ] && echo "#SBATCH --account=${account}" || echo "# no account set")
qos_directive=$([ -n "${qos}" ] && echo "#SBATCH --qos=${qos}" || echo "# no qos set")

count=0
while read dataset; do
for classifier in $classifiers_to_run; do

# Skip to the script start point
((count++))
if ((count>=start_point)); then

# This is the loop to keep from dumping everything in the queue which is maintained around max_num_submitted jobs
num_jobs=$(squeue -u ${username} --format="%20P %5t" -r | awk '{print $2, $1}' | grep -e "R ${queue}" -e "PD ${queue}" | wc -l)
while [ "${num_jobs}" -ge "${max_num_submitted}" ]
do
    echo Waiting 60s, ${num_jobs} currently submitted on ${queue}, user-defined max is ${max_num_submitted}
    sleep 60
    num_jobs=$(squeue -u ${username} --format="%20P %5t" -r | awk '{print $2, $1}' | grep -e "R ${queue}" -e "PD ${queue}" | wc -l)
done

mkdir -p "${out_dir}${classifier}/${dataset}/"

# This skips jobs which have test/train files already written to the results directory. Only looks for Resamples, not Folds (old file name)
array_jobs=""
for (( i=start_fold-1; i<max_folds; i++ ))
do
    if [ -f "${results_dir}${classifier}/Predictions/${dataset}/testResample${i}.csv" ]; then
        if [ "${generate_train_files}" == "-tr" ] && ! [ -f "${results_dir}${classifier}/Predictions/${dataset}/trainResample${i}.csv" ]; then
            array_jobs="${array_jobs}${array_jobs:+,}$((i + 1))"
        fi
    else
        array_jobs="${array_jobs}${array_jobs:+,}$((i + 1))"
    fi
done

if [ "${array_jobs}" != "" ]; then

# This creates the script to run the job based on the info above
echo "#!/bin/bash
#SBATCH --gres=${gres}
${account_directive}
${qos_directive}
#SBATCH --mail-type=${mail}
#SBATCH --mail-user=${mailto}
#SBATCH -p ${queue}
#SBATCH -t ${max_time}
#SBATCH --job-name=${classifier}${dataset}
#SBATCH --array=${array_jobs}
#SBATCH --mem=${max_memory}M
#SBATCH --cpus-per-task=${cpus_per_task}
#SBATCH -o ${out_dir}/${classifier}/${dataset}/%A-%a.out
#SBATCH -e ${out_dir}/${classifier}/${dataset}/%A-%a.err
#SBATCH --nodes=1

. /etc/profile

module load ${conda_module}
source activate ${env_name}

# Do not unset CUDA_VISIBLE_DEVICES here, Slurm sets it for the allocated GPU
export TF_NUM_INTEROP_THREADS=${cpus_per_task}
export TF_NUM_INTRAOP_THREADS=${cpus_per_task}
export PYTHONUNBUFFERED=1

nvidia-smi

# Input args to the default classification_experiments are in main method of
# https://github.com/time-series-machine-learning/tsml-eval/blob/main/tsml_eval/experiments/classification_experiments.py
python -u ${script_file_path} ${data_dir} ${results_dir} ${classifier} ${dataset} \$((\$SLURM_ARRAY_TASK_ID - 1)) ${generate_train_files} ${predefined_folds} ${normalise_data}" > generatedFile.sub

echo "${count} ${classifier}/${dataset}"

sbatch < generatedFile.sub

else
    echo "${count} ${classifier}/${dataset}" has finished all required resamples, skipping
fi

fi
done
done < ${datasets}

echo Finished submitting jobs
