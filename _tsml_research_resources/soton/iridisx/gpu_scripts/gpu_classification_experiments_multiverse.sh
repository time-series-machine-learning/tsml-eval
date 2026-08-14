#!/bin/bash
# GPU classification experiments on IridisX for the Multiverse multivariate archive.
# Check and edit all options before the first run!
# While reading is fine, please dont write anything to the default directories in this script
#
# Settings marked TODO-IRIDISX have not been confirmed on IridisX yet. Run
# iridisx_probe.sh on the login node and fill them in before the first submission.

# Start and end for resamples. 30 is the full resample protocol, 1 is the default-resample pass
max_folds=1
start_fold=1

# To avoid hitting the cluster queue limit we have a higher level queue.
# Keep this low for GPU jobs, there are far fewer GPUs than CPU cores
max_num_submitted=12

# IridisX GPU partitions. Do NOT use "batch", it does not exist on IridisX, and "amd"
# is the general AMD CPU partition with no GPUs.
#   swarm_a100      5 nodes, 4x A100 80GB NV-Linked,  ECS/ORC staff and PGR only
#   swarm_h100      2 nodes, 8x H100 80GB NV-Switch,  ECS/ORC staff and PGR only
#   a100           13 nodes, 2x A100 80GB NV-Linked,  open to all
#   scavenger_4a100 idle swarm_a100 nodes, PREEMPTIBLE, 8 nodes max per user
#   scavenger_8h100 idle swarm_h100 nodes, PREEMPTIBLE, 8 nodes max per user
# Preemption is tolerable here: a killed resample leaves no results file, so rerunning
# this script resubmits exactly the missing work
queue="swarm_a100"

# TODO-IRIDISX: confirm the exact gres type string with "sinfo -o '%P %G' -e".
# Requesting the type rather than a bare "gpu:1" keeps jobs on the intended hardware
gres="gpu:a100:1"

# TODO-IRIDISX: if IridisX requires an account and/or QoS, set them here.
# Leave empty to omit the directive (Iridis 6 did not require either)
account=""
qos=""

# Enter your username and email here
username="ajb2u23"
mail="NONE"
mailto="$username@soton.ac.uk"

# MB for jobs, increase incrementally and try not to use more than you need.
# The Multiverse problems are larger than UCR, the Hali CPU passes needed 32-64GB.
# swarm_a100 nodes have 48 cores and 900000 MB usable, default 17700 MB/core
max_memory=32000

# CPUs per job for the data pipeline. The GPU does the training, this is for loading.
# 4 of 48 cores alongside 1 of 4 GPUs leaves the node usable by others
cpus_per_task=4

# TODO-IRIDISX: confirm the maximum walltime for the GPU partition ("sinfo -o '%P %l'")
max_time="60:00:00"

# Start point for the script i.e. 3 datasets, 3 classifiers = 9 jobs to submit, start_point=5 will skip to job 5
start_point=1

# IridisX exposes /home and /scratch globally through Storage Scale on both login and
# compute nodes. TODO-IRIDISX: confirm the exact user paths under these
local_path="/home/$username/"

# Datasets to use and directory of data files.
# Multivariate133Classification-MultiverseClean.txt is the usual list, the 66 problem
# subset is MultivariateClassification66-MultiverseMini.txt. Both are in ../../../dataset_lists/
data_dir="$local_path/Data/Multiverse/"
datasets="$local_path/DataSetLists/Multivariate133Classification-MultiverseClean.txt"

# Results and output file write location. Change these to reflect your own file structure
results_dir="$local_path/MultiverseResults/results/"
out_dir="$local_path/MultiverseResults/output/"

# The python script we are running
script_file_path="$local_path/tsml-eval/tsml_eval/experiments/classification_experiments.py"

# GPU conda environment. Keep this separate from the CPU "tsml-eval" environment,
# see iridisx_python.md for how to build it
env_name="tsml-eval-gpu"

# TODO-IRIDISX: confirm the conda module name with "module avail conda"
conda_module="conda/python3"

# Classifiers to loop over. Must be separated by a space
# LITETime-MV is the multivariate LITE variant used for the Multiverse paper.
# Other deep learning options in tsml_eval/experiments/_get_classifier.py:
#   CNN FCN MLP Encoder ResNet SingleInception InceptionTime H-InceptionTime
#   LITETime IndividualLITE DisjointCNN
classifiers_to_run="InceptionTime LITETime-MV"

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
