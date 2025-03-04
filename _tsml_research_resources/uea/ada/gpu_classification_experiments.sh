#!/bin/bash
# CHECK before each new run:
#   datasets (list of problems)
#   results_dir (where to check/write results)
#   classifiers_to_run (list of classifiers to run)
# While reading is fine, please dont write anything to the default directories in this script

# To use GPU resources you need to be given access (gpu qos), which involves emailing hpc.admin@uea.ac.uk
# Ask Tony or on slack, and read the GPU section in https://my.uea.ac.uk/divisions/it-and-computing-services/service-catalogue/research-it-services/hpc/ada-cluster/using-ada/jobs

# Start and end for resamples
max_folds=30
start_fold=1

# To avoid dumping 1000s of jobs in the queue we have a higher level queue
max_num_submitted=10

# Queue options are https://my.uea.ac.uk/divisions/it-and-computing-services/service-catalogue/research-it-services/hpc/ada-cluster/using-ada
# Make sure GPU jobs are on one of the "gpu-" queues, .sub file qos may need to change for ones other than "gpu-rtx6000-2"
queue="gpu-rtx6000-2"

# Enter your username and email here
username="ajb"
mail="NONE"
mailto="$username@uea.ac.uk"

# MB for jobs, this is less important for GPU jobs but if you swap nodes check how much is available and how many jobs can be submitted
max_memory=90000

# Max allowable is 7 days - 168 hours
max_time="168:00:00"

# Start point for the script i.e. 3 datasets, 3 classifiers = 9 jobs to submit, start_point=5 will skip to job 5
start_point=1

# Put your home directory here
local_path="/gpfs/home/$username/"

# Datasets to use and directory of data files. Default is Tony's work space, all should be able to read these. Change if you want to use different data or lists
data_dir="$local_path/Data/"
datasets="$local_path/DataSetLists/Classification.txt"


# Results and output file write location. Change these to reflect your own file structure
results_dir="$local_path/ClassificationResults/results/"
out_dir="$local_path/ClassificationResults/output/"

# The python script we are running
script_file_path="$local_path/tsml-eval/tsml_eval/experiments/classification_experiments.py"

# Environment name, change accordingly, for set up, see https://github.com/time-series-machine-learning/tsml-eval/blob/main/_tsml_research_resources/uea/ada/ada_python.md
# Separate environments for GPU and CPU are recommended
env_name="tsml-eval-gpu"

# Classifiers to loop over. Must be seperated by a space
# See list of potential classifiers in set_classifier
classifiers_to_run="CNNClassifier FCNClassifier"

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
# ======================================================================================
# Dont change anything under here (unless you want to change how the experiment
# is working)
# ======================================================================================
# ======================================================================================

# Set to -tr to generate test files
generate_train_files=$([ "${generate_train_files,,}" == "true" ] && echo "-tr" || echo "")

# Set to -pr to use predefined folds
predefined_folds=$([ "${predefined_folds,,}" == "true" ] && echo "-pr" || echo "")

# Set to -rn to normalise data
normalise_data=$([ "${normalise_data,,}" == "true" ] && echo "-rn" || echo "")

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
        if [ "${generate_train_files}" == "true" ] && ! [ -f "${results_dir}${classifier}/Predictions/${dataset}/trainResample${i}.csv" ]; then
            array_jobs="${array_jobs}${array_jobs:+,}$((i + 1))"
        fi
    else
        array_jobs="${array_jobs}${array_jobs:+,}$((i + 1))"
    fi
done

if [ "${array_jobs}" != "" ]; then

# This creates the scrip to run the job based on the info above
echo "#!/bin/bash
#SBATCH --qos=gpu-rtx #gpu-rtx-reserved
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mail-type=${mail}
#SBATCH --mail-user=${mailto}
#SBATCH -p ${queue}
#SBATCH -t ${max_time}
#SBATCH --job-name=${classifier}${dataset}
#SBATCH --array=${array_jobs}
#SBATCH --mem=${max_memory}M
#SBATCH -o ${out_dir}${classifier}/${dataset}/%A-%a.out
#SBATCH -e ${out_dir}${classifier}/${dataset}/%A-%a.err

. /etc/profile

module add python/anaconda/2019.10/3.7
module add cuda/10.2.89
module add cudnn/7.6.5
source activate $env_name
export LD_LIBRARY_PATH=\$LD_LIBRARY_PATH:/gpfs/home/${username}/.conda/envs/${env_name}/lib/

# Input args to the default classification_experiments are in main method of
# https://github.com/time-series-machine-learning/tsml-eval/blob/main/tsml_eval/experiments/classification_experiments.py
python -u ${script_file_path} ${data_dir} ${results_dir} ${classifier} ${dataset} \$((\$SLURM_ARRAY_TASK_ID - 1)) ${generate_train_files} ${predefined_folds} ${normalise_data}"  > generatedFileGPU.sub

echo "${count} ${classifier}/${dataset}"

sbatch < generatedFileGPU.sub

else
    echo "${count} ${classifier}/${dataset}" has finished all required resamples, skipping
fi

fi
done
done < ${datasets}

echo Finished submitting jobs
