#!/bin/bash
# CHECK:
#   datasets (list of problems)
#   results_dir (where to check/write results),
#   for classifier in (the classifiers we are running)

# While reading is fine, please dont write anything to the default directories in this script

# Start and end for resamples
max_folds=30
start_fold=1

# To avoid dumping 1000s of jobs in the queue we have a higher level queue
max_num_submitted=100

# Queue options are https://my.uea.ac.uk/divisions/it-and-computing-services/service-catalogue/research-it-services/hpc/ada-cluster/using-ada
queue="compute-64-512"

# Enter your username and email here
username="ajb"
mail="NONE"
mailto=$username"@uea.ac.uk"

# MB for jobs, increase incrementally and try not to use more than you need. If you need hundreds of GB consider the huge memory queue.
max_memory=8000

# Max allowable is 7 days - 168 hours
max_time="168:00:00"

# Start point for the script i.e. 3 datasets, 3 classifiers = 9 jobs to submit, start_point=5 will skip to job 5
start_point=1

# Datasets to use and directory of data files. Default is Tony's work space, all should be able to read these. Change if you want to use different data or lists
data_dir="/gpfs/home/ajb/Data/"
datasets="/gpfs/home/ajb/DataSetLists/TSC_112_2019.txt"

# Put your home directory here
local_path="/gpfs/home/$username/"

# Results and output file write location. Change these to reflect your own file structure
results_dir=$local_path"ClassificationResults/results/"
out_dir=$local_path"ClassificationResults/output/"

# The python script we are running
script_file_path=$local_path"Code/tsml-eval/tsml_eval/experiments/classification_experiments.py"

# Environment name, change accordingly, for set up, see https://hackmd.io/ds5IEK3oQAquD4c6AP2xzQ
# Separate environments for GPU (Python 3.8) and CPU (Python 3.10) are recommended
env_name="tsml-eval"

# You can add extra arguments here. See tsml_eval/utils/experiments.py parse_args
# You will have to add any variable to the python call close to the bottom of the script

# Generating train folds is usually slower, set to empty string unless you need them
generate_train_files="-tr"

# If set to -pr, looks for <problem><resample>_TRAIN.ts files. This is useful for running tsml-java resamples
predefined_folds=""

# List valid classifiers e.g DrCIF TDE Arsenal STC MUSE ROCKET Mini-ROCKET Multi-ROCKET
# See set_classifier for aliases
count=0
while read dataset; do
for classifier in ROCKET DrCIF
do

# Dont change anything after here for regular runs

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

mkdir -p ${out_dir}${classifier}/${dataset}/

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
mkdir -p ${out_dir}${classifier}/${dataset}/

# This creates the scrip to run the job based on the info above
echo "#!/bin/bash
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
source activate $env_name

# Input args to the default classification_experiments are in main method of
# https://github.com/time-series-machine-learning/tsml-eval/blob/main/tsml_eval/experiments/classification_experiments.py
python -u ${script_file_path} ${data_dir} ${results_dir} ${classifier} ${dataset} \$((\$SLURM_ARRAY_TASK_ID - 1)) ${generate_train_files} ${predefined_folds}"  > generatedFile.sub

echo ${count} ${classifier}/${dataset}

sbatch < generatedFile.sub

else
    echo ${count} ${classifier}/${dataset} has finished all required resamples, skipping
fi

fi
done
done < ${datasets}

echo Finished submitting jobs
