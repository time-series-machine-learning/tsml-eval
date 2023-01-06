#!/bin/bash
# CHECK:
#   datasets (list of problems)
#   results_dir (where to check/write results),
#   for clusterer in (the clusterers we are running)

# While reading from them is fine, please dont write anything to the default directories in this script

# To use GPU resources you need to be given access (gpu qos), which involves emailing hpc.admin@uea.ac.uk
# Ask Tony or on slack, and read the GPU section in https://my.uea.ac.uk/divisions/it-and-computing-services/service-catalogue/research-it-services/hpc/ada-cluster/using-ada/jobs

# Start and end for resamples
max_folds=30
start_fold=1

# To avoid dumping 1000s of jobs in the queue we have a higher level queue
max_num_submitted=100

# Queue options are https://my.uea.ac.uk/divisions/it-and-computing-services/service-catalogue/research-it-services/hpc/ada-cluster/using-ada
# Make sure GPU jobs are on one of the "gpu-" queues, .sub file qos may need to change for ones other than "gpu-rtx6000-2"
queue="gpu-rtx6000-2"

# Enter your username and email here
username="ajb"
mail="NONE"
mailto=$username"@uea.ac.uk"

# MB for jobs, max is maybe 64000 before you need to use huge memory queue. Do not use more than you need
max_memory=8000

# Max allowable is 7 days - 168 hours
max_time="168:00:00"

# Start point for the script i.e. 3 datasets, 3 clusterers = 9 jobs to submit, start_point=5 will skip to job 5
start_point=1

# Datasets to use and directory of data files. Default is Tony's work space, all should be able to read these. Change if you want to use different data or lists
data_dir="/gpfs/home/ajb/Data/"
datasets="/gpfs/home/ajb/DataSetLists/TSC_112_2019.txt"

# Put your home directory here
local_path="/gpfs/home/$username/"

# Results and output file write location. Change these to reflect your own file structure
results_dir=$local_path"ClusteringResults/results/"
out_dir=$local_path"ClusteringResults/output/"

# The python script we are running
script_file_path=$local_path"Code/tsml-eval/tsml_eval/experiments/clustering_experiments.py"

# Environment name, change accordingly, for set up, see https://hackmd.io/ds5IEK3oQAquD4c6AP2xzQ
# Separate environments for GPU (Python 3.8) and CPU (Python 3.10) are recommended
env_name="tsml-eval-gpu"

# generate a results file for the test data as well as train
generate_test_files="true"

# If set for true, looks for <problem><fold>_TRAIN.ts file. This is useful for running tsml resamples
predefined_folds="false"

# List valid clusterers e.g ???
# See set_clusterer for aliases
count=0
while read dataset; do
for clusterer in ???
do

# Dont change anything after here for regular runs

# This is the loop to keep from dumping everything in the queue which is maintained around max_num_submitted jobs
num_pending=$(squeue -u ${username} --format="%10i %15P %20j %10u %10t %10M %10D %20R" -r | awk '{print $5, $2}' | grep "PD ${queue}" | wc -l)
num_running=$(squeue -u ${username} --format="%10i %15P %20j %10u %10t %10M %10D %20R" -r | awk '{print $5, $2}' | grep "R ${queue}" | wc -l)
while [ "$((num_pending+num_running))" -ge "${max_num_submitted}" ]
do
    echo Waiting 90s, $((num_pending+num_running)) currently submitted on ${queue}, user-defined max is ${max_num_submitted}
	sleep 90
	num_pending=$(squeue -u ${username} --format="%10i %15P %20j %10u %10t %10M %10D %20R" -r | awk '{print $5, $2}' | grep "PD ${queue}" | wc -l)
	num_running=$(squeue -u ${username} --format="%10i %15P %20j %10u %10t %10M %10D %20R" -r | awk '{print $5, $2}' | grep "R ${queue}" | wc -l)
done

# Skip to the script start point
((count++))
if ((count>=start_point)); then

mkdir -p ${out_dir}${clusterer}/${dataset}/

# This skips jobs which have test/train files already written to the results directory. Only looks for Resamples, not Folds (old file name)
array_jobs=""
for (( i=start_fold-1; i<max_folds; i++ ))
do
    if [ -f "${results_dir}${clusterer}/Predictions/${dataset}/trainResample${i}.csv" ]; then
        if [ "${generate_test_files}" == "true" ] && ! [ -f "${results_dir}${clusterer}/Predictions/${dataset}/testResample${i}.csv" ]; then
            array_jobs="${array_jobs}${array_jobs:+,}$((i + 1))"
        fi
    else
        array_jobs="${array_jobs}${array_jobs:+,}$((i + 1))"
    fi
done

if [ "${array_jobs}" != "" ]; then

# This creates the scrip to run the job based on the info above
echo "#!/bin/bash
#SBATCH --qos=gpu-rtx
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mail-type=${mail}
#SBATCH --mail-user=${mailto}
#SBATCH -p ${queue}
#SBATCH -t ${max_time}
#SBATCH --job-name=${clusterer}${dataset}
#SBATCH --array=${array_jobs}
#SBATCH --mem=${max_memory}M
#SBATCH -o ${out_dir}${clusterer}/${dataset}/%A-%a.out
#SBATCH -e ${out_dir}${clusterer}/${dataset}/%A-%a.err

. /etc/profile

module add python/anaconda/2019.10/3.7
module add cuda/10.2.89
module add cudnn/7.6.5
source activate $env_name
export LD_LIBRARY_PATH=\$LD_LIBRARY_PATH:/gpfs/home/${username}/.conda/envs/${env_name}/lib/


# Input args to the default clustering_experiments are in main method of
# https://github.com/time-series-machine-learning/tsml-eval/blob/main/tsml_eval/experiments/clustering_experiments.py
python -u ${script_file_path} ${data_dir} ${results_dir} ${clusterer} ${dataset} \$SLURM_ARRAY_TASK_ID ${generate_test_files} ${predefined_folds}"  > generatedFileGPU.sub

echo ${count} ${clusterer}/${dataset}

sbatch < generatedFileGPU.sub

else
    echo ${count} ${clusterer}/${dataset} has finished all required resamples, skipping
fi

fi
done
done < ${datasets}

echo Finished submitting jobs
