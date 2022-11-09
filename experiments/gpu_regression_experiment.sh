#!/bin/bash
# CHECK: datasets (list of problems), results_dir (where to check/write results), for classifier in MLPClassifier
# also  queue="gpu-rtx6000-2" (GPU) queue="compute-64-512" (normal),
# SBATCH --qos=gpu-rtx (gpu) or --qos=ht (normal job)

# Start and end for resamples
max_folds=30
start_fold=1
# To avoid dumping 1000s of jobs in the queue we have a higher level queue
max_num_submitted=500
# Queue options are https://my.uea.ac.uk/divisions/it-and-computing-services/service-catalogue/research-it-services/hpc/ada-cluster/using-ada
# By default use "compute-64-512". For deep learning classifiers, use "gpu-rtx6000-2"
queue="gpu-rtx6000-2"
# Enter your username and email here
username="ajb"
mail="NONE"
mailto="ajb@uea.ac.uk"
# MB for jobs, max is maybe 64000 before you need to use huge memory queue
max_memory=75000

# Max allowable is 7 days - 168 hours
max_time="168:00:00"

# Start point for the script i.e. 3 datasets, 3 classifiers = 9 jobs to submit, start_point=5 will skip to job 5
start_point=1

# CHECK: Datasets to use and directory of data files. Default is Tony's work space, all should be able to read these. Change if you want to use different data or lists
data_dir="/gpfs/home/ajb/Data/"
datasets="/gpfs/home/ajb/DataSetLists/temp.txt"
datasets="/gpfs/home/ajb/DataSetLists/Regression.txt"


# Put your home directory here
local_path="/gpfs/home/ajb/"

# CHECK Change these to reflect your own file structure
results_dir=$local_path"temp/"
results_dir=$local_path"RegressionResults/sktime/"
out_dir=$local_path"Code/output/multivariate/"
script_file_path=$local_path"Code/tsml-estimator-evaluation/tsml_estimator_evaluation/experiments/regression_experiments.py"

# Environment, to set up, see https://hackmd.io/ds5IEK3oQAquD4c6AP2xzQ
env_name="eval"

# Generating train folds is usually slower, set to false unless you need them
generate_train_files="false"
# If set for true, looks for <problem><fold>_TRAIN.ts file. This is useful for running tsml resamples
predefined_folds="false"

# List valid regressors e.g CNNRegressor, TapNetRegressor
count=0
while read dataset; do
for classifier in CNNRegressor TapNetRegressor
do

# DO NOT CHANGE FROM HERE

# This is the loop to keep from dumping everything in the queue which is maintained around max_num_submitted jobs
#num_pending=$(squeue -u ${username} --format="%10i %15P %20j %10u %10t %10M %10D %20R" -r | awk '{print $5, $2}' | grep "PD ${queue}" | wc -l)
#num_running=$(squeue -u ${username} --format="%10i %15P %20j %10u %10t %10M %10D %20R" -r | awk '{print $5, $2}' | grep "R ${queue}" | wc -l)
#while [ "$((num_pending+num_running))" -ge "${max_num_submitted}" ]
#do
#    echo Waiting 60s, $((num_pending+num_running)) currently submitted on ${queue}, user-defined max is ${max_num_submitted}
#	sleep 60
#	num_pending=$(squeue -u ${username} --format="%10i %15P %20j %10u %10t %10M %10D %20R" -r | awk '{print $5, $2}' | grep "PD ${queue}" | wc -l)
#	num_running=$(squeue -u ${username} --format="%10i %15P %20j %10u %10t %10M %10D %20R" -r | awk '{print $5, $2}' | grep "R ${queue}" | wc -l)
#done

# Skip to the script start point
((count++))
if ((count>=start_point)); then

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

# This creates the scrip to run the job based on the info above
echo "#!/bin/bash
#SBATCH --qos=gpu-rtx-reserved
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

module add python/anaconda/2020.11/3.8
module add cudnn/8.2.0
#source /gpfs/software/ada/python/anaconda/2019.10/3.7/etc/profile.d/conda.sh
source activate $env_name
export LD_LIBRARY_PATH=\$LD_LIBRARY_PATH:/gpfs/home/${username}/.conda/envs/${env_name}/lib
#export PYTHONPATH=$(pwd)
# Input args to classification_experiments are in main method of
# https://github.com/uea-machine-learning/estimator-evaluation/blob/main/sktime_estimator_evaluation/experiments/classification_experiments.py
python -u ${script_file_path} ${data_dir} ${results_dir} ${classifier} ${dataset} \$SLURM_ARRAY_TASK_ID ${generate_train_files} ${predefined_folds}"  > generatedFileGPU.sub

echo ${count} ${classifier}/${dataset}

sbatch < generatedFileGPU.sub

else
    echo ${count} ${classifier}/${dataset} has finished all required resamples, skipping
fi

fi
done
done < ${datasets}

echo Finished submitting jobs
