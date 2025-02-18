#!/bin/bash

max_folds=10
start_fold=1
max_num_submitted=400
queue="batch"
n_tasks_per_node=40
username="mbm1g23"
mail="NONE"
mailto=$username"@soton.ac.uk"
max_time="60:00:00"
start_point=1
local_path="/mainfs/home/$username/"
data_dir=$local_path"scratch/UnivariateDataTSC/"
dataset_list=$local_path"DatasetLists/datasetsUV112Batch/"
results_dir=$local_path"ClassificationResults/results/"
out_dir=$local_path"ClassificationResults/output/"
script_file_path=$local_path"tsml-eval/tsml_eval/experiments/classification_experiments.py"
env_name="eval-py11"
generate_train_files="false"
predefined_folds="false"
normalise_data="false"
classifiers_to_run="Arsenal STC DrCIF-500 TDE"


# ======================================================================================
# 	Experiment configuration end
# ======================================================================================


# Set to -tr to generate test files
generate_train_files=$([ "${generate_train_files,,}" == "true" ] && echo "-tr" || echo "")

# Set to -pr to use predefined folds
predefined_folds=$([ "${predefined_folds,,}" == "true" ] && echo "-pr" || echo "")

# Set to -rn to normalise data
normalise_data=$([ "${normalise_data,,}" == "true" ] && echo "-rn" || echo "")

mkdir -p "${out_dir}/"

# This creates the submission file to run and does clean up
submit_jobs () {

echo "#!/bin/bash
#SBATCH --mail-type=${mail}
#SBATCH --mail-user=${mailto}
#SBATCH --job-name=batch-${dt}
#SBATCH -p ${queue}
#SBATCH -t ${max_time}
#SBATCH -o ${out_dir}/%A-${dt}.out
#SBATCH -e ${out_dir}/%A-${dt}.err
#SBATCH --nodes=1
#SBATCH --ntasks=${cmdCount}

. /etc/profile

module load anaconda/py3.10
source activate $env_name

staskfarm ${out_dir}/generatedCommandList-${dt}.txt" > generatedSubmissionFile-${dt}.sub

echo "At experiment ${expCount}, ${totalCount} jobs submitted total"

sbatch < generatedSubmissionFile-${dt}.sub

rm generatedSubmissionFile-${dt}.sub

}

totalCount=0
expCount=0

# turn a directory of files into a list
if [[ -d $dataset_list ]]; then
    file_names=""
    for file in ${dataset_list}/*; do
        file_names="$file_names$dataset_list$(basename "$file") "
    done
    dataset_list=$file_names
fi

for dataset_file in $dataset_list; do

echo "Dataset list ${dataset_file}"

for classifier in $classifiers_to_run; do

# we use time for unique names
sleep 1
cmdCount=0
dt=$(date +%Y%m%d%H%M%S)

while read dataset; do

# Skip to the script start point
((expCount++))
if ((expCount>=start_point)); then

# This finds the resamples to run and skips jobs which have test/train files already written to the results directory.
resamples_to_run=""
for (( i=start_fold-1; i<max_folds; i++ ))
do
    if [ -f "${results_dir}${classifier}/Predictions/${dataset}/testResample${i}.csv" ]; then
        if [ "${generate_train_files}" == "true" ] && ! [ -f "${results_dir}${classifier}/Predictions/${dataset}/trainResample${i}.csv" ]; then
            resamples_to_run="${resamples_to_run}${i} "
        fi
    else
        resamples_to_run="${resamples_to_run}${i} "
    fi
done

for resample in $resamples_to_run; do

# add to the command list if
if ((cmdCount>=n_tasks_per_node)); then
    submit_jobs

    # This is the loop to stop you from dumping everything in the queue at once, see max_num_submitted jobs
    num_jobs=$(squeue -u ${username} --format="%20P %5t" -r | awk '{print $2, $1}' | grep -e "R ${queue}" -e "PD ${queue}" | wc -l)
    while [ "${num_jobs}" -ge "${max_num_submitted}" ]
    do
        echo Waiting 60s, "${num_jobs}" currently submitted on ${queue}, user-defined max is ${max_num_submitted}
        sleep 60
        num_jobs=$(squeue -u ${username} --format="%20P %5t" -r | awk '{print $2, $1}' | grep -e "R ${queue}" -e "PD ${queue}" | wc -l)
    done

    sleep 1
    cmdCount=0
    dt=$(date +%Y%m%d%H%M%S)
fi

# Input args to the default classification_experiments are in main method of
# https://github.com/time-series-machine-learning/tsml-eval/blob/main/tsml_eval/experiments/classification_experiments.py
echo "python -u ${script_file_path} ${data_dir} ${results_dir} ${classifier} ${dataset} ${resample} ${generate_train_files} ${predefined_folds} ${normalise_data}" >> ${out_dir}/generatedCommandList-${dt}.txt

((cmdCount++))
((totalCount++))

done
fi
done < ${dataset_file}

if ((cmdCount>0)); then
    # final submit for this dataset list
    submit_jobs
fi

done
done

echo Finished submitting jobs
