#!/bin/bash
# Check and edit all options before the first run!
# While reading is fine, please dont write anything to the default directories in this script

# Start and end for resamples
max_folds=10
start_fold=1

# To avoid hitting the cluster queue limit we have a higher level queue
max_num_submitted=900

# Queue options are https://sotonac.sharepoint.com/teams/HPCCommunityWiki/SitePages/Iridis%205%20Job-submission-and-Limits-Quotas.aspx
queue="batch"

# The number of tasks/threads to use in each job. 40 is the number of cores on batch nodes
n_tasks_per_node=40

# Enter your username and email here
username="ajb2u23"
mail="NONE"
mailto=$username"@soton.ac.uk"

# Max allowable is 60 hours
max_time="60:00:00"

# Start point for the script i.e. 3 datasets, 3 classifiers = 9 experiments to submit, start_point=5 will skip to job 5
start_point=1

# Put your home directory here
local_path="/mainfs/home/$username/"

# Datasets to use and directory of data files. This can either be a text file or directory of text files
# Separate text files will not run jobs of the same dataset in the same node. This is good to keep large and small datasets separate
data_dir="$local_path/Data/"
dataset_list="$local_path/DataSetLists/ClassificationBatch/"

# Results and output file write location. Change these to reflect your own file structure
results_dir="$local_path/ClassificationResults/results/"
out_dir="$local_path/ClassificationResults/output/"

# The python script we are running
script_file_path="$local_path/tsml-eval/tsml_eval/experiments/classification_experiments.py"

# Environment name, change accordingly, for set up, see https://github.com/time-series-machine-learning/tsml-eval/blob/main/_tsml_research_resources/soton/iridis/iridis_python.md
# Separate environments for GPU and CPU are recommended
env_name="eval-py11"

# Classifiers to loop over. Must be separated by a space. Different classifiers will not run in the same node
# See list of potential classifiers in set_classifier
classifiers_to_run="ROCKET DrCIF"

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

# This creates the submission file to run and does clean up
submit_jobs () {

echo "#!/bin/bash
#SBATCH --mail-type=${mail}
#SBATCH --mail-user=${mailto}
#SBATCH --job-name=batch-${dt}
#SBATCH -p ${queue}
#SBATCH -t ${max_time}
#SBATCH -o ${out_dir}/${classifier}/%A-${dt}.out
#SBATCH -e ${out_dir}/${classifier}/%A-${dt}.err
#SBATCH --nodes=1
#SBATCH --ntasks=${cmdCount}

. /etc/profile

module load anaconda/py3.10
source activate $env_name

staskfarm ${out_dir}/${classifier}/generatedCommandList-${dt}.txt" > generatedSubmissionFile-${dt}.sub

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

mkdir -p "${out_dir}/${classifier}/"

# create a new command list for each classifier and dataset list
# we use time for unique names
sleep 1
cmdCount=0
dt=$(date +%Y%m%d%H%M%S)

while read dataset; do

# Skip to the script start point
((expCount++))
if ((expCount>=start_point)); then

# This finds the resamples to run and skips jobs which have test/train files already written to the results directory.
# This can result in uneven sized command lists
resamples_to_run=""
for (( i=start_fold-1; i<max_folds; i++ ))
do
    if [ -f "${results_dir}${classifier}/Predictions/${dataset}/testResample${i}.csv" ]; then
        if [ "${generate_train_files}" == "-tr" ] && ! [ -f "${results_dir}${classifier}/Predictions/${dataset}/trainResample${i}.csv" ]; then
            resamples_to_run="${resamples_to_run}${i} "
        fi
    else
        resamples_to_run="${resamples_to_run}${i} "
    fi
done

for resample in $resamples_to_run; do

# submit the command list if
if ((cmdCount>=n_tasks_per_node)); then
    submit_jobs

    # This is the loop to stop you from dumping everything in the queue at once, see max_num_submitted
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
echo "python -u ${script_file_path} ${data_dir} ${results_dir} ${classifier} ${dataset} ${resample} ${generate_train_files} ${predefined_folds} ${normalise_data}" >> ${out_dir}/${classifier}/generatedCommandList-${dt}.txt

((cmdCount++))
((totalCount++))

done
fi
done < ${dataset_file}

if ((cmdCount>0)); then
    # final submit for this dataset list and classifier
    submit_jobs
fi

done
done

echo Finished submitting jobs
