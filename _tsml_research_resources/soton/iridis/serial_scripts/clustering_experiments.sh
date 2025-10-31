#!/bin/bash
# Check and edit all options before the first run!
# While reading is fine, please dont write anything to the default directories in this script

# Start and end for resamples
max_folds=30
start_fold=1

# To avoid hitting the cluster queue limit we have a higher level queue
max_num_submitted=100

# Queue options are https://sotonac.sharepoint.com/teams/HPCCommunityWiki/SitePages/Iridis%205%20Job-submission-and-Limits-Quotas.aspx
queue="batch"

# Enter your username and email here
username="ajb2u23"
mail="NONE"
mailto="$username@soton.ac.uk"

# MB for jobs, increase incrementally and try not to use more than you need. If you need hundreds of GB consider the huge memory queue
max_memory=8000

# Max allowable is 60 hours
max_time="60:00:00"

# Start point for the script i.e. 3 datasets, 3 clusterers = 9 jobs to submit, start_point=5 will skip to job 5
start_point=1

# Put your home directory here
local_path="/mainfs/home/$username/"

# Datasets to use and directory of data files. Default is Tony's work space, all should be able to read these. Change if you want to use different data or lists
data_dir="$local_path/Data/"
datasets="$local_path/DataSetLists/Clustering.txt"

# Results and output file write location. Change these to reflect your own file structure
results_dir="$local_path/ClusteringResults/results/"
out_dir="$local_path/ClusteringResults/output/"

# The python script we are running
script_file_path="$local_path/tsml-eval/tsml_eval/experiments/clustering_experiments.py"

# Environment name, change accordingly, for set up, see https://github.com/time-series-machine-learning/tsml-eval/blob/main/_tsml_research_resources/soton/iridis/iridis_python.md
# Separate environments for GPU and CPU are recommended
env_name="tsml-eval"

# Clusterers to loop over. Must be separated by a space
# See list of potential clusterers in set_clusterer
clusterers_to_run="kmedoids-squared kmedoids-euclidean"

# You can add extra arguments here. See tsml_eval/utils/arguments.py parse_args
# You will have to add any variable to the python call close to the bottom of the script
# and possibly to the options handling below

# generate a results file for the test data as well as train, usually slower
generate_test_files="true"

# If set for true, looks for <problem><fold>_TRAIN.ts file. This is useful for running tsml-java resamples
predefined_folds="false"

# Boolean on if to combine the test/train split
combine_test_train_split="false"

# Normalise data before fit/predict
normalise_data="true"

# ======================================================================================
# 	Experiment configuration end
# ======================================================================================

# Set to -te to generate test files
generate_test_files=$([ "${generate_test_files,,}" == "true" ] && echo "-te" || echo "")

# Set to -pr to use predefined folds
predefined_folds=$([ "${predefined_folds,,}" == "true" ] && echo "-pr" || echo "")

# Update result path to split combined test train split and test train split
results_dir="${results_dir}$([ "${combine_test_train_split,,}" == "true" ] && echo "combine-test-train-split/" || echo "test-train-split/")"

# Update out path to split combined test train split and test train split
out_dir="${out_dir}$([ "${combine_test_train_split,,}" == "true" ] && echo "combine-test-train-split/" || echo "test-train-split/")"

# Set to -utts to combine test train split
combine_test_train_split=$([ "${combine_test_train_split,,}" == "true" ] && echo "-ctts" || echo "")

# Set to -rn to normalise data
normalise_data=$([ "${normalise_data,,}" == "true" ] && echo "-rn" || echo "")

# dont submit to serial directly
queue=$([ "$queue" == "serial" ] && echo "batch" || echo "$queue")
queue_alias=$([ "$queue" == "batch" ] && echo "serial" || echo "$queue")

count=0
while read dataset; do
for clusterer in $clusterers_to_run; do

# Skip to the script start point
((count++))
if ((count>=start_point)); then

# This is the loop to keep from dumping everything in the queue which is maintained around max_num_submitted jobs
num_jobs=$(squeue -u ${username} --format="%20P %5t" -r | awk '{print $2, $1}' | grep -e "R ${queue_alias}" -e "PD ${queue_alias}" | wc -l)
while [ "${num_jobs}" -ge "${max_num_submitted}" ]
do
    echo Waiting 60s, ${num_jobs} currently submitted on ${queue}, user-defined max is ${max_num_submitted}
    sleep 60
    num_jobs=$(squeue -u ${username} --format="%20P %5t" -r | awk '{print $2, $1}' | grep -e "R ${queue_alias}" -e "PD ${queue_alias}" | wc -l)
done

mkdir -p "${out_dir}${clusterer}/${dataset}/"

# This skips jobs which have test/train files already written to the results directory. Only looks for Resamples, not Folds (old file name)
array_jobs=""
for (( i=start_fold-1; i<max_folds; i++ ))
do
    if [ -f "${results_dir}${clusterer}/Predictions/${dataset}/trainResample${i}.csv" ]; then
        if [ "${generate_test_files}" == "-te" ] && ! [ -f "${results_dir}${clusterer}/Predictions/${dataset}/testResample${i}.csv" ]; then
            array_jobs="${array_jobs}${array_jobs:+,}$((i + 1))"
        fi
    else
        array_jobs="${array_jobs}${array_jobs:+,}$((i + 1))"
    fi
done

if [ "${array_jobs}" != "" ]; then

# This creates the script to run the job based on the info above
echo "#!/bin/bash
#SBATCH --mail-type=${mail}
#SBATCH --mail-user=${mailto}
#SBATCH -p ${queue}
#SBATCH -t ${max_time}
#SBATCH --job-name=${clusterer}${dataset}
#SBATCH --array=${array_jobs}
#SBATCH --mem=${max_memory}M
#SBATCH -o ${out_dir}/${clusterer}/${dataset}/%A-%a.out
#SBATCH -e ${out_dir}/${clusterer}/${dataset}/%A-%a.err
#SBATCH --nodes=1

. /etc/profile

module load anaconda/py3.10
source activate $env_name

# Input args to the default clustering_experiments are in main method of
# https://github.com/time-series-machine-learning/tsml-eval/blob/main/tsml_eval/experiments/clustering_experiments.py
python -u ${script_file_path} ${data_dir} ${results_dir} ${clusterer} ${dataset} \$((\$SLURM_ARRAY_TASK_ID - 1)) ${generate_test_files} ${predefined_folds} ${combine_test_train_split} ${normalise_data}" > generatedFile.sub

echo "${count} ${clusterer}/${dataset}"

sbatch < generatedFile.sub

else
    echo "${count} ${clusterer}/${dataset}" has finished all required resamples, skipping
fi

fi
done
done < ${datasets}

echo Finished submitting jobs
