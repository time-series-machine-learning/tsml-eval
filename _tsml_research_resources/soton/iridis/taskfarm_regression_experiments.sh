#!/bin/bash
# Check and edit all options before the first run!
# While reading is fine, please dont write anything to the default directories in this script
set -eu
# ======================================================================================
# 	Default experiment configuration start
#   Create your own config.local or pass in --config <config_file> to override these settings
#   Use config.local.example as a template
# ======================================================================================
# Start and end for resamples
max_folds=10
start_fold=1

# To avoid hitting the cluster queue limit we have a higher level queue
max_num_submitted=900

gpu_job="false"
iridis_version="5"

# The number of tasks/threads to use in each job. 40 is the number of cores on batch nodes
n_tasks_per_node=40

# The number of threads per task. Usually 1 unless using a regressor that can multithread internally
# use with threaded_regression_experiments.py
n_threads_per_task=1

# The number of cores to request from the node. Don't go over the number of cores for the node. 40 is the number of cores on batch nodes
# If you are not using the whole node, please make sure you are requesting memory correctly
max_cpus_to_use=40

# MB for jobs, increase incrementally and try not to use more than you need. If you need hundreds of GB consider the huge memory queue
max_memory=8000

# Create a separate submission list for each regressor. This will stop the mixing of
# large and small jobs in the same node, but results in some smaller scripts submitted
# to serial when moving between regressors.
# For small workloads i.e. single resample 10 datasets, turning this off will be the only way to get on the batch queue realistically
split_regressors="true"

# Enter your username and email here
mail="NONE"

# Max allowable is 60 hours
max_time="60:00:00"

# Start point for the script i.e. 3 datasets, 3 regressorss = 9 experiments to submit, start_point=5 will skip to job 5
start_point=1

# Datasets to use and directory of data files. This can either be a text file or directory of text files
# Separate text files will not run jobs of the same dataset in the same node. This is good to keep large and small datasets separate
relative_data_dir="Data/forecasting"
relative_dataset_list="Data/windowed_series.txt"

# Results and output file write location. Change these to reflect your own file structure
relative_results_dir="RegressionResults/results/"
relative_out_dir="RegressionResults/output/"

# The python script we are running
relative_script_file_path="tsml-eval/tsml_eval/experiments/forecasting_experiments.py"

extra_args=""

# Environment name, change accordingly, for set up, see https://github.com/time-series-machine-learning/tsml-eval/blob/main/_tsml_research_resources/soton/iridis/iridis_python.md
# Separate environments for GPU and CPU are recommended #regress_gpu regression_experiments
# env_name="regression_experiments"
container_path="scratch/tensorflow_sandbox/"

# Regressors to loop over. Must be seperated by a space. Different regressors will not run in the same node
# See list of potential regressors in set_regressor  InceptionTimeRegressor
# regressors_to_run="ETSForecaster, AutoETSForecaster, SktimeETS, StatsForecastETS" # RocketRegressor MultiRocketRegressor ResNetRegressor fpcregressor fpcr-b-spline TimeCNNRegressor FCNRegressor 1nn-ed 1nn-dtw 5nn-ed 5nn-dtw FreshPRINCERegressor TimeSeriesForestRegressor DrCIFRegressor Ridge SVR RandomForestRegressor RotationForestRegressor xgboost

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

# ======================================================================================
# 	Read in config files and CLI args
# ======================================================================================

# Helper
maybe_source() { [ -f "$1" ] && . "$1"; }

# Resolve script dir
SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)

# 1) Load configs in increasing priority
maybe_source "$SCRIPT_DIR/../../config.default"
maybe_source "$SCRIPT_DIR/../../config.local"

# 2) Parse CLI overrides (highest priority)
CONFIG_FILE=""
while [ "$#" -gt 0 ]; do
  case "$1" in
    --config) CONFIG_FILE=$2; shift 2 ;;
    --regressors_to_run) regressors_to_run=$2; shift 2 ;;
    --debug) DEBUG=1; shift ;;
    --) shift; break ;;
    *) echo "Unknown option: $1" >&2; exit 2 ;;
  esac
done
[ -n "${CONFIG_FILE:-}" ] && maybe_source "$CONFIG_FILE"

# 3) Validate required vars
: "${username:?Set username in config file}"
: "${env_name:?Set env_name in config file}"
: "${regressors_to_run:?Set regressors_to_run in config file or CLI}"

[ "$DEBUG" = "1" ] && echo "Running $regressors_to_run"

# ======================================================================================
# 	Read in config files and CLI args
# ======================================================================================

mailto=$username"@soton.ac.uk"

# Queue options are https://sotonac.sharepoint.com/teams/HPCCommunityWiki/SitePages/Iridis%205%20Job-submission-and-Limits-Quotas.aspx
if [ "${gpu_job}" == "true" ]; then
    if [ "${iridis_version}" == "5" ]; then
        queue="gpu"
    else
        queue="a100"
    fi
else
    queue="batch"
fi

# Different home paths on iridis5 and iridis6/iridisX
if [ "${iridis_version}" == "5" ]; then
    local_path="/ECShome/$username"
else
    local_path="/home/$username"
fi

# staskfarm doesn't exist on iridis6 or iridisX
if [ "${iridis_version}" == "5" ]; then
    taskfarm_file_path="staskfarm"
else
    taskfarm_file_path="$local_path/tsml-eval/_tsml_research_resources/soton/iridis/staskfarm.sh"
fi

# The python script we are running
full_script_file_path="$local_path/$relative_script_file_path"

# Datasets to use and directory of data files. This can either be a text file or directory of text files
# Separate text files will not run jobs of the same dataset in the same node. This is good to keep large and small datasets separate
data_dir="$local_path/$relative_data_dir"
dataset_list="$local_path/$relative_dataset_list"

# Results and output file write location.
results_dir="$local_path/$relative_results_dir"
out_dir="$local_path/$relative_out_dir"

# Set to -tr to generate test files
generate_train_files=$([ "${generate_train_files,,}" == "true" ] && echo "-tr" || echo "")

# Set to -pr to use predefined folds
predefined_folds=$([ "${predefined_folds,,}" == "true" ] && echo "-pr" || echo "")

# Set to -rn to normalise data
normalise_data=$([ "${normalise_data,,}" == "true" ] && echo "-rn" || echo "")

mkdir -p "${out_dir}/"

if [ "${iridis_version}" == "5" ]; then
    conda_instruction="anaconda/py3.10"
else
    conda_instruction="conda/python3"
fi

if [ "${gpu_job}" == "true" ]; then
    gpu_instruction="#SBATCH --gres=gpu:1"
else
    gpu_instruction=""
fi

# This creates the submission file to run and does clean up
submit_jobs () {

if ((cmdCount>=max_cpus_to_use)); then
    cpuCount=$max_cpus_to_use
else
    cpuCount=$cmdCount
fi

if [ "${gpu_job}" == "true" ]; then
    environment_instructions="module load apptainer/1.3.3"
    apptainer_instruction="apptainer exec --nv ${container_path} echo "Running Apptainer job."; "
else
    environment_instructions="module load $conda_instruction && source activate $env_name"
    apptainer_instruction=""
fi

echo "#!/bin/bash
${gpu_instruction}
#SBATCH --mail-type=${mail}
#SBATCH --mail-user=${mailto}
#SBATCH --job-name=batch-${dt}
#SBATCH -p ${queue}
#SBATCH -t ${max_time}
#SBATCH -o ${out_dir}/%A-${dt}.out
#SBATCH -e ${out_dir}/%A-${dt}.err
#SBATCH --nodes=1
#SBATCH --ntasks=${cpuCount}
#SBATCH --mem=${max_memory}M

. /etc/profile

${environment_instructions}



${taskfarm_file_path} ${out_dir}/generatedCommandList-${dt}.txt" > generatedSubmissionFile-${dt}.sub

echo "At experiment ${expCount}, ${totalCount} jobs submitted total"

sbatch < generatedSubmissionFile-${dt}.sub

rm generatedSubmissionFile-${dt}.sub

}

totalCount=0
expCount=0
dt=$(date +%Y%m%d%H%M%S)

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

for regressor in $regressors_to_run; do

mkdir -p "${out_dir}/${regressor}/"

if [ "${split_regressors,,}" == "true" ]; then
    # we use time for unique names
    sleep 1
    cmdCount=0
    dt=$(date +%Y%m%d%H%M%S)
    outDir=${out_dir}/${regressor}
else
    outDir=${out_dir}
fi

while read dataset; do

# Skip to the script start point
((expCount++))
if ((expCount>=start_point)); then

# This finds the resamples to run and skips jobs which have test/train files already written to the results directory.
# This can result in uneven sized command lists
resamples_to_run=""
for (( i=start_fold-1; i<max_folds; i++ ))
do
    if [ -f "${results_dir}${regressor}/Predictions/${dataset}/testResample${i}.csv" ]; then
        if [ "${generate_train_files}" == "true" ] && ! [ -f "${results_dir}${regressor}/Predictions/${dataset}/trainResample${i}.csv" ]; then
            resamples_to_run="${resamples_to_run}${i} "
        fi
    else
        resamples_to_run="${resamples_to_run}${i} "
    fi
done

for resample in $resamples_to_run; do

# add to the command list if
if ((cmdCount>(n_tasks_per_node-n_threads_per_task))); then
    submit_jobs

    # This is the loop to stop you from dumping everything in the queue at once, see max_num_submitted jobs
    num_jobs=$(squeue -u ${username} --format="%20P %5t" -r | awk '{print $2, $1}' | grep -e "R ${queue}" -e "PD ${queue}" | wc -l)
    echo "Number of Jobs currently running on the cluster: ${num_jobs}"
    while [ "${num_jobs}" -ge "${max_num_submitted}" ]
    do
        echo Waiting 60s, ${num_jobs} currently submitted on ${queue}, user-defined max is ${max_num_submitted}
        sleep 60
        num_jobs=$(squeue -u ${username} --format="%20P %5t" -r | awk '{print $2, $1}' | grep -e "R ${queue}" -e "PD ${queue}" | wc -l)
    done

    sleep 1
    cmdCount=0
    dt=$(date +%Y%m%d%H%M%S)
fi

# Input args to the default classification_experiments are in main method of
# https://github.com/time-series-machine-learning/tsml-eval/blob/main/tsml_eval/experiments/classification_experiments.py
echo "${apptainer_instruction}python -u ${full_script_file_path} ${data_dir} ${results_dir} ${regressor} ${dataset} ${resample} ${generate_train_files} ${predefined_folds} ${normalise_data} ${extra_args} > ${out_dir}/${regressor}/output-${dataset}-${resample}-${dt}.txt 2>&1" >> ${out_dir}/generatedCommandList-${dt}.txt

((cmdCount=cmdCount+n_threads_per_task))
((totalCount++))

done
fi
done < ${dataset_file}

if [[ "${split_regressors,,}" == "true" && $cmdCount -gt 0 ]]; then
    # final submit for this regressor
    submit_jobs
fi

done

if [[ "${split_regressors,,}" != "true" && $cmdCount -gt 0 ]]; then
    # final submit for this dataset list
    submit_jobs
fi

done

echo Finished submitting jobs