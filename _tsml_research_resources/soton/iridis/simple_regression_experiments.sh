#!/bin/bash
# Local execution of forecasting experiments using venv
set -eu
# ======================================================================================
# 	Default experiment configuration start
#   Create your own config.local or pass in --config <config_file> to override these settings
#   Use config.local.example as a template
# ======================================================================================
# Start and end for resamples
max_folds=10
start_fold=1

# The number of tasks/threads to use in each job. 40 is the number of cores on batch nodes
n_jobs=40

# The number of threads per task. Usually 1 unless using a regressor that can multithread internally
# use with threaded_regression_experiments.py
n_threads_per_job=1

# Enter your username and email here
mail="NONE"

# Start point for the script i.e. 3 datasets, 3 regressorss = 9 experiments to submit, start_point=5 will skip to job 5
start_point=1

# Datasets to use and directory of data files. This can either be a text file or directory of text files
# Separate text files will not run jobs of the same dataset in the same node. This is good to keep large and small datasets separate
relative_data_dir="RegressionResults/Data/forecasting"
relative_dataset_list="RegressionResults/Data/forecasting/windowed_series.txt"

# Results and output file write location. Change these to reflect your own file structure
relative_results_dir="RegressionResults/results/"
relative_out_dir="RegressionResults/output/"
relative_aeon_dir="aeon/"
relative_tsml_eval_dir="tsml-eval/"

# The python script we are running
relative_script_file_path="tsml-eval/tsml_eval/experiments/forecasting_experiments.py"

relative_preprocessing_file_path=""

extra_args=""

# Optional seasonal period/frequency to pass to forecasting estimators.
# SCUM uses aeon's "season_length" parameter; override seasonal_period_parameter
# in a config file if a different forecaster expects another parameter name.
seasonal_period=""
seasonal_period_parameter="season_length"
# relative_seasonal_period_file="tsml-eval/_tsml_research_resources/soton/seasonal_periods.windowed_series.txt"
relative_seasonal_period_file=""

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

# -----------------------------
# Helper to maybe source a config file
maybe_source() {
    if [ -f "$1" ]; then
        . "$1"
    fi
}

# -----------------------------
# Resolve script directory
SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)

# -----------------------------
# Load configs (default < local < CLI override)
maybe_source "$SCRIPT_DIR/../config.default"
maybe_source "$SCRIPT_DIR/../config.local"

# Load any CLI-provided config, then parse all CLI overrides.
ORIGINAL_ARGS=("$@")
CONFIG_FILE=""
for (( arg_i=0; arg_i<${#ORIGINAL_ARGS[@]}; arg_i++ )); do
    if [ "${ORIGINAL_ARGS[$arg_i]}" = "--config" ]; then
        CONFIG_FILE="${ORIGINAL_ARGS[$((arg_i + 1))]:-}"
        break
    fi
done
[ -n "${CONFIG_FILE:-}" ] && maybe_source "$CONFIG_FILE"

set -- "${ORIGINAL_ARGS[@]}"
DEBUG=0
while [ "$#" -gt 0 ]; do
    case "$1" in
        --config) CONFIG_FILE=$2; shift 2 ;;
        --regressors_to_run) regressors_to_run=$2; shift 2 ;;
        --seasonal_period|--seasonal-period) seasonal_period=$2; shift 2 ;;
        --seasonal_period_parameter|--seasonal-period-parameter) seasonal_period_parameter=$2; shift 2 ;;
        --seasonal_period_file|--seasonal-period-file) relative_seasonal_period_file=$2; shift 2 ;;
        --debug)
          DEBUG=1
          if [ "${2:-}" = "on" ] || [ "${2:-}" = "true" ] || [ "${2:-}" = "1" ]; then
            shift 2
          else
            shift
          fi
          ;;
        --) shift; break ;;
        *) echo "Unknown option: $1" >&2; exit 2 ;;
    esac
done

# -----------------------------
# Validate required variables
: "${username:?Set username in config file}"
: "${env_name:?Set env_name in config file}"
: "${regressors_to_run:?Set regressors_to_run in config file or CLI}"

[ "$DEBUG" = "1" ] && echo "Running regressors: $regressors_to_run"

mailto=$username"@soton.ac.uk"
# -----------------------------
# Directories
local_path="/home/$username"

script_file_path="$local_path/$relative_script_file_path"
preprocessing_file_path="$local_path/$relative_preprocessing_file_path"
aeon_path="$local_path/$relative_aeon_dir"
tsml_eval_path="$local_path/$relative_tsml_eval_dir"

if [ -n "${relative_seasonal_period_file:-}" ]; then
    case "$relative_seasonal_period_file" in
        /*) seasonal_period_file_path="$relative_seasonal_period_file" ;;
        *) seasonal_period_file_path="$local_path/$relative_seasonal_period_file" ;;
    esac
else
    seasonal_period_file_path=""
fi

data_dir="$local_path/$relative_data_dir"
dataset_list="$local_path/$relative_dataset_list"
results_dir="$local_path/$relative_results_dir"
out_dir="$local_path/$relative_out_dir"

# -----------------------------
# Flags
generate_train_files_flag=$([ "${generate_train_files,,}" == "true" ] && echo "-tr" || echo "")
predefined_folds_flag=$([ "${predefined_folds,,}" == "true" ] && echo "-pr" || echo "")
normalise_data_flag=$([ "${normalise_data,,}" == "true" ] && echo "-rn" || echo "")

mkdir -p "${out_dir}"

# -----------------------------
# Activate Python venv
source "$aeon_path/$env_name/bin/activate"

get_seasonal_period () {
    series_name=$1
    if [ -n "${seasonal_period:-}" ]; then
        echo "$seasonal_period"
    elif [ -n "${seasonal_period_file_path:-}" ] && [ -f "$seasonal_period_file_path" ]; then
        awk -v series="$series_name" '$1 == series { print $2; exit }' "$seasonal_period_file_path"
    else
        echo ""
    fi
}

# -----------------------------
# Turn a directory into a list if needed
if [[ -d $dataset_list ]]; then
    file_names=""
    for file in ${dataset_list}/*; do
        file_names="$file_names$dataset_list$(basename "$file") "
    done
    dataset_list=$file_names
fi

# -----------------------------
# Main loop
expCount=0
totalCount=0
dt=$(date +%Y%m%d%H%M%S)

for dataset_file in $dataset_list; do

    # Optional preprocessing
    if [ -n "${preprocessing_file_path:-}" ]; then
        list_of_series="${dataset_file}_compiled.txt"
        echo "Running preprocessing for $dataset_file..."
        python -u "${preprocessing_file_path}" "${data_dir}" "${dataset_file}" "${list_of_series}"
    fi

    for regressor in $regressors_to_run; do
        mkdir -p "${out_dir}/${regressor}/"

        while read dataset; do
            expCount=$((expCount + 1))
            if ((expCount < start_point)); then
                continue
            fi

            # Determine which resamples to run
            resamples_to_run=""
            for ((i=start_fold-1; i<max_folds; i++)); do
                if [ -f "${results_dir}${regressor}/Predictions/${dataset}/testResample${i}.csv" ]; then
                    if [ "${generate_train_files,,}" == "true" ] && ! [ -f "${results_dir}${regressor}/Predictions/${dataset}/trainResample${i}.csv" ]; then
                        resamples_to_run="${resamples_to_run}${i} "
                    fi
                else
                    resamples_to_run="${resamples_to_run}${i} "
                fi
            done

            for resample in $resamples_to_run; do
                dataset_seasonal_period=$(get_seasonal_period "$dataset")
                if [ -n "${dataset_seasonal_period:-}" ]; then
                    seasonal_period_args="-kw ${seasonal_period_parameter} ${dataset_seasonal_period} int"
                else
                    seasonal_period_args=""
                fi
                echo "Running $regressor on $dataset fold $resample..."
                python -u "${script_file_path}" "${data_dir}" "${results_dir}" "${regressor}" "${dataset}" "${resample}" \
                    ${generate_train_files_flag} ${predefined_folds_flag} ${normalise_data_flag} -nj ${n_threads_per_job} ${seasonal_period_args} ${extra_args} \
                    > "${out_dir}/${regressor}/output-${dataset}-${resample}-${dt}.txt" 2>&1 &
                
                totalCount=$((totalCount + 1))

                # Optional: limit number of parallel jobs
                while (( $(jobs -r | wc -l) >= n_jobs )); do
                    sleep 1
                done

            done
        done < "$dataset_file"
    done
done

# Wait for all background jobs to finish
wait

echo "Finished all $totalCount jobs."
