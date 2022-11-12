#!/bin/bash
# GPU jobs require two changes:
#queue="gpu-rtx6000-2"
#SBATCH --qos=gpu-rtx
#ALSO CHECK: datasets (list of problems), results_dir (where to check/write results),
# for classifier in MLPClassifier

# Start and end for resamples
max_folds=30
start_fold=1
# To avoid dumping 1000s of jobs in the queue we have a higher level queue
max_num_submitted=300
# Queue options are https://my.uea.ac.uk/divisions/it-and-computing-services/service-catalogue/research-it-services/hpc/ada-cluster/using-ada
# For tensorflow/GPU jobs, use "gpu-rtx6000-2"
queue="compute-64-512"
# Enter your username and email here
username="ajb"
mail="NONE"
mailto="ajb@uea.ac.uk"
# MB for jobs, max is maybe 64000 before you need to use huge memory queue
max_memory=8000

# Max allowable is 7 days - 168 hours
max_time="168:00:00"

# Start point for the script i.e. 3 datasets, 3 classifiers = 9 jobs to submit, start_point=5 will skip to job 5
start_point=1

# Datasets to use and directory of data files. Default is Tony's work space, all should be able to read these. Change if you want to use different data or lists
data_dir="/gpfs/home/ajb/Data/"
datasets="/gpfs/home/ajb/DataSetLists/temp.txt"

# Put your home directory here
local_path="/gpfs/home/ajb/"

# Change these to reflect your own file structure
results_dir=$local_path"ClassificationResults/MultivariateReferenceResults/sktime/"
out_dir=$local_path"Code/output/multivariate/"
script_file_path=$local_path"Code/tsml-estimator-evaluation/tsml_estimator_evaluation
/experiments/classification_experiments.py"
# environment name, change accordingly, for set up, see https://hackmd.io/ds5IEK3oQAquD4c6AP2xzQ
env_name="eval"
# Generating train folds is usually slower, set to false unless you need them
generate_train_files="false"
# If set for true, looks for <problem><fold>_TRAIN.ts file. This is useful for running tsml resamples
predefined_folds="false"

# List valid classifiers e.g DrCIF TDE Arsenal STC MUSE ROCKET Mini-ROCKET Multi-ROCKET
# See set_classifier for aliases
#Arsenal BOSSEnsemble(BOSS) CanonicalIntervalForest (CIF) Catch22Classifier
# ContractableBOSS  DrCIF ElasticEnsemble FreshPRINCE HIVECOTEV1 HIVECOTEV2
# KNeighborsTimeSeriesClassifier RandomIntervalClassifier
# 'RandomIntervalSpectralEnsemble RocketClassifier
# ShapeDTWW ShapeletTransformClassifier TemporalDictionaryEnsemble
# All multivariate classifiers can be listed like this (remove filter_tags for
# univariate
#    from sktime.registry import all_estimators
#    cls = all_estimators(
#        estimator_types="classifier", filter_tags={"capability:multivariate": True}
#        )
#    names = [i for i, _ in cls]

count=0
while read dataset; do
for classifier in ShapeletTransformClassifier
do
# Dont change anything after here
# This is the loop to keep from dumping everything in the queue which is maintained around max_num_submitted jobs
num_pending=$(squeue -u ${username} --format="%10i %15P %20j %10u %10t %10M %10D %20R" -r | awk '{print $5, $2}' | grep "PD ${queue}" | wc -l)
num_running=$(squeue -u ${username} --format="%10i %15P %20j %10u %10t %10M %10D %20R" -r | awk '{print $5, $2}' | grep "R ${queue}" | wc -l)
while [ "$((num_pending+num_running))" -ge "${max_num_submitted}" ]
do
    echo Waiting 60s, $((num_pending+num_running)) currently submitted on ${queue}, user-defined max is ${max_num_submitted}
	sleep 60
	num_pending=$(squeue -u ${username} --format="%10i %15P %20j %10u %10t %10M %10D %20R" -r | awk '{print $5, $2}' | grep "PD ${queue}" | wc -l)
	num_running=$(squeue -u ${username} --format="%10i %15P %20j %10u %10t %10M %10D %20R" -r | awk '{print $5, $2}' | grep "R ${queue}" | wc -l)
done

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
#SBATCH --qos=ht
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
source /gpfs/software/ada/python/anaconda/2019.10/3.7/etc/profile.d/conda.sh
conda activate $env_name
export PYTHONPATH=$(pwd)
# Input args to classification_experiments are in main method of
# https://github.com/uea-machine-learning/estimator-evaluation/blob/main/sktime_estimator_evaluation/experiments/classification_experiments.py
python ${script_file_path} ${data_dir} ${results_dir} ${classifier} ${dataset} \$SLURM_ARRAY_TASK_ID ${generate_train_files} ${predefined_folds}"  > generatedFile.sub

echo ${count} ${classifier}/${dataset}

sbatch < generatedFile.sub

else
    echo ${count} ${classifier}/${dataset} has finished all required resamples, skipping
fi

fi
done
done < ${datasets}

echo Finished submitting jobs
