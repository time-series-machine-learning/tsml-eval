#!/bin/bash

max_folds=30
start_fold=1
# To avoid dumping 1000s of jobs in the queue we have a higher level queue
maxNumSubmitted=700
# queue options are https://my.uea.ac.uk/divisions/it-and-computing-services/service-catalogue/research-it-services/hpc/ada-cluster/using-ada
queue="compute-64-512"
username="eej17ucu"
mail="NONE"
mailto="eej17ucu@uea.ac.uk"
# MB for jobs, max is maybe 128000 before you need ot use huge memory queue
max_memory=32000
# Max allowable is 7 days  - 168 hours
max_time="168:00:00"
start_point=1
data_dir="/gpfs/home/ajb/Data/"
# Tony's work space, all should be able to read these.
# Change if you want to use different data or lists
local_path="/gpfs/home/ajb/"
data_dir=$local_path"Data/"
#dont write results to my file space, it causes problems
my_path="/gpfs/home/eej17ucu/"
datasets=$my_path"code/estimator-evaluation/experiments/Univariate.txt"
results_dir=$my_path"experiment-results/averaging/elastic-dba/"
out_dir=$my_path"experiment-logs/averaging/elastic-dba/"
script_file_path=$my_path"code/estimator-evaluation/sktime_estimator_evaluation/experiments/clustering_experiments.py"
# For env set up, see https://hackmd.io/ds5IEK3oQAquD4c6AP2xzQ
env_name="averaging-elastic-dba"
generate_train_files="false"
clusterer="kmeans"
averaging="dba"
count=0
# dtw ddtw erp edr wdtw lcss twe msm dwdtw euclidean
while read dataset; do
for distance in dtw msm ddtw wdtw wddtw erp twe
do

numPending=$(squeue -u ${username} --format="%10i %15P %20j %10u %10t %10M %10D %20R" -r | awk '{print $5, $2}' | grep "PD ${queue}" | wc -l)
numRunning=$(squeue -u ${username} --format="%10i %15P %20j %10u %10t %10M %10D %20R" -r | awk '{print $5, $2}' | grep "R ${queue}" | wc -l)
while [ "$((numPending+numRunning))" -ge "${maxNumSubmitted}" ]
do
    echo Waiting 60s, $((numPending+numRunning)) currently submitted on ${queue}, user-defined max is ${maxNumSubmitted}
	sleep 60
	numPending=$(squeue -u ${username} --format="%10i %15P %20j %10u %10t %10M %10D %20R" -r | awk '{print $5, $2}' | grep "PD ${queue}" | wc -l)
	numRunning=$(squeue -u ${username} --format="%10i %15P %20j %10u %10t %10M %10D %20R" -r | awk '{print $5, $2}' | grep "R ${queue}" | wc -l)
done

((count++))

if ((count>=start_point)); then

mkdir -p ${out_dir}${clusterer}/${dataset}/

echo "#!/bin/bash

#SBATCH --mail-type=${mail}
#SBATCH --mail-user=${mailto}
#SBATCH -p ${queue}
#SBATCH -t ${max_time}
#SBATCH --job-name=${clusterer}${dataset}
#SBATCH --array=${start_fold}-${max_folds}
#SBATCH --mem=${max_memory}M
#SBATCH -o ${out_dir}${clusterer}/${dataset}/%A-%a.out
#SBATCH -e ${out_dir}${clusterer}/${dataset}/%A-%a.err

. /etc/profile

module add python/anaconda/2019.10/3.7
source /gpfs/software/ada/python/anaconda/2019.10/3.7/etc/profile.d/conda.sh
conda activate $env_name
export PYTHONPATH=$(pwd)

python ${script_file_path} ${data_dir} ${results_dir} ${distance} ${dataset} \$SLURM_ARRAY_TASK_ID ${generate_train_files} ${clusterer} ${averaging}"  > generatedFile.sub
#                         data_dir = sys.argv[1]
#                         results_dir = sys.argv[2]
#                         distance = sys.argv[3]
#                         dataset = sys.argv[4]
#                         resample = int(sys.argv[5]) - 1
#                         generate_train_files = sys.argv[6]
#                         clusterer = sys.argv[7]
#                         averaging = sys.argv[8]

echo ${count} ${clusterer}/${dataset}

sbatch < generatedFile.sub --qos=ht

fi

done
done < ${datasets}

echo Finished submitting jobs
