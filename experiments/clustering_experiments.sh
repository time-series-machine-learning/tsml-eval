#!/bin/bash

max_folds=1                                               
start_fold=1                                                
maxNumSubmitted=700
queue="compute-64-512"
username="ajb"
mail="NONE"
mailto="ajb@uea.ac.uk"
max_memory=8000
max_time="168:00:00"
start_point=1
data_dir="/gpfs/home/ajb/Data/"
datasets="/gpfs/home/ajb/DataSetLists/temp.txt"
# Put your home directory here
local_path="/gpfs/home/ajb/"

data_dir=$local_path"sktimeData/"
datasets=$local_path"DataSetLists/temp.txt"
results_dir=$local_path"ClusteringResults/kmeans/"
out_dir=$local_path"Code/output/Clustering/"
script_file_path=$local_path"Code/estimator-evaluation/sktime_estimator_evaluation/experiments/clustering_experiments.py"
env_name="sktime"
generate_train_files="false"
clusterer="kmeans"
averaging="mean"
count=0
# dtw ddtw erp edr wdtw lcss twe msm dwdtw euclidean
while read dataset; do
for distance in euclidean
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
#                         tf = sys.argv[6]
#                         clusterer = sys.argv[8]
#                         averaging = sys.argv[9]

echo ${count} ${clusterer}/${dataset}

sbatch < generatedFile.sub --qos=ht

fi

done
done < ${datasets}

echo Finished submitting jobs