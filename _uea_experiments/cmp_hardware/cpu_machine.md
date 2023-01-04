# Kraken

aka cmpcpusvr.uea.ac.uk

### machine names for putty/winscp

ADA
ada.uea.ac.uk

Kraken


Beast https://www.overleaf.com/dash
cmpresearchsvr.uea.ac.uk\ueatsc

https://docs.conda.io/projects/conda/en/4.6.0/_downloads/52a95608c49671267e40c689e0bc00ca/conda-cheatsheet.pdf


# Kraken and ADA commands

Tonys solutions to running and managing sktime

## Installing sktime using conda:



### on the cluster
#### install the modules
module add python/anaconda/2019.10/3.7
source /gpfs/software/ada/python/anaconda/2019.10/3.7/etc/profile.d/conda.sh

### get the code

#### on kraken and cluster,

git clone https://github.com/alan-turing-institute/sktime
cd sktime
#### on windows
install anaconda
install git
clone repository through desktop

### create an sktime development environment

conda create -n sktime python=3.8 (3.7 for cluster)
conda activate sktime
conda install -c conda-forge pystan
conda install -c conda-forge prophet
pip install -e .[all_extras,dev]

the extra ones may not be necessary.

### create an experimental

## Running sktime on kraken
organisation:
code in Code directory
have a file for each input parameter
Code/ClassificationInputFiles
Code/ClusteringInputFiles

conda activate sktime

conda create -n exp-eval python=3.8 (3.7 for cluster)
conda activate exp-eval
pip install sktime


cd ClassificationInputFiles
parallel -d "\n" --verbose --jobs 90% --memfree 30G -a data_dir.txt -a results_dir.txt -a classifier.txt -a dataset.txt -a resamples.txt -a generate_train_files.txt -a predefined_folds.txt python sktime_estimator_evaluation/experiments/classification_experiments.py > output.txt

need to activate environment?

Example: for clustering

parallel -d "\n" --verbose --jobs 90% --memfree 30G -a data_dir.txt -a results_dir.txt -a distances.txt -a dataset_list.txt -a resamples.txt -a clusterer.txt -a averaging.txt python --wd ignore sktime/_contrib/clustering_experiments.py > kmedoids_twe.txt



## Running sktime on ADA
base it on script /gpfs/home/code/clustering_distances.sh

## Monitoring jobs on Kraken
list processes         >htop -u ajb
kills all processes    >pkill -u ajb

## Monitoring jobs on ADA

list processes

>squeue -u ajb --format="%12i %15P %20j %10u %10t %10M %10D %20R" -r

GPU queue
>squeue -p gpu-rtx6000-2

to kill all ajb jobs
>scancel -u ajb

To delete all jobs on a queue it’s:

>scancel -p gpu-rtx6000-2

To delete one job it’s:

>scancel 11133013_1
