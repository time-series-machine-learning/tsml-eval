#!/bin/sh

#slurm script for requesting batch jobs

#SBATCH --nodes=1
#SBATCH --mem=4000          # Request 4GB of memory
#SBATCH --job-name=classification_experiments
#SBATCH --output=classification_experiments.out
#SBATCH --error=classification_experiments.err
#SBATCH --partition=batch   # Default partition

# For example run the classification experiments
./classification_experiments.sh