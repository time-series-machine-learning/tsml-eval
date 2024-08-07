#!/bin/sh

#slurm script for requesting batch jobs

#SBATCH --nodes=1
#SBATCH --mem=4000          # Request 4GB of memory
#SBATCH --job-name=classification_experiments
#SBATCH --output=classification_experiments.out
#SBATCH --error=classification_experiments.err
#SBATCH --partition=batch   # Default partition

# Ensure single-threaded execution to avoid resource conflicts
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

# For example run the classification experiments
./classification_experiments.sh