#!/bin/bash
#SBATCH --job-name=auto_bakeoff
#SBATCH --output=auto_bakeoff.out
#SBATCH --error=auto_bakeoff.err
#SBATCH --time=00:03:00               # Maximum runtime (HH:MM:SS)
#SBATCH --partition=lyceum           # Partition to submit to
#SBATCH --ntasks=1                    # Number of tasks (typically 1)
#SBATCH --cpus-per-task=4             # Number of CPU cores per task
#SBATCH --nodes=1                     # Number of nodes

#SBATCH --mail-type=ALL               # Send email on job start, end, fail
#SBATCH --mail-user=as14n23@soton.ac.uk  # Email address for notifications

# Load Miniconda and activate the environment
source /path/to/your/python/environment #REPLACE WITH THE APPROPRIATE LOCATION OF YOUR ENVIRONMENT
#example: source /lyceum/as14n23/miniconda3/etc/profile.d/conda.sh

conda activate your_python_environment # ACTIVATE THE ENVIRONMENT 

# Change to the directory where the automated bakeoff script is located
cd /path/to/bakeoff/script
#example: cd /lyceum/as14n23/my_tsml/tsml-eval/tsml_eval/publications/y2023/tsc_bakeoff

# Run the Python script
python auto_bakeoff.py


