#!/usr/bin/env bash
# Run build_hc2_from_files.py as a batch job on Iridis 6 instead of a login node.
# Arguments are passed to the script. Submit from this directory:
#
#   sbatch build_hc2_job.sh                     # HC2 from all four components
#   sbatch build_hc2_job.sh --name HC2-noTDE --components \
#       ConvolutionBased/Arsenal IntervalBased/DrCIF ShapeletBased/STC-aeon
#
# Output goes to ~/Results/EEG/Hybrid/<name>/Predictions, and the log to
# build_hc2-<jobid>.out here. Rerun as components finish; built datasets are skipped.
#SBATCH --job-name=build-hc2
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=04:00:00
#SBATCH --output=build_hc2-%j.out
set -eo pipefail
export DISPLAY="${DISPLAY:-}"  # some Iridis profile hooks read it
source /etc/profile
unset CONDA_DEFAULT_ENV CONDA_PREFIX CONDA_SHLVL CONDA_PROMPT_MODIFIER PYTHONPATH
module purge
module load conda/python3
source "$(dirname "$(dirname "$(command -v conda)")")/etc/profile.d/conda.sh"
conda activate tsml-eval
cd "${SLURM_SUBMIT_DIR:-.}"
echo "$(date -Iseconds) $(hostname) python build_hc2_from_files.py $*"
python -u build_hc2_from_files.py "$@"
