#!/bin/bash
# Build the Clinical window-length variants (one process per source problem, in parallel)
# into ~/DataClinWin. Sources and flags are in clinwin_sources.txt.
#SBATCH --job-name=clinwin-build
#SBATCH --partition=amd
#SBATCH --nodes=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=64G
#SBATCH --time=10:00:00
#SBATCH --output=/home/ajb2u23/tsml-configs/clinwin/build-%j.log

module load conda/python3 2>/dev/null
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
C=/home/ajb2u23/tsml-configs/clinwin
P=/home/ajb2u23/.conda/envs/tsml-eval/bin/python
OUT=/home/ajb2u23/DataClinWin
mkdir -p $OUT
while read -r name flags; do
  [ -z "$name" ] && continue
  $P -u $C/build_clinical_windows.py /home/ajb2u23/Data/EEG/$name $name $OUT $flags \
      > $C/build-$name.log 2>&1 &
done < $C/clinwin_sources.txt
wait
echo "all builds finished $(date)"
grep -L "test:" $C/build-*.log && echo "SOME BUILDS FAILED" || echo "all builds wrote test"
