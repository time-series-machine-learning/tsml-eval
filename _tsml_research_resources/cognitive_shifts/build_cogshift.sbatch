#!/bin/bash
# Build the Cognitive shift/surrogate variants (one process per source problem) into
# ~/Data/EEG-CogShift, then start both experiments.
#SBATCH --job-name=cogshift-build
#SBATCH --partition=compute
#SBATCH --account=cmp
#SBATCH --qos=uea-core-default
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=/gpfs/home/ajb/tsml-configs/cogshift/build-%j.log

export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
C=/gpfs/home/ajb/tsml-configs/cogshift
P=/gpfs/home/ajb/.conda/envs/tsml-eval/bin/python
S=/gpfs/home/ajb/Code/tsml-eval/_tsml_research_resources/slurm
OUT=/gpfs/home/ajb/Data/EEG-CogShift
mkdir -p $OUT
while read -r name; do
  [ -z "$name" ] && continue
  $P -u $C/build_cognitive_shifts.py /gpfs/home/ajb/Data/EEG/$name $name $OUT > $C/build-$name.log 2>&1 &
done < $C/cogshift_sources.txt
wait
echo "all builds finished $(date)"
missing=0
for n in $(grep -v '^#' $C/cogshift_datasets.txt); do
  for s in TRAIN TEST; do [ -s $OUT/$n/${n}_$s.ts ] || { echo "missing $n $s"; missing=1; }; done
done
echo "missing=$missing; starting experiments anyway (missing problems fail and are reported)"
for cfg in hali_cogshift_hc2 hali_cogshift_other; do
  $P $S/tsml_slurm.py start $C/$cfg.toml 2>&1 | grep -v -i warn | tail -15
done
