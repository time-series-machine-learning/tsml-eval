#!/bin/bash
# Wait for the clinwin build job, check every variant in clinwin_datasets.txt has both .ts
# files, then start the EEGNeX-NB and REVE-FT supervisor loops on the login node.
# Usage (on loginx003): setsid nohup bash start_after_build.sh JOBID > watcher.log 2>&1 < /dev/null &
C=/home/ajb2u23/tsml-configs/clinwin
PY=/home/ajb2u23/.conda/envs/tsml-eval-gpu/bin/python
S=/home/ajb2u23/Code/tsml-eval-gpu/_tsml_research_resources/slurm
D=/home/ajb2u23/DataClinWin
echo "$(date) waiting for job $1"
while squeue -h -j "$1" 2>/dev/null | grep -q .; do sleep 300; done
echo "$(date) build job finished"
missing=0
for n in $(grep -v '^#' $C/clinwin_datasets.txt); do
  for s in TRAIN TEST; do
    [ -s $D/$n/${n}_$s.ts ] || { echo "missing $D/$n/${n}_$s.ts"; missing=1; }
  done
done
if [ $missing = 1 ]; then
  # start anyway: tasks for missing problems fail and are listed in the reports
  echo "$(date) some variants missing; starting loops for what exists"
fi
for cfg in iridisx_clinwin_eegnex iridisx_clinwin_reve; do
  name=$(grep '^name' $C/$cfg.toml | cut -d'"' -f2)
  mkdir -p /home/ajb2u23/tsml-slurm/$name/supervisor
  $PY $S/tsml_slurm.py check $C/$cfg.toml > $C/check-$cfg.log 2>&1
  setsid nohup $PY -u $S/tsml_slurm.py loop $C/$cfg.toml \
      >> /home/ajb2u23/tsml-slurm/$name/supervisor/loop.log 2>&1 < /dev/null &
  echo "$(date) started $cfg loop $!"
done
