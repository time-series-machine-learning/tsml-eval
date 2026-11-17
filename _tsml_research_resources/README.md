This directory contains instructions and utilities for running experiments using
tsml-eval on University of Southampton (Soton), University of East Anglia (UEA) and
University of Bradford hardware.

**To run experiments on Hali, Iridis 6 or Iridis X, start with
[`slurm/README.md`](slurm/README.md).** It describes the standard controller
(`slurm/tsml_slurm.py`), the cluster profiles, and example experiments for CPU and GPU
work on each cluster. The scripts under `soton/` and `uea/` are the older
copy-and-edit submission scripts, kept for reference. The environment setup guides
there (`soton/iridis/iridis_python.md`) are still current.

For non-Soton/UEA/Bradford users, the directory contents are likely irrelevant.
Some of the contents could be adapted for other purposes or could be generalised
for other Slurm/Linux devices, but running the scripts or following the hardware
instructions will likely achieve nothing without alterations.
