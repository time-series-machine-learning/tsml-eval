#!/bin/bash
# Print the IridisX Slurm and module settings needed to fill in the TODO-IRIDISX
# entries in the gpu_scripts. Run this on the login node, it submits nothing.
#
#   sh iridisx_probe.sh 2>&1 | tee iridisx_probe.txt

echo "=============================================================="
echo " Host and Slurm version"
echo "=============================================================="
hostname
sinfo --version

echo
echo "=============================================================="
echo " Partitions, GPU resources, walltime limits"
echo " -> queue, gres and max_time in the gpu scripts"
echo "=============================================================="
sinfo -o "%20P %10l %10D %10G %t" -e

echo
echo "=============================================================="
echo " GPU node detail (gres per node)"
echo " -> gres in the gpu scripts. IridisX has BOTH NVIDIA and AMD"
echo "    GPU nodes, so note which types are which. A CUDA TensorFlow"
echo "    build must request an NVIDIA type explicitly"
echo "=============================================================="
sinfo -o "%20N %20P %40G" -e | grep -i gpu

echo
echo "=============================================================="
echo " SWARM and departmental partitions"
echo " -> ECS staff and PGRs can use SWARM (H100 80GB, A100 80GB)"
echo "=============================================================="
sinfo -o "%20P %10l %10D %40G" -e | grep -i -e swarm -e h100 -e a100 -e l4

echo
echo "=============================================================="
echo " Account and QoS associations"
echo " -> account and qos in the gpu scripts, empty means omit them"
echo "=============================================================="
sacctmgr show assoc user="${USER}" format=Account,Partition,QOS%40 --noheader

echo
echo "=============================================================="
echo " Queue limits for this user"
echo "=============================================================="
sacctmgr show qos format=Name,MaxSubmitJobsPU,MaxJobsPU,MaxTRESPU%30 --noheader

echo
echo "=============================================================="
echo " Filesystem roots"
echo " -> local_path in the gpu scripts"
echo "=============================================================="
echo "HOME:  ${HOME}"
for candidate in /home/"${USER}" /scratch/"${USER}" "${HOME}"/scratch \
    /iridisfs/home/"${USER}" /iridisfs/scratch/"${USER}"; do
    ls -ld "${candidate}" 2>/dev/null || echo "no ${candidate}"
done
echo "Home quota:"
quota -s 2>/dev/null || echo "quota unavailable"

echo
echo "=============================================================="
echo " Conda modules"
echo " -> conda_module in the gpu scripts"
echo "=============================================================="
module avail conda 2>&1 | head -30
module avail anaconda 2>&1 | head -30

echo
echo "=============================================================="
echo " CUDA, cuDNN and container modules"
echo " -> needed to build the tsml-eval-gpu environment"
echo "=============================================================="
module avail cuda 2>&1 | head -30
module avail cudnn 2>&1 | head -20
module avail apptainer 2>&1 | head -20
module avail singularity 2>&1 | head -20

echo
echo "Probe complete. Paste the output into the setup discussion so the"
echo "TODO-IRIDISX entries in gpu_scripts can be filled in."
