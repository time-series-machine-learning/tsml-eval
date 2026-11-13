# Getting set up on Iridis, the way we work

This is the practical route from a Windows desktop to running experiments on Iridis 6
(CPU) and Iridis X (GPU, plus the amd CPU partition), driven by short ssh commands and
the scripts in this directory. `README.md` here is the reference for the controller
(`tsml_slurm.py`) itself; this guide covers getting to the point where you can use it,
and the habits that keep it working.

Throughout, `<user>` is your Southampton username (ours is `ajb2u23`).

## Contents

- [The shape of it](#the-shape-of-it)
- [Rules we keep](#rules-we-keep)
- [1. The desktop](#1-the-desktop)
- [2. An ssh key](#2-an-ssh-key)
- [3. The cluster side, once](#3-the-cluster-side-once)
- [4. Data](#4-data)
- [5. Running commands from the desktop](#5-running-commands-from-the-desktop)
- [6. Running an experiment](#6-running-an-experiment)
- [7. Getting results back](#7-getting-results-back)
- [Iridis quirks](#iridis-quirks)
- [Cheat sheet](#cheat-sheet)

## The shape of it

```
desktop (Git Bash, ssh key)  --VPN-->  Iridis 6 login   iridis6_b.soton.ac.uk
                                       Iridis X login   loginX001.iridis.soton.ac.uk
                                              \             /
                                           one shared home: ~/Code, ~/Data, ~/Results
```

- All the editing, git commits and analysis happen on the desktop.
- The clusters only run jobs. Code reaches them through `git pull` of a pushed branch.
  Data and config files reach them by `scp` or a tar stream from the desktop.
- Results come back the same way, into one master copy on the desktop (ours is
  `D:/Results/EEG`).
- Iridis 6 and Iridis X share one home filesystem. Results written by either appear in
  the same `~/Results/...`, so you pull from one login only.

## Rules we keep

1. **No outgoing connections from the clusters by scripts or agents.** Nothing run
   over ssh does `git pull`, `pip install`, `conda install`, `curl` or a model
   download. The account holder does pulls and installs themselves, interactively.
   Desktop to cluster (`ssh`, `scp`, tar streams) is fine. Jobs that might reach for
   Hugging Face run with `HF_HUB_OFFLINE=1`.
2. **Heavy work only through Slurm.** Login nodes are for editing, `check`, `status`
   and the supervisor loop. Never train or evaluate on a login node.
3. **Your own files and jobs only.**
4. **Read freely, ask before changing.** `squeue`, `sacct`, `status`, reading logs,
   `ls` and `du` need no sign-off. Submitting, `start`, `stop`, `cancel`, `retry`, and
   moving or deleting results are decided by the account holder.
5. **Never run jobs from a checkout you are editing.** Jobs import the code in the
   checkout, so a half-made change there leaks into running work.

## 1. The desktop

- **Git for Windows**, which brings Git Bash with `ssh`, `scp`, `tar` and `gzip`.
  There is no `rsync`; `pull_results.sh` and `push_data.sh` exist to work around that.
- **The University of Southampton VPN.** Iridis is reachable only through it. In
  practice you can be on one university VPN at a time, so Iridis and Hali (UEA) are not
  reachable together.
- **A local clone of tsml-eval** on the runner branch (`ajb/slurm-runner`), for
  editing experiment files and pushing them.
- Python 3.11 or later, with pandas, for the results tools.

Line endings: a file written on Windows with CRLF breaks bash on the cluster
(`$'\r': command not found`). Save `.sh` files with LF endings (or set
`git config core.autocrlf input`, as nothing in the repo enforces it); for a stray
file, run `dos2unix file` on the cluster.

## 2. An ssh key

Everything below assumes a key, so that commands run without a password prompt.
`BatchMode=yes` then makes any prompt fail at once instead of hanging a script.

```bash
# on the desktop, in Git Bash: make a key if you have none
[ -f ~/.ssh/id_ed25519 ] || ssh-keygen -q -t ed25519 -N "" -C "my-desktop" -f ~/.ssh/id_ed25519

# install it (asks for your password once); one install covers Iridis 6 and X
ssh -o StrictHostKeyChecking=accept-new <user>@iridis6_b.soton.ac.uk \
  "mkdir -p ~/.ssh && chmod 700 ~/.ssh && cat >> ~/.ssh/authorized_keys && chmod 600 ~/.ssh/authorized_keys" \
  < ~/.ssh/id_ed25519.pub

# test: must print a host name with no prompt
ssh -o BatchMode=yes <user>@iridis6_b.soton.ac.uk hostname
ssh -o BatchMode=yes <user>@loginX001.iridis.soton.ac.uk hostname
```

If the key is offered but refused, it is nearly always permissions. sshd ignores keys
when the home folder is group- or world-writable, or when `~/.ssh` is not
`drwx------`. Ours was `drwxrwxrwx` on Iridis. Fix it from an interactive login:

```bash
chmod go-w ~; chmod 700 ~/.ssh; chmod 600 ~/.ssh/authorized_keys
```

To revoke a machine later, delete its line (the `-C` comment) from
`~/.ssh/authorized_keys`.

If you cannot paste into Git Bash, put the install command in a small script in your
home folder (`bash iridis_key.sh`), or paste
`echo '<contents of id_ed25519.pub>' >> ~/.ssh/authorized_keys` into a PuTTY session.

## 3. The cluster side, once

Log in interactively for this part: it needs the network and your judgement, so it
is the account holder's job, not a script's.

**Checkouts.** Both live in the shared home:

| Checkout | Used by | Branch |
| --- | --- | --- |
| `~/Code/tsml-eval` | Iridis 6 CPU, and Iridis X amd CPU experiments | `ajb/slurm-runner` |
| `~/Code/tsml-eval-gpu` | Iridis X GPU experiments | the GPU branch the experiment names |

```bash
mkdir -p ~/Code && cd ~/Code
git clone https://github.com/time-series-machine-learning/tsml-eval.git
cd tsml-eval && git checkout ajb/slurm-runner
```

Clone aeon-neuro beside it if you need the EEG estimators, on the branch the
experiments use.

**Environments.** Conda comes from a module on both clusters:

```bash
module load conda/python3
conda create -n tsml-eval python=3.12
source activate tsml-eval
export PYTHONNOUSERSITE=1          # ignore anything in ~/.local
pip install --editable ~/Code/tsml-eval
pip install --editable ~/Code/aeon-neuro    # for the EEG estimators
```

| Env | Cluster | For |
| --- | --- | --- |
| `tsml-eval` | Iridis 6, Iridis X amd | CPU estimators |
| `tsml-eval-gpu` | Iridis X | TensorFlow deep learners (`make_tensorflow_only.sh`) |
| `tsml-eval-torch` | Iridis X | PyTorch / Braindecode learners (`make_torch_env.sh`) |

Keep TensorFlow and PyTorch in separate envs: two CUDA stacks in one env break both.
Iridis 6 GPU jobs instead use an Apptainer sandbox at `~/scratch/tensorflow_sandbox`
(see `../soton/iridis/iridis_python.md`). Test a GPU env with a short interactive job,
for example:

```bash
srun -p i7_h200 --nodes=1 --gres=gpu:h200:1 --time=00:10:00 --pty \
  ~/.conda/envs/tsml-eval-torch/bin/python -c "import torch; print(torch.cuda.is_available())"
```

**An alias** for the controller, in `~/.bashrc`:

```bash
alias tsml-slurm='~/.conda/envs/tsml-eval/bin/python ~/Code/tsml-eval/_tsml_research_resources/slurm/tsml_slurm.py'
```

From then on, **updating code** is one line, run by the account holder:

```bash
cd ~/Code/tsml-eval && git pull
```

New dependencies are the only reason to touch the env again.

## 4. Data

Data goes up from the desktop with `push_data.sh`. It sends one dataset folder at a
time as a gzip tar stream, smallest first, checks the size on arrival, and records
what is done, so a dropped VPN costs only the dataset in flight. Rerun it to resume.

```bash
bash push_data.sh /d/Data/EEG <user>@iridis6_b.soton.ac.uk /iridisfs/home/<user>/Data/EEG
```

The layout is the usual tsml one, `~/Data/<Archive>/<Dataset>/<Dataset>_TRAIN.ts`.
Subject-wise CV also needs `<Dataset>_id_TRAIN.txt` and `_id_TEST.txt` beside the
`.ts` files. Dataset lists do not need copying: experiments read them from
`_tsml_research_resources/dataset_lists/` in the checkout.

For a single file, `scp` is enough:

```bash
scp eeg_extra.txt <user>@iridis6_b.soton.ac.uk:Code/tsml-eval/_tsml_research_resources/dataset_lists/
```

Iridis 6's home is `/iridisfs/home/<user>`, Iridis X's is `/home/<user>`. They are the
same files under two mount names. Paths relative to `~` work on both.

## 5. Running commands from the desktop

Every interaction is one non-interactive ssh call. The pattern:

```bash
ssh -o BatchMode=yes <user>@iridis6_b.soton.ac.uk '<commands>'
```

Things that make this reliable:

- **Load the environment inside the command.** A non-interactive shell does not read
  your aliases. Either load the module:

  ```bash
  ssh -o BatchMode=yes <user>@iridis6_b.soton.ac.uk \
    'module load conda/python3 >/dev/null 2>&1; source activate tsml-eval >/dev/null 2>&1;
     cd ~/Code/tsml-eval/_tsml_research_resources/slurm && python tsml_slurm.py status experiments/iridis6_eeg_aeon.toml'
  ```

  or call the env's interpreter directly, which needs no module:

  ```bash
  ssh -o BatchMode=yes <user>@iridis6_b.soton.ac.uk \
    'cd ~/Code/tsml-eval/_tsml_research_resources/slurm &&
     PYTHONNOUSERSITE=1 ~/.conda/envs/tsml-eval/bin/python tsml_slurm.py status experiments/iridis6_eeg_aeon.toml'
  ```

- **Batch commands into one login.** Logins can take tens of seconds, so check five
  experiments in one call, not five calls:

  ```bash
  ssh -o BatchMode=yes <user>@iridis6_b.soton.ac.uk 'cd ~/Code/tsml-eval/_tsml_research_resources/slurm;
    for e in iridis6_eeg_aeon iridis6_eeg_hc2_components iridis6_eeg_rotf_ch; do
      PYTHONNOUSERSITE=1 ~/.conda/envs/tsml-eval/bin/python tsml_slurm.py status experiments/$e.toml | head -3; done'
  ```

- **Long-running commands** (a big pull, a `check` that builds every estimator): run
  them in the background on the desktop and wait, rather than holding a terminal.
- **Scripts over stdin.** For anything longer than a line, pipe a script in instead of
  quoting it: `ssh host 'python -' < probe.py` or `ssh host bash -s < script.sh`. This
  is how the Neuroverse registry probes the clusters.
- **Read-only first.** `squeue -u $USER`, `sacct -j <job>`, `status`, `tail` of a log
  and `ls`/`du` are safe at any time. See [Rules we keep](#rules-we-keep) for the rest.

## 6. Running an experiment

The cycle we use for every run:

1. **Write or copy an experiment file on the desktop**, in `experiments/`, named
   `<cluster>_<project>_<what>.toml`. Copy the nearest existing one (for example
   `iridis6_eeg_aeon.toml` for a CPU farm or `iridisx_eeg_gpu.toml` for H200s) and change
   `name`, the dataset list and the estimators.
2. **Check it locally** for typos: `python tsml_slurm.py check experiments/<file>.toml`
   (a data-path error is expected on the desktop).
3. **Commit and push** the file to the runner branch.
4. **Pull on the cluster** (account holder): `cd ~/Code/tsml-eval && git pull`.
5. **Check on the cluster**, with the env's Python so it also builds every estimator:

   ```bash
   tsml-slurm check experiments/<file>.toml
   tsml-slurm submit experiments/<file>.toml --dry-run
   ```

6. **Start it** (account holder's decision): `tsml-slurm start experiments/<file>.toml`.
7. **Watch it** from the desktop with `status`; see the cheat sheet.

To try something without a commit, `scp` the toml to the cluster (for example to
`~/tsml-slurm/<name>/`) and point the commands at that copy. It still needs
`require_clean = false` if the checkout has changes, and the registry finds configs
there too.

**Which resource to choose:**

| You want | Cluster and resource | Notes |
| --- | --- | --- |
| Many CPU tasks | Iridis 6 `cpu` (farm) | 192-core batch nodes, up to 4 node jobs, 60 h. Supervisor is a Slurm job. |
| Many CPU tasks, Iridis 6 full | Iridis X `cpu` (farm, amd) | 64-core, 240 GB nodes, up to 8, 2.5 days. Set `repo_dir = "{home}/Code/tsml-eval"`. |
| A few slow or very uneven CPU tasks | Iridis 6 `cpu-array` | One Slurm task each; a farm would idle waiting for its slowest task. |
| More than 240 GB | Iridis 6 | Memory tiers go to 620 GB there. |
| GPU | Iridis X `gpu` (H200), `gpu-a100`, `gpu-swarm` | Each has its own `gres` string, set in the profile. Swarm needs `account = "ecs"`. |
| Longer than 2.5 days or 60 h | Hali (7 days) | Or checkpoint (`-ch`) and `retry_timeouts`. |

**Out of memory** is handled for you: the task is resubmitted at the next memory tier
until the top tier. To restart OOM failures from a higher tier after raising the
tiers:

```bash
tsml-slurm retry experiments/<file>.toml --reason OOM --memory-tier 4
```

**Time limits are not retried.** Move those tasks to a cluster with longer jobs, or
use checkpointing.

## 7. Getting results back

Results land in `~/Results/<Archive>/<Category>/<Estimator>/Predictions/<Dataset>/`.

- **EEG benchmark:** use the Neuroverse registry on the desktop. It asks the cluster
  which cells are done, fetches only those the desktop lacks, and rebuilds the list
  of what is missing and why:

  ```bash
  cd C:/Code/EEGBench
  python tools/results_registry.py pull iridis           # refresh, fetch, rebuild
  python tools/results_registry.py pull iridis --dry-run
  ```

  `results/registry/summary.md` then says what is still running, failed or
  unscheduled.

- **Anything else:** `pull_results.sh` copies new or changed `Predictions` files in
  two logins, never deleting or changing anything:

  ```bash
  bash pull_results.sh <user>@iridis6_b.soton.ac.uk Results/EEG /d/Results/EEG
  bash pull_results.sh <user>@iridis6_b.soton.ac.uk Results/UCR /d/Results/UCR "./*/QUANT/Predictions/*"
  ```

Skip `Checkpoints/`: they are large and are deleted when a task finishes.

To stop a cluster rerunning work whose files you have moved off it, build
`completed.txt` from the master copy and copy it up:

```bash
python registry.py D:/Results/EEG
scp D:/Results/EEG/completed.txt <user>@iridis6_b.soton.ac.uk:Results/EEG/
```

## Iridis quirks

- **Two logins, two queues.** The Iridis 6 login cannot see Iridis X jobs. `status` of
  an `iridisx_*` experiment, and `squeue` for its jobs, must run on `loginX001`.
- **Iridis X supervisors live in `screen` on one login node.** `start` opens a
  detached session `tsml-<name>` on whichever login node you are on. `screen -ls` on
  another node shows nothing, so always use the same node (`loginX001`) for Iridis X.
  After a login-node reboot, run `start` again; nothing is lost.
- **Iridis 6 supervisors are Slurm jobs** that resubmit themselves every 15 minutes,
  so nothing needs to stay on the login node.
- **Farm progress.** A farm runs many tasks inside one job, so `squeue` shows the
  farm, not its tasks. Use `status --running` to see the tasks running inside farms.
- **Near-full-node farms wait for an empty node.** If Iridis 6 farms sit pending,
  lower `node_memory_gb` or `node_cpus` in the experiment's `[slurm]` table.
- **Shared home, two mount names.** `/iridisfs/home/<user>` on Iridis 6,
  `/home/<user>` on Iridis X.
- **Planned shutdowns.** Pull results before a maintenance window, and `stop` the
  supervisors beforehand so they do not resubmit into a draining queue.

## Cheat sheet

From the desktop, with `I6=<user>@iridis6_b.soton.ac.uk`, `IX=<user>@loginX001.iridis.soton.ac.uk`
and `R='cd ~/Code/tsml-eval/_tsml_research_resources/slurm && PYTHONNOUSERSITE=1 ~/.conda/envs/tsml-eval/bin/python tsml_slurm.py'`:

```bash
ssh -o BatchMode=yes $I6 "$R status experiments/iridis6_eeg_aeon.toml"             # progress
ssh -o BatchMode=yes $I6 "$R status experiments/iridis6_eeg_aeon.toml --running"   # what is running, longest first
ssh -o BatchMode=yes $I6 "$R status experiments/iridis6_eeg_aeon.toml --details"   # every missing task
ssh -o BatchMode=yes $IX "$R status experiments/iridisx_eeg_mi_cv.toml"            # Iridis X: use its own login
ssh -o BatchMode=yes $I6 'squeue -u $USER'                                          # the queue
ssh -o BatchMode=yes $I6 'tail -50 ~/tsml-slurm/<name>/logs/<Estimator>/<Dataset>/<jobid>_0.err'
```

Changes, once agreed (`start`, `stop`, `cancel`, `retry`):

```bash
ssh -o BatchMode=yes $I6 "$R start experiments/<file>.toml"
ssh -o BatchMode=yes $I6 "$R stop experiments/<file>.toml"            # add --cancel to kill its jobs
ssh -o BatchMode=yes $I6 "$R cancel experiments/<file>.toml --estimator X --dataset Y"
ssh -o BatchMode=yes $I6 "$R retry experiments/<file>.toml --reason OOM --memory-tier 3"
```

Account holder only, interactively on the cluster:

```bash
cd ~/Code/tsml-eval && git pull
```
