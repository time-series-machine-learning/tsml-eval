"""Run tsml_slurm experiments on one machine, without Slurm (e.g. Hydra).

    python run_local.py experiments/hydra_*.toml            # run until done
    python run_local.py experiments/hydra_*.toml --dry-run  # list what would run
    python run_local.py experiments/hydra_*.toml --status   # progress and failures

The experiment files are the usual ones, with a cluster profile for the machine
(clusters/hydra.toml), so results land in the same layout as on the clusters and
can be merged by copying. Every experiment given shares one budget of CPUs, memory
and GPUs: a task needs max(n_jobs, cpus_per_task) CPUs, its memory tier in GB, and
one whole GPU if its resources have a gres. GPU tasks see only their own GPU
(CUDA_VISIBLE_DEVICES); CPU tasks see none.

As on the clusters, a task is complete when its results file exists, so stopping
and starting again is safe and nothing is rerun. Failures are diagnosed from the
logs: an out-of-memory task moves up a memory tier, a task over the resources'
time_limit is stopped and counts as timed out, others are retried up to
max_attempts. Tasks that failed for good are skipped until --retry. The dataset
list is re-read every --rescan minutes, so added datasets are picked up.

State is in <state_dir>/local.json and logs in <state_dir>/logs/<Estimator>/
<Dataset>/local-<time>_<resample>.{out,err,status}. Run it in tmux or screen, or
with nohup; Ctrl-C stops the running tasks and charges them no attempt.
"""

import argparse
import json
import os
import signal
import socket
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import tsml_slurm as ts

HERE = Path(__file__).resolve().parent
POLL_SECONDS = 5


# --------------------------------------------------------------------------------------
# Machine and environments
# --------------------------------------------------------------------------------------


def machine_memory_gb():
    """Total memory in GB from /proc/meminfo, or 0 if unknown."""
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemTotal:"):
                return int(line.split()[1]) // (1024 * 1024)
    except OSError:
        pass
    return 0


def machine_gpus():
    """GPU indices from nvidia-smi, or [] without one."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        return []
    return [line.strip() for line in out.splitlines() if line.strip()]


def env_python(env_name):
    """Path of a conda env's python, searching the usual places."""
    roots = []
    if os.environ.get("CONDA_EXE"):
        roots.append(Path(os.environ["CONDA_EXE"]).resolve().parent.parent)
    home = Path.home()
    roots += [home / "miniforge3", home / "miniconda3", home / "anaconda3"]
    candidates = [home / ".conda" / "envs" / env_name / "bin" / "python"]
    candidates += [r / "envs" / env_name / "bin" / "python" for r in roots]
    for c in candidates:
        if c.is_file():
            return c
    return None


_GPU_CHECKS = {
    "tensorflow": "import tensorflow as tf, sys; "
    "sys.exit(0 if tf.config.list_physical_devices('GPU') else 1)",
    "torch": "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)",
}


def gpu_problem(python, framework, gpu):
    """Why the framework cannot see the GPU in this env, or '' if it can.

    Checked once per env before its first GPU task, so a broken CUDA install fails
    fast instead of every task training silently on the CPU.
    """
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu, TF_CPP_MIN_LOG_LEVEL="2")
    env.pop("PYTHONPATH", None)
    done = subprocess.run(
        [str(python), "-c", _GPU_CHECKS[framework]],
        env=env,
        capture_output=True,
        text=True,
    )
    if done.returncode == 0:
        return ""
    last = ts._last_error_line(done.stderr)
    return f"{framework} sees no GPU" + (f" ({last})" if last else "")


# --------------------------------------------------------------------------------------
# Tasks
# --------------------------------------------------------------------------------------


def load_local_state(e):
    """Per-task records of this machine's runs: attempts, tier, terminal, events."""
    path = e.state_dir / "local.json"
    if path.is_file():
        return json.loads(path.read_text(encoding="utf-8"))
    return {"experiment": e.name, "tasks": {}}


def save_local_state(e, state):
    """Write local.json atomically."""
    e.state_dir.mkdir(parents=True, exist_ok=True)
    path = e.state_dir / "local.json"
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=1, sort_keys=True), encoding="utf-8")
    tmp.replace(path)


def needs(e, record):
    """(cpus, memory GB, gpus) one task of this experiment takes."""
    r = e.resources
    cpus = max(e.n_jobs, r.cpus_per_task)
    tier = min(record.get("tier", 0), len(r.memory_gb) - 1)
    return cpus, r.memory_gb[tier], 1 if r.gpu else 0


def pending_tasks(e, state, running_keys):
    """Incomplete tasks in priority order, minus those running or failed for good."""
    e.datasets = tuple(ts.read_datasets(e))
    missing = [t for t in ts.all_tasks(e) if not ts.is_complete(e, t)]
    keep = [
        t
        for t in missing
        if t.key not in running_keys
        and not state["tasks"].get(t.key, {}).get("terminal")
    ]
    return ts.order_tasks(e, keep)


class Run:
    """One task running as a child process."""

    def __init__(self, e, task, record, gpu, python, commit):
        self.e, self.task, self.record, self.gpu = e, task, record, gpu
        self.cpus, self.memory, _ = needs(e, record)
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.log = ts.task_log(e, task, f"local-{stamp}")
        self.log.parent.mkdir(parents=True, exist_ok=True)
        self.limit = ts.time_limit_seconds(e.resources.time_limit)
        self.start = time.time()
        self.timed_out = False
        env = dict(os.environ)
        for var in ("PYTHONPATH", "CONDA_PREFIX", "CONDA_DEFAULT_ENV"):
            env.pop(var, None)
        env.update(
            PYTHONPATH=str(e.repo_dir),
            NUMBA_CACHE_DIR=str(e.numba_cache_dir),
            PYTHONUNBUFFERED="1",
            TF_CPP_MIN_LOG_LEVEL="2",
            LOKY_MAX_CPU_COUNT=str(self.cpus),
            CUDA_VISIBLE_DEVICES=gpu if gpu is not None else "",
        )
        for var in ts._THREAD_VARS:
            env[var] = "1"
        e.numba_cache_dir.mkdir(parents=True, exist_ok=True)
        cmd = [str(python), "-m", ts._entry_module(e), str(e.data_dir)]
        cmd += ts._experiment_arguments(e, task)
        self.out = open(f"{self.log}.out", "w", encoding="utf-8")  # noqa: SIM115
        self.err = open(f"{self.log}.err", "w", encoding="utf-8")  # noqa: SIM115
        self.out.write(
            f"host={socket.gethostname()} job=local commit={commit} "
            f"memory_gb={self.memory} cpus={self.cpus} gpu={gpu} "
            f"python={python}\n{' '.join(cmd)}\n"
        )
        self.out.flush()
        Path(f"{self.log}.status").write_text(f"started {ts._now()}\n")
        self.proc = subprocess.Popen(
            cmd,
            cwd=str(e.repo_dir),
            env=env,
            stdout=self.out,
            stderr=self.err,
            start_new_session=True,  # its own process group, killed as one
        )

    def poll(self):
        """Exit code, or None while running; kills the task past its time limit."""
        code = self.proc.poll()
        if code is None and time.time() - self.start > self.limit:
            self.timed_out = True
            self.kill()
            code = self.proc.wait()
        return code

    def kill(self):
        """Stop the task's whole process group."""
        try:
            os.killpg(self.proc.pid, signal.SIGKILL)
        except (OSError, AttributeError):
            self.proc.kill()

    def close(self, code):
        """Write the status file and close the logs."""
        with open(f"{self.log}.status", "a", encoding="utf-8") as f:
            f.write(f"exit {code} {ts._now()}\n")
        self.out.close()
        self.err.close()


def outcome(run, code):
    """Diagnose a finished run: None when its results exist, else (reason, detail)."""
    e, task = run.e, run.task
    if ts.is_complete(e, task):
        return None
    err = ts._read_tail(Path(f"{run.log}.err"))
    text = err + ts._read_tail(Path(f"{run.log}.out"))
    detail = ts._last_error_line(err)
    if run.timed_out:
        return ts.TIMEOUT, detail
    reason = ts._classify_text(text)
    if code in (137, -9):  # SIGKILL: how the kernel OOM killer ends a process
        return (ts.GPU_OOM if reason == ts.GPU_OOM else ts.OOM), detail
    if code == 0:
        return ts.NO_RESULT, detail
    return reason or ts.UNKNOWN, detail


# --------------------------------------------------------------------------------------
# The scheduler
# --------------------------------------------------------------------------------------


class Scheduler:
    """Runs the tasks of several experiments within one machine's budget."""

    def __init__(self, experiments, cpus, memory_gb, gpus, rescan_minutes, log=print):
        self.experiments = experiments
        self.cpus, self.memory_gb, self.gpus = cpus, memory_gb, list(gpus)
        self.rescan = rescan_minutes * 60
        self.log = log
        self.states = {e.name: load_local_state(e) for e in experiments}
        self.pythons = {}
        self.commits = {}
        self.running = []
        self.queues = {}
        self.last_scan = 0.0
        self.stopping = False

    def scan(self):
        """Rebuild each experiment's queue of pending tasks."""
        keys = {(r.e.name, r.task.key) for r in self.running}
        for e in self.experiments:
            state = self.states[e.name]
            running = {k for n, k in keys if n == e.name}
            self.queues[e.name] = pending_tasks(e, state, running)
        self.last_scan = time.time()

    def free(self):
        """(cpus, memory, gpu ids) not in use."""
        cpus = self.cpus - sum(r.cpus for r in self.running)
        memory = self.memory_gb - sum(r.memory for r in self.running)
        used = {r.gpu for r in self.running if r.gpu is not None}
        return cpus, memory, [g for g in self.gpus if g not in used]

    def fits(self, e, record):
        """Whether a task of e with this record could ever run on the machine."""
        cpus, memory, gpu = needs(e, record)
        return cpus <= self.cpus and memory <= self.memory_gb and gpu <= len(self.gpus)

    def launch(self):
        """Start tasks while they fit, taking experiments in turn."""
        started = True
        while started and not self.stopping:
            started = False
            for e in self.experiments:
                queue = self.queues.get(e.name, [])
                state = self.states[e.name]
                while queue:
                    task = queue[0]
                    record = state["tasks"].setdefault(
                        task.key, {"attempts": 0, "tier": 0}
                    )
                    if not self.fits(e, record):
                        record["terminal"] = "too big for this machine"
                        save_local_state(e, state)
                        queue.pop(0)
                        continue
                    break
                if not queue:
                    continue
                cpus, memory, gpu = needs(e, record)
                free_cpus, free_memory, free_gpus = self.free()
                if cpus > free_cpus or memory > free_memory or gpu > len(free_gpus):
                    continue
                if ts.is_complete(e, task):  # finished elsewhere since the scan
                    queue.pop(0)
                    started = True
                    continue
                python = self.python(e)
                if python is None:
                    queue.clear()
                    continue
                queue.pop(0)
                record["attempts"] = record.get("attempts", 0) + 1
                save_local_state(e, state)
                run = Run(
                    e,
                    task,
                    record,
                    free_gpus[0] if gpu else None,
                    python,
                    self.commit(e),
                )
                self.running.append(run)
                self.log(
                    f"start  {e.name} {task.key} {memory}G {cpus}cpu"
                    + (f" gpu{run.gpu}" if gpu else "")
                )
                started = True

    def python(self, e):
        """Return the env's python for e; None (logged once) if missing or GPU-blind."""
        r = e.resources
        key = (r.env_name, r.gpu_check)
        if key not in self.pythons:
            python = env_python(r.env_name)
            if python is None:
                self.log(f"ERROR  no python for conda env {r.env_name!r}: skipping")
            elif r.gpu_check != "none" and self.gpus:
                problem = gpu_problem(python, r.gpu_check, self.gpus[0])
                if problem:
                    self.log(f"ERROR  {r.env_name}: {problem}: skipping its GPU work")
                    python = None
            self.pythons[key] = python
        return self.pythons[key]

    def commit(self, e):
        """Return the repository commit, for the logs."""
        if e.repo_dir not in self.commits:
            try:
                self.commits[e.repo_dir] = ts.git_revision(e.repo_dir)[1]
            except (OSError, subprocess.CalledProcessError):
                self.commits[e.repo_dir] = "unknown"
        return self.commits[e.repo_dir]

    def reap(self):
        """Collect finished runs and decide their retries."""
        for run in list(self.running):
            code = run.poll()
            if code is None:
                continue
            run.close(code)
            self.running.remove(run)
            e, state = run.e, self.states[run.e.name]
            record = state["tasks"][run.task.key]
            result = outcome(run, code)
            took = ts._duration(time.time() - run.start)
            if result is None:
                record.pop("terminal", None)
                record["done"] = ts._now()
                self.log(f"done   {e.name} {run.task.key} in {took}")
            else:
                reason, detail = result
                record["job"] = "local"
                ts.apply_outcome(e, record, reason, detail)
                if not self.stopping and not record.get("terminal"):
                    self.queues.setdefault(e.name, []).insert(0, run.task)
                terminal = record.get("terminal")
                self.log(
                    f"FAIL   {e.name} {run.task.key} after {took}: {reason}"
                    + (f", failed for good: {terminal}" if terminal else ", will retry")
                    + (f" ({detail})" if detail else "")
                )
            save_local_state(e, state)

    def stop(self, *_):
        """Stop every running task; they are not charged an attempt."""
        self.stopping = True
        for run in self.running:
            run.kill()
        for run in self.running:
            run.proc.wait()
            run.close("cancelled")
            record = self.states[run.e.name]["tasks"][run.task.key]
            record["attempts"] = max(0, record.get("attempts", 1) - 1)
            save_local_state(run.e, self.states[run.e.name])
        self.running = []
        self.log("stopped")

    def idle(self):
        """Whether nothing is running and nothing is waiting."""
        return not self.running and not any(self.queues.values())

    def run(self):
        """Loop until every task is complete or failed for good."""
        self.scan()
        while not self.stopping:
            self.reap()
            if time.time() - self.last_scan > self.rescan:
                self.scan()
            self.launch()
            if self.idle():
                self.scan()  # catch datasets added since the last scan
                if self.idle():
                    break
            time.sleep(POLL_SECONDS)
        self.log("finished: nothing left to run")


# --------------------------------------------------------------------------------------
# Reports
# --------------------------------------------------------------------------------------


def status_lines(e):
    """Completion and failures of one experiment on this machine."""
    state = load_local_state(e)
    e.datasets = tuple(ts.read_datasets(e))
    tasks = list(ts.all_tasks(e))
    complete = sum(1 for t in tasks if ts.is_complete(e, t))
    failed = {
        t.key: state["tasks"][t.key]
        for t in tasks
        if state["tasks"].get(t.key, {}).get("terminal") and not ts.is_complete(e, t)
    }
    lines = [
        f"{e.name}: {complete}/{len(tasks)} complete "
        f"({100 * complete / max(len(tasks), 1):.0f}%), {len(failed)} failed for good"
    ]
    for key, record in sorted(failed.items()):
        events = record.get("events") or [{}]
        detail = events[-1].get("detail", "")
        lines.append(
            f"  {key}: {record['terminal']}" + (f"  {detail}" if detail else "")
        )
    return lines


def retry(e, reason=None):
    """Clear failed-for-good records (of one reason, if given); returns how many."""
    state = load_local_state(e)
    n = 0
    for record in state["tasks"].values():
        terminal = record.get("terminal")
        if terminal and (reason is None or reason.lower() in terminal.lower()):
            record.pop("terminal")
            record["attempts"] = 0
            n += 1
    save_local_state(e, state)
    return n


def main(argv=None):
    """Run, list or report the experiments given."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("configs", nargs="+", help="experiment TOMLs")
    parser.add_argument("--cpus", type=int, default=os.cpu_count() or 1)
    parser.add_argument(
        "--memory-gb",
        type=int,
        default=max(machine_memory_gb() - 8, 1),
        help="memory budget (default: the machine's, less 8 GB)",
    )
    parser.add_argument(
        "--gpus", default=None, help="GPU ids to use, e.g. 0,1,2 (default: all)"
    )
    parser.add_argument("--rescan", type=float, default=10, help="minutes")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--status", action="store_true")
    parser.add_argument(
        "--retry",
        nargs="?",
        const="",
        help="clear failed-for-good tasks (optionally only this reason) first",
    )
    args = parser.parse_args(argv)
    experiments = [ts.load_experiment(c) for c in args.configs]

    if args.status:
        for e in experiments:
            print("\n".join(status_lines(e)))
        return 0
    if args.retry is not None:
        for e in experiments:
            print(f"{e.name}: {retry(e, args.retry or None)} tasks cleared for retry")
    gpus = args.gpus.split(",") if args.gpus else machine_gpus()
    if args.dry_run:
        for e in experiments:
            tasks = pending_tasks(e, load_local_state(e), set())
            cpus, memory, gpu = needs(e, {})
            print(
                f"{e.name}: {len(tasks)} to run, each {cpus} cpu, {memory}G"
                + (", 1 GPU" if gpu else "")
                + f", python {env_python(e.resources.env_name)}"
            )
            for t in tasks[:10]:
                print(f"  {t.key}")
            if len(tasks) > 10:
                print(f"  ... and {len(tasks) - 10} more")
        print(f"budget: {args.cpus} cpus, {args.memory_gb}G, GPUs {gpus or 'none'}")
        return 0

    def log(message):
        print(f"{datetime.now():%Y-%m-%d %H:%M:%S} {message}", flush=True)

    scheduler = Scheduler(
        experiments, args.cpus, args.memory_gb, gpus, args.rescan, log=log
    )
    signal.signal(signal.SIGTERM, scheduler.stop)
    log(f"budget: {args.cpus} cpus, {args.memory_gb}G, GPUs {gpus or 'none'}")
    try:
        scheduler.run()
    except KeyboardInterrupt:
        scheduler.stop()
    for e in experiments:
        print("\n".join(status_lines(e)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
