"""Run and monitor the Iridis 6 UCR reference task farms.

Only results under the configured UCR root count. CPU jobs use staskfarm;
GPU jobs run sequential commands with one allocated GPU and Apptainer.
The same config and completeness check drive submission, workers and monitoring.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import re
import shlex
import subprocess
import sys
import time
import traceback
import uuid
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
TERMINAL = {
    "COMPLETED", "FAILED", "CANCELLED", "TIMEOUT", "OUT_OF_MEMORY",
    "NODE_FAIL", "PREEMPTED", "BOOT_FAIL", "DEADLINE", "REVOKED",
}
THREAD_VARS = (
    "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS", "NUMBA_NUM_THREADS", "LOKY_MAX_CPU_COUNT",
    "TF_NUM_INTEROP_THREADS", "TF_NUM_INTRAOP_THREADS",
)


def read_json(path, default=None):
    """Read JSON, using a default only for an absent file."""
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else default


def save_json(path, value):
    """Atomically replace a state or status file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def load_config(path, args=None):
    """Resolve paths and validate the explicit benchmark manifest."""
    c = read_json(path)
    substitutions = {"username": c["username"], "repo_dir": c["repo_dir"]}
    substitutions["repo_dir"] = c["repo_dir"].format(**substitutions)
    for key in ("repo_dir", "aeon_dir", "data_dir", "results_root", "dataset_list", "python"):
        c[key] = c[key].format(**substitutions)
    c["gpu"]["container"] = c["gpu"]["container"].format(**substitutions)
    if args:
        for key in ("results_root", "data_dir", "dataset_list"):
            if getattr(args, key, None):
                c[key] = str(Path(getattr(args, key)).resolve())
    c["state_dir"] = str(Path(c["results_root"]) / ".ucr-reference-state")
    names = [r["name"] for r in c["classifiers"]]
    classes = [r["class"] for r in c["classifiers"]]
    if not names or len(names) != len(set(names)) or len(classes) != len(set(classes)):
        raise ValueError("Classifier names and classes must be nonempty and unique")
    safe = re.compile(r"[A-Za-z0-9_.-]+\Z")
    for row in c["classifiers"]:
        if not all(safe.fullmatch(row[k]) for k in ("name", "class", "category")):
            raise ValueError(f"Invalid classifier entry: {row}")
        if row["device"] not in ("cpu", "gpu") or not isinstance(row["train"], bool):
            raise ValueError(f"Invalid device/train flag: {row}")
    for key in ("resamples", "max_attempts", "max_failures", "max_rounds", "refill_minutes"):
        if not isinstance(c[key], int) or c[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    if not safe.fullmatch(c["job_prefix"]):
        raise ValueError("Invalid job prefix")
    for device in ("cpu", "gpu"):
        p = c[device]
        if p["max_jobs"] < 1 or not safe.fullmatch(p["partition"]):
            raise ValueError(f"Invalid {device} partition/job limit")
        levels = p["memory_gib"]
        if not levels or levels != sorted(set(levels)) or min(levels) < 1:
            raise ValueError(f"Invalid {device} memory tiers")
    if max(c["cpu"]["memory_gib"]) > c["cpu"]["node_memory_gib"]:
        raise ValueError("CPU memory tiers exceed the node memory budget")
    for value in (c["cpu"]["max_cpus"], c["cpu"]["waves_per_job"],
                  c["gpu"]["cpus"], c["gpu"]["commands_per_job"]):
        if not isinstance(value, int) or value < 1:
            raise ValueError("CPU counts and batch sizes must be positive integers")
    # Paths enter SBATCH directives as well as quoted shell commands.
    for key in ("repo_dir", "aeon_dir", "data_dir", "results_root", "python"):
        if any(ch.isspace() for ch in c[key]):
            raise ValueError(f"Cluster paths must not contain whitespace: {key}")
    return c


def datasets(c):
    """Read the clean UCR list without changing its ordering."""
    rows = [s.strip() for s in Path(c["dataset_list"]).read_text(encoding="utf-8-sig").splitlines()
            if s.strip() and not s.lstrip().startswith("#")]
    if not rows or len(rows) != len(set(rows)):
        raise ValueError("Dataset list must be nonempty and unique")
    if any(not re.fullmatch(r"[A-Za-z0-9_.-]+", d) for d in rows):
        raise ValueError("Invalid dataset name")
    return rows


def key_for(name, dataset, resample):
    """Return a stable experiment identifier."""
    return f"{name}|{dataset}|{resample}"


def result_paths(c, row, dataset, resample):
    """Return just the splits required for this reference collection."""
    base = Path(c["results_root"]) / row["category"] / row["name"] / "Predictions" / dataset
    return [base / f"{split}Resample{resample}.csv"
            for split in (["test", "train"] if row["train"] else ["test"])]


def nonempty(path):
    """Treat an absent or empty prediction file as incomplete."""
    try:
        return path.is_file() and path.stat().st_size > 0
    except FileNotFoundError:
        return False


def inventory(c, ds):
    """Scan dataset directories once, excluding ZIPs and other result roots."""
    complete = set()
    counts = {}
    pattern = re.compile(r"(test|train)Resample(\d+)\.csv\Z")
    for row in c["classifiers"]:
        tests = trains = done = full = 0
        for d in ds:
            directory = result_paths(c, row, d, 0)[0].parent
            splits = {"test": set(), "train": set()}
            if directory.exists():
                for f in directory.iterdir():
                    m = pattern.fullmatch(f.name)
                    if m and int(m[2]) < c["resamples"] and nonempty(f):
                        splits[m[1]].add(int(m[2]))
            tests += len(splits["test"])
            trains += len(splits["train"])
            indices = splits["test"] & splits["train"] if row["train"] else splits["test"]
            done += len(indices)
            full += len(indices) == c["resamples"]
            complete.update(key_for(row["name"], d, i) for i in indices)
        counts[row["name"]] = dict(test=tests, train=trains, complete=done, datasets=full)
    return complete, counts


def command(args, **kwargs):
    """Execute a command without suppressing scheduler or import failures."""
    result = subprocess.run(args, text=True, capture_output=True, **kwargs)
    if result.returncode:
        raise RuntimeError(f"{shlex.join(map(str, args))}:\n{result.stderr or result.stdout}")
    return result.stdout.strip()


def query_slurm(c):
    """Return all live user jobs; completing allocations remain reserved."""
    output = command(["squeue", "--noheader", f"--user={c['username']}",
                      "--format=%i|%100j|%T|%R"])
    return {parts[0]: dict(name=parts[1], state=parts[2], node=parts[3])
            for line in output.splitlines() if (parts := line.strip().split("|", 3)) and len(parts) == 4}


def accounting(job_id):
    """Read the allocation state, distinguishing accounting lag from failure."""
    output = command(["sacct", "--noheader", "--parsable2", f"--jobs={job_id}",
                      "--format=JobIDRaw,State"])
    for line in output.splitlines():
        fields = line.split("|")
        if fields[0] == job_id:
            return fields[1].split()[0].rstrip("+")
    return "UNKNOWN"


@contextlib.contextmanager
def locked(path):
    """Serialize submission cycles or executions on the Linux cluster."""
    import fcntl

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        yield


def provenance(c):
    """Fingerprint aeon and the experiment code used by the task farms."""
    aeon = command(["git", "-C", c["aeon_dir"], "rev-parse", "HEAD"])
    diff = command(["git", "-C", c["aeon_dir"], "diff", "HEAD", "--", "aeon"])
    source_hash = hashlib.sha256()
    for rel in ("tsml_eval/experiments/_get_classifier.py",
                "tsml_eval/experiments/classification_experiments.py",
                "tsml_eval/experiments/experiments.py", "tsml_eval/utils/experiments.py"):
        source_hash.update((Path(c["repo_dir"]) / rel).read_bytes())
    return {"aeon": aeon, "aeon_diff": hashlib.sha256(diff.encode()).hexdigest(),
            "experiment_sources": source_hash.hexdigest()}


def runtime(c, device, gpu=False):
    """Build the CPU or container interpreter command."""
    if device == "cpu":
        return [c["python"]]
    return ["apptainer", "exec", *(["--nv"] if gpu else []),
            "--bind", f"{c['repo_dir']}:{c['repo_dir']}",
            "--bind", f"{c['aeon_dir']}:{c['aeon_dir']}",
            "--bind", f"{c['data_dir']}:{c['data_dir']}",
            "--bind", f"{c['results_root']}:{c['results_root']}",
            c["gpu"]["container"], c["gpu"]["python"]]


def check_estimators(c, device):
    """Construct every configured estimator in its execution environment."""
    # This command is called inside the target Python/container, before submission.
    import aeon

    if not Path(aeon.__file__).resolve().is_relative_to(Path(c["aeon_dir"]).resolve()):
        raise RuntimeError(f"aeon imported from unexpected location: {aeon.__file__}")
    errors = {}
    for row in c["classifiers"]:
        if row["device"] != device:
            continue
        try:
            with contextlib.redirect_stdout(sys.stderr):
                make_classifier(row, 0)
        except Exception as error:
            errors[row["name"]] = f"{type(error).__name__}: {error}"
    print(json.dumps(errors))


def make_classifier(row, resample):
    """Resolve a factory key while retaining the existing result-directory name."""
    from tsml_eval.experiments import get_classifier_by_name

    estimator = get_classifier_by_name(row.get("key", row["name"]), random_state=resample, n_jobs=1)
    if type(estimator).__name__ != row["class"] or not type(estimator).__module__.startswith("aeon."):
        raise ValueError(f"{row['name']} resolved to {type(estimator).__module__}.{type(estimator).__name__}, expected aeon {row['class']}")
    return estimator


def preflight(c, config_path, devices):
    """Validate data and dependencies without fitting classifiers on the login node."""
    for d in datasets(c):
        for split in ("TRAIN", "TEST"):
            if not nonempty(Path(c["data_dir"]) / d / f"{d}_{split}.ts"):
                raise ValueError(f"Missing/empty data: {d}_{split}.ts under {c['data_dir']}")
    env = os.environ.copy()
    env.update({v: "1" for v in THREAD_VARS})
    env.update(PYTHONPATH=os.pathsep.join([c["aeon_dir"], c["repo_dir"]]), PYTHONNOUSERSITE="1")
    cache = Path(c["state_dir"]) / "numba-cache"
    cache.mkdir(parents=True, exist_ok=True)
    env["NUMBA_CACHE_DIR"] = str(cache)
    for name in (*THREAD_VARS, "PYTHONPATH", "PYTHONNOUSERSITE", "NUMBA_CACHE_DIR"):
        env["APPTAINERENV_" + name] = env[name]
    env.pop("PYTHONHOME", None)
    for device in devices:
        args = runtime(c, device) + [str(HERE / "controller.py"), "check-estimators",
                                    "--config", str(config_path), "--device", device]
        if device == "gpu":
            # Apptainer comes from the same module used by existing GPU scripts.
            args = ["bash", "-lc", f"module load {shlex.quote(c['gpu']['module'])}; exec {shlex.join(args)}"]
        errors = json.loads(command(args, env=env))
        if errors:
            raise RuntimeError(f"{device.upper()} preflight failed:\n" +
                               "\n".join(f"  {k}: {v}" for k, v in errors.items()))


def failure_reason(text, allocation, device):
    """Distinguish GPU VRAM, host-memory and ordinary failures."""
    if device == "gpu" and re.search(r"ResourceExhaustedError|CUDA.*out of memory|OOM when allocating|GPU.*out.of.memory", text, re.I):
        return "GPU_OOM"
    if re.search(r"MemoryError|std::bad_alloc|Cannot allocate memory|Unable to allocate|out.of.memory|oom.kill", text, re.I):
        return "OOM"
    if allocation == "OUT_OF_MEMORY":
        return "OOM"
    if allocation == "TIMEOUT":
        return "TIMEOUT"
    return "FAILED"


def reconcile(c, state, complete, live):
    """Reconcile finished allocations once; never resubmit live or unknown jobs."""
    for job in state["jobs"].values():
        if job["stage"] == "RECONCILED":
            continue
        if not job.get("id"):
            matches = [jid for jid, info in live.items() if info["name"] == job["name"]]
            if len(matches) == 1:
                job.update(id=matches[0], stage="SUBMITTED")
            else:
                raise RuntimeError(f"Submission outcome unknown for {job['name']}. Check squeue/sacct and record its job ID in state.json before resuming; no duplicate will be submitted.")
        if job["id"] in live:
            continue
        allocation = accounting(job["id"])
        if allocation not in TERMINAL:
            job["stage"] = "ACCOUNTING"
            continue
        if (allocation in {"FAILED", "BOOT_FAIL", "OUT_OF_MEMORY"}
                and any(t["key"] not in complete for t in job["tasks"])
                and not any(Path(t["status"]).exists() for t in job["tasks"])):
            raise RuntimeError(
                f"Allocation {job['id']} failed before any worker started. "
                f"Inspect {Path(job['tasks'][0]['log']).parent}/*.err and fix "
                "the job environment before releasing this batch in state.json."
            )
        for task in job["tasks"]:
            k = task["key"]
            record = state["attempts"].setdefault(k, {"attempts": 0, "failures": 0, "tier": job["tier"]})
            if k in complete:
                record["reason"] = "COMPLETE"
                continue
            status = read_json(Path(task["status"]), {})
            if not status:
                # Commands that never started consume no attempt or memory tier.
                record["reason"] = "NOT_STARTED"
                continue
            record["attempts"] += 1
            log = Path(task["log"])
            text = log.read_text(errors="replace")[-100000:] if log.exists() else ""
            reason = failure_reason(text, allocation, job["device"])
            record.update(reason=reason, log=str(log), job=job["id"])
            if reason == "OOM" and record["tier"] + 1 < len(c[job["device"]]["memory_gib"]):
                record["tier"] += 1
            else:
                record["failures"] += 1
            if (record["attempts"] >= c["max_attempts"] or record["failures"] >= c["max_failures"]):
                record["blocked"] = True
        job.update(stage="RECONCILED", outcome=allocation)


def active_tasks(state):
    """Reserve every task belonging to an unreconciled allocation."""
    return {t["key"] for j in state["jobs"].values() if j["stage"] != "RECONCILED" for t in j["tasks"]}


def plan(c, ds, state, complete, devices):
    """Interleave missing resamples across classifiers and memory tiers."""
    reserved = active_tasks(state)
    groups = defaultdict(list)
    sizes = {}
    for d in ds:
        sizes[d] = sum(p.stat().st_size for p in (Path(c["data_dir"]) / d).glob(f"{d}_*.ts"))
    for i in range(c["resamples"]):
        for d in sorted(ds, key=lambda d: (sizes[d], d)):
            for row in c["classifiers"]:
                k = key_for(row["name"], d, i)
                record = state["attempts"].get(k, {})
                if row["device"] not in devices or k in complete or k in reserved or record.get("blocked"):
                    continue
                initial = 2 if sizes[d] > 300 * 1024**2 else 1 if sizes[d] > 60 * 1024**2 else 0
                tier = record.get("tier", min(initial, len(c[row["device"]]["memory_gib"]) - 1))
                groups[row["device"], tier].append(dict(key=k, classifier=row["name"], dataset=d, resample=i))
    return groups


def shell_environment(c):
    """Set the same single-thread policy as existing Iridis CPU scripts."""
    lines = [". /etc/profile", "set -euo pipefail", f"cd {shlex.quote(c['repo_dir'])}",
             "unset PYTHONHOME", "export PYTHONNOUSERSITE=1", "export PYTHONUNBUFFERED=1",
             f"export PYTHONPATH={shlex.quote(c['aeon_dir'] + ':' + c['repo_dir'])}"]
    lines += [f"export {v}=1" for v in THREAD_VARS]
    cache = str(Path(c["state_dir"]) / "numba-cache")
    lines += [f"export NUMBA_CACHE_DIR={shlex.quote(cache)}", 'mkdir -p "$NUMBA_CACHE_DIR"']
    for name in (*THREAD_VARS, "PYTHONPATH", "PYTHONNOUSERSITE", "NUMBA_CACHE_DIR"):
        lines.append(f'export APPTAINERENV_{name}="${{{name}}}"')
    return lines


def batch_script(c, job, config_path, command_file):
    """Render a CPU task farm or a serial one-GPU allocation."""
    device = job["device"]
    memory = c[device]["memory_gib"][job["tier"]]
    folder = Path(command_file).parent
    lines = ["#!/bin/bash", f"#SBATCH --job-name={job['name']}",
             f"#SBATCH --partition={c[device]['partition']}",
             f"#SBATCH --time={c['time_limit']}", "#SBATCH --nodes=1",
             f"#SBATCH --output={folder}/%j.out", f"#SBATCH --error={folder}/%j.err",
             "#SBATCH --mail-type=NONE"]
    if device == "cpu":
        cpus = min(len(job["tasks"]), c["cpu"]["max_cpus"], c["cpu"]["node_memory_gib"] // memory)
        lines += [f"#SBATCH --ntasks={cpus}", f"#SBATCH --mem-per-cpu={memory}G"]
    else:
        lines += ["#SBATCH --ntasks=1", f"#SBATCH --cpus-per-task={c['gpu']['cpus']}",
                  "#SBATCH --gres=gpu:1", f"#SBATCH --mem={memory}G"]
    lines += [""] + shell_environment(c)
    lines += [shlex.join([c["python"], str(HERE / "controller.py"), "verify",
                         "--config", str(config_path)])]
    if device == "cpu":
        lines += ['export CUDA_VISIBLE_DEVICES=""', f"staskfarm {shlex.quote(str(command_file))}"]
    else:
        lines += [f"module load {shlex.quote(c['gpu']['module'])}",
                  'export TF_FORCE_GPU_ALLOW_GROWTH=true',
                  'export APPTAINERENV_TF_FORCE_GPU_ALLOW_GROWTH=true',
                  'export APPTAINERENV_CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:?Slurm did not assign a GPU}"',
                  # Every line is generated with shlex.join; a failed experiment
                  # is recorded by its worker and must not discard later tasks.
                  'while IFS= read -r task_command; do',
                  '    bash -c "$task_command" || true',
                  f"done < {shlex.quote(str(command_file))}"]
    return "\n".join(lines) + "\n"


def submit_batch(c, state, state_path, config_path, device, tier, tasks):
    """Journal a batch before sbatch so interrupted submissions cannot duplicate work."""
    token = uuid.uuid4().hex[:12]
    folder = Path(c["state_dir"]) / "batches" / token
    folder.mkdir(parents=True)
    job = dict(name=f"{c['job_prefix']}-{device}-{token}", device=device, tier=tier,
               stage="SUBMITTING", tasks=tasks, id=None)
    lines = []
    for index, task in enumerate(tasks):
        task.update(status=str(folder / f"{index}.status.json"), log=str(folder / f"{index}.log"))
        task_path = folder / f"{index}.task.json"
        save_json(task_path, task)
        args = runtime(c, device, gpu=device == "gpu") + [str(HERE / "controller.py"), "worker",
                "--config", str(config_path), "--task", str(task_path)]
        lines.append(f"{shlex.join(args)} > {shlex.quote(task['log'])} 2>&1")
    command_file = folder / "commands.txt"
    command_file.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    script = folder / "submit.sh"
    script.write_text(batch_script(c, job, config_path, command_file), encoding="utf-8", newline="\n")
    state["jobs"][token] = job
    save_json(state_path, state)
    job_id = command(["sbatch", "--parsable", str(script)]).split(";")[0]
    if not job_id.isdigit():
        raise RuntimeError(f"Unexpected sbatch response: {job_id}")
    job.update(id=job_id, stage="SUBMITTED")
    save_json(state_path, state)
    print(f"Submitted {device} {len(tasks)} commands, {c[device]['memory_gib'][tier]} GiB/task: {job_id}", flush=True)


def schedule_successor(c, state, state_path, config_path, live, device):
    """Keep one delayed, short reconciliation job alive while work remains."""
    previous = state.get("supervisor", {})
    if previous.get("id") in live and previous["id"] != os.environ.get("SLURM_JOB_ID"):
        return
    if previous.get("stage") == "SUBMITTING":
        matches = [jid for jid, info in live.items() if info["name"] == previous["name"]]
        if matches:
            previous.update(id=matches[0], stage="SUBMITTED")
            save_json(state_path, state)
            return
        raise RuntimeError("Supervisor submission outcome unknown; inspect state.json and Slurm before restarting.")
    name = f"{c['job_prefix']}-supervisor-{uuid.uuid4().hex[:12]}"
    script = Path(c["state_dir"]) / f"{name}.sh"
    lines = ["#!/bin/bash", f"#SBATCH --job-name={name}",
             f"#SBATCH --partition={c['cpu']['partition']}", "#SBATCH --nodes=1",
             "#SBATCH --ntasks=1", "#SBATCH --mem=2G", "#SBATCH --time=00:30:00",
             f"#SBATCH --begin=now+{c['refill_minutes']}minutes",
             f"#SBATCH --output={c['state_dir']}/%j-supervisor.out",
             f"#SBATCH --error={c['state_dir']}/%j-supervisor.err", "#SBATCH --mail-type=NONE", ""]
    lines += shell_environment(c)
    lines += [shlex.join([c["python"], str(HERE / "controller.py"), "run",
                         "--config", str(config_path), "--device", device])]
    script.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    state["supervisor"] = dict(name=name, stage="SUBMITTING", id=None)
    save_json(state_path, state)
    job_id = command(["sbatch", "--parsable", str(script)]).split(";")[0]
    if not job_id.isdigit():
        raise RuntimeError(f"Unexpected supervisor sbatch response: {job_id}")
    state["supervisor"].update(id=job_id, stage="SUBMITTED")
    save_json(state_path, state)
    print(f"Next reconciliation: {job_id}, eligible in {c['refill_minutes']} minutes")


def print_report(c, ds, counts, state, complete, live=None, details=False):
    """Report file coverage, active commands and exhausted attempts together."""
    print(f"UCR reference results: {c['results_root']}\n{time.strftime('%Y-%m-%d %H:%M:%S %Z')}\n{len(ds)} datasets x {c['resamples']} resamples")
    print(f"{'CLASSIFIER':32} {'TEST':>7} {'TRAIN*':>7} {'DONE':>7} {'DATASETS':>9} {'DEVICE':>7}")
    for row in c["classifiers"]:
        r = counts[row["name"]]
        print(f"{row['name']:32} {r['test']:7} {str(r['train']) if row['train'] else '-':>7} {r['complete']:7} {r['datasets']:9} {row['device']:>7}")
    total = len(ds) * c["resamples"] * len(c["classifiers"])
    blocked = {k: v for k, v in state["attempts"].items() if v.get("blocked") and k not in complete}
    reserved = active_tasks(state) - complete
    percentage = 100 * len(complete) / total if total else 100
    print(f"Complete: {len(complete):,}/{total:,}; percentage: {percentage:.2f}%; missing: {total-len(complete):,}; reserved: {len(reserved):,}; blocked: {len(blocked):,}")
    print("* Train files are required only where the collection's train flag is true.")
    if live is not None:
        relevant = {jid: j for jid, j in live.items() if j["name"].startswith(c["job_prefix"] + "-")}
        for jid, j in relevant.items():
            print(f"Job {jid}: {j['state']} {j['node']} {j['name']}")
        if not relevant and not reserved and total > len(complete):
            print("No live reference jobs: restart the runner after addressing any blocked failures.")
        elif relevant and total > len(complete) and not any("-supervisor-" in j["name"] for j in relevant.values()):
            print("No live refill supervisor: workers can finish, but further submissions need a runner launch.")
    activity = Counter()
    for job in state["jobs"].values():
        if job["stage"] == "RECONCILED":
            continue
        for task in job["tasks"]:
            if task["key"] in complete:
                continue
            status = read_json(Path(task["status"]), {})
            label = status.get("state", "QUEUED")
            if label == "RUNNING" and live is not None and job.get("id") not in live:
                label = "AWAITING_RECONCILIATION"
            activity[label] += 1
    if activity:
        print("Activity:", ", ".join(f"{k}={v}" for k, v in sorted(activity.items())))
    if (Path(c["state_dir"]) / "STOP").exists():
        print("Refills are STOPPED; existing allocations may still be running.")
    for job in state["jobs"].values():
        if job["stage"] in {"SUBMITTING", "ACCOUNTING"}:
            print(f"Needs reconciliation: {job['name']} {job['stage']} job={job.get('id')}")
    if blocked:
        print("Blocked reasons:", dict(Counter(r["reason"] for r in blocked.values())))
    if details:
        for row in c["classifiers"]:
            for d in ds:
                gaps = [str(i) for i in range(c["resamples"]) if key_for(row["name"], d, i) not in complete]
                if gaps:
                    print(f"Missing {row['name']} / {d}: {','.join(gaps)}")
        for k, r in blocked.items():
            print(f"BLOCKED {k}: {r['reason']}; attempts={r['attempts']}; log={r.get('log', '-')}")


def run_cycle(c, args):
    """Reconcile and refill bounded CPU/GPU allocations."""
    ds = datasets(c)
    complete, counts = inventory(c, ds)
    state_path = Path(c["state_dir"]) / "state.json"
    empty_state = dict(jobs={}, attempts={}, round=0)
    devices = ("cpu", "gpu") if args.device == "all" else (args.device,)
    if args.dry_run:
        state = read_json(state_path, empty_state)
        print_report(c, ds, counts, state, complete)
        groups = plan(c, ds, state, complete, devices)
        for (device, tier), tasks in sorted(groups.items()):
            print(f"Plan: {device}, {c[device]['memory_gib'][tier]} GiB/task, {len(tasks):,} missing unreserved experiments")
        print("Dry run: no scheduler calls, imports, submissions or state changes. Use --check on Iridis to validate data and environments.")
        return
    with locked(Path(c["state_dir"]) / "controller.lock"):
        if (Path(c["state_dir"]) / "STOP").exists():
            print("STOP file present: no submission or successor.")
            return
        state = read_json(state_path, empty_state)
        live = query_slurm(c)
        if any(j["name"].startswith("ucr-completion-") for j in live.values()):
            raise RuntimeError("The old ucr-completion workflow is active. Finish or stop that workflow before starting overlapping reference work.")
        known_names = {j["name"] for j in state["jobs"].values()}
        orphaned = [jid for jid, job in live.items()
                    if any(job["name"].startswith(f"{c['job_prefix']}-{d}-") for d in ("cpu", "gpu"))
                    and job["name"] not in known_names]
        if orphaned:
            raise RuntimeError(f"Live reference allocations are missing from this state: {orphaned}. Restore their batch records before refilling.")
        snapshot = Path(c["state_dir"]) / "run-config.json"
        if (snapshot.exists() and state["jobs"] and read_json(snapshot) != c):
            raise RuntimeError(f"Configuration changed during a run. Use the saved config {snapshot}; resolve changes before resuming.")
        current = provenance(c)
        pinned_path = Path(c["state_dir"]) / "provenance.json"
        if pinned_path.exists() and read_json(pinned_path) != current:
            raise RuntimeError("aeon or experiment sources changed; restore the pinned sources before continuing.")
        save_json(snapshot, c)
        # First launch validates every selected CPU/GPU environment before sbatch.
        unchecked = [d for d in devices if d not in state.get("checked_devices", [])]
        if unchecked or args.check:
            preflight(c, snapshot, devices if args.check else unchecked)
        if args.check:
            print("Data, source paths and all selected estimator constructors passed; nothing submitted.")
            return
        save_json(pinned_path, current)
        state["checked_devices"] = sorted(set(state.get("checked_devices", [])) | set(devices))
        reconcile(c, state, complete, live)
        # A device-specific manual refill must not narrow an existing full run.
        state["devices"] = sorted(set(state.get("devices", [])) | set(devices))
        devices = tuple(state["devices"])
        state["round"] += 1
        save_json(state_path, state)
        if state["round"] > c["max_rounds"]:
            raise RuntimeError("Maximum reconciliation rounds reached; inspect outstanding failures.")
        groups = plan(c, ds, state, complete, devices)
        for device in devices:
            occupied = sum(j["device"] == device and j["stage"] != "RECONCILED" for j in state["jobs"].values())
            for _ in range(max(0, c[device]["max_jobs"] - occupied)):
                tiers = [tier for (dev, tier), tasks in groups.items() if dev == device and tasks]
                if not tiers:
                    break
                # Give each memory tier a turn rather than starving sparse failures.
                tier = tiers[(state["round"] - 1 + _) % len(tiers)]
                memory = c[device]["memory_gib"][tier]
                capacity = (min(c["cpu"]["max_cpus"], c["cpu"]["node_memory_gib"] // memory)
                            * c["cpu"]["waves_per_job"] if device == "cpu" else c["gpu"]["commands_per_job"])
                batch, groups[device, tier] = groups[device, tier][:capacity], groups[device, tier][capacity:]
                submit_batch(c, state, state_path, snapshot, device, tier, batch)
        print_report(c, ds, counts, state, complete, live)
        has_work = any(groups.values()) or bool(active_tasks(state) - complete)
        if has_work and not args.no_chain and state["round"] < c["max_rounds"]:
            scope = "all" if len(devices) == 2 else devices[0]
            schedule_successor(c, state, state_path, snapshot, live, scope)
        elif not has_work:
            print("No runnable work remains. Blocked failures, if listed, are incomplete results.")


def worker(c, task_path):
    """Execute one experiment, recording starts and failures for the monitor."""
    task = read_json(task_path)
    row = next(r for r in c["classifiers"] if r["name"] == task["classifier"])
    for v in THREAD_VARS:
        os.environ[v] = "1"
    lock_name = hashlib.sha256(task["key"].encode()).hexdigest() + ".lock"
    with locked(Path(c["state_dir"]) / "experiment-locks" / lock_name):
        status_path = Path(task["status"])
        status = dict(state="RUNNING", started=time.time(), key=task["key"], job=os.environ.get("SLURM_JOB_ID"))
        save_json(status_path, status)
        try:
            paths = result_paths(c, row, task["dataset"], task["resample"])
            if not all(nonempty(p) for p in paths):
                # The library checks existence, so preserve empty placeholders
                # under a different name before asking it to fill missing splits.
                for p in paths:
                    if p.exists() and p.stat().st_size == 0:
                        p.rename(p.with_name(p.name + f".empty-{uuid.uuid4().hex[:12]}"))
                import aeon
                if not Path(aeon.__file__).resolve().is_relative_to(Path(c["aeon_dir"]).resolve()):
                    raise RuntimeError(f"Unexpected aeon import: {aeon.__file__}")
                if row["device"] == "gpu":
                    import tensorflow as tf
                    if not tf.config.list_physical_devices("GPU"):
                        raise RuntimeError("No TensorFlow GPU visible inside the allocation/container")
                import numba
                from tsml_eval.experiments import load_and_run_classification_experiment

                numba.set_num_threads(1)
                load_and_run_classification_experiment(
                    c["data_dir"], str(Path(c["results_root"]) / row["category"]),
                    task["dataset"], make_classifier(row, task["resample"]),
                    classifier_name=row["name"], resample_id=task["resample"],
                    build_train_file=row["train"], benchmark_time=False, overwrite=False,
                    predefined_resample=False, load_equal_length=True, load_no_missing=True,
                )
            if not all(nonempty(p) for p in paths):
                raise RuntimeError("Experiment exited without all required nonempty result files")
            status["state"] = "COMPLETE"
        except BaseException as error:
            status.update(state="FAILED", error=f"{type(error).__name__}: {error}")
            traceback.print_exc()
            raise
        finally:
            status["finished"] = time.time()
            save_json(status_path, status)


def main(argv=None):
    """Dispatch the runner, read-only monitor, preflight and batch worker."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["run", "monitor", "worker", "check-estimators", "verify", "stop"])
    parser.add_argument("--config", type=Path, default=HERE / "ucr_reference.json")
    parser.add_argument("--device", choices=["all", "cpu", "gpu"], default="all")
    parser.add_argument("--results-root")
    parser.add_argument("--data-dir")
    parser.add_argument("--dataset-list")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--no-chain", action="store_true")
    parser.add_argument("--details", action="store_true")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--watch", type=int, default=0)
    parser.add_argument("--task", type=Path)
    args = parser.parse_args(argv)
    c = load_config(args.config.resolve(), args)
    if args.action == "run":
        run_cycle(c, args)
    elif args.action == "monitor":
        if args.watch and args.watch < 5:
            parser.error("--watch must be zero or at least five seconds")
        while True:
            ds = datasets(c)
            complete, counts = inventory(c, ds)
            state = read_json(Path(c["state_dir"]) / "state.json", dict(jobs={}, attempts={}))
            live = None if args.offline else query_slurm(c)
            print_report(c, ds, counts, state, complete, live, args.details)
            if not args.watch:
                break
            time.sleep(args.watch)
    elif args.action == "worker":
        if args.task is None:
            parser.error("worker requires --task")
        worker(c, args.task)
    elif args.action == "check-estimators":
        check_estimators(c, args.device)
    elif args.action == "verify":
        if read_json(Path(c["state_dir"]) / "provenance.json") != provenance(c):
            raise RuntimeError("Source provenance changed after submission")
    elif args.action == "stop":
        with locked(Path(c["state_dir"]) / "controller.lock"):
            (Path(c["state_dir"]) / "STOP").touch()
        print("Refills stopped. Existing jobs can finish; no jobs were cancelled.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr)
        sys.exit(1)
