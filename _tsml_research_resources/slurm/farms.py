"""One-screen overview of every running task farm, across experiments.

    python farms.py                       # every farm experiment in experiments/
    python farms.py experiments/iridis6_eeg_*.toml

For each experiment with a farm in the queue: overall completion, then one row per
farm job (Slurm state, time used and left, memory per slot, slots, and its tasks
running, waiting to start, done and failed), then which estimators are running.
Read only: it never submits, cancels or changes the experiment's state.
"""

import argparse
import subprocess
import sys
from collections import Counter
from pathlib import Path

import tsml_slurm as ts

HERE = Path(__file__).resolve().parent


def _farm_jobs(user):
    """{job id: (state, time used, time left, node)} for the user's jobs."""
    out = subprocess.run(
        ["squeue", "--noheader", f"--user={user}", "--format=%A|%T|%M|%L|%N"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    jobs = {}
    for line in out.splitlines():
        parts = [p.strip() for p in line.split("|")]
        if len(parts) == 5:
            jobs[parts[0]] = tuple(parts[1:])
    return jobs


def _task_state(e, key, job):
    """Return running, waiting, done or failed, from the task status file."""
    estimator, dataset, resample = key.split("|")
    log = ts.task_log(e, ts.Task(estimator, "", dataset, int(resample)), job)
    status = ts._read_tail(Path(f"{log}.status"))
    if not status:
        return "waiting"
    if "exit 0" in status:
        return "done"
    if "exit " in status or "cancelled" in status:
        return "failed"
    return "running"


def overview(e, jobs):
    """Lines describing this experiment's farms in the queue, or [] if none."""
    state = ts.load_state(e)
    farms = [
        s for s in state["submissions"] if s.get("kind") == "farm" and s["job"] in jobs
    ]
    if not farms:
        return []
    ts.read_datasets(e)
    tasks = list(ts.all_tasks(e))
    missing = [t for t in tasks if not ts.is_complete(e, t)]
    complete = len(tasks) - len(missing)
    records = [state["tasks"].get(t.key, {}) for t in missing]
    failed = sum(1 for r in records if r.get("terminal"))
    waiting = sum(1 for r in records if not r.get("job") and not r.get("terminal"))
    lines = [
        f"{e.name}: {complete}/{len(tasks)} complete "
        f"({100 * complete / max(len(tasks), 1):.0f}%), {waiting} not yet "
        f"submitted, {failed} failed",
        f"  {'job':>9} {'state':<9}{'used':>12}{'left':>12}{'mem':>6}{'slots':>6}"
        f"{'run':>5}{'wait':>6}{'done':>6}{'fail':>6}  node",
    ]
    for info in farms:
        job = info["job"]
        slurm_state, used, left, node = jobs[job]
        keys = [k for k, r in state["tasks"].items() if r.get("job") == job]
        counts = Counter(_task_state(e, k, job) for k in keys)
        running = Counter(
            k.split("|")[0] for k in keys if _task_state(e, k, job) == "running"
        )
        inbox = ts.farm_inbox(e, job)
        top_up = ""
        if info.get("top_up"):
            if (inbox / "closed").exists():
                top_up = "  top-up closed"
            elif inbox.is_dir():
                top_up = "  top-up open"
            else:
                top_up = "  top-up not started"
        # farms submitted before slots were recorded: the same rule that sized them
        slots = info.get("slots") or ts.farm_shape(e, info["memory_gb"], info["tasks"])
        lines.append(
            f"  {job:>9} {slurm_state:<9}{used:>12}{left:>12}"
            f"{str(info['memory_gb']) + 'G':>6}{slots:>6}"
            f"{counts['running']:>5}{counts['waiting']:>6}{counts['done']:>6}"
            f"{counts['failed']:>6}  {node}{top_up}"
        )
        if running:
            names = ", ".join(f"{n} x{c}" for n, c in sorted(running.items()))
            lines.append(f"  {'':>9} running: {names}")
    return lines


def main(argv=None):
    """Print the overview for each farm experiment given, or found."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("configs", nargs="*", help="experiment TOMLs")
    args = parser.parse_args(argv)
    configs = args.configs or sorted((HERE / "experiments").glob("*.toml"))
    jobs = None
    printed = False
    for config in configs:
        try:
            e = ts.load_experiment(config)
        except (ts.ConfigError, OSError) as error:
            if args.configs:
                print(f"{config}: {error}", file=sys.stderr)
            continue
        if e.resources.mode != "farm" or not (e.state_dir / "state.json").is_file():
            continue
        if jobs is None:
            jobs = _farm_jobs(e.user)
        lines = overview(e, jobs)
        if lines:
            print("\n".join(lines) + "\n")
            printed = True
    if not printed:
        print("No farm jobs in the queue.")


if __name__ == "__main__":
    main()
