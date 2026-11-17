"""Offline tests for run_local.py: real child processes, a fake tsml-eval.

python -m pytest _tsml_research_resources/slurm/tests -q
"""

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for name in ("tsml_slurm", "run_local"):
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, ROOT / f"{name}.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
ts = sys.modules["tsml_slurm"]
rl = sys.modules["run_local"]

CLUSTER = """
name = "box"
home = "{home}"
repo_dir = "{home}/repo"

[supervisor]
mode = "login"

[resources.cpu]
partition = "local"
time_limit = "1:00:00"
memory_gb = [4, 8]
env_name = "tsml-eval"

[resources.gpu]
partition = "local"
gres = "gpu:1"
time_limit = "1:00:00"
memory_gb = [4]
env_name = "tsml-eval-gpu"
"""

# Stands in for tsml_eval.experiments.classification_experiments. Behaviour per
# dataset name: Ok writes results, Boom raises, Oom1 runs out of memory on its first
# run only, Slow sleeps. It records CUDA_VISIBLE_DEVICES next to its results.
FAKE = """
import os, sys, time
from pathlib import Path
data, results, est, dataset, resample = sys.argv[1:6]
out = Path(results) / est / "Predictions" / dataset
marker = out / "tried"
if dataset == "Boom":
    raise ValueError("bad data")
if dataset == "Oom1" and not marker.exists():
    out.mkdir(parents=True, exist_ok=True)
    marker.write_text("x")
    raise MemoryError("Unable to allocate 9 GiB")
if dataset == "Slow":
    time.sleep(30)
out.mkdir(parents=True, exist_ok=True)
(out / "gpu.txt").write_text(os.environ.get("CUDA_VISIBLE_DEVICES", "unset"))
(out / f"testResample{resample}.csv").write_text("results")
"""


@pytest.fixture
def box(tmp_path, monkeypatch):
    home = tmp_path / "home"
    repo = home / "repo"
    pkg = repo / "tsml_eval" / "experiments"
    pkg.mkdir(parents=True)
    (repo / "tsml_eval" / "__init__.py").write_text("")
    (pkg / "__init__.py").write_text("")
    (pkg / "classification_experiments.py").write_text(FAKE)
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    clusters = tmp_path / "clusters"
    clusters.mkdir()
    (clusters / "box.toml").write_text(CLUSTER.replace("{home}", home.as_posix()))
    monkeypatch.setattr(rl, "POLL_SECONDS", 0.05)
    monkeypatch.setattr(rl, "env_python", lambda name: Path(sys.executable))
    monkeypatch.setattr(rl, "gpu_problem", lambda *a: "")
    return {"home": home, "clusters": clusters, "tmp": tmp_path}


def experiment(box, datasets, name="exp", resources="cpu", body=""):
    home = box["home"]
    for d in datasets:
        (home / "Data" / d).mkdir(parents=True, exist_ok=True)
        (home / "Data" / d / f"{d}_TRAIN.ts").write_text("x")
    (home / f"{name}.txt").write_text("\n".join(datasets) + "\n")
    path = box["tmp"] / f"{name}.toml"
    path.write_text(f"""
[experiment]
name = "{name}"
cluster = "box"
resources = "{resources}"
task = "classification"
data_dir = "{{home}}/Data"
datasets = "{{home}}/{name}.txt"
results_dir = "{{home}}/Results"
resamples = 1
estimators = ["A"]
{body}
""")
    return ts.load_experiment(path, cluster_dir=box["clusters"], user="u")


def run(experiments, cpus=4, memory_gb=16, gpus=()):
    lines = []
    s = rl.Scheduler(experiments, cpus, memory_gb, gpus, 10, log=lines.append)
    s.run()
    return lines


def result(box, dataset):
    return box["home"] / "Results" / "A" / "Predictions" / dataset


def test_runs_until_complete_and_skips_done_work(box):
    e = experiment(box, ["Ok", "Ok2"])
    run([e])
    assert all(ts.is_complete(e, t) for t in ts.all_tasks(e))
    assert (result(box, "Ok") / "gpu.txt").read_text() == ""  # CPU: no GPU
    assert not any(line.startswith("start") for line in run([e]))


def test_failures_are_retried_then_stop(box):
    e = experiment(box, ["Boom", "Ok"])
    lines = run([e])
    record = rl.load_local_state(e)["tasks"]["A|Boom|0"]
    assert record["terminal"] == ts.ERROR and record["attempts"] == 2
    assert sum("start  exp A|Boom|0" in line for line in lines) == 2
    status = "\n".join(rl.status_lines(e))
    assert "1/2 complete" in status and "ValueError: bad data" in status
    assert not any(line.startswith("start") for line in run([e]))  # stays failed
    assert rl.retry(e) == 1
    assert sum("start" in line for line in run([e])) == 2


def test_out_of_memory_moves_up_a_tier(box):
    e = experiment(box, ["Oom1"])
    lines = run([e])
    assert ts.is_complete(e, ts.Task("A", "", "Oom1", 0))
    assert any("start  exp A|Oom1|0 4G" in line for line in lines)
    assert any("start  exp A|Oom1|0 8G" in line for line in lines)
    assert rl.load_local_state(e)["tasks"]["A|Oom1|0"]["tier"] == 1


def test_time_limit_stops_a_task(box, monkeypatch):
    monkeypatch.setattr(ts, "time_limit_seconds", lambda limit: 0.5)
    e = experiment(box, ["Slow"])
    lines = run([e])
    assert rl.load_local_state(e)["tasks"]["A|Slow|0"]["terminal"] == ts.TIMEOUT
    assert any(ts.TIMEOUT in line for line in lines)


def test_too_big_for_the_machine(box):
    e = experiment(box, ["Ok"], body="n_jobs = 8")
    run([e], cpus=4)
    assert rl.load_local_state(e)["tasks"]["A|Ok|0"]["terminal"] == (
        "too big for this machine"
    )


def test_budget_limits_concurrency(box):
    e = experiment(box, ["Ok", "Ok2", "Ok3"])
    s = rl.Scheduler([e], 4, 8, [], 10, log=lambda m: None)
    s.scan()
    s.launch()
    assert len(s.running) == 2  # 4G each in 8G
    s.stop()


def test_gpu_tasks_get_their_own_gpu(box):
    e = experiment(box, ["Ok", "Ok2", "Ok3"], name="g", resources="gpu")
    s = rl.Scheduler([e], 8, 64, ["0", "1"], 10, log=lambda m: None)
    s.scan()
    s.launch()
    assert sorted(r.gpu for r in s.running) == ["0", "1"]  # third waits for a GPU
    s.run()
    seen = {(result(box, d) / "gpu.txt").read_text() for d in ("Ok", "Ok2", "Ok3")}
    assert seen <= {"0", "1"}


def test_gpu_experiment_without_gpus_is_too_big(box):
    e = experiment(box, ["Ok"], name="g", resources="gpu")
    run([e], gpus=())
    assert rl.load_local_state(e)["tasks"]["A|Ok|0"]["terminal"]


def test_pending_and_status(box):
    e = experiment(box, ["Ok"])
    tasks = rl.pending_tasks(e, rl.load_local_state(e), set())
    assert [t.key for t in tasks] == ["A|Ok|0"]
    assert rl.status_lines(e)[0] == "exp: 0/1 complete (0%), 0 failed for good"
