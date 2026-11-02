"""Offline tests for tsml_slurm.py. No Slurm, cluster or tsml-eval import needed.

python -m pytest _tsml_research_resources/slurm/tests -q
"""

import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
spec = importlib.util.spec_from_file_location("tsml_slurm", ROOT / "tsml_slurm.py")
ts = importlib.util.module_from_spec(spec)
sys.modules["tsml_slurm"] = ts
spec.loader.exec_module(ts)

BASH = shutil.which("bash")


class FakeSlurm(ts.Slurm):
    def __init__(self):
        self.entries = []
        self.acct = {}
        self.scripts = {}
        self.next_id = 1000

    def available(self):
        return True

    def queue(self, user):
        return list(self.entries)

    def accounting(self, job_ids):
        return {k: v for k, v in self.acct.items() if k[0] in job_ids}

    def submit(self, script):
        self.next_id += 1
        self.scripts[str(self.next_id)] = script
        return str(self.next_id)

    def cancel(self, job_ids):
        self.entries = [q for q in self.entries if q.job_id not in job_ids]


CLUSTER = """
name = "testcluster"
home = "{home}"
repo_dir = "{home}/repo"

[supervisor]
mode = "slurm"
partition = "short"

[resources.cpu]
mode = "array"
partition = "compute"
account = "acct"
time_limit = "1:00:00"
memory_gb = [4, 8, 16]
max_active = 50
module = "python"
env_name = "tsml-eval"

[resources.farm]
mode = "farm"
partition = "batch"
time_limit = "1:00:00"
memory_gb = [4, 8, 64]
max_jobs = 2
node_cpus = 8
node_memory_gb = 64
waves = 2
env_name = "tsml-eval"

[resources.gpu]
partition = "gpu"
gres = "gpu:1"
time_limit = "1:00:00"
memory_gb = [32]
env_name = "tsml-eval-gpu"
gpu_check = "torch"
"""


def _git(repo, *args):
    subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True)


@pytest.fixture
def world(tmp_path):
    home = tmp_path / "home"
    repo = home / "repo"
    repo.mkdir(parents=True)
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "t@example.com")
    _git(repo, "config", "user.name", "t")
    (repo / "README").write_text("x")
    _git(repo, "add", "README")
    _git(repo, "commit", "-q", "-m", "init")
    data = home / "Data"
    for dataset, size in (("Small", 10), ("Big", 1000), ("Mid", 100)):
        (data / dataset).mkdir(parents=True)
        (data / dataset / f"{dataset}_TRAIN.ts").write_text("x" * size)
    (home / "list.txt").write_text("# comment\nSmall\nBig\n\nMid\n")
    clusters = tmp_path / "clusters"
    clusters.mkdir()
    (clusters / "testcluster.toml").write_text(
        CLUSTER.replace("{home}", home.as_posix())
    )
    return {"home": home, "repo": repo, "clusters": clusters, "tmp": tmp_path}


def make(
    world,
    body="",
    resources="cpu",
    estimators='estimators = ["A", "B"]',
    extra_sections="",
    task="classification",
):
    path = world["tmp"] / "exp.toml"
    path.write_text(f"""
[experiment]
name = "exp"
cluster = "testcluster"
resources = "{resources}"
task = "{task}"
data_dir = "{{home}}/Data"
datasets = "{{home}}/list.txt"
results_dir = "{{home}}/Results"
{"" if "resamples" in body else "resamples = 2"}
{estimators}
{body}
{extra_sections}
""")
    return ts.load_experiment(path, cluster_dir=world["clusters"], user="tester")


def write_result(e, task, train=False):
    for path in ts.result_files(e, task)[: 2 if train else 1]:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("result")


def age_submissions(e):
    state = ts.load_state(e)
    for record in state["tasks"].values():
        record["submitted_at"] = 0
    ts.save_state(e, state)


# --------------------------------------------------------------------------- config


@pytest.mark.parametrize(
    "config", sorted((ROOT / "experiments").glob("*.toml")), ids=lambda p: p.stem
)
def test_shipped_examples_load(config):
    e = ts.load_experiment(config, user="tester")
    assert e.resources.partition
    assert str(e.results_dir).startswith(str(e.home))


def test_placeholders_and_defaults(world):
    e = make(world)
    assert e.data_dir == world["home"] / "Data"
    assert e.state_dir == world["home"] / ".tsml-slurm" / "exp"
    assert e.resamples == (0, 1)
    assert [x.name for x in e.estimators] == ["A", "B"]
    assert e.resources.account == "acct"


def test_unknown_keys_are_rejected(world):
    with pytest.raises(ts.ConfigError, match="memroy_gb"):
        make(world, extra_sections="[slurm]\nmemroy_gb = [1]")
    with pytest.raises(ts.ConfigError, match="resampels"):
        make(world, body="resampels = 3")


def test_bad_settings_are_rejected(world):
    with pytest.raises(ts.ConfigError, match="overwrite"):
        make(world, body='extra_args = ["-ow"]')
    with pytest.raises(ts.ConfigError, match="train_files"):
        make(world, body='extra_args = ["-tr"]')
    with pytest.raises(ts.ConfigError, match="increasing"):
        make(world, extra_sections="[slurm]\nmemory_gb = [8, 4]")
    with pytest.raises(ts.ConfigError, match="farm mode"):
        make(world, resources="farm", extra_sections='[slurm]\ngres = "gpu:1"')


def test_categories_set_results_layout(world):
    e = make(world, estimators="", extra_sections='[estimators]\nCat = ["A"]')
    ts.read_datasets(e)
    task = next(ts.all_tasks(e))
    assert ts.result_files(e, task)[0] == (
        world["home"] / "Results/Cat/A/Predictions/Small/testResample0.csv"
    )


def test_clustering_requires_train_file(world):
    with pytest.raises(ts.ConfigError, match="only applies to clustering"):
        make(world, body="test_files = true")
    e = make(world, body="test_files = true", task="clustering")
    ts.read_datasets(e)
    files = ts.result_files(e, next(ts.all_tasks(e)))
    assert [f.name for f in files] == ["trainResample0.csv", "testResample0.csv"]


def test_dataset_list_and_ordering(world):
    e = make(world, body="smallest_datasets_first = true")
    assert ts.read_datasets(e) == ("Small", "Big", "Mid")
    order = ts.order_tasks(e, list(ts.all_tasks(e)))
    assert [(t.resample, t.estimator, t.dataset) for t in order[:3]] == [
        (0, "A", "Small"),
        (0, "A", "Mid"),
        (0, "A", "Big"),
    ]


# --------------------------------------------------------------------------- scripts


@pytest.mark.skipif(BASH is None, reason="bash not available")
@pytest.mark.parametrize("resources", ["cpu", "farm", "gpu"])
def test_generated_scripts_are_valid_bash(world, tmp_path, resources):
    e = make(
        world,
        resources=resources,
        extra_sections='[estimator_kwargs.A]\nn = 5\nname = "it\'s"\nf = 0.5\n',
    )
    ts.read_datasets(e)
    tasks = list(ts.all_tasks(e))[:3]
    scripts = [ts.supervisor_script(e, "/usr/bin/python3")]
    if e.resources.mode == "array":
        scripts.append(ts.array_script(e, tasks[:1], 4, "abc"))
    else:
        scripts.append(ts.farm_script(e, tasks, 4, "abc", 2))
    for i, script in enumerate(scripts):
        path = tmp_path / f"s{i}.sh"
        path.write_text(script, newline="\n")
        result = subprocess.run([BASH, "-n", str(path)], capture_output=True, text=True)
        assert result.returncode == 0, result.stderr + script


def test_array_script_contents(world):
    e = make(
        world,
        resources="gpu",
        body="train_files = true",
        extra_sections="[estimator_kwargs.A]\nn_estimators = 500\nverbose = true",
    )
    ts.read_datasets(e)
    tasks = [t for t in ts.all_tasks(e) if t.estimator == "A" and t.dataset == "Big"]
    script = ts.array_script(e, tasks, 32, "deadbeef")
    assert "#SBATCH --array=0,1" in script
    assert "#SBATCH --gres=gpu:1" in script
    assert "#SBATCH --mem=32G" in script
    assert "import torch" in script
    assert "CUDA_VISIBLE_DEVICES" not in script
    assert "-kw n_estimators 500 int -kw verbose true bool" in script
    assert "-tr" in script
    assert "deadbeef" in script
    assert "tsml_eval.experiments.classification_experiments" in script


def test_threaded_entry_point(world):
    e = make(world, body="n_jobs = 4")
    ts.read_datasets(e)
    script = ts.array_script(e, list(ts.all_tasks(e))[:1], 4, "c")
    assert "threaded_classification_experiments" in script
    assert "-nj 4" in script
    assert "--cpus-per-task=4" in script


# --------------------------------------------------------------------------- cycles


def test_cycle_submits_only_missing_work(world):
    e = make(world)
    ts.read_datasets(e)
    write_result(e, ts.Task("A", "", "Small", 0))
    slurm = FakeSlurm()
    result = ts.run_cycle(e, slurm)
    assert result.submitted == 11
    # one array per estimator/dataset
    assert len(slurm.scripts) == 6
    assert not any(
        "--array=0,1" in s and "-A-Small" in s for s in slurm.scripts.values()
    )
    assert "Complete 1/12" in result.report


def test_capacity_counts_live_tasks(world):
    e = make(world, extra_sections="[slurm]\nmax_active = 5")
    slurm = FakeSlurm()
    assert ts.run_cycle(e, slurm).submitted == 5
    state = ts.load_state(e)
    slurm.entries = [
        ts.QueueEntry(r["job"], int(k.rsplit("|", 1)[1]), "PENDING", "x", "compute")
        for k, r in state["tasks"].items()
    ]
    assert ts.run_cycle(e, slurm).submitted == 0


def test_oom_escalates_memory_and_timeout_is_terminal(world):
    e = make(world, body="resamples = 1", estimators='estimators = ["A"]')
    slurm = FakeSlurm()
    ts.run_cycle(e, slurm)
    age_submissions(e)
    state = ts.load_state(e)
    jobs = {k.split("|")[1]: r["job"] for k, r in state["tasks"].items()}
    slurm.acct = {
        (jobs["Small"], 0): "OUT_OF_MEMORY",
        (jobs["Big"], 0): "TIMEOUT",
        (jobs["Mid"], 0): "COMPLETED",
    }
    write_result(e, ts.Task("A", "", "Mid", 0))
    slurm.scripts.clear()
    result = ts.run_cycle(e, slurm)
    assert result.submitted == 1
    (script,) = slurm.scripts.values()
    assert "-A-Small" in script and "--mem=8G" in script
    state = ts.load_state(e)
    assert state["tasks"]["A|Big|0"]["terminal"] == ts.TIMEOUT
    assert state["tasks"]["A|Small|0"]["tier"] == 1
    assert state["tasks"]["A|Small|0"]["attempts"] == 1
    assert "Time limit" in result.report and not result.settled

    # Small fails again at the top tier: settled with failures
    age_submissions(e)
    state = ts.load_state(e)
    state["tasks"]["A|Small|0"]["tier"] = 2
    ts.save_state(e, state)
    slurm.acct[(state["tasks"]["A|Small|0"]["job"], 0)] = "OUT_OF_MEMORY"
    result = ts.run_cycle(e, slurm)
    assert result.submitted == 0
    assert result.settled
    assert "OOM at 16G" in result.report


def test_python_errors_retry_until_max_attempts(world, tmp_path):
    e = make(
        world, body="resamples = 1\nmax_attempts = 2", estimators='estimators = ["A"]'
    )
    slurm = FakeSlurm()
    ts.run_cycle(e, slurm)
    for _attempt in range(2):
        age_submissions(e)
        record = ts.load_state(e)["tasks"]["A|Small|0"]
        log = ts.task_log(e, ts.Task("A", "", "Small", 0), record["job"])
        Path(f"{log}.err").write_text(
            "Traceback (most recent call last):\nValueError: bad input\n"
        )
        slurm.acct[(record["job"], 0)] = "FAILED"
        for other in ("Big", "Mid"):
            write_result(e, ts.Task("A", "", other, 0))
        result = ts.run_cycle(e, slurm)
    state = ts.load_state(e)["tasks"]["A|Small|0"]
    assert state["terminal"] == ts.ERROR
    assert state["events"][-1]["detail"] == "ValueError: bad input"
    assert result.settled


def test_lost_job_is_resubmitted_without_using_an_attempt(world):
    e = make(world, body="resamples = 1", estimators='estimators = ["A"]')
    slurm = FakeSlurm()
    ts.run_cycle(e, slurm)
    age_submissions(e)
    result = ts.run_cycle(e, slurm)  # vanished: no logs, no accounting
    assert result.submitted == 3
    assert ts.load_state(e)["tasks"]["A|Small|0"]["attempts"] == 1


def test_recent_submission_not_yet_in_queue_is_active(world):
    e = make(world, body="resamples = 1", estimators='estimators = ["A"]')
    slurm = FakeSlurm()
    ts.run_cycle(e, slurm)
    assert ts.run_cycle(e, slurm).submitted == 0


def test_farm_packing_and_status_files(world):
    e = make(
        world,
        resources="farm",
        estimators='estimators = ["A", "B", "C"]',
        body="resamples = 4",
    )
    slurm = FakeSlurm()
    result = ts.run_cycle(e, slurm)
    # max_jobs 2, 8 slots at 4G, 2 waves -> 16 tasks per job
    assert len(slurm.scripts) == 2
    assert result.submitted == 32
    first = slurm.scripts["1001"]
    assert "--cpus-per-task=8" in first and "--mem=32G" in first
    assert "slots=8" in first

    age_submissions(e)
    state = ts.load_state(e)
    keys = [k for k, r in state["tasks"].items() if r["job"] == "1001"]
    oom_key, done_key, quiet_key = keys[0], keys[1], keys[2]

    def log(key):
        est, ds, rs = key.split("|")
        return ts.task_log(e, ts.Task(est, "", ds, int(rs)), "1001")

    Path(f"{log(oom_key)}.status").write_text("started now\nexit 137\n")
    Path(f"{log(done_key)}.status").write_text("started now\nexit 1\n")
    Path(f"{log(done_key)}.err").write_text("Traceback (most recent call last):\nX\n")
    slurm.acct = {("1001", None): "COMPLETED", ("1002", None): "TIMEOUT"}
    ts.run_cycle(e, slurm, submit=False)
    state = ts.load_state(e)
    assert state["tasks"][oom_key]["tier"] == 1
    assert state["tasks"][done_key]["events"][-1]["reason"] == ts.ERROR
    assert state["tasks"][quiet_key]["attempts"] == 0  # never started
    # tasks of the timed-out farm that never started are also refunded
    assert all(
        r["attempts"] == 0
        for k, r in state["tasks"].items()
        if r.get("events") and r["events"][-1]["job"] == "1002"
    )


def test_farm_shape_respects_memory():
    class E:
        n_jobs = 1
        resources = ts.Resources(
            partition="b",
            time_limit="1",
            memory_gb=(4, 64),
            node_cpus=192,
            node_memory_gb=630,
        )

    assert ts.farm_shape(E, 4, 1000) == 157
    assert ts.farm_shape(E, 64, 1000) == 9
    assert ts.farm_shape(E, 4, 3) == 3


def test_stop_file_blocks_submission(world):
    e = make(world)
    e.state_dir.mkdir(parents=True)
    (e.state_dir / "STOP").write_text("x")
    result = ts.run_cycle(e, FakeSlurm())
    assert result.submitted == 0 and "STOP" in result.report


def test_changed_kwargs_are_refused_but_new_estimators_allowed(world):
    e = make(world, extra_sections="[estimator_kwargs.A]\nn = 1")
    ts.run_cycle(e, FakeSlurm())
    e = make(
        world,
        estimators='estimators = ["A", "B", "C"]',
        extra_sections="[estimator_kwargs.A]\nn = 1",
    )
    ts.run_cycle(e, FakeSlurm())
    e = make(world, extra_sections="[estimator_kwargs.A]\nn = 2")
    with pytest.raises(RuntimeError, match="changed since"):
        ts.run_cycle(e, FakeSlurm())


def test_commit_is_pinned(world):
    e = make(world)
    ts.run_cycle(e, FakeSlurm())
    (world["repo"] / "README").write_text("y")
    _git(world["repo"], "commit", "-qam", "next")
    with pytest.raises(RuntimeError, match="--accept-commit"):
        ts.run_cycle(e, FakeSlurm())
    ts.run_cycle(e, FakeSlurm(), accept_commit=True)


def test_dirty_repository_is_refused(world):
    e = make(world)
    (world["repo"] / "README").write_text("dirty")
    with pytest.raises(RuntimeError, match="uncommitted"):
        ts.run_cycle(e, FakeSlurm())


def test_dry_run_changes_nothing(world):
    e = make(world)
    slurm = FakeSlurm()
    result = ts.run_cycle(e, slurm, dry_run=True)
    assert result.submitted == 12 and not slurm.scripts
    assert "#!/bin/bash" in result.report
    assert not (e.state_dir / "state.json").exists()


def test_retry_command_clears_failures(world, capsys):
    e = make(world, body="resamples = 1", estimators='estimators = ["A"]')
    e.state_dir.mkdir(parents=True)
    ts.save_state(
        e,
        {
            "version": 1,
            "experiment": "exp",
            "fingerprints": {},
            "commit": None,
            "submissions": [],
            "last_email": 0,
            "tasks": {
                "A|Small|0": {"attempts": 2, "tier": 2, "terminal": "OOM at 16G"},
                "A|Big|0": {"attempts": 2, "tier": 0, "terminal": ts.ERROR},
            },
        },
    )

    class Args:
        reason = "oom"
        memory_tier = 1

    ts.cmd_retry(e, Args)
    state = ts.load_state(e)["tasks"]
    assert "terminal" not in state["A|Small|0"] and state["A|Small|0"]["tier"] == 1
    assert state["A|Big|0"]["terminal"] == ts.ERROR


def test_queue_failure_submits_nothing(world):
    class Broken(FakeSlurm):
        def available(self):
            return False

    result = ts.run_cycle(make(world), Broken())
    assert result.submitted == 0 and "squeue not found" in result.report
    assert not result.settled


def test_diagnosis_from_logs_without_accounting(world):
    e = make(world)
    task = ts.Task("A", "", "Small", 0)
    record = {"job": "7", "kind": "array"}
    log = ts.task_log(e, task, "7")
    log.parent.mkdir(parents=True)
    Path(f"{log}.err").write_text("slurmstepd: error: Detected 1 oom_kill event")
    assert ts.diagnose(e, task, record, None)[0] == ts.OOM
    Path(f"{log}.err").write_text("torch.OutOfMemoryError: CUDA out of memory.")
    assert ts.diagnose(e, task, record, None)[0] == ts.GPU_OOM
    Path(f"{log}.err").write_text("CANCELLED AT 2026 DUE TO TIME LIMIT")
    assert ts.diagnose(e, task, record, None)[0] == ts.TIMEOUT


def test_state_file_is_json(world):
    e = make(world)
    ts.run_cycle(e, FakeSlurm())
    data = json.loads((e.state_dir / "state.json").read_text())
    assert data["commit"] and len(data["tasks"]) == 12
