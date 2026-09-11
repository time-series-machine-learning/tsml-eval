"""Offline tests for UCR coverage, submission safety and recovery."""

import contextlib
import copy
import io
import re
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import controller as ctl


class ControllerTests(unittest.TestCase):
    """Exercise submission and monitoring against a small synthetic results tree."""

    def setUp(self):
        """Create two classifiers, two datasets and two resamples."""
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.c = ctl.load_config(ctl.HERE / "ucr_reference.json")
        self.c.update(results_root=str(self.root / "results"), data_dir=str(self.root / "data"),
                      state_dir=str(self.root / "results" / ".ucr-reference-state"),
                      dataset_list=str(self.root / "datasets.txt"), resamples=2)
        Path(self.c["dataset_list"]).write_text("A\nB\n")
        for d in ("A", "B"):
            folder = Path(self.c["data_dir"]) / d
            folder.mkdir(parents=True)
            for s in ("TRAIN", "TEST"):
                (folder / f"{d}_{s}.ts").write_text("data")
        self.c["classifiers"] = [
            dict(name="Dummy", category="Other", device="cpu", train=True, **{"class": "DummyClassifier"}),
            dict(name="TimeCNNClassifier", category="DeepLearning", device="gpu", train=False, **{"class": "TimeCNNClassifier"}),
        ]
        self.c["reference_manifest"] = str(self.root / "reference-manifest.json")
        target_keys = [ctl.key_for(row["name"], d, i)
                       for row in self.c["classifiers"] for d in ("A", "B") for i in range(2)]
        ctl.save_json(Path(self.c["reference_manifest"]), {
            "schema": 1, "resamples": 2, "reference_complete": [],
            "target_tasks": target_keys, "total_tasks": len(target_keys),
        })
        self.c["cpu"].update(max_jobs=1, max_cpus=2, waves_per_job=1)
        self.c["gpu"].update(max_jobs=1, commands_per_job=2)
        self.state = dict(jobs={}, attempts={}, round=0)
        self.args = SimpleNamespace(device="all", dry_run=False, check=False, no_chain=True)

    def put_result(self, row=0, dataset="A", resample=0, test="test", train=None):
        """Write only the requested result splits."""
        paths = ctl.result_paths(self.c, self.c["classifiers"][row], dataset, resample)
        paths[0].parent.mkdir(parents=True, exist_ok=True)
        if test is not None:
            paths[0].write_text(test)
        if train is not None:
            paths[1].write_text(train)
        return paths

    def job(self, device="cpu", allocation_id="123", started=True):
        """Build one previously submitted task with a durable start marker."""
        row = self.c["classifiers"][device == "gpu"]
        task = dict(key=ctl.key_for(row["name"], "A", 0), classifier=row["name"],
                    dataset="A", resample=0, status=str(self.root / "status.json"), log=str(self.root / "task.log"))
        if started:
            ctl.save_json(Path(task["status"]), {"state": "RUNNING"})
        job = dict(id=allocation_id, name=f"ucr-reference-{device}-token", stage="SUBMITTED",
                   device=device, tier=0, tasks=[task])
        self.state["jobs"]["token"] = job
        return job, task

    def test_only_declared_root_and_required_splits_count(self):
        """Ignore duplicate test-only trees, unknown datasets, extra folds and empties."""
        paths = self.put_result(train="")
        self.put_result(1)
        self.put_result(dataset="Outside", train="train")
        self.put_result(resample=30, train="train")
        other = Path(self.c["results_root"]) / "TestOnly" / "Other" / "Dummy" / "Predictions" / "B"
        other.mkdir(parents=True)
        (other / "testResample0.csv").write_text("not the target collection")
        done, counts = ctl.inventory(self.c, ["A", "B"])
        self.assertEqual(done, {"TimeCNNClassifier|A|0"})
        self.assertEqual(counts["Dummy"]["test"], 1)
        paths[1].write_text("train")
        done, _ = ctl.inventory(self.c, ["A", "B"])
        self.assertIn("Dummy|A|0", done)

    def test_manifest_scope_and_original_folders(self):
        """All configured classes have unique explicit targets."""
        c = ctl.load_config(ctl.HERE / "ucr_reference.json")
        self.assertEqual(len(c["classifiers"]), 54)
        self.assertEqual(sum(r["device"] == "gpu" for r in c["classifiers"]), 10)
        by_name = {r["name"]: r for r in c["classifiers"]}
        self.assertEqual(by_name["MrHydra"]["category"], "ConvolutionBased")
        self.assertFalse(by_name["HC2"]["train"])
        self.assertTrue(by_name["BOSS"]["train"])
        self.assertEqual(by_name["STC"]["key"], "stc-aeon")
        self.assertEqual(by_name["PF"]["key"], "pf-aeon")
        self.assertEqual(by_name["ProximityTree"]["key"], "proximitytree-aeon")
        self.assertFalse(any("PreVal" in str(r) or "Experimental" in str(r) for r in c["classifiers"]))

    def test_category_filter_only_plans_dictionary_results(self):
        """A staged launch can fill DictionaryBased while copied results settle."""
        self.c["classifiers"][0]["category"] = "DictionaryBased"
        groups = ctl.plan(self.c, ["A", "B"], self.state, set(), ("cpu", "gpu"), ("DictionaryBased",))
        tasks = [task for batch in groups.values() for task in batch]
        self.assertTrue(tasks)
        self.assertTrue(all(task["classifier"] == "Dummy" for task in tasks))

    def test_reference_manifest_excludes_d_drive_baseline(self):
        """Only target keys are eligible when the reference inventory is complete."""
        baseline_key = ctl.key_for("Dummy", "A", 0)
        target_key = ctl.key_for("Dummy", "A", 1)
        ctl.save_json(Path(self.c["reference_manifest"]), {
            "schema": 1, "resamples": 2,
            "reference_complete": [baseline_key],
            "target_tasks": [k for k in [
                ctl.key_for(row["name"], d, i)
                for row in self.c["classifiers"] for d in ("A", "B") for i in range(2)
            ] if k != baseline_key],
            "total_tasks": 8,
        })
        baseline, targets, _ = ctl.load_manifest(self.c, ["A", "B"])
        self.assertEqual(baseline, {baseline_key})
        groups = ctl.plan(self.c, ["A", "B"], self.state, baseline,
                          ("cpu", "gpu"), task_keys=targets)
        keys = {task["key"] for batch in groups.values() for task in batch}
        self.assertNotIn(baseline_key, keys)
        self.assertIn(target_key, keys)

    def test_live_and_accounting_lag_are_reserved(self):
        """A temporarily absent scheduler record must not create duplicate work."""
        job, task = self.job()
        with patch.object(ctl, "accounting", return_value="UNKNOWN") as acct:
            ctl.reconcile(self.c, self.state, set(), {"123": {}})
            acct.assert_not_called()
            ctl.reconcile(self.c, self.state, set(), {})
        self.assertEqual(job["stage"], "ACCOUNTING")
        groups = ctl.plan(self.c, ["A", "B"], self.state, set(), ("cpu", "gpu"))
        self.assertNotIn(task["key"], {t["key"] for ts in groups.values() for t in ts})

    def test_oom_escalates_only_once(self):
        """Repeated monitor/refill cycles must not charge one failure repeatedly."""
        job, task = self.job()
        Path(task["log"]).write_text("MemoryError: allocation failed")
        with patch.object(ctl, "accounting", return_value="FAILED"):
            ctl.reconcile(self.c, self.state, set(), {})
            ctl.reconcile(self.c, self.state, set(), {})
        record = self.state["attempts"][task["key"]]
        self.assertEqual((record["tier"], record["attempts"], record["failures"]), (1, 1, 0))
        self.assertEqual(job["stage"], "RECONCILED")

    def test_unstarted_timeout_consumes_no_attempt(self):
        """Time-limited task farms can safely requeue commands they never reached."""
        _, task = self.job(started=False)
        with patch.object(ctl, "accounting", return_value="TIMEOUT"):
            ctl.reconcile(self.c, self.state, set(), {})
        record = self.state["attempts"][task["key"]]
        self.assertEqual(record["attempts"], 0)
        self.assertEqual(record["reason"], "NOT_STARTED")

    def test_gpu_oom_does_not_increase_host_memory(self):
        """Increasing host RAM cannot fix exhausted GPU VRAM."""
        _, task = self.job(device="gpu")
        Path(task["log"]).write_text("ResourceExhaustedError: OOM when allocating tensor")
        with patch.object(ctl, "accounting", return_value="FAILED"):
            ctl.reconcile(self.c, self.state, set(), {})
        record = self.state["attempts"][task["key"]]
        self.assertEqual((record["tier"], record["failures"], record["reason"]), (0, 1, "GPU_OOM"))

    def test_completed_files_override_job_failure(self):
        """A later failed command in a farm must not invalidate earlier successes."""
        _, task = self.job()
        self.put_result(train="train")
        complete, _ = ctl.inventory(self.c, ["A", "B"])
        with patch.object(ctl, "accounting", return_value="FAILED"):
            ctl.reconcile(self.c, self.state, complete, {})
        self.assertEqual(self.state["attempts"][task["key"]]["reason"], "COMPLETE")

    def test_exhausted_failure_is_not_replanned(self):
        """Repeated errors stop at the configured bound and remain visibly incomplete."""
        _, task = self.job()
        self.c["max_failures"] = 1
        with patch.object(ctl, "accounting", return_value="FAILED"):
            ctl.reconcile(self.c, self.state, set(), {})
        self.assertTrue(self.state["attempts"][task["key"]]["blocked"])
        groups = ctl.plan(self.c, ["A", "B"], self.state, set(), ("cpu",))
        self.assertNotIn(task["key"], {t["key"] for ts in groups.values() for t in ts})

    def test_startup_failure_stops_instead_of_endless_requeue(self):
        """A broken container or taskfarm launch must not burn thousands of jobs."""
        self.job(started=False)
        with patch.object(ctl, "accounting", return_value="FAILED"):
            with self.assertRaisesRegex(RuntimeError, "before any worker"):
                ctl.reconcile(self.c, self.state, set(), {})

    def test_ambiguous_submission_recovers_from_live_unique_name(self):
        """Recover a successful sbatch interrupted before the job ID was persisted."""
        job, _ = self.job(allocation_id=None)
        job["stage"] = "SUBMITTING"
        ctl.reconcile(self.c, self.state, set(), {"456": {"name": job["name"]}})
        self.assertEqual(job["id"], "456")

    def test_ambiguous_submission_without_evidence_stops(self):
        """An unknown submission outcome is not assumed to mean no job exists."""
        self.job(allocation_id=None)
        with self.assertRaisesRegex(RuntimeError, "outcome unknown"):
            ctl.reconcile(self.c, self.state, set(), {})

    def test_cpu_and_gpu_submission_resources(self):
        """Keep one GPU per serial allocation and CPU requests within node RAM."""
        job, _ = self.job()
        job["tasks"] *= 300
        self.c["cpu"]["max_cpus"] = 192
        cpu = ctl.batch_script(self.c, job, self.root / "config.json", self.root / "commands.txt")
        self.assertIn("#SBATCH --ntasks=157", cpu)
        self.assertIn("#SBATCH --mem-per-cpu=4G", cpu)
        self.assertIn("staskfarm", cpu)
        job["device"] = "gpu"
        gpu = ctl.batch_script(self.c, job, self.root / "config.json", self.root / "commands.txt")
        self.assertIn("#SBATCH --gres=gpu:1", gpu)
        self.assertIn("#SBATCH --ntasks=1", gpu)
        self.assertNotIn("staskfarm", gpu)
        self.assertIn("while IFS= read", gpu)
        self.assertIn("--nv", ctl.runtime(self.c, "gpu", gpu=True))

    def test_dry_run_does_not_touch_state_or_slurm(self):
        """The local D-drive inventory must be a read-only preview."""
        self.args.dry_run = True
        with patch.object(ctl, "command", side_effect=AssertionError("scheduler called")), contextlib.redirect_stdout(io.StringIO()):
            ctl.run_cycle(self.c, self.args)
        self.assertFalse(Path(self.c["state_dir"]).exists())

    def test_refill_is_idempotent_and_bounded(self):
        """Two launch cycles reserve the same work and respect CPU/GPU job limits."""
        live = {}
        submitted = []

        def fake_command(args, **kwargs):
            self.assertEqual(args[0], "sbatch")
            text = Path(args[-1]).read_text()
            name = re.search(r"--job-name=(.*)", text)[1]
            jid = str(100 + len(submitted))
            submitted.append(args)
            live[jid] = dict(name=name, state="PENDING", node="Priority")
            return jid

        with (patch.object(ctl, "locked", lambda _: contextlib.nullcontext()),
              patch.object(ctl, "provenance", return_value={"aeon": "fixed"}),
              patch.object(ctl, "preflight"),
              patch.object(ctl, "query_slurm", side_effect=lambda _: copy.deepcopy(live)),
              patch.object(ctl, "command", side_effect=fake_command),
              contextlib.redirect_stdout(io.StringIO())):
            ctl.run_cycle(self.c, self.args)
            ctl.run_cycle(self.c, self.args)
        self.assertEqual(len(submitted), 2)
        state = ctl.read_json(Path(self.c["state_dir"]) / "state.json")
        self.assertEqual(len(ctl.active_tasks(state)), 4)
        self.assertEqual(len(state["jobs"]), 2)

    def test_sbatch_failure_is_journalled(self):
        """Persist the prepared manifest even when sbatch's outcome is unknown."""
        _, task = self.job(started=False)
        self.state["jobs"] = {}
        state_path = Path(self.c["state_dir"]) / "state.json"
        with patch.object(ctl, "command", side_effect=RuntimeError("connection failed")):
            with self.assertRaisesRegex(RuntimeError, "connection failed"):
                ctl.submit_batch(self.c, self.state, state_path, self.root / "config.json", "cpu", 0, [task])
        state = ctl.read_json(state_path)
        self.assertEqual(next(iter(state["jobs"].values()))["stage"], "SUBMITTING")

    def test_one_supervisor_and_persistent_device_scope(self):
        """A manual CPU refill retains GPU scope and creates no second supervisor."""
        self.args.no_chain = False
        live = {}
        scripts = []

        def submit(args, **kwargs):
            text = Path(args[-1]).read_text()
            scripts.append(text)
            jid = str(len(scripts) + 700)
            live[jid] = dict(name=re.search(r"--job-name=(.*)", text)[1], state="PENDING", node="Priority")
            return jid

        with (patch.object(ctl, "locked", lambda _: contextlib.nullcontext()),
              patch.object(ctl, "provenance", return_value={"aeon": "fixed"}),
              patch.object(ctl, "preflight"),
              patch.object(ctl, "query_slurm", side_effect=lambda _: copy.deepcopy(live)),
              patch.object(ctl, "command", side_effect=submit),
              contextlib.redirect_stdout(io.StringIO())):
            ctl.run_cycle(self.c, self.args)
            self.args.device = "cpu"
            ctl.run_cycle(self.c, self.args)
        self.assertEqual(len(scripts), 3)
        self.assertIn("#SBATCH --begin=now+15minutes", scripts[-1])
        self.assertIn("--device all", scripts[-1])
        state = ctl.read_json(Path(self.c["state_dir"]) / "state.json")
        self.assertEqual(state["devices"], ["cpu", "gpu"])

    def test_preflight_failure_submits_nothing(self):
        """Validate both selected runtimes before allocating any worker resources."""
        with (patch.object(ctl, "locked", lambda _: contextlib.nullcontext()),
              patch.object(ctl, "provenance", return_value={"aeon": "fixed"}),
              patch.object(ctl, "query_slurm", return_value={}),
              patch.object(ctl, "preflight", side_effect=RuntimeError("GPU dependencies missing")),
              patch.object(ctl, "command", side_effect=AssertionError("submitted"))):
            with self.assertRaisesRegex(RuntimeError, "GPU dependencies missing"):
                ctl.run_cycle(self.c, self.args)

    def test_orphan_live_batch_stops_refill(self):
        """Lost state must not turn an existing reference allocation into new work."""
        live = {"100": dict(name="ucr-reference-cpu-orphan", state="RUNNING", node="node1")}
        with (patch.object(ctl, "locked", lambda _: contextlib.nullcontext()),
              patch.object(ctl, "query_slurm", return_value=live)):
            with self.assertRaisesRegex(RuntimeError, "missing from this state"):
                ctl.run_cycle(self.c, self.args)

    def test_stop_file_prevents_scheduler_calls(self):
        """A delayed supervisor honours STOP without cancelling active jobs."""
        stop = Path(self.c["state_dir"]) / "STOP"
        stop.parent.mkdir(parents=True)
        stop.touch()
        with (patch.object(ctl, "locked", lambda _: contextlib.nullcontext()),
              patch.object(ctl, "query_slurm", side_effect=AssertionError("scheduler called")),
              contextlib.redirect_stdout(io.StringIO())):
            ctl.run_cycle(self.c, self.args)


if __name__ == "__main__":
    unittest.main()
