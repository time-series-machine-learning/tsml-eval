from aeon.datasets import load_classification
from aeon.datasets.tsc_datasets import multiverse_core
from pathlib import Path
from joblib import Parallel, delayed
from aeon.datasets import load_classification
from aeon.datasets.tsc_datasets import multiverse_core
import traceback

data_dir = Path.home() / "Data"
n_jobs = 8  # change as needed

def download_dataset(dataset_name):
    try:
        print(f"Downloading {dataset_name}", flush=True)

        X, y = load_classification(
            dataset_name,
            extract_path=str(data_dir),
        )

        n_cases = len(y)
        n_classes = len(set(y))

        print(
            f"Finished {dataset_name}: cases={n_cases}, classes={n_classes}",
            flush=True,
        )

        return {
            "dataset": dataset_name,
            "success": True,
            "n_cases": n_cases,
            "n_classes": n_classes,
        }

    except Exception as e:
        print(
            f"FAILED {dataset_name}: {type(e).__name__}: {e}",
            flush=True,
        )
        traceback.print_exc()

        return {
            "dataset": dataset_name,
            "success": False,
            "error": f"{type(e).__name__}: {e}",
        }


results = Parallel(n_jobs=n_jobs, prefer="threads")(
    delayed(download_dataset)(d) for d in multiverse_core
)

print("\nSummary")
for r in results:
    if r["success"]:
        print(f"OK    {r['dataset']}")
    else:
        print(f"FAIL  {r['dataset']}  {r['error']}")