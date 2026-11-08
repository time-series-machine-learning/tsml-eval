"""Extract HC2 level-1 (stacking) problems from component results files.

For each dataset and resample, the train problem is the components' train-file
probabilities (their own out-of-bag or cross-validation estimates) and the test
problem is their test-file probabilities. Each is a flat (tabular) problem: one row
per case, with n_components * n_classes features, each component's class
probabilities in turn (component order in components.txt, classes by index). Labels
are the class indices used in the results files.

Each resample is its own pair of files, in the predefined-resample layout that
tsml-eval's load_experiment_data reads with predefined_resample=True:

    C:/Results/Stacking/Data/<archive>/<dataset>/<dataset><resample>_TRAIN.ts

    python extract_stacking_data.py            # all archives
    python extract_stacking_data.py UCR EEG
"""

import sys
import warnings
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from aeon.datasets import save_to_ts_file

from tsml_eval.estimators.classification.hybrid.hivecote_from_results import (
    _load_split,
    component_prediction_paths,
)

OUT = Path("C:/Results/Stacking/Data")
ARCHIVES = {
    # name: (results root, components in channel order, resamples)
    "UCR": (
        "D:/Results/UCR",
        ["ConvolutionBased/Arsenal", "IntervalBased/DrCIF", "ShapeletBased/STC",
         "DictionaryBased/TDE"],
        30,
    ),
    "Multiverse": (
        "D:/Results/Multiverse",
        ["ConvolutionBased/Arsenal", "IntervalBased/DrCIF-500", "ShapeletBased/STC"],
        1,
    ),
    "EEG": (
        "C:/Results/EEG",
        ["ConvolutionBased/Arsenal", "IntervalBased/DrCIF", "ShapeletBased/STC-aeon",
         "DictionaryBased/TDE"],
        1,
    ),
}


def to_X(results):
    """Component probabilities side by side, as (n_cases, 1, n_comps * n_classes).

    The single channel is only how the .ts format stores a flat feature vector.
    """
    X = np.hstack([cr.probabilities for cr in results])
    return X[:, np.newaxis, :]


def extract_dataset(args):
    """Write every resample of one dataset; returns (dataset, complete, written)."""
    warnings.filterwarnings("ignore")
    archive, ds = args
    root, comps, n_res = ARCHIVES[archive]
    paths = component_prediction_paths([f"{root}/{c}" for c in comps], ds)
    out = OUT / archive / ds
    ok = written = 0
    for r in range(n_res):
        if (out / f"{ds}{r}_TEST.ts").exists():
            ok += 1
            continue
        try:
            tr = _load_split(paths, r, "TRAIN")
            te = _load_split(paths, r, "TEST")
        except (FileNotFoundError, ValueError) as e:
            print(f"skip {archive} {ds} r{r}: {type(e).__name__} {str(e)[:90]}")
            continue
        for split, res in (("TRAIN", tr), ("TEST", te)):
            save_to_ts_file(
                to_X(res),
                res[0].class_labels.astype(int),
                label_type="classification",
                path=str(out),
                problem_name=f"{ds}{r}",
                file_suffix=f"_{split}",
            )
        ok += 1
        written += 1
    return ds, ok == n_res, written


def extract(archive, n_jobs=8):
    root, comps, n_res = ARCHIVES[archive]
    first = Path(root) / comps[0] / "Predictions"
    datasets = sorted(d.name for d in first.iterdir() if d.is_dir())
    with Pool(n_jobs) as pool:
        found = pool.map(extract_dataset, [(archive, ds) for ds in datasets])
    done = [ds for ds, complete, _ in found if complete]
    written = sum(w for _, _, w in found)
    (OUT / archive).mkdir(parents=True, exist_ok=True)
    (OUT / archive / "datasets.txt").write_text("\n".join(done) + "\n")
    (OUT / archive / "components.txt").write_text(
        f"results root: {root}\nresamples: {n_res}\nchannels, in order:\n"
        + "\n".join(comps) + "\n"
    )
    print(f"{archive}: {len(done)} datasets complete, {written} resamples written")


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    for a in sys.argv[1:] or ARCHIVES:
        extract(a)
