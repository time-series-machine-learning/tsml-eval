# Multiverse reference results on Iridis

Fills the **66 Multiverse-core datasets at resample 0** for the 27 classifiers the
archive reports on, running only what a D-drive inventory says is missing. It is the
multivariate counterpart of `ucr_reference` and shares its `controller.py`, manifest
model, task farms and monitor; only the configuration and the manifest differ.

Resample 0 only. From 2026-09-09 the Multiverse archive runs one resample, so
`resamples` is 1 and the keys are all `|0`. Do not raise it here without changing
that decision first.

## The inventory, 11 September 2026

`multiverse_reference_manifest.json` records which `classifier|dataset|resample`
keys exist under `D:/Results/Multiverse`. It is metadata only and copies no
predictions.

| | |
|---|---:|
| Required experiments | 1,782 |
| Complete on D | 1,671 (93.8%) |
| Missing | 111 |

Eleven classifiers are complete. What is missing concentrates in a few places:

| Classifier | Complete | Missing |
|---|---:|---|
| ConvTran | 0/66 | **see below — the results are not on D, not absent** |
| RankSCL | 50/66 | the timeout set, awaiting the dense loss and probe fixes |
| TS2Vec, MUSE | 61/66 | probe timeouts; `chi2` densification |
| HC2 | 62/66 | AustraliaRainfall_disc, STEW, Tiselac, USCActivity |
| MRHydra, RDST | 63/66 | aeon issue 3738, and PenDigits at length 8 |
| nine others | 65/66 | EmoPain, or AustraliaRainfall_disc |

### ConvTran needs a sync, not a rerun

`D:/Results/Multiverse/DeepLearning/ConvTran/` is an empty directory. The job logs
are still there, 70 of them under `output/ConvTran`, and the ingested accuracies for
62 datasets are in the multiverse repository, so these results existed and were
lost from the reference tree rather than never produced.

That is **66 of the 111 missing keys**. Sync `Results/Multiverse/DeepLearning/ConvTran`
back from Iridis and regenerate the manifest before launching, or the run will spend
two thirds of its work reproducing results that already exist.

## Regenerating the manifest

The UCR manifest was built by hand, which is why its README could only say
"regenerate the manifest" without naming a tool. `make_reference_manifest.py`, in
`../ucr_reference/`, is that tool and serves both runs:

```bash
python ../ucr_reference/make_reference_manifest.py \
    --config multiverse_reference.json \
    --source D:/Results/Multiverse \
    --out multiverse_reference_manifest.json
```

Run it on the machine holding the D drive, not on Iridis. Add `--check` to compare
against the checked-in manifest and print what moved without writing anything —
that is what to run after a sync from Iridis, to see what arrived before deciding
whether a rerun is still needed.

A result counts as complete when its prediction file exists and is nonempty, which
is the controller's own rule, and the controller independently validates that the
manifest's keys exactly cover the configured universe. A mismatch is an error, not
a warning.

## Running

```bash
bash run_multiverse_reference.sh --check      # resolve every classifier, submit nothing
bash run_multiverse_reference.sh --dry-run    # show the plan
bash run_multiverse_reference.sh              # CPU work on batch
bash monitor_multiverse_reference.sh          # progress and blocked failures
```

`--category` restricts to one family, repeatable, for example
`--category DeepLearning --category Hybrid`.

### CPU is the supported path

The 17 CPU classifiers run on Iridis 6 `batch` through the shared task farm, with
the memory ladder 4 to 620 GiB per experiment and the usual 60 hour limit.

The 10 GPU classifiers are configured for completeness but **the GPU block is
inherited from the UCR run and has not been exercised here**: it expects an
Apptainer TensorFlow sandbox on Iridis 6's `gpu` partition, whereas the Multiverse
deep learners have been running on IridisX `i7_h200` under conda, through
`start_multiverse_core_*_gpu_iridisx.sh`. Until that container is confirmed to hold
torch, TensorFlow and the ported estimators, keep the deep learners on the IridisX
starters and leave this run on `--device cpu`, which is the wrapper default.

### Ported estimators

Eight of the classifiers are Multiverse ports living in `tsml_eval._wip`, not aeon:
ConvTran, DisjointCNN, PatchMTSC, RankSCL, TimesNet, TimesURL, TS2Vec and XCM. The
controller verifies that a factory key resolves to the expected class in the
expected package, and those rows carry `"module": "tsml_eval."` for that check. Rows
without the field default to `aeon.`, so the UCR run is unaffected.

## What this will not fix

Some of the 111 are not waiting on compute, and a reference run will keep
rediscovering that. Recorded here so the failures are expected rather than
investigated twice:

- **EmoPain** — aeon's `check_collection_variance` raises before fit for every aeon
  classifier. Fixed upstream in aeon by #3598, which relegated it to a warning, but
  that landed after v1.5.0 was tagged, so it needs an aeon newer than any release.
- **PenDigits** — MRHydra requires `n_timepoints >= 9` and the series are length 8.
- **AustraliaRainfall_disc, Tiselac** — RDST, ROCKET and MRHydra hit LAPACK integer
  overflow in `RidgeClassifierCV`'s SVD, aeon issue 3738.
- **RankSCL, TS2Vec** — their timeouts were a Python loop in the ranking loss and
  Platt scaling carried into the probe's grid search, both fixed on `ajb/gpu`. Those
  reruns belong on IridisX, and the controller records a timeout as terminal, so
  their state has to be cleared before they will be submitted again.
