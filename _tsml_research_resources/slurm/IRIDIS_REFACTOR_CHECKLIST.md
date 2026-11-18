# Deferred Iridis dataset refactors

Apply these together after the user finishes the naming decisions and switches
to the Southampton VPN. Do not update Iridis during the Hali refactor batches.
Iridis 6 and Iridis X share Data, Results and runner state, but have two
tsml-eval checkouts and separate queues.

## Pending names

| Previous name | Canonical name |
|---|---|
| LowCost | ImaginedDominantHandRest |
| SitStand | ImaginedSitStand |
| PsychosisRestFirst | PsychosisRestCohort1 |
| PsychosisRestSecond | PsychosisRestCohort2 |
| AlcoholBandit | AlcoholUseBandit |
| ShortIntervalTask | ParkinsonsShortInterval |
| SleepDeprivation | SleepDeprivationRest |
| FuzzySemantics | SemanticTargets |
| PhotoStimulation | DementiaPhotic |
| CovidSurvivors | PostCovidRest |

LongIntervalTask is excluded from the main benchmark. Preserve its archived data
and results; cancel unfinished EEG experiment tasks through the standard runner
before changing the lists. The selected clinical group currently contains 29
problems, awaiting Neuroverse's replacement. Do not add a placeholder.

ImaginedRightHandFeetRest and ImaginedLeftRightHandRest are also awaiting
Neuroverse replacements. Leave them unchanged until those replacements are
explicitly supplied.

## Migration sequence

1. Inspect both queues, both repository revisions, active supervisors, original
   STOP files, cancellations and checkpoints. Back up all affected state and
   metadata before modifying them. Preserve active REVE/LUNA work and its
   checksummed aeon-neuro snapshot.
2. Temporarily pause affected controllers without cancelling unrelated work.
   Use the same controller locks as the standard Slurm runner.
3. Validate source exports and subject-ID files against the canonical desktop
   exports. For LowCost/SitStand, compare the bytes after `@data`: the desktop
   header names changed, while Hali retained historical headers. Other renames
   retained their raw `.ts` bytes. Do not reuse results if the actual signals,
   labels or splits differ.
4. Rename Data folders and filenames, Predictions and Checkpoints folders, and
   task log folders. Keep Linux old-name symlinks for existing jobs. Update
   current sidecars/manifests, preserving original provenance and backups.
5. Remap persisted task keys without changing attempts, job IDs, memory tiers,
   events or terminal reasons. Preserve cancellation markers and farm work.
   Update first-line result names only; preserve prediction values. Add canonical
   paths to `completed.txt` while retaining old entries for old-name jobs.
6. Update external EEG dataset lists and deploy current tsml-eval metadata in
   **both** `~/Code/tsml-eval` and `~/Code/tsml-eval-gpu`. Keep historical
   published Multiverse archive lists unchanged. Use the canonical short-interval
   name and exclude LongIntervalTask from active EEG benchmark lists.
7. Verify old/new result completion agrees; confirm all earlier cancellations,
   existing results, checkpoints and original STOP settings are preserved.
   Restore only supervisors that were previously active. Use
   `tsml_slurm.py` and TOML configs for all experiment operations. Never use a
   blanket `retry --reason cancelled`.
8. Refresh the Iridis Neuroverse registry and rebuild desktop summaries. Pull
   newly completed results through the registry. Do not rerun completed work.

Desktop migration reports and Hali backup locations are recorded in the local,
untracked `CODEX_HANDOVER.md` and `analysis/`. No cluster downloads or package
installations are needed for these name changes.

## MI replacements (18 Nov)

ImaginedArmGraspElbow and ImaginedLimbActions replace ImaginedLeftRightHandRest and
ImaginedRightHandFeetRest in the benchmark and MI lists (commit 522c3e8). Their data is on
the desktop (`C:/Data/EEG`) and Hali only. They run on Hali through `eeg_not_on_iridis.txt`.
Before restarting any Iridis controller on the new commit, either copy their data to Iridis or
exclude them from Iridis configs that read `eeg_core90.txt`, `eeg_eligible.txt` or the MI
lists. Otherwise tasks will fail on missing data or duplicate Hali's work. Cancel unfinished
Iridis tasks on the two old problems with the standard runner (Hali's were cancelled
18 Nov). Keep them in `eeg_within_subject.txt`.
