# EEGPT and BIOT deployment, 19 November 2026

`eegpt_biot_20261119.json` records every Python source hash in the isolated
aeon-neuro snapshot (`/home/ajb2u23/Code/aeon-neuro-fm-20261119`), checkpoint
hashes, upstream Hub revisions and run settings. Keeping a separate snapshot
directory leaves any in-flight `aeon-neuro-fm-20261118` (`REVE-FT` / `LUNA-FT`)
jobs completely undisturbed. Checkpoints are copied from `D:/hf_models` on the
desktop; clusters perform no external downloads or package installations.

Configurations for the standard Slurm runner:

- `experiments/iridisx_eeg_eegpt_biot_pilot.toml`: one epoch on
  `ImaginedRightHandFeet` and `Alzheimers`, both pretrained models, writing to
  `~/Results/EEG-FM-Pilot`.
- `experiments/iridisx_eeg_eegpt.toml`: `EEGPT-FT` (`batch_size = 32`), 40 epochs
  maximum, patience 5, all 90/90 Core problems (`ch_names_any_length`), writing
  to `~/Results/EEG`.
- `experiments/iridisx_eeg_biot.toml`: `BIOT-FT` (`variant = "shhs-prest-18chs"`,
  `batch_size = 64`), 40 epochs maximum, patience 5, all 90/90 Core problems,
  writing to `~/Results/EEG` (`gpu-a100` partition).

Pretraining overlap notes:

- `EEGPT-FT` (`braindecode/eegpt-pretrained`, Wang et al., NeurIPS 2024) included
  PhysioNet `eegmmidb` (`OpenCloseFist` / `FeetHands` / `ImaginedOpenCloseFist` /
  `ImaginedFeetHands`), `Wang2016` (`FlickerSymbols`), `SEED`, and `M3CV` during
  pretraining; flag overlapping datasets when interpreting results.
- `BIOT-FT` uses `braindecode/biot-pretrained-shhs-prest-18chs` (unsupervised
  contrastive pretraining on `SHHS` and private MGH `PREST`), which has zero
  overlap with the Core 90 (`six-datasets-18chs` was avoided as the default
  because it trained supervisedly on `CHB-MIT` / `ChildhoodSeizures`).
