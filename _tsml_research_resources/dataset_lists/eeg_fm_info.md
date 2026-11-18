# Foundation-model dataset metadata

`eeg_fm_info.csv` is read by the Slurm runner. Estimator kwargs can reference any
column with `{dataset.<column>}`; a blank required cell excludes that task.

- `ch_names` retains the existing LaBraM eligibility convention. It is blank for
  windows longer than LaBraM's 16 s limit, even when electrode names are known.
- `ch_names_any_length` contains ordered electrode names without that length
  restriction. REVE and LUNA configurations should set
  `ch_names = "{dataset.ch_names_any_length}"`, including their random controls.
  Existing LaBraM configurations continue to use `{dataset.ch_names}`.

On 18 November 2026, the Core 90 have 86 populated `ch_names` cells and 89 populated
`ch_names_any_length` cells. The additional datasets are Alzheimers,
ImaginedHandMusicTempo and PhotoStimulation. Their ordered 19-channel names were
read from `D:/Data/EEG/<dataset>/channels.txt` (Alzheimers and PhotoStimulation) or
`channels.tsv` (ImaginedHandMusicTempo), and checked against both train and test
case dimensions. Existing cells, including the 12 Option 1 channel mappings,
were preserved. PostCovidVigilance remains unnamed in both columns.

These counts describe channel-metadata availability, not completed experiments
or a guarantee of model compatibility or sufficient GPU memory. Model-specific
position lookup and training validation remain necessary. Preserve the source
provenance and interpretation of the inherited Option 1 mappings.
