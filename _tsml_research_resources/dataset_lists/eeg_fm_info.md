# Foundation-model dataset metadata

`eeg_fm_info.csv` is read by the Slurm runner. Estimator kwargs can reference any
column with `{dataset.<column>}`; a blank required cell excludes that task.

- `ch_names` retains the existing LaBraM eligibility convention. It is blank for
  windows longer than LaBraM's 16 s limit, even when electrode names are known.
- `ch_names_any_length` contains ordered electrode names without that length
  restriction. REVE and LUNA configurations should set
  `ch_names = "{dataset.ch_names_any_length}"`, including their random controls.
  Existing LaBraM configurations continue to use `{dataset.ch_names}`.

On 19 November 2026, the Core 90 (`30 MI / 30 Clinical / 30 Cognitive`) have 87
populated `ch_names` cells and all 90 populated `ch_names_any_length` cells. The
three long-window datasets populated only in `ch_names_any_length` are
Alzheimers, ImaginedHandMusicTempo and DementiaPhotic; their ordered 19-channel
names were read from `D:/Data/EEG/<dataset>/channels.txt` (Alzheimers and
DementiaPhotic) or `channels.tsv` (ImaginedHandMusicTempo), and checked against
both train and test case dimensions. `PostCovidVigilance` (`OpenNeuro ds007605`,
127 EasyCap `BC-128-pass` equidistant channels `BrainVision RDA_1..19, 21..128`
at 1,000 Hz, 4.0 s windows) uses the Option-1 3D nearest-trained-LaBraM `10-20`
projection computed from the authors' upstream FieldTrip electrode file
`BC-128-pass-lay.mat` (`christianneumannCAU/EPOC-Kiel`, commit `5190e4af`),
preserving native `channels.tsv` labels untouched. Existing Option 1 channel
mappings were preserved.

These counts describe channel-metadata availability, not completed experiments
or a guarantee of model compatibility or sufficient GPU memory. Model-specific
position lookup and training validation remain necessary. Preserve the source
provenance and interpretation of the inherited Option 1 mappings.
