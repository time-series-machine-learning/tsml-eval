# Multiverse Core resample 0 completeness

Generated on 2026-08-04 from `C:\Temp\MultiverseCore` using the 66 problems in
`_tsml_research_resources/dataset_lists/MultiverseCore.txt`.

A result is complete when its expected `testResample0.csv` exists, is nonempty, and
passes `tsml_eval.utils.results_validation.validate_results_file`.

## Summary

- Classifiers: 17
- Expected classifier/problem results: 1,122
- Complete expected results: 997 (88.9%)
- Missing expected results: 125 (11.1%)
- Empty or invalid results: 0
- Unexpected results: 5, all for `FeedbackButton`
- Fully complete classifiers: `Catch22`, `CIF`, `DrCIF`, and `QUANT`

| Classifier | Complete | Missing | Completion |
|---|---:|---:|---:|
| 1NN-DTW | 63 | 3 | 95.5% |
| Arsenal | 55 | 11 | 83.3% |
| Catch22 | 66 | 0 | 100.0% |
| CIF | 66 | 0 | 100.0% |
| DrCIF | 66 | 0 | 100.0% |
| Dummy | 60 | 6 | 90.9% |
| FreshPRINCE | 58 | 8 | 87.9% |
| HC2 | 58 | 8 | 87.9% |
| H-InceptionTime | 46 | 20 | 69.7% |
| LiteTIME | 62 | 4 | 93.9% |
| MRHydra | 49 | 17 | 74.2% |
| QUANT | 66 | 0 | 100.0% |
| RDST | 63 | 3 | 95.5% |
| RIST | 54 | 12 | 81.8% |
| ROCKET | 53 | 13 | 80.3% |
| STC | 65 | 1 | 98.5% |
| TDE | 47 | 19 | 71.2% |
| **Total** | **997** | **125** | **88.9%** |

The largest cross-classifier gaps are `AustraliaRainfall_disc` (missing for eight
classifiers); `BIDMC32SpO2_disc`, `CounterMovementJump`, and `Tiselac` (seven each);
and `AsphaltObstaclesCoordinates`, `AsphaltRegularityCoordinates`,
`BenzeneConcentration_disc`, `BIDMC32HR_disc`, `CharacterTrajectories`, both
`HouseholdPowerConsumption` problems, `JapaneseVowels`, `SpokenArabicDigits`, `STEW`,
and `USCActivity` (six each).

## Missing problems by classifier

### 1NN-DTW (3)

`BIDMC32HR_disc`, `BIDMC32SpO2_disc`, `STEW`

### Arsenal (11)

`AsphaltObstaclesCoordinates`, `AsphaltRegularityCoordinates`,
`BeijingPM10Quality_disc`, `BeijingPM25Quality_disc`,
`BenzeneConcentration_disc`, `CharacterTrajectories`, `CounterMovementJump`,
`HouseholdPowerConsumption1_disc`, `HouseholdPowerConsumption2_disc`,
`JapaneseVowels`, `SpokenArabicDigits`

### Dummy (6)

`AsphaltObstaclesCoordinates`, `AsphaltRegularityCoordinates`,
`CharacterTrajectories`, `CounterMovementJump`, `JapaneseVowels`,
`SpokenArabicDigits`

### FreshPRINCE (8)

`AustraliaRainfall_disc`, `DuckDuckGeese`, `FaceDetection`, `FordChallenge`,
`PEMS-SF`, `Skoda`, `STEW`, `Tiselac`

### HC2 (8)

`AustraliaRainfall_disc`, `BIDMC32HR_disc`, `BIDMC32SpO2_disc`, `CrowdSourced`,
`FordChallenge`, `STEW`, `Tiselac`, `USCActivity`

### H-InceptionTime (20)

`AustraliaRainfall_disc`, `BenzeneConcentration_disc`, `BIDMC32HR_disc`,
`BIDMC32SpO2_disc`, `CounterMovementJump`, `CrowdSourced`, `EigenWorms`,
`FordChallenge`, `HouseholdPowerConsumption1_disc`,
`HouseholdPowerConsumption2_disc`, `IEEEPPG_disc`, `MotionSenseHAR`,
`MotorImagery`, `Skoda`, `STEW`, `TactileTextureRecognition`, `Tiselac`,
`UCIActivity`, `USCActivity`, `WISDM`

### LiteTIME (4)

`BIDMC32HR_disc`, `BIDMC32SpO2_disc`, `STEW`, `USCActivity`

### MRHydra (17)

`AsphaltObstaclesCoordinates`, `AsphaltRegularityCoordinates`,
`AustraliaRainfall_disc`, `BeijingPM10Quality_disc`, `BeijingPM25Quality_disc`,
`BenzeneConcentration_disc`, `BIDMC32HR_disc`, `BIDMC32SpO2_disc`,
`CharacterTrajectories`, `CounterMovementJump`,
`HouseholdPowerConsumption1_disc`, `HouseholdPowerConsumption2_disc`,
`JapaneseVowels`, `PenDigits`, `SpokenArabicDigits`, `Tiselac`, `USCActivity`

### RDST (3)

`AustraliaRainfall_disc`, `Tiselac`, `USCActivity`

### RIST (12)

`AsphaltObstaclesCoordinates`, `AsphaltRegularityCoordinates`,
`BeijingPM10Quality_disc`, `BeijingPM25Quality_disc`,
`BenzeneConcentration_disc`, `BIDMC32SpO2_disc`, `CharacterTrajectories`,
`CounterMovementJump`, `HouseholdPowerConsumption1_disc`,
`HouseholdPowerConsumption2_disc`, `JapaneseVowels`, `SpokenArabicDigits`

### ROCKET (13)

`AsphaltObstaclesCoordinates`, `AsphaltRegularityCoordinates`,
`AustraliaRainfall_disc`, `BeijingPM10Quality_disc`, `BeijingPM25Quality_disc`,
`BenzeneConcentration_disc`, `CharacterTrajectories`, `CounterMovementJump`,
`HouseholdPowerConsumption1_disc`, `HouseholdPowerConsumption2_disc`,
`JapaneseVowels`, `SpokenArabicDigits`, `Tiselac`

### STC (1)

`AustraliaRainfall_disc`

### TDE (19)

`AsphaltObstaclesCoordinates`, `AsphaltRegularityCoordinates`,
`AustraliaRainfall_disc`, `BeijingPM10Quality_disc`, `BeijingPM25Quality_disc`,
`BenzeneConcentration_disc`, `BIDMC32HR_disc`, `BIDMC32SpO2_disc`,
`CharacterTrajectories`, `CounterMovementJump`, `CrowdSourced`,
`HouseholdPowerConsumption1_disc`, `HouseholdPowerConsumption2_disc`,
`JapaneseVowels`, `Skoda`, `SpokenArabicDigits`, `STEW`, `Tiselac`,
`USCActivity`

## Unexpected results

`FeedbackButton/testResample0.csv` exists for `Catch22`, `Dummy`, `MRHydra`,
`ROCKET`, and `STC`, but `FeedbackButton` is not in the current 66-problem
`MultiverseCore.txt` list. These five files are valid but are excluded from the
completion percentages above.

## Failure analysis

The latest resample-0 logs for the 125 missing results show:

| Diagnosis | Count | Retry assessment |
|---|---:|---|
| Confirmed OOM | 19 | Retry with more than the original 64 GB request |
| Unsupported missing values | 25 | Retry with the current forced no-missing loader |
| Unsupported unequal length | 20 | Retry with the current forced equal-length loader |
| Missing local `TRAIN.ts` | 10 | Retry after `load_classification` obtains the clean data |
| Cancelled | 9 | Retry |
| LAPACK integer overflow | 1 | Do not retry unchanged |
| Inconclusive/nonterminal logs | 34 | Retry only when no matching job is active |
| No logs | 7 | Retry when no matching job is active |

No time-limit failures were found. The 25 logs initially matching a generic runtime
error contain only TensorFlow CPU/CUDA initialization messages or Slurm node-energy
telemetry errors, not evidence that the experiment terminated. They are included in
the 34 inconclusive/nonterminal entries above.

The deterministic error is `ROCKET/AustraliaRainfall_disc`: SciPy's Ridge SVD tries
to index a 2,243,832,186-element matrix and exceeds LAPACK's integer indexing limit.
The targeted retry script therefore leaves this pair for a code-level fix.
