#!/bin/bash
# Cancel only the explicitly identified newer duplicates from the resample-0 rerun.

set -euo pipefail

username=${USER:-ajb}
partition="compute"

for command in squeue scancel; do
    if ! command -v "${command}" >/dev/null 2>&1; then
        echo "ERROR: ${command} was not found. Run this on a Slurm login node." >&2
        exit 1
    fi
done

# newer array ID | expected full job name | older array ID to retain
duplicate_pairs=(
    "3811192_1|H-InceptionTime_AustraliaRainfall_disc|3784401_1"
    "3811194_1|H-InceptionTime_BIDMC32HR_disc|3784416_1"
    "3811195_1|H-InceptionTime_BIDMC32SpO2_disc|3784419_1"
    "3811196_1|H-InceptionTime_CounterMovementJump|3784434_1"
    "3811197_1|H-InceptionTime_CrowdSourced|3784440_1"
    "3811198_1|H-InceptionTime_EigenWorms|3784446_1"
    "3811199_1|H-InceptionTime_FordChallenge|3784467_1"
    "3811200_1|H-InceptionTime_HouseholdPowerConsumption1_disc|3784479_1"
    "3811201_1|H-InceptionTime_HouseholdPowerConsumption2_disc|3784482_1"
    "3811202_1|H-InceptionTime_IEEEPPG_disc|3784485_1"
    "3811203_1|H-InceptionTime_MotionSenseHAR|3784518_1"
    "3811204_1|H-InceptionTime_MotorImagery|3784521_1"
    "3811205_1|H-InceptionTime_Skoda|3784545_1"
    "3811206_1|H-InceptionTime_STEW|3784554_1"
    "3811207_1|H-InceptionTime_TactileTextureRecognition|3784558_1"
    "3811208_1|H-InceptionTime_Tiselac|3784561_1"
    "3811209_1|H-InceptionTime_UCIActivity|3784567_1"
    "3811210_1|H-InceptionTime_USCActivity|3784573_1"
    "3811211_1|H-InceptionTime_WISDM|3784579_1"
    "3811212_1|HC2_BIDMC32HR_disc|3785695_1"
    "3811213_1|HC2_BIDMC32SpO2_disc|3785696_1"
    "3811218_1|LiteTIME_BIDMC32HR_disc|3784417_1"
    "3811219_1|LiteTIME_BIDMC32SpO2_disc|3784420_1"
    "3811172_1|1NN-DTW_BIDMC32HR_disc|3784047_1"
    "3811173_1|1NN-DTW_BIDMC32SpO2_disc|3784052_1"
)

queue_output=$(
    squeue --noheader --array --user="${username}" \
        --partition="${partition}" --states=RUNNING,PENDING \
        --format='%i|%200j'
)

declare -A active_names=()
while IFS='|' read -r raw_id raw_name; do
    [[ -z "${raw_id}" ]] && continue
    job_id=${raw_id//[[:space:]]/}
    job_name=${raw_name//[[:space:]]/}
    active_names["${job_id}"]=${job_name}
done <<< "${queue_output}"

cancel_ids=()
retained=0
skipped=0

for pair in "${duplicate_pairs[@]}"; do
    IFS='|' read -r newer_id expected_name older_id <<< "${pair}"

    if [[ -z "${active_names[${newer_id}]+present}" ]]; then
        echo "SKIP: newer job ${newer_id} is no longer active"
        ((skipped += 1))
        continue
    fi
    if [[ "${active_names[${newer_id}]}" != "${expected_name}" ]]; then
        echo "SKIP: ${newer_id} has unexpected name "\
            "${active_names[${newer_id}]}" >&2
        ((skipped += 1))
        continue
    fi

    if [[ -z "${active_names[${older_id}]+present}" ]]; then
        echo "KEEP: ${newer_id}; old counterpart ${older_id} is no longer active"
        ((retained += 1))
        continue
    fi
    if [[ "${active_names[${older_id}]}" != "${expected_name}" ]]; then
        echo "KEEP: ${newer_id}; ${older_id} has unexpected name "\
            "${active_names[${older_id}]}" >&2
        ((retained += 1))
        continue
    fi

    echo "CANCEL: ${newer_id} ${expected_name}; retaining ${older_id}"
    cancel_ids+=("${newer_id}")
done

echo
echo "Confirmed newer duplicates: ${#cancel_ids[@]}"
echo "New jobs retained:          ${retained}"
echo "Pairs already gone/skipped: ${skipped}"

if ((${#cancel_ids[@]})); then
    scancel "${cancel_ids[@]}"
    echo "Cancellation requested for: ${cancel_ids[*]}"
else
    echo "Nothing was cancelled."
fi
