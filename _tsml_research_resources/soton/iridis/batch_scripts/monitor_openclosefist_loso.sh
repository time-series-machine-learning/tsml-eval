#!/bin/bash

set -u

username="${USER:-ajb2u23}"
results_dir="/iridisfs/home/${username}/Results/ChannelSelectionLOSO"
dataset="OpenCloseFistLOSO"
components=("Arsenal" "STC" "DrCIF" "TDE")
first_subject=0
last_subject=104
expected=$((last_subject - first_subject + 1))

printf "OpenCloseFist LOSO progress - %s\n" "$(date)"
printf "%-10s %9s %9s %10s %9s %11s\n" \
    "COMPONENT" "COMPLETE" "TEST_ONLY" "TRAIN_ONLY" "STARTED" "NOT_STARTED"

for component in "${components[@]}"; do
    complete=0
    test_only=0
    train_only=0
    started=0
    not_started=0

    for ((subject = first_subject; subject <= last_subject; subject++)); do
        prediction_dir="${results_dir}/${component}/Predictions/${dataset}"
        test_file="${prediction_dir}/testResample${subject}.csv"
        train_file="${prediction_dir}/trainResample${subject}.csv"

        if [[ -s "${test_file}" && -s "${train_file}" ]]; then
            complete=$((complete + 1))
        elif [[ -s "${test_file}" ]]; then
            test_only=$((test_only + 1))
        elif [[ -s "${train_file}" ]]; then
            train_only=$((train_only + 1))
        elif compgen -G \
            "${results_dir}/output/${component}/output-${dataset}-${subject}-*.txt" \
            > /dev/null; then
            started=$((started + 1))
        else
            not_started=$((not_started + 1))
        fi
    done

    printf "%-10s %4d/%-4d %9d %10d %9d %11d\n" \
        "${component}" "${complete}" "${expected}" "${test_only}" \
        "${train_only}" "${started}" "${not_started}"
done

hc2_ready=0
for ((subject = first_subject; subject <= last_subject; subject++)); do
    ready=true
    for component in "${components[@]}"; do
        prediction_dir="${results_dir}/${component}/Predictions/${dataset}"
        if [[ ! -s "${prediction_dir}/testResample${subject}.csv" ||
              ! -s "${prediction_dir}/trainResample${subject}.csv" ]]; then
            ready=false
            break
        fi
    done
    if [[ "${ready}" == true ]]; then
        hc2_ready=$((hc2_ready + 1))
    fi
done

printf "\nHC2-from-file ready: %d/%d folds\n" "${hc2_ready}" "${expected}"

for selector in DetachRocket GEAR-Comp; do
    printf "\n%s component progress\n" "${selector}"
    printf "%-22s %9s %9s %10s %9s %11s\n" \
        "COMPONENT" "COMPLETE" "TEST_ONLY" "TRAIN_ONLY" "STARTED" "NOT_STARTED"

    selector_hc2_ready=0
    for component in "${components[@]}"; do
        pipeline="${selector}-${component}"
        complete=0
        test_only=0
        train_only=0
        started=0
        not_started=0

        for ((subject = first_subject; subject <= last_subject; subject++)); do
            prediction_dir="${results_dir}/${pipeline}/Predictions/${dataset}"
            test_file="${prediction_dir}/testResample${subject}.csv"
            train_file="${prediction_dir}/trainResample${subject}.csv"

            if [[ -s "${test_file}" && -s "${train_file}" ]]; then
                complete=$((complete + 1))
            elif [[ -s "${test_file}" ]]; then
                test_only=$((test_only + 1))
            elif [[ -s "${train_file}" ]]; then
                train_only=$((train_only + 1))
            elif compgen -G \
                "${results_dir}/output/${pipeline}/output-${dataset}-${subject}-*.txt" \
                > /dev/null; then
                started=$((started + 1))
            else
                not_started=$((not_started + 1))
            fi
        done

        printf "%-22s %4d/%-4d %9d %10d %9d %11d\n" \
            "${pipeline}" "${complete}" "${expected}" "${test_only}" \
            "${train_only}" "${started}" "${not_started}"
    done

    for ((subject = first_subject; subject <= last_subject; subject++)); do
        ready=true
        for component in "${components[@]}"; do
            prediction_dir="${results_dir}/${selector}-${component}/Predictions/${dataset}"
            if [[ ! -s "${prediction_dir}/testResample${subject}.csv" ||
                  ! -s "${prediction_dir}/trainResample${subject}.csv" ]]; then
                ready=false
                break
            fi
        done
        if [[ "${ready}" == true ]]; then
            selector_hc2_ready=$((selector_hc2_ready + 1))
        fi
    done
    printf "%s HC2-from-file ready: %d/%d folds\n" \
        "${selector}" "${selector_hc2_ready}" "${expected}"
done

printf "\nGEAR-Auto progress\n"
gear_auto_complete=0
gear_auto_started=0
for ((subject = first_subject; subject <= last_subject; subject++)); do
    test_file="${results_dir}/GEAR-Auto-HC2/Predictions/${dataset}/testResample${subject}.csv"
    if [[ -s "${test_file}" ]]; then
        gear_auto_complete=$((gear_auto_complete + 1))
    elif compgen -G \
        "${results_dir}/output/GEAR-Auto-HC2/output-${dataset}-${subject}-*.txt" \
        > /dev/null; then
        gear_auto_started=$((gear_auto_started + 1))
    fi
done
printf "GEAR-Auto-HC2:      %d/%d folds (%d started)\n" \
    "${gear_auto_complete}" "${expected}" "${gear_auto_started}"

mrhydra_complete=0
mrhydra_started=0
for ((subject = first_subject; subject <= last_subject; subject++)); do
    test_file="${results_dir}/MrHydra/Predictions/${dataset}/testResample${subject}.csv"
    if [[ -s "${test_file}" ]]; then
        mrhydra_complete=$((mrhydra_complete + 1))
    elif compgen -G \
        "${results_dir}/output/MrHydra/output-${dataset}-${subject}-*.txt" \
        > /dev/null; then
        mrhydra_started=$((mrhydra_started + 1))
    fi
done
printf "MrHydra complete:    %d/%d folds (%d started)\n" \
    "${mrhydra_complete}" "${expected}" "${mrhydra_started}"

printf "\nActive LOSO Slurm jobs\n"
printf "%-12s %-2s %-10s %-32s %s\n" "JOBID" "ST" "TIME" "NAME" "NODE/REASON"
squeue --user="${username}" --noheader \
    --format="%.12i %.2t %.10M %.32j %R" |
    grep -E "eeg-ocf-loso-" || true
