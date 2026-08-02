#!/bin/bash

set -euo pipefail

# One-shot migration of active Iridis result directories from the development
# name GMARv5 to the paper name GEAR. Historical prediction-file metadata and
# batch-submission logs are intentionally retained unchanged as provenance.

username="${USER:-ajb2u23}"
results_root="/iridisfs/home/${username}/Results"

if command -v squeue > /dev/null 2>&1; then
    active_gmarv5_jobs=$(
        squeue --noheader --user="${username}" --states=RUNNING,PENDING \
            --format="%j" | grep -i -c "gmarv5" || true
    )
    if ((active_gmarv5_jobs > 0)); then
        echo "ERROR: ${active_gmarv5_jobs} GMARv5 Slurm job(s) are still active."
        echo "Wait for them to finish before migrating result directories."
        exit 1
    fi
fi

moves=(
    "ChannelSelectionPipeline/GMARv5-HC2|ChannelSelectionPipeline/GEAR-Auto-HC2"
    "ChannelSelectionPipeline/GMARv5-Arsenal|ChannelSelectionPipeline/GEAR-Comp-Arsenal"
    "ChannelSelectionPipeline/GMARv5-DrCIF|ChannelSelectionPipeline/GEAR-Comp-DrCIF"
    "ChannelSelectionPipeline/GMARv5-STC|ChannelSelectionPipeline/GEAR-Comp-STC"
    "ChannelSelectionPipeline/GMARv5-TDE|ChannelSelectionPipeline/GEAR-Comp-TDE"
    "ChannelSelectionPipeline/output/GMARv5-HC2|ChannelSelectionPipeline/output/GEAR-Auto-HC2"
    "ChannelSelectionPipeline/output/GMARv5-Arsenal|ChannelSelectionPipeline/output/GEAR-Comp-Arsenal"
    "ChannelSelectionPipeline/output/GMARv5-DrCIF|ChannelSelectionPipeline/output/GEAR-Comp-DrCIF"
    "ChannelSelectionPipeline/output/GMARv5-STC|ChannelSelectionPipeline/output/GEAR-Comp-STC"
    "ChannelSelectionPipeline/output/GMARv5-TDE|ChannelSelectionPipeline/output/GEAR-Comp-TDE"
    "ChannelSelectionLOSO/GMARv5-Arsenal|ChannelSelectionLOSO/GEAR-Comp-Arsenal"
    "ChannelSelectionLOSO/GMARv5-DrCIF|ChannelSelectionLOSO/GEAR-Comp-DrCIF"
    "ChannelSelectionLOSO/GMARv5-STC|ChannelSelectionLOSO/GEAR-Comp-STC"
    "ChannelSelectionLOSO/GMARv5-TDE|ChannelSelectionLOSO/GEAR-Comp-TDE"
    "ChannelSelectionLOSO/output/GMARv5-Arsenal|ChannelSelectionLOSO/output/GEAR-Comp-Arsenal"
    "ChannelSelectionLOSO/output/GMARv5-DrCIF|ChannelSelectionLOSO/output/GEAR-Comp-DrCIF"
    "ChannelSelectionLOSO/output/GMARv5-STC|ChannelSelectionLOSO/output/GEAR-Comp-STC"
    "ChannelSelectionLOSO/output/GMARv5-TDE|ChannelSelectionLOSO/output/GEAR-Comp-TDE"
)

for entry in "${moves[@]}"; do
    IFS="|" read -r source_relative target_relative <<< "${entry}"
    source="${results_root}/${source_relative}"
    target="${results_root}/${target_relative}"

    if [[ ! -e "${source}" ]]; then
        if [[ -e "${target}" ]]; then
            echo "Already migrated: ${target_relative}"
        else
            echo "Not present:      ${source_relative}"
        fi
        continue
    fi

    if [[ -e "${target}" ]]; then
        echo "ERROR: both source and target exist:"
        echo "  ${source}"
        echo "  ${target}"
        exit 1
    fi

    mv -- "${source}" "${target}"
    echo "Renamed: ${source_relative} -> ${target_relative}"
done

echo "GEAR result-directory migration complete."
