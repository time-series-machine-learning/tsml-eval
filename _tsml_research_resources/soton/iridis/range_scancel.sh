#!/bin/bash

# start and end job IDs to cancel
start=7615996
end=7616025

for ((n=start; n<=end; n++)); do
    echo "Cancelling job ID $n"
    if ! scancel "$n" 2>/dev/null; then
        echo "Failed to cancel job ID $n"
    fi
done
