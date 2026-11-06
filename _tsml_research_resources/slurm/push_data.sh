#!/usr/bin/env bash
# Copy an archive to a cluster one dataset folder at a time, smallest first, over
# ssh. Safe to rerun: datasets already copied (and size-checked) are skipped, so a
# dropped VPN only costs the dataset that was in flight.
#
#   bash push_data.sh <local_dir> <user@host> <remote_dir>
#   bash push_data.sh /d/Data/EEG abc12xyz@hali.uea.ac.uk /gpfs/home/abc12xyz/Data/EEG
#
# Each folder goes as a gzip-compressed tar stream (.ts files are text and compress
# well), then the remote size is compared with the local one. Done datasets are
# recorded in <local_dir>/.pushed_<host>. Set up an ssh key first, or you are asked
# for a password for every dataset. SKIP lists folders that are never copied.
set -u
src=${1:?local dir}; dest=${2:?user@host}; remote=${3:?remote dir}
SKIP=${SKIP:-"_quarantine nm000245 ButtonPress"}
host=${dest#*@}
done_file="$src/.pushed_$host"
touch "$done_file"
cd "$src" || exit 1

ssh "$dest" "mkdir -p '$remote'" || { echo "cannot reach $dest"; exit 1; }

# dataset folders, smallest first
for name in $(du -s --apparent-size --block-size=1 -- */ 2>/dev/null \
    | sort -n | awk '{print $2}' | sed 's#/$##'); do
    case "$name" in .*) continue ;; esac
    case " $SKIP " in *" $name "*) continue ;; esac
    grep -qx "$name" "$done_file" && continue
    size=$(du -s --apparent-size --block-size=1 -- "$name" | cut -f1)
    echo "$(date +%H:%M) $name: $((size / 1000000)) MB"
    if tar cf - -- "$name" | gzip -1 | ssh "$dest" "cd '$remote' && tar xzf -"; then
        got=$(ssh "$dest" "cd '$remote' && du -s --apparent-size --block-size=1 -- '$name'" | cut -f1)
        if [ "$got" = "$size" ]; then
            echo "$name" >> "$done_file"
            echo "    done"
        else
            echo "    SIZE MISMATCH: local $size, remote ${got:-none}; rerun to retry"
        fi
    else
        echo "    FAILED; rerun to retry"
    fi
done
echo "$(wc -l < "$done_file") dataset(s) copied; list in $done_file"
