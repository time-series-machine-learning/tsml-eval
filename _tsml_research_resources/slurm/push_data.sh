#!/usr/bin/env bash
# Copy an archive to a cluster one dataset folder at a time, smallest first, over
# ssh. Safe to rerun: datasets already copied (and size-checked) are skipped, so a
# dropped VPN only costs the dataset that was in flight.
#
#   bash push_data.sh <local_dir> <user@host> <remote_dir>
#   bash push_data.sh /d/Data/EEG abc12xyz@hali.uea.ac.uk /gpfs/home/abc12xyz/Data/EEG
#
# Each folder goes as a gzip-compressed tar stream (.ts files are text and compress
# well), then the total size of its files on the cluster is compared with the
# local total, in the same connection. Done datasets are recorded in
# <local_dir>/.pushed_<host>. Set up an ssh key first, or you are asked for a
# password for every dataset. SKIP lists folders that are never copied.
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
    size=$(find "$name" -type f -printf '%s\n' | awk '{s+=$1} END {print s+0}')
    echo "$(date +%H:%M) $name: $((size / 1000000)) MB"
    # one connection per dataset (logins can be slow): unpack, then print the size
    got=$(tar cf - -- "$name" | gzip -1 | ssh "$dest" \
        "cd '$remote' && tar xzf - && find '$name' -type f -printf '%s\n' | awk '{s+=\$1} END {print s+0}'")
    if [ "$got" = "$size" ]; then
        echo "$name" >> "$done_file"
        echo "    done"
    else
        echo "    FAILED or SIZE MISMATCH: local $size, remote ${got:-none}; rerun to retry"
    fi
done
echo "$(wc -l < "$done_file") dataset(s) copied; list in $done_file"
