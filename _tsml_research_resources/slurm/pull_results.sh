#!/usr/bin/env bash
# Pull new or changed files from a cluster results directory, over ssh, without
# rsync (Git Bash on Windows has none). Two logins in all: one lists the remote
# files with their sizes, one streams the missing or changed ones as a gzip tar.
# Safe to rerun; nothing local is deleted and nothing remote is changed.
#
#   bash pull_results.sh <user@host> <remote_dir> <local_dir> [pattern]
#   bash pull_results.sh ajb2u23@iridis6.soton.ac.uk Results/EEG /d/Results/EEG
#
# remote_dir is relative to the remote home unless it starts with /. pattern is a
# find -path pattern relative to remote_dir; the default takes Predictions only,
# skipping Checkpoints (large, deleted when a task finishes) and Workspace.
set -u
dest=${1:?user@host}; remote=${2:?remote dir}; local=${3:?local dir}
pattern=${4:-"./*/*/Predictions/*"}
mkdir -p "$local" || exit 1
SSH=(ssh -o BatchMode=yes -o ServerAliveInterval=60)
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT

echo "$(date +%H:%M) listing $dest:$remote"
"${SSH[@]}" "$dest" "cd '$remote' && find . -path '$pattern' -type f -printf '%s %p\n'" \
    > "$tmp/remote" || { echo "cannot list $dest:$remote"; exit 1; }
(cd "$local" && find . -path "$pattern" -type f -printf '%s %p\n') > "$tmp/local"
# files absent locally or of a different size; FILENAME, not NR==FNR, so an
# empty local list still works
awk 'FILENAME == ARGV[1] {have[$2]=$1; next} !($2 in have) || have[$2] != $1 {print $2}' \
    "$tmp/local" "$tmp/remote" > "$tmp/need"
n=$(wc -l < "$tmp/need")
echo "$(wc -l < "$tmp/remote") remote files, $n to fetch"
[ "$n" -eq 0 ] && exit 0
"${SSH[@]}" "$dest" "cd '$remote' && tar cf - -T - | gzip -1" < "$tmp/need" \
    | gzip -dc | tar xf - -C "$local" || { echo "transfer failed; rerun"; exit 1; }
echo "$(date +%H:%M) fetched $n file(s) into $local"
