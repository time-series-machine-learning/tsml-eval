"""Write <Name>_id_TRAIN.txt subject ID files for the EEG archive.

One subject ID per TRAIN case, in TRAIN file order, for CTNet-NB's subject-wise
early stopping split (see slurm/experiments/hali_eeg_ctnet.toml). The IDs come
from, in order: an existing <Name>_id_TRAIN.txt (left as it is), cases_train.tsv,
<Name>_rows.tsv (rows with split == train, in order), cases_train.parquet.

A file is written only when the IDs are usable: none blank or containing
whitespace, more than one subject, and as many IDs as TRAIN cases where that count
is known (from the metadata sidecar, or by counting a .ts file under 2 GB).
Existing files are never changed. Without --write it only reports.

    python make_train_groups.py ~/Data/EEG            # report
    python make_train_groups.py ~/Data/EEG --write    # write missing files

At the end it lists the files eeg_ctnet_info.csv refers to that do not exist.
"""

import csv
import json
import sys
from pathlib import Path

SKIP = {"_quarantine", "ButtonPress", "nm000245"}
INFO = Path(__file__).with_name("eeg_ctnet_info.csv")


def find_count(obj):
    """Search a metadata sidecar for the number of TRAIN cases."""
    if isinstance(obj, dict):
        for key in ("train_cases", "n_train", "n_train_cases", "train_size"):
            value = obj.get(key)
            if isinstance(value, (int, str)) and str(value).isdigit():
                return int(value)
        shapes = obj.get("shapes")
        if isinstance(shapes, dict) and isinstance(shapes.get("train"), list):
            return int(shapes["train"][0])
        for value in obj.values():
            count = find_count(value)
            if count is not None:
                return count
    return None


def train_count(folder, name):
    """Return (number of TRAIN cases or None, where it came from)."""
    meta = folder / f"{name}_metadata.json"
    if meta.exists():
        try:
            with open(meta, encoding="utf-8") as file:
                count = find_count(json.load(file))
            if count is not None:
                return count, "metadata"
        except (OSError, ValueError):
            pass
    ts = folder / f"{name}_TRAIN.ts"
    if ts.exists() and ts.stat().st_size < 2e9:
        count, data = 0, False
        with open(ts, encoding="utf-8", errors="replace") as file:
            for line in file:
                if data:
                    count += bool(line.strip())
                elif line.strip().lower().startswith("@data"):
                    data = True
        return count, "ts"
    return None, "not checked"


def _participants(path, train_only):
    with open(path, encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file, delimiter="\t"))
    if not rows or "participant_id" not in rows[0]:
        return None
    if train_only:
        rows = [r for r in rows if r.get("split", "").lower() == "train"]
    return [r["participant_id"] for r in rows]


def read_ids(folder, name):
    """Return (IDs or None, source)."""
    path = folder / f"{name}_id_TRAIN.txt"
    if path.exists():
        return path.read_text(encoding="utf-8").split(), "id file"
    path = folder / "cases_train.tsv"
    if path.exists():
        ids = _participants(path, train_only=False)
        if ids is not None:
            return ids, "cases_train.tsv"
    path = folder / f"{name}_rows.tsv"
    if path.exists():
        ids = _participants(path, train_only=True)
        if ids is not None:
            return ids, "rows.tsv"
    path = folder / "cases_train.parquet"
    if path.exists():
        import pandas as pd

        frame = pd.read_parquet(path, columns=["participant_id"])
        return frame["participant_id"].astype(str).tolist(), "cases_train.parquet"
    return None, "none"


def main(argv):
    """Report on, and with --write create, the ID files under a data directory."""
    root = Path(argv[1]).expanduser()
    write = "--write" in argv
    folders = sorted(
        p
        for p in root.iterdir()
        if p.is_dir() and p.name not in SKIP and not p.name.startswith(".")
    )
    for folder in folders:
        name = folder.name
        ids, source = read_ids(folder, name)
        if ids is None:
            print(f"{name}: no subject IDs")
            continue
        bad = sum(
            1
            for i in ids
            if not i.strip() or i.split() != [i] or i.lower() in ("nan", "none")
        )
        count, how = train_count(folder, name)
        matches = count is None or count == len(ids)
        usable = matches and bad == 0 and len(set(ids)) > 1
        note = f"{len(ids)} IDs, {len(set(ids))} subjects, {bad} bad; "
        note += f"TRAIN cases {count} ({how})"
        if not matches:
            note += " MISMATCH"
        action = ""
        if usable and source != "id file":
            action = "write" if write else "would write"
            if write:
                text = "\n".join(ids) + "\n"
                (folder / f"{name}_id_TRAIN.txt").write_text(text, encoding="utf-8")
        print(f"{name}: {source}, {note}{', ' + action if action else ''}")
        if not usable:
            print(f"  NOT USABLE: {name}")
    if INFO.exists():
        with open(INFO, encoding="utf-8", newline="") as file:
            missing = [
                r["dataset"]
                for r in csv.DictReader(file)
                if r["train_groups"] != "none"
                and not (root / r["dataset"] / f"{r['dataset']}_id_TRAIN.txt").exists()
            ]
        print(f"{INFO.name}: {len(missing)} ID file(s) missing", *missing)


if __name__ == "__main__":
    main(sys.argv)
