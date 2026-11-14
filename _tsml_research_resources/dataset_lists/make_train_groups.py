"""Write <Name>_id_TRAIN.txt and <Name>_id_TEST.txt subject ID files for the EEG archive.

One subject ID per case, in file order. The TRAIN files are for CTNet-NB's
subject-wise early stopping split (see slurm/experiments/hali_eeg_ctnet.toml); the
TEST files complete the pair that subject-wise cross-validation (-cv) needs. For a
split, the IDs come from, in order: an existing <Name>_id_<SPLIT>.txt (left as it
is), cases_<split>.tsv, <Name>_rows.tsv (rows with that split, in order),
cases_<split>.parquet.

A file is written only when the IDs are usable: none blank or containing
whitespace, and as many IDs as cases where that count is known (from the metadata
sidecar, or by counting a .ts file under 2 GB). A TRAIN file also needs more than
one subject; a TEST file may hold one. Existing files are never changed. Without
--write it only reports.

    python make_train_groups.py ~/Data/EEG                        # report TRAIN
    python make_train_groups.py ~/Data/EEG --write                # write missing TRAIN
    python make_train_groups.py ~/Data/EEG --split test --write   # write missing TEST
    python make_train_groups.py ~/Data/EEG --datasets A,B         # only these folders

For TRAIN over the whole archive, it ends by listing the files eeg_ctnet_info.csv
refers to that do not exist.
"""

import csv
import json
import sys
from pathlib import Path

SKIP = {"_quarantine", "ButtonPress", "nm000245"}
INFO = Path(__file__).with_name("eeg_ctnet_info.csv")


def find_count(obj, split="train"):
    """Search a metadata sidecar for the number of cases in a split."""
    if isinstance(obj, dict):
        for key in (f"{split}_cases", f"n_{split}", f"n_{split}_cases", f"{split}_size"):
            value = obj.get(key)
            if isinstance(value, (int, str)) and str(value).isdigit():
                return int(value)
        shapes = obj.get("shapes")
        if isinstance(shapes, dict) and isinstance(shapes.get(split), list):
            return int(shapes[split][0])
        for value in obj.values():
            count = find_count(value, split)
            if count is not None:
                return count
    return None


def case_count(folder, name, split="train"):
    """Return (number of cases in the split or None, where it came from)."""
    meta = folder / f"{name}_metadata.json"
    if meta.exists():
        try:
            with open(meta, encoding="utf-8") as file:
                count = find_count(json.load(file), split)
            if count is not None:
                return count, "metadata"
        except (OSError, ValueError):
            pass
    ts = folder / f"{name}_{split.upper()}.ts"
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


def _participants(path, split=None):
    with open(path, encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file, delimiter="\t"))
    if not rows or "participant_id" not in rows[0]:
        return None
    if split:
        rows = [r for r in rows if r.get("split", "").lower() == split]
    return [r["participant_id"] for r in rows]


def read_ids(folder, name, split="train"):
    """Return (IDs or None, source)."""
    path = folder / f"{name}_id_{split.upper()}.txt"
    if path.exists():
        return path.read_text(encoding="utf-8").split(), "id file"
    path = folder / f"cases_{split}.tsv"
    if path.exists():
        ids = _participants(path)
        if ids is not None:
            return ids, path.name
    path = folder / f"{name}_rows.tsv"
    if path.exists():
        ids = _participants(path, split)
        if ids is not None:
            return ids, "rows.tsv"
    path = folder / f"cases_{split}.parquet"
    if path.exists():
        import pandas as pd

        frame = pd.read_parquet(path, columns=["participant_id"])
        return frame["participant_id"].astype(str).tolist(), path.name
    return None, "none"


def main(argv):
    """Report on, and with --write create, the ID files under a data directory."""
    root = Path(argv[1]).expanduser()
    write = "--write" in argv
    split = argv[argv.index("--split") + 1].lower() if "--split" in argv else "train"
    if split not in ("train", "test"):
        raise SystemExit("--split must be train or test")
    only = None
    if "--datasets" in argv:
        only = set(argv[argv.index("--datasets") + 1].split(","))
    folders = sorted(
        p
        for p in root.iterdir()
        if p.is_dir()
        and p.name not in SKIP
        and not p.name.startswith(".")
        and (only is None or p.name in only)
    )
    for folder in folders:
        name = folder.name
        ids, source = read_ids(folder, name, split)
        if ids is None:
            print(f"{name}: no subject IDs")
            continue
        bad = sum(
            1
            for i in ids
            if not i.strip() or i.split() != [i] or i.lower() in ("nan", "none")
        )
        count, how = case_count(folder, name, split)
        matches = count is None or count == len(ids)
        usable = matches and bad == 0 and (split == "test" or len(set(ids)) > 1)
        note = f"{len(ids)} IDs, {len(set(ids))} subjects, {bad} bad; "
        note += f"{split.upper()} cases {count} ({how})"
        if not matches:
            note += " MISMATCH"
        action = ""
        if usable and source != "id file":
            action = "write" if write else "would write"
            if write:
                text = "\n".join(ids) + "\n"
                out = folder / f"{name}_id_{split.upper()}.txt"
                out.write_text(text, encoding="utf-8")
        print(f"{name}: {source}, {note}{', ' + action if action else ''}")
        if not usable:
            print(f"  NOT USABLE: {name}")
    if INFO.exists() and split == "train" and only is None:
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
