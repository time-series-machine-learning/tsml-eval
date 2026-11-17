"""Build the completed-results registry that tsml_slurm.py reads.

tsml_slurm.py counts a task as done when its result files exist or are listed
in <results_dir>/completed.txt. Build the registry from the master copy of the
results (the desktop), then copy it to the same results directory on every
machine. Results can then be deleted from a machine short of space without the
runner submitting them again.

The registry only grows: entries already in the output file are kept even if
their files are no longer under the scanned directory.

    python registry.py D:/Results/EEG
    python registry.py D:/Results/EEG --dry-run

Then, for example:

    scp D:/Results/EEG/completed.txt ajb@hali.uea.ac.uk:Results/EEG/
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tsml_slurm import REGISTRY_NAME  # noqa: E402

PATTERNS = ("*/*/Predictions/*/*Resample*.csv", "*/Predictions/*/*Resample*.csv")


def scan(results_dir):
    """Relative paths of every non-empty result file under results_dir."""
    found = set()
    for pattern in PATTERNS:
        for path in results_dir.glob(pattern):
            if path.is_file() and path.stat().st_size > 0:
                found.add(path.relative_to(results_dir).as_posix())
    return found


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("results_dir", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        help=f"registry to update (default <results_dir>/{REGISTRY_NAME})",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    output = args.output or args.results_dir / REGISTRY_NAME
    old = set()
    if output.is_file():
        old = {
            line.strip()
            for line in output.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        }
    new = scan(args.results_dir)
    merged = old | new
    print(f"{len(old)} in registry, {len(new)} result files found, "
          f"{len(merged - old)} added, {len(merged)} total")
    if args.dry_run:
        return 0
    tmp = output.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8", newline="\n") as file:
        file.write("# Result files known to be complete; see registry.py\n")
        file.writelines(f"{line}\n" for line in sorted(merged))
    tmp.replace(output)
    print(f"wrote {output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
