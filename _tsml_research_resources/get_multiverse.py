from aeon.datasets import load_classification
from aeon.datasets.tsc_datasets import multiverse_core

path = "/gpfs/home/ajb/Data/Multiverse"

for s in multiverse_core:
    load_classification(s, extract_path=path)

