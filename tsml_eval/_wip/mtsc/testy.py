print("Hello world, up the arsenal")
from aeon.datasets import load_classification
from tsml_eval.experiments import load_and_run_classification_experiment
from pathlib import Path

path = str(Path("~/Data").expanduser())
results = str(Path("~/Results").expanduser())
data = "Chinatown"

load_and_run_classification_experiment(problem_path=path,results_path=results,
                                       dataset=data, classifier="Arsenal")

