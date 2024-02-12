"""Working area for summarising results.

1. Stages in validating and collating results, using tsml tools to generate all
summary stats and extracting stats from these for the website

2. Using existing results on website for new comparison.

"""

from tsml_eval.evaluation.multiple_estimator_evaluation import \
     evaluate_clusterers_by_problem, evaluate_classifiers_by_problem, evaluate_regressors_by_problem
from aeon.datasets.tsc_data_lists import univariate_equal_length
from aeon.benchmarking import plot_critical_difference
from aeon.benchmarking.results_loaders import get_estimator_results

import pandas as pd
import numpy as np

import numpy as np

from aeon.classification.tests.test_classification_reproduction import _print_results_for_classifier
from aeon.transformations.tests.test_transformer_reproduction import _print_results_for_transformer


from aeon.datasets.tsc_data_lists import univariate_equal_length
import os
import glob

def count_files(directory, pattern):
    """Count the number of files in a directory matching a pattern."""
    path = os.path.join(directory, pattern)
    return len(glob.glob(path))

def count_lines(full_path):
    line_count = 0
    with open(full_path) as file:
        for line in file:
            line_count += 1
    return line_count

def count_complete(root, datasets, algorithms, resamples =30):
    """Count the number of results files in each directory."""
    table=[]
    # Function to count files matching the pattern in a directory
    for d in datasets:
        count =0
        counts = np.zeros(len(algorithms))
        for r in algorithms:
            # Creating a table of counts

            directory = f"{root}\\{r}\\Predictions\\{d}"
            counts[count]= count_files(directory, "TestResample*.csv")
            count=count+1
        table.append(counts)
    df = pd.DataFrame(counts, index =tser_new, columns=algorithms)
    # Compare each element to expected resample
    n_resamples = df >= resamples
    # Sum up the True values in each column
    counts = n_resamples.sum()
    return counts


def collate_by_algorithm(estimators, measures, destination, location):
    """Extract the individual files for all estimators/measures combinations."""
    ## Set up destination files
    if not os.path.exists(destination):
        os.makedirs(destination)
    for m in measures:
        m= m.lower()
        path = destination+"\\"+m
        if not os.path.exists(path):
            os.makedirs(path)
        for e in estimators:
            source = location + "\\" + m+"\\all_resamples\\"+e+"_"+m+".csv"
            dest = destination + "\\" + m+"\\"+e+"_"+m+".csv"
            shutil.copy(source,dest)



print(univariate_equal_length)
print(list(univariate_equal_length))

from aeon.benchmarking import get_estimator_results, get_estimator_results_as_array
est = ["STC","RDST","ROCKET","MrSQM"]
import pandas as pd

new_res = pd.read_csv("C:\\Temp\\Accuracy_mean.csv")
data_names = new_res.iloc[:,0].tolist()
cls_names = new_res.columns[1:].tolist()

print(cls_names)

new_results = new_res.iloc[:,3].values.tolist()

array = np.zeros((len(data_names),len(est)+1))
published_results = get_estimator_results(estimators=est, default_only=False)
for i in range(len(data_names)):
    for j in range(len(est)):
        array[i][j] =np.average(published_results[est[j]][data_names[i]])
    array[i][len(est)] = new_results[i]
from aeon.benchmarking import plot_critical_difference
est.append(cls_names[2])
cd = plot_critical_difference(array, est)
cd.show()

print(est)

print(np.mean(array,axis=0))


def compare_to_reference(new_path, est):
    """Compare new results to reference results."""
    new_res = pd.read_csv(new_path)
    data_names = new_res.iloc[:,0].tolist()
    cls_names = new_res.columns[1:].tolist()
    new_results = new_res.iloc[:,1:].values
    array = np.zeros((len(data_names),len(est)+len(cls_names)))
    published_results = get_estimator_results(estimators=est, default_only=False)
    for i in range(len(data_names)):
     for j in range(len(est)):
         array[i,j] =np.average(published_results[est[j]][data_names[i]])
     for j in range(len(cls_names)):
          array[i,len(est)+j] = new_results[i,j]

    for i in range(len(cls_names)):
     est.append(cls_names[i])

    cd = plot_critical_difference(array, est)
    cd.show()


if __name__ == "__main__":
    print("AFC")
